# -*- coding: utf-8 -*-
"""SCAPE探索 (Structured Cached Abliteration with Prefix Evaluation)。

3つの柱:
  1. データ駆動の初期化: 層顕著度 (salience) のピークからカーネル中心を決める
  2. 構造化信任領域座標探索: 少数の意味ある座標 (中心・半径・強度・方向) に対して
     パターンサーチ+適応ステップで探索し、行き詰まるとダター再始動する
  3. 弾性キャッシュ評価: surrogate.FastEvaluator による高速評価

目的関数:  J = KL + λ * max(0, refusal_soft - target)^2
(目標拒否率を達成しつつ、モデル品質の劣化 (KL) を最小化する制約付き最適化)

v3の重点修正 (チェックポイント, 条件⑬):
  - チェックポイントを「1試行ごと」に必ず保存する (v2は初期化フェーズ終了後
    まで保存されず、初期化だけで数時間かかるCPU環境では <workdir>/cache/runs が
    一切作られない問題があった)。
  - 探索開始時に runs/ ディレクトリを即時作成し、チェックポイントのパスを
    冒頭で表示する。
  - 各試行の進捗行 (n/N + 経過 + ETA + 現在のbest) を表示。
  - --time-budget-min で時間予算に達した時点で最良解を確定して検証へ進む。
  - Ctrl+C で探索を中断してもチェックポイントは保存済み (再実行で自動再開)。
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Callable

import torch

from .engine import AblationParams, KernelParams, PatchEngine
from .msg import tr
from .surrogate import FastEvaluator
from .ui import TrialProgress, fmt_duration, print_line


def _is_oom_error(e: BaseException) -> bool:
    if isinstance(e, MemoryError):
        return True
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


class ScapeSearch:
    """SCAPEアルゴリズムによる非検閲化パラメータ探索。"""

    def __init__(self, engine: PatchEngine, evaluator: FastEvaluator,
                 salience: torch.Tensor, checkpoint_path: Path,
                 trials: int, target_refusal: float, penalty: float,
                 seed: int = 0, components: str = "both",
                 focus_init: AblationParams | None = None,
                 focus_radius_frac: float = 0.15,
                 use_cache: bool = True, quiet: bool = True,
                 monitor: Callable[[], str] | None = None,
                 time_budget_sec: float = 0.0,
                 live_cb: Callable[[str], None] | None = None,
                 on_ready: Callable[[], None] | None = None):
        self.engine = engine
        self.ev = evaluator
        self.on_ready = on_ready   # [v7] 探索開始直前 (評価構造確定後) に1回だけ呼ぶ
        self.salience = salience
        self.ckpt_path = checkpoint_path
        self.trials = trials
        self.target = target_refusal
        self.penalty = penalty
        self.components = components
        self.use_cache = use_cache
        self.quiet = quiet
        self.monitor = monitor
        self.time_budget_sec = float(time_budget_sec or 0.0)
        self.rng = random.Random(seed)

        # [v9①] 代理指標の「文法床」較正 (シードグリッド評価後に確定する)。
        # 質量加重多位置ソフト拒否率は、基底モデル自身の拒否文を教師強制する
        # 限り完全に非検閲化されたモデルでも ~0.3-0.4 の拒否語彙質量が残る
        # (拒否文を文法的に続けるための "I" / "As" / "However" 等が拒否語彙
        # リストに含まれるため)。目標 r* がこの床より下にあると J の罰則項が
        # 床で飽和し、探索は真の拒否低減の方向へ勾配を受け取れなくなる。
        # シードグリッド (全層一様の強い直交化を含む) の実測最小値を床とし、
        # 実効目標 target_eff = max(r*, 床+0.03) で J を再定義する。
        self.surrogate_floor: float | None = None
        self.target_eff = float(target_refusal)

        self.n_layers = engine.registry.n_layers
        L = self.n_layers

        # [v3] runs/を即時作成し、中断・OOM後も必ずチェックポイントが書ける状態にする
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        # 座標ごとの探索範囲
        self.bounds = {
            "center": (0.3 * L, 0.98 * L),
            "radius": (1.5, 0.45 * L),
            "strength": (0.0, 2.5),
            "direction": (0.0, max(0.0, L - 1)),
        }
        # [v9①] flatカーネルの半径上限は全レイヤを覆える値 (center=L/2,
        # radius=L でレイヤ0..L-1全体に一様強度 = Arditi et al. の全層直交化)。
        self.flat_radius_hi = float(max(0.45 * L, L))
        # 基本ステップ幅
        self.base_step = {
            "center": max(1.0, 0.06 * L),
            "radius": max(1.0, 0.05 * L),
            "strength": 0.2,
            "direction": max(1.0, 0.08 * L),
        }

        # 初期パラメータ: 顕著度ピーク (データ駆動) を中心に
        peak = int(torch.argmax(salience[1:]).item()) + 1  # レイヤ番号 (1..L)
        comps = ["attn_out", "mlp_down"] if components == "both" else (
            ["attn_out"] if components == "attn" else ["mlp_down"])
        init_kernels = {}
        for c in comps:
            kp = KernelParams(center=float(peak), radius=float(max(2.0, 0.15 * L)), strength=0.0)
            init_kernels[c] = kp
        self.init_params = AblationParams(
            direction_mode="per_layer", direction_index=0.0, kernels=init_kernels)

        # 外部パラメータ (focusモード) の初期値上書きと探索範囲の絞り込み
        self.comp_bounds: dict[tuple[str, str], tuple[float, float]] = {}
        self.focus_init = focus_init
        if focus_init is not None:
            self.init_params = focus_init
            for c, kp in focus_init.kernels.items():
                self.comp_bounds[(c, "center")] = (
                    max(self.bounds["center"][0], kp.center * (1 - focus_radius_frac)),
                    min(self.bounds["center"][1], kp.center * (1 + focus_radius_frac)))
                self.comp_bounds[(c, "radius")] = (
                    max(self.bounds["radius"][0], kp.radius * (1 - focus_radius_frac)),
                    min(self.bounds["radius"][1], kp.radius * (1 + focus_radius_frac)))
                self.comp_bounds[(c, "strength")] = (
                    max(0.0, kp.strength * (1 - 2 * focus_radius_frac)),
                    min(self.bounds["strength"][1], kp.strength * (1 + 2 * focus_radius_frac)))
            if focus_init.direction_mode == "global":
                di = focus_init.direction_index
                self.bounds["direction"] = (max(0.0, di - 2), min(L - 1, di + 2))

        self.history: list[dict[str, Any]] = []
        self.best_params: AblationParams | None = None
        self.best_metrics: dict[str, Any] | None = None
        self.best_J = float("inf")
        self.attempt = 0
        self.interrupted = False
        self.budget_stopped = False

        # 評価中のlive進捗: v5はcliの生書き込みハックを廃止し、
        # TrialProgress自身が (残骸なしの) live行と1秒カウントダウンを描画する。
        # [v7] live_cbは (text, eta) を受け取る (条件⑤: 評価残りの推定を表示)。
        self.ev.live_cb = live_cb if live_cb is not None else (
            (lambda text, eta=None: self.progress.live_step(text, eta)) if not quiet else None)

        # 試行ごとの統計行UI
        self.progress = TrialProgress(trials, quiet=quiet)

    # ------------------------------------------------------------ 目的関数

    def objective(self, refusal_soft: float, kl: float) -> float:
        over = max(0.0, refusal_soft - self.target_eff)
        return kl + self.penalty * over * over

    def objective_lower_bound(self, refusal_lb: float) -> float:
        """Jの厳密下界 (KL≥0を利用)。早期スクリーニングで使う。"""
        over = max(0.0, refusal_lb - self.target_eff)
        return self.penalty * over * over

    # ------------------------------------------------------------ [v5] 早期スクリーニング

    def _screen_refusal(self, refusal_lb: float) -> tuple[bool, float]:
        """評価器から呼ばれる: 拒否率下界でJ下界を計算し、絶望的なら棄却する。

        J = KL + λ·max(0, refusal−target)² ≥ λ·max(0, refusal_lb−target)² なので、
        この下界が現best_J以上なら候補はbestを上回れない (数学的に厳密)。
        """
        if self.best_metrics is None or self.best_J == float("inf"):
            return (False, 0.0)
        j_lb = self.objective_lower_bound(refusal_lb)
        return (j_lb >= self.best_J - 1e-12, j_lb)

    def _screen_kl_partial(self, kl_partial: float) -> bool:
        """評価器から呼ばれる: KL第1バッチの部分値で絶望的に悪い候補を棄却。

        ヒューリスティック (バッチ間のKL差は小さいという経験則)。誤って
        棄却した場合でもbest更新に影響はない (bestになり得ない候補だけを切る)。
        """
        if self.best_metrics is None or self.best_J == float("inf"):
            return False
        return kl_partial > max(4.0 * self.best_J, self.best_J + 0.15)

    def _install_screens(self, enabled: bool) -> None:
        self.ev.screen_refusal_fn = self._screen_refusal if enabled else None
        self.ev.screen_kl_fn = self._screen_kl_partial if enabled else None

    # ------------------------------------------------------------ [v5] 多境界キャッシュ

    def _ensure_boundary_cache(self) -> None:
        """探索開始時 (とメモリ圧で解放された直後) に1回だけスイープする。

        スイープは摂動なしのモデルで行う必要があるため suspend_patches で囲む。
        メモリ圧でキャッシュが解放された場合は次の試行適用前に再構築される。
        """
        try:
            if not getattr(self.ev, "needs_boundary_sweep", lambda: False)():
                return
            with self.engine.suspend_patches():
                self.ev.build_boundary_cache()
        except Exception:
            # スイープ失敗は致命的ではない (フルフォワードに退化するだけ)
            pass

    def _clip(self, name: str, value: float, comp: str | None = None,
              kp: KernelParams | None = None) -> float:
        if comp is not None and (comp, name) in self.comp_bounds:
            lo, hi = self.comp_bounds[(comp, name)]
        else:
            lo, hi = self.bounds[name]
        # [v9①] flatカーネルは帯全域を覆えるよう半径上限を拡張する
        if (name == "radius" and kp is not None and kp.shape == "flat"
                and not (comp is not None and (comp, name) in self.comp_bounds)):
            hi = max(hi, self.flat_radius_hi)
        return float(max(lo, min(hi, value)))

    def _params_variant(self, base: AblationParams, comp: str, coord: str, delta: float) -> AblationParams:
        """baseの指定座標をdeltaだけ動かした新しいパラメータを作る。"""
        new = AblationParams.from_dict(base.to_dict())
        if comp is None:
            if coord == "direction":
                if new.direction_mode == "per_layer":
                    new.direction_mode = "global"
                    new.direction_index = self._clip("direction", self.init_params.kernels.get(
                        "attn_out", KernelParams()).center)
                else:
                    new.direction_index = self._clip("direction", new.direction_index + delta)
            elif coord == "direction_mode":
                new.direction_mode = "global" if base.direction_mode == "per_layer" else "per_layer"
            return new
        kp = new.kernels.get(comp)
        if kp is None:
            return new
        if coord == "center":
            kp.center = self._clip("center", kp.center + delta, comp, kp)
        elif coord == "radius":
            kp.radius = self._clip("radius", kp.radius + delta, comp, kp)
        elif coord == "strength":
            kp.strength = self._clip("strength", kp.strength + delta, comp, kp)
        elif coord == "toggle":
            kp.strength = 0.0 if kp.strength > 0 else 1.0
        return new

    # ------------------------------------------------------------ 評価ラッパ

    def _evaluate_params(self, params: AblationParams) -> dict[str, Any]:
        """候補を安全に評価する。OOM時はメモリ削減して再試行、
        非有限重みなど不適格な候補はペナルティ扱い (J=inf) として記録する。

        [v5] スクリーニング棄却された候補は KL=NaN / screened=True で返り、
        JにはJ下界 (またはinf) を使う。best更新に影響しない。
        """
        try:
            self.engine.reset()
            self.engine.apply(params)
            m = self.ev.evaluate(params, use_cache=self.use_cache)
        except Exception as e:  # noqa: BLE001
            if _is_oom_error(e):
                self.ev.reduce_memory(True)
                try:
                    self.engine.reset()
                    self.engine.apply(params)
                    m = self.ev.evaluate(params, use_cache=self.use_cache)
                except Exception:
                    m = None
            else:
                # [v5] 失敗理由を1行だけ表示 (デバッグしやすさ。継続はする)
                print_line(tr("eval_failed_why", a=self.attempt + 1,
                              t=type(e).__name__, m=str(e)[:120]))
                m = None
        finally:
            self.engine.reset()
        if m is None:
            m = {"refusal_soft": 1.0, "refusal_first": 1.0, "kl": 9.9, "kl_first": 9.9,
                 "nll_ratio": None,
                 "margin_mean": 0.0, "band_start": params.band_start(self.n_layers),
                 "eval_seconds": 0.0, "failed": True}
        if m.get("screened"):
            # 棄却済み: Jは下界値 (下界≥bestなのでbestが更新されることはない)
            m["J"] = m.get("j_lower_bound", float("inf"))
            if not (m["J"] < float("inf")):
                m["J"] = float("inf")
        else:
            m["J"] = self.objective(m["refusal_soft"], m["kl"])
        m["params"] = params.to_dict()
        return m

    def _run_one_eval(self, cand: AblationParams, phase: str) -> dict[str, Any]:
        """1候補を評価し、履歴記録・best更新・チェックポイント保存・進捗表示を
        すべて行う (v3: ここを一本化したことで1試行ごとの保存が保証される)。

        [v5] 試行ごとに境界キャッシュの解放を確認し、必要なら再スイープする
        (摂動適用前なので suspend なしで安全)。フック方式のapply/resetは
        数ミリ秒のため逐次実行しても問題にならない。
        """
        self._ensure_boundary_cache()
        m = self._evaluate_params(cand)
        self.attempt += 1
        self.history.append(self._record(m, phase=phase))
        self._maybe_update_best(m, cand)
        self._save_ckpt()
        self._tick_monitor()
        self._report_trial(m)
        return m

    def _report_trial(self, m: dict[str, Any]) -> None:
        """試行統計行 (n/N, 所要, 指標, best, 経過, ETA)。"""
        if self._first_report_done is False:
            self._first_report_done = True
            rate = m.get("eval_seconds", 0.0) or 1.0
            est = rate * self.trials
            if self.trials > 1 and not m.get("cached") and not m.get("screened"):
                print_line(tr("search_est_total", t=fmt_duration(est), r=rate))
        self.progress.trial(
            self.attempt, m.get("eval_seconds", 0.0),
            m.get("refusal_soft", 1.0), m.get("kl", 9.9), m.get("J", float("inf")),
            self.best_J if self.best_metrics else float("nan"),
            (self.best_metrics or {}).get("attempt", 0),
            cached=bool(m.get("cached")), screened=bool(m.get("screened")),
            refusal_first=m.get("refusal_first"))

    _first_report_done: bool = False

    def _tick_monitor(self) -> None:
        """メモリ監視フック。

        [v7] v6はswap増加の警告で評価キャッシュ (多境界プレフィックス)
        を破棄し、次の試行で6分の再スイープを毎回発生させていた。ここでは
        何も破棄しない (危険時のINT4重みキャッシュ縮小はMemoryGuard側で
        行う)。キャッシュの喪失はOOM再試行 (_evaluate_params) のみ。
        """
        if self.monitor is None:
            return
        try:
            self.monitor()
        except Exception:
            pass

    # ------------------------------------------------------------ チェックポイント

    def _save_ckpt(self) -> None:
        """チェックポイントを原子書き込みで保存する (1試行ごとに呼ばれる)。"""
        data = {
            "attempt": self.attempt,
            "trials": self.trials,
            "best_J": self.best_J,
            "best_params": self.best_params.to_dict() if self.best_params else None,
            "best_metrics": self.best_metrics,
            "history": self.history,
            "saved_at": time.time(),
            # [v9①] 床較正の再現 (再開時に再評価不要)
            "surrogate_floor": self.surrogate_floor,
            "target_eff": self.target_eff,
        }
        tmp = self.ckpt_path.with_suffix(".tmp")
        try:
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, self.ckpt_path)
        except Exception:
            # 保存失敗で探索自体を落とさない (次の試行で再試行される)
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def try_resume(self) -> bool:
        """チェックポイントがあれば履歴を復元する。"""
        if not self.ckpt_path.exists():
            return False
        try:
            with open(self.ckpt_path, encoding="utf-8") as f:
                data = json.load(f)
            self.attempt = data.get("attempt", 0)
            self.history = data.get("history", [])
            # [v7] 過去試行の所要秒でETA統計をシード (条件⑤: 再開直後から
            # 正確な残り時間を出す)
            try:
                rates = [h.get("eval_seconds", 0.0) for h in self.history[-16:]
                         if h and not h.get("cached")]
                if rates:
                    self.progress.seed_rates([r for r in rates if r and r > 0])
            except Exception:
                pass
            if data.get("best_params"):
                self.best_params = AblationParams.from_dict(data["best_params"])
                self.best_metrics = data.get("best_metrics")
                if self.best_metrics is not None and "attempt" not in self.best_metrics:
                    self.best_metrics["attempt"] = self.attempt
                self.best_J = data.get("best_J", float("inf"))
            # [v9①] 床較正の復元 (v8以前のチェックポイントには無い → None)
            try:
                fl = data.get("surrogate_floor")
                self.surrogate_floor = (float(fl) if isinstance(fl, (int, float)) else None)
                te = data.get("target_eff")
                self.target_eff = (float(te) if isinstance(te, (int, float))
                                   else float(self.target))
            except Exception:
                self.surrogate_floor = None
                self.target_eff = float(self.target)
            if self.attempt >= self.trials:
                return True
            print_line(tr("resume_found", a=self.attempt, n=self.trials))
            return True
        except Exception as e:
            print_line(tr("resume_fail", e=e))
            return False

    # ------------------------------------------------------------ 予算・中断

    def _budget_exceeded(self) -> bool:
        if self.time_budget_sec <= 0:
            return False
        return (time.perf_counter() - self._t_start) >= self.time_budget_sec

    _t_start: float = 0.0

    # ------------------------------------------------------------ メインループ

    def run(self, resume: bool = False) -> dict[str, Any]:
        """探索を実行する。戻り値: 最良候補の情報。

        KeyboardInterruptは「1回目」を優雅に処理し、チェックポイントを保存した
        上で結果 (interrupted=True) を返す。2連続のCtrl+Cは通常どおり伝播する。

        [v5] 探索開始時に (1) 多境界キャッシュをスイープし、
        (2) 早期スクリーニングを有効化する。終了時 (return/except) には
        スクリーニングを解除する — 呼び出し側 (最終検証) が同一評価器を
        使うため。
        """
        self._t_start = time.perf_counter()
        self.interrupted = False
        self.budget_stopped = False
        self.progress.start()   # [v5] 再実行時にティッカーを復活させる
        self._install_screens(True)
        if not self.quiet:
            # [v5] 評価中live表示を接続 (終了時に解除 — 最終検証と混線させない)
            # [v7] (text, eta) シグネチャ (条件⑤)
            self.ev.live_cb = (lambda text, eta=None: self.progress.live_step(text, eta))
        try:
            return self._run_inner(resume)
        except KeyboardInterrupt:
            self.interrupted = True
            self._save_ckpt()
            result = self._result()
            result["interrupted"] = True
            return result
        finally:
            self._install_screens(False)
            self.ev.live_cb = None
            self.progress.finish()

    def _run_inner(self, resume: bool) -> dict[str, Any]:
        if resume and self.try_resume():
            if self.attempt >= self.trials:
                print_line(tr("resume_done"))
                return self._result()
            # 再開時は現bestから出発
            current = AblationParams.from_dict(self.best_params.to_dict()) if self.best_params \
                else AblationParams.from_dict(self.init_params.to_dict())
            current_J = self.best_J if self.best_metrics else float("inf")
        else:
            current = AblationParams.from_dict(self.init_params.to_dict())
            current_J = float("inf")

        # [v5] 多境界プレフィックスキャッシュを構築 (1回だけ, 摂動なしで)
        self._ensure_boundary_cache()
        # [v7] 評価構造 (境界キャッシュ等) が確定した時点で1回だけ呼ぶ
        # (INT4重みキャッシュ予算の最終化など — 早期決定の二重計上を防ぐ)
        if getattr(self, "on_ready", None) is not None:
            try:
                self.on_ready()
            except Exception:
                pass
            self.on_ready = None

        steps = {k: v for k, v in self.base_step.items()}
        comps = [c for c in current.kernels.keys()]
        coord_cycle: list[tuple[str, str]] = []
        for c in comps:
            coord_cycle += [(c, "strength"), (c, "center"), (c, "radius")]
        coord_cycle += [(None, "direction"), (None, "direction_mode")]

        # ---- [v9①] シードフェーズ: 強度スペクトルを覆う参照候補グリッド ----
        if self.attempt == 0:
            current = self._seed_phase(current)
            current_J = self.best_J if self.best_metrics else float("inf")
            if self._budget_exceeded() or self.interrupted:
                return self._result()

        since_improve = 0

        while self.attempt < self.trials:
            if self._budget_exceeded():
                self.budget_stopped = True
                print_line(tr("search_budget_stop", a=self.attempt))
                self._save_ckpt()
                break
            self._tick_monitor()
            # 座標を巡回
            for comp, coord in coord_cycle:
                if self.attempt >= self.trials:
                    break
                if self._budget_exceeded():
                    self.budget_stopped = True
                    print_line(tr("search_budget_stop", a=self.attempt))
                    self._save_ckpt()
                    return self._result()
                best_variant = None
                best_vJ = float("inf")
                best_m: dict[str, Any] | None = None
                if coord == "direction_mode":
                    # モード切替 (per_layer <-> global) は1回だけ評価
                    cand = self._params_variant(current, comp, coord, 0.0)
                    m = self._run_one_eval(cand, phase="explore")
                    if m["J"] < best_vJ:
                        best_vJ, best_variant, best_m = m["J"], cand, m
                else:
                    # 候補: +step と -step を両方試す (パターンサーチ)
                    for delta in (steps[coord], -steps[coord]):
                        cand = self._params_variant(current, comp, coord, delta)
                        m = self._run_one_eval(cand, phase="explore")
                        if m["J"] < best_vJ:
                            best_vJ, best_variant, best_m = m["J"], cand, m
                        if self.attempt >= self.trials:
                            break
                if best_variant is not None and best_m is not None and best_vJ < current_J - 1e-12:
                    current = best_variant
                    current_J = best_vJ
                    since_improve = 0
                    self._maybe_update_best(best_m, best_variant)
                elif coord != "direction_mode":
                    # 改善なし → ステップを縮小
                    steps[coord] = max(steps[coord] * 0.6, self._min_step(coord))
                self._save_ckpt()
                since_improve += 1

            # 行き詰まり検出 → ベスト周辺のダター再始動
            if since_improve >= 2 * len(coord_cycle):
                current = self._dither_restart(self.best_params or current)
                current_J = self.best_J
                steps = {k: v for k, v in self.base_step.items()}
                since_improve = 0

        self._save_ckpt()
        return self._result()

    def _seed_phase(self, current: AblationParams) -> AblationParams:
        """[v9①] 構造化初期化: 摂動族の「強度×帯幅」スペクトルを覆うシード候補を
        最初に評価し、(1) 代理指標の文法床を実測し、(2) パターンサーチの起点を
        全域の中から選ぶ。

        v8の初期化は「顕著度ピークの狭い帯でコンポーネント単独の強度三分割」
        だけだったため、強い全層直交化 (Arditi et al. の参照解) が探索族の初期
        位置から遠く、パターンサーチがそこへ道を見つけるまでに数十試行を消費
        していた (4B実測: 31試行かけて代理拒否率0.689→0.401 — 床付近で未達)。
        v9は最初の ~8 評価でスペクトル全体 (弱い全帯・強い全帯・単一コンポー
        ネント・ピーク周辺の狭帯) を抑え、最良を起点にする。

        各評価は _run_one_eval を通すため、進捗表示とチェックポイント保存
        (1試行ごと) が保証される。シード評価中は早期スクリーニングを無効化
        する (best_J未確定のまま候補を切り捨てると床推定が歪むため)。
        """
        L = self.n_layers
        comps = list(current.kernels.keys())
        peak = int(torch.argmax(self.salience[1:]).item()) + 1

        def _flat(strength: float, only: str | None = None) -> AblationParams:
            # 全層に近い一様帯 (レイヤ2..L-1)。レイヤ0-1は残差方向の形成が
            # 未熟で顕著度も低いため除外する — これにより band_start=2 となり
            # 多境界前置キャッシュの最初の境界 (レイヤ2) が使える。
            span_lo, span_hi = 2.0, float(L - 1)
            kers = {}
            for c in comps:
                on = (only is None or c == only)
                kers[c] = KernelParams(
                    center=(span_lo + span_hi) / 2.0,
                    radius=max(1.5, (span_hi - span_lo) / 2.0),
                    strength=(strength if on else 0.0), shape="flat")
            return AblationParams(direction_mode=current.direction_mode,
                                  direction_index=current.direction_index,
                                  kernels=kers)

        def _peak(strength: float) -> AblationParams:
            kers = {c: KernelParams(center=float(peak), radius=max(2.0, 0.15 * L),
                                    strength=strength, shape="cosine") for c in comps}
            return AblationParams(direction_mode=current.direction_mode,
                                  direction_index=current.direction_index,
                                  kernels=kers)

        if self.focus_init is not None:
            # --params-from focus モード: 読み込みパラメータを起点に、床較正用の
            # 強い全帯参照候補だけを追加評価する。
            cands = [AblationParams.from_dict(self.focus_init.to_dict()), _flat(2.1)]
        else:
            cands = [
                _peak(0.0),                       # ベースライン (無摂動)
                _flat(0.7),                       # 弱い全層一様
                _flat(1.4),                       # 中間全層一様
                _flat(2.1),                       # 強い全層一様 ≈ 床の実測点
                _flat(1.4, only=comps[0]),        # 単一コンポーネント (attn)
                _flat(1.4, only=comps[-1]),       # 単一コンポーネント (mlp)
                _peak(1.0),                       # v8族: ピーク狭帯
                _peak(2.0),                       # v8族: ピーク狭帯・強
            ]

        self._install_screens(False)   # シード中はスクリーニング無効
        if not self.quiet:
            print_line(tr("v9_seed_phase"))
        best_params = AblationParams.from_dict(current.to_dict())
        best_J = float("inf")
        for cand in cands:
            if self.attempt >= self.trials or self._budget_exceeded():
                break
            m = self._run_one_eval(cand, phase="seed")
            if m["J"] < best_J:
                best_J = m["J"]
                best_params = AblationParams.from_dict(cand.to_dict())
        self._calibrate_floor()
        self._install_screens(True)
        self._save_ckpt()
        # 床較正後のJが最小の候補 (self.best_params) から パターンサーチを開始する
        if self.best_params is not None:
            best_params = AblationParams.from_dict(self.best_params.to_dict())
        return best_params

    def _calibrate_floor(self) -> None:
        """[v9①] シード評価の実測から代理指標の床を推定し実効目標を再設定する。

        床が持ち上がった場合は既記録の J を新しい実効目標で再計算し、
        best も履歴から選び直す (較正前の J は床未達の罰則で強すぎる候補に
        片寄ってしまうため)。
        """
        masses = [h.get("refusal_soft") for h in self.history
                  if h.get("phase") == "seed"
                  and isinstance(h.get("refusal_soft"), (int, float))]
        if not masses:
            return
        floor = float(min(masses))
        self.surrogate_floor = floor
        self.target_eff = max(float(self.target), floor + 0.03)
        try:
            print_line(tr("v9_floor_calibrated", f=floor, t=self.target_eff,
                          r=self.target))
        except Exception:
            pass
        # Jの再計算 (罰則項が変わるため)
        for h in self.history:
            rs, kl, j = h.get("refusal_soft"), h.get("kl"), h.get("J")
            if (isinstance(rs, (int, float)) and isinstance(kl, (int, float))
                    and kl == kl and j is not None and not h.get("screened")):
                h["J"] = round(self.objective(float(rs), float(kl)), 6)
        self._recalibrate_best()

    def _recalibrate_best(self) -> None:
        """履歴から (較正後Jが最小の) 最良候補を選び直す。"""
        cands = []
        for h in self.history:
            j = h.get("J")
            rs, kl = h.get("refusal_soft"), h.get("kl")
            if (isinstance(j, (int, float)) and j == j and j < float("inf")
                    and isinstance(rs, (int, float)) and isinstance(kl, (int, float))
                    and kl == kl and not h.get("screened") and h.get("params")):
                cands.append((float(j), h))
        if not cands:
            return
        j, h = min(cands, key=lambda x: x[0])
        self.best_J = j
        self.best_params = AblationParams.from_dict(h["params"])
        self.best_metrics = {"refusal_soft": h.get("refusal_soft"),
                             "refusal_first": h.get("refusal_first"),
                             "kl": h.get("kl"), "J": j, "margin_mean": 0.0,
                             "attempt": h.get("attempt", 0)}

    def _dither_restart(self, base: AblationParams) -> AblationParams:
        """最良解周辺へガウス摂動を加えた再始動点を作る。"""
        cand = AblationParams.from_dict(base.to_dict())
        for comp, kp in cand.kernels.items():
            kp.center += self.rng.gauss(0, self.base_step["center"])
            kp.radius += self.rng.gauss(0, self.base_step["radius"] * 0.7)
            kp.strength += self.rng.gauss(0, 0.25)
            kp.center = self._clip("center", kp.center, comp, kp)
            kp.radius = self._clip("radius", kp.radius, comp, kp)
            kp.strength = self._clip("strength", kp.strength, comp, kp)
        if cand.direction_mode == "global":
            cand.direction_index = self._clip("direction",
                                              cand.direction_index + self.rng.gauss(0, self.base_step["direction"]))
        return cand

    def _min_step(self, coord: str) -> float:
        if coord == "strength":
            return 0.02
        if coord == "direction":
            return 0.25
        return 0.5

    def _maybe_update_best(self, m: dict[str, Any], params: AblationParams) -> bool:
        if m["J"] < self.best_J - 1e-9:
            self.best_J = m["J"]
            self.best_params = AblationParams.from_dict(params.to_dict())
            self.best_metrics = {"refusal_soft": m["refusal_soft"],
                                 "refusal_first": m.get("refusal_first", m["refusal_soft"]),
                                 "kl": m["kl"],
                                 "J": m["J"], "margin_mean": m["margin_mean"],
                                 "attempt": self.attempt}
            return True
        return False

    def _record(self, m: dict[str, Any], phase: str) -> dict[str, Any]:
        kl_v = m["kl"]
        kl_first_v = m.get("kl_first", m["kl"])
        j_v = m["J"]
        rf_v = m.get("refusal_first")
        return {
            "attempt": self.attempt,
            "phase": phase,
            "refusal_soft": round(m["refusal_soft"], 5),
            "refusal_first": (round(rf_v, 5) if isinstance(rf_v, float) else None),
            "kl": (round(kl_v, 6) if kl_v == kl_v else None),
            "kl_first": (round(kl_first_v, 6) if kl_first_v == kl_first_v else None),
            "nll_ratio": (round(m["nll_ratio"], 4)
                          if isinstance(m.get("nll_ratio"), float) else None),
            "J": (round(j_v, 6) if j_v == j_v and j_v not in (float("inf"), float("-inf"))
                  else None),
            "eval_seconds": round(m["eval_seconds"], 4),
            "cached": bool(m.get("cached")),
            "screened": bool(m.get("screened")),
            "params": m["params"],
        }

    def _result(self) -> dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_metrics": self.best_metrics,
            "best_J": self.best_J,
            "history": self.history,
            "attempts": self.attempt,
            "interrupted": self.interrupted,
            "budget_stopped": self.budget_stopped,
        }


class RandomSearchBaseline:
    """比較実験用: 同一評価器・同一予算のランダム探索ベースライン。

    v4: SCAPEと同じく 1試行ごとのチェックポイント保存・再開・時間予算に対応
    (ベースライン実験も長時間実行の中断に耐えるようにした)。
    """

    def __init__(self, engine: PatchEngine, evaluator: FastEvaluator,
                 trials: int, target_refusal: float, penalty: float, seed: int = 0,
                 checkpoint_path: Path | None = None,
                 time_budget_sec: float = 0.0, monitor: Callable[[], str] | None = None):
        self.engine = engine
        self.ev = evaluator
        self.trials = trials
        self.target = target_refusal
        self.penalty = penalty
        self.rng = random.Random(seed)
        self.n_layers = engine.registry.n_layers
        self.history: list[dict[str, Any]] = []
        self.best_J = float("inf")
        self.best_metrics: dict[str, Any] | None = None
        self.best_params: AblationParams | None = None
        self.progress = TrialProgress(trials)
        self.ckpt_path = checkpoint_path
        self.time_budget_sec = float(time_budget_sec or 0.0)
        self.monitor = monitor
        self.attempt = 0
        self.budget_stopped = False
        self.interrupted = False
        self._t_start = 0.0
        if self.ckpt_path is not None:
            self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        # [v5] live進捗はTrialProgress自身が描画 (評価バッチごとに更新)
        self.ev.live_cb = (lambda text: self.progress.live_step(text))

    def _ensure_boundary_cache(self) -> None:
        """[v5] 多境界キャッシュ (SCAPEと同じ仕組み)。"""
        try:
            if not getattr(self.ev, "needs_boundary_sweep", lambda: False)():
                return
            with self.engine.suspend_patches():
                self.ev.build_boundary_cache()
        except Exception:
            pass

    # ------------------------------------------------------------ チェックポイント

    def _save_ckpt(self) -> None:
        if self.ckpt_path is None:
            return
        data = {
            "attempt": self.attempt,
            "trials": self.trials,
            "best_J": self.best_J,
            "best_params": self.best_params.to_dict() if self.best_params else None,
            "best_metrics": self.best_metrics,
            "history": self.history,
            "saved_at": time.time(),
        }
        tmp = self.ckpt_path.with_suffix(".tmp")
        try:
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, self.ckpt_path)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def try_resume(self) -> bool:
        if self.ckpt_path is None or not self.ckpt_path.exists():
            return False
        try:
            with open(self.ckpt_path, encoding="utf-8") as f:
                data = json.load(f)
            self.attempt = data.get("attempt", 0)
            self.history = data.get("history", [])
            if data.get("best_params"):
                self.best_params = AblationParams.from_dict(data["best_params"])
                self.best_metrics = data.get("best_metrics")
                self.best_J = data.get("best_J", float("inf"))
            # RNG状態は保存していないため、試行数分だけ棄却して独立同一分布を担保
            for _ in range(self.attempt):
                self._sample()
            if self.attempt >= self.trials:
                return True
            print_line(tr("resume_found", a=self.attempt, n=self.trials))
            return True
        except Exception as e:
            print_line(tr("resume_fail", e=e))
            return False

    def _tick_monitor(self) -> None:
        if self.monitor is None:
            return
        try:
            status = self.monitor()
            if status == "critical":
                self.ev.reduce_memory(True)
        except Exception:
            pass

    def _budget_exceeded(self) -> bool:
        if self.time_budget_sec <= 0:
            return False
        return (time.perf_counter() - self._t_start) >= self.time_budget_sec

    def _sample(self) -> AblationParams:
        L = self.n_layers
        k = {
            "attn_out": KernelParams(
                center=self.rng.uniform(0.3 * L, 0.98 * L),
                radius=self.rng.uniform(1.5, 0.45 * L),
                strength=self.rng.uniform(0.0, 2.5)),
            "mlp_down": KernelParams(
                center=self.rng.uniform(0.3 * L, 0.98 * L),
                radius=self.rng.uniform(1.5, 0.45 * L),
                strength=self.rng.uniform(0.0, 2.5)),
        }
        mode = "per_layer" if self.rng.random() < 0.5 else "global"
        return AblationParams(direction_mode=mode,
                              direction_index=self.rng.uniform(0, max(0, L - 1)), kernels=k)

    def run(self, resume: bool = True) -> dict[str, Any]:
        if resume and self.ckpt_path is not None:
            self.try_resume()
        self._t_start = time.perf_counter()
        self.interrupted = False
        self.budget_stopped = False
        self.progress.start()   # [v5] 再実行時にティッカーを復活させる
        # [v5] 多境界キャッシュを摂動なしで構築してから探索する
        self._ensure_boundary_cache()
        try:
            while self.attempt < self.trials:
                if self._budget_exceeded():
                    self.budget_stopped = True
                    break
                params = self._sample()
                self.engine.reset()
                self.engine.apply(params)
                m = self.ev.evaluate(params, use_cache=True)
                over = max(0.0, m["refusal_soft"] - self.target)
                m["J"] = m["kl"] + self.penalty * over * over
                self.engine.reset()
                self.attempt += 1
                self.history.append({
                    "attempt": self.attempt, "phase": "random",
                    "refusal_soft": round(m["refusal_soft"], 5),
                    "refusal_first": round(m.get("refusal_first", m["refusal_soft"]), 5),
                    "kl": round(m["kl"], 6),
                    "J": round(m["J"], 6), "eval_seconds": round(m["eval_seconds"], 4),
                    "params": params.to_dict(),
                })
                if m["J"] < self.best_J:
                    self.best_J = m["J"]
                    self.best_metrics = {"refusal_soft": m["refusal_soft"],
                                         "refusal_first": m.get("refusal_first", m["refusal_soft"]),
                                         "kl": m["kl"],
                                         "J": m["J"]}
                    self.best_params = params
                self._save_ckpt()
                self._tick_monitor()
                self._ensure_boundary_cache()
                self.progress.trial(
                    self.attempt, m.get("eval_seconds", 0.0), m["refusal_soft"], m["kl"], m["J"],
                    self.best_J, 0)
        except KeyboardInterrupt:
            self.interrupted = True
            self._save_ckpt()
        finally:
            self.progress.finish()
            self.ev.live_cb = None
        return {"best_params": self.best_params, "best_metrics": self.best_metrics,
                "history": self.history, "attempts": self.attempt,
                "budget_stopped": self.budget_stopped, "interrupted": self.interrupted}


def best_so_far_curve(history: list[dict[str, Any]], key: str = "refusal_soft") -> list[float]:
    """履歴から「その時点までの最良値」の曲線を作る (レポート用)。"""
    out = []
    cur = float("inf")
    for h in history:
        v = h.get(key, float("nan"))
        if v is None:
            v = float("nan")
        if v == v and v < cur:
            cur = v
        out.append(cur if cur != float("inf") else float("nan"))
    return out
