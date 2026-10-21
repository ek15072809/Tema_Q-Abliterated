# -*- coding: utf-8 -*-
"""CLIエントリポイント (v3)。

環境検出 → メモリ計画 (自動量子化/ストリーミング読み込み) → モデル解決 →
方向計算 → SCAPE探索 (1試行ごとのチェックポイント+進捗表示+時間予算) →
100プロンプト全文生成検証 → 候補選択 (対話/自動) → 出力 → レポート。

v3の主な変更:
  - 出力ログは既定で英語。--language jp 指定時のみ日本語 (msg.py経由)
  - SCAPE探索は1試行ごとにチェックポイント保存 + n/N・経過・ETA表示 (ui.py)
  - --time-budget-min で探索の時間予算を指定可能 (超過時は最良解で検証へ)
  - 探索をCtrl+Cで中断しても チェックポイントが残り再実行で自動再開
  - 部分フォワードの差し替え方式化 (surrogate.py) による評価高速化
  - transformers の誤検知警告・ノイズ警告を抑制 (envcheck/modelio)

v5の主な変更:
  - 評価中live進捗は ui.TrialProgress が自前描画 (旧cliの生\r書き込みハック
    による表示残骸を廃止。残り時間は1秒ごとのカウントダウン)
  - 摂動エンジンがランク1フック方式に (engine.py)。起動時に完全重みパッチ
    との自己検証を行い、不一致時は従来方式へ自動フォールバック
  - INT4重みキャッシュは「可能なら全レイヤbf16化」へ拡大 (メモリ余裕次第)
  - 最終検証はプロンプト件数ベースのn/Nバー + 拒否マーカー/EOSでの
    系列打ち切り + 生成バッチ16で高速化 (指標は従来どおり)
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

from . import __version__
from .config import Settings, load_settings
from .envcheck import (EnvReport, _ensure_utf8_stdio, apply_runtime_settings,
                       detect_environment, noise_summary_lines, suppress_library_noise)
from .msg import tr
from .probes import load_probe_set, refusal_marker_list
from .ui import fmt_duration, print_line

# [v7条件⑧][v9②改訂] 代理指標ゲート: この値を超える候補は全文生成検証をスキップ
# (拒否率が高すぎて測定する価値がないため)。元モデル (baseline) の測定は対象外。
# v8までは質量加重多位置比 (refusal_soft) で判定していたが、この指標には
# 文法床 (~0.3-0.4) があり完全に非検閲化されたモデルでも床を下回れない。
# そのため4B級の実測では「全候補がしきい値超過でスキップ」が起こり、検証
# なしのモデルが無断で書き出されていた。v9は床のない初頭トークン比
# (refusal_first) でゲートし、全候補スキップ時はユーザーの判断を仰ぐ。
# 値は --verify-gate で変更できる。
VERIFY_REFUSAL_GATE = 0.4


def _log(msg: str = "") -> None:
    # [v5] 進捗live行 (ui.LINES) と衝突しない出力経路を使う
    try:
        print_line(msg)
    except Exception:
        print(msg, flush=True)


def _banner() -> None:
    _log("==========================================================")
    _log(tr("banner_title", version=__version__))
    _log(tr("banner_sub"))
    _log("==========================================================")


def _safetensors_key_map(model, model_dir: Path, registry) -> dict[str, tuple[str, int]]:
    """摂動対象モジュールとsafetensorsキーの対応表を作る。

    [v6] 融合3D専門家はチェックポイント上で2種類の書式がありうる:
      - 融合形式:   "...experts.down_proj"            (3Dテンソル, 実Flash-Next形式)
      - 分割形式:   "...experts.{e}.down_proj.weight" (専門家ごとの2D)
    両方を key_map に登録する (エクスポートは sd[key] の次元で自動判別)。

    [v11] transformers 5.x には「メモリ上の名前」と「チェックポイント上の名前」が
    異なるアーキテクチャがある (変換マッピング)。例:
      - DeepSeek V4系:  self_attn.o_b_proj → attn.wo_b / mlp → ffn
      - Lfm2Moe (LFM2.5-24B-A2B): 融合専門家 mlp/feed_forward.experts.down_proj
        はチェックポイントでは "...experts.{e}.w2.weight" (w1/w2/w3分割形式)
    これらにも出力書き換えが届くよう、transformersの逆変換
    (revert_weight_conversion) で導かれるチェックポイント形式名と
    w2分割形式名を候補に加える。変換を持たないモデルでは恒等変換となり
    従来候補と重複するだけで挙動は変わらない。
    """
    from safetensors import safe_open
    all_keys: set[str] = set()
    for f in sorted(model_dir.iterdir()):
        if f.name.endswith(".safetensors"):
            with safe_open(str(f), framework="pt") as fh:
                all_keys.update(fh.keys())
    path_of = {id(mod): path for path, mod in model.named_modules()}

    def _ckpt_alias(model, name: str, tensor) -> list[str]:
        """[v11] メモリ上パラメータ名1つをチェックポイント形式名へ逆変換する。"""
        try:
            from transformers.core_model_loading import revert_weight_conversion
            return list(revert_weight_conversion(model, {name: tensor}).keys())
        except Exception:
            return []

    key_map: dict[str, tuple[str, int]] = {}
    missed = 0
    for comp in registry.components:
        for layer, tms in registry.map[comp].items():
            for tm in tms:
                path = path_of.get(id(tm.module))
                if path is None:
                    continue
                if tm.fused:
                    base_name = f"{path}.{tm.weight_attr}"
                    # 分割形式 (専門家ごとの2Dキー)。gate_up は対象外 (downのみ)
                    candidates = [base_name]
                    for e in range(tm.n_experts):
                        candidates.append(f"{path}.{e}.{tm.weight_attr}.weight")
                        # [v11] Lfm2Moe等: 専門家2Dは w2 命名で保存される
                        candidates.append(f"{path}.{e}.w2.weight")
                    try:
                        candidates += _ckpt_alias(model, base_name, tm.weight())
                    except Exception:
                        pass
                else:
                    base_name = f"{path}.{tm.weight_attr}"
                    candidates = [f"{path}.{tm.weight_attr}",
                                  f"{path}.{tm.weight_attr}.weight"]
                    try:
                        w = tm.weight()
                        if torch.is_tensor(w):
                            # base_name が既に完全なパラメータ名 (weight_attr="weight")
                            candidates += _ckpt_alias(model, base_name, w)
                    except Exception:
                        pass
                hit = False
                for key in candidates:
                    if key in all_keys:
                        key_map[key] = (comp, layer)
                        hit = True
                if not hit:
                    missed += 1
    if missed:
        _log(tr("params_key_missed", n=missed))
    return key_map


def run_pipeline(settings: Settings) -> int:
    """メインパイプライン。戻り値: 終了コード。"""
    _banner()

    # ワークスペースの位置を冒頭に明示する (既定では ./tq_workspace/<モデル名>)
    _log(tr("workspace_info", dir=str(Path(settings.workdir).resolve())))

    # ---------------------------------------------------------- 環境検出 (条件⑦)
    suppress_library_noise()   # [v3] 既知の無害な繰り返し警告を抑制
    prefer = None if settings.device == "auto" else settings.device
    env: EnvReport = detect_environment(auto_device=True, prefer_device=prefer)
    apply_runtime_settings(env)
    _log("\n" + tr("env_section"))
    _log(env.summary())
    device = env.device

    torch.set_grad_enabled(False)
    if settings.seed is not None:
        torch.manual_seed(settings.seed)

    # ---------------------------------------------------------- モデル解決 (条件⑤)
    from .modelio import is_local_model, model_param_count, resolve_model_dir

    _log("\n" + tr("model_section", model=settings.model))
    model_dir = resolve_model_dir(settings.model, settings.cache_root)
    _log(tr("model_local_path", path=model_dir))
    # モデルカード用: 元モデルのHuggingFaceリポジトリID (ローカルパス指定時は不明)
    base_model_ref = (settings.model if ("/" in settings.model
                                          and not is_local_model(settings.model))
                      else None)

    # ---------------------------------------------------------- [v9⑦] 再実行メニュー (ロード前)
    # 同じモデルを指定して再実行した場合、モデルのロード (数十分かかる
    # こともある) の前にどうするかを詳しく尋ねる:
    #   1) 探索を続行 (チェックポイントから再開 / 完了済みなら最終検証へ)
    #   2) 試行を追加 (探索を延長)
    #   3) 記録済み候補からモデルを書き出す (探索フェーズをスキップ)
    #   4) 最初からやり直す (チェックポイントを破棄)
    #   5) 終了
    # 探索は1試行ごと・検証は候補ごとにチェックポイントへ保存されるため、
    # 誤中断しても任意の時点から再開できる。非対話環境 (パイプ/CI) では
    # 従来どおり自動再開する (バッチ処理の挙動を変えない)。
    trials = settings.trials
    if trials <= 0:
        trials = settings.auto_trials_gpu if device.startswith("cuda") else settings.auto_trials_cpu
    run_key = model_dir.name
    ckpt_path = settings.cache_root / "runs" / f"{run_key}.json"
    vpath = settings.cache_root / "runs" / f"{run_key}.verified.json"
    # [v3] runs/ディレクトリはここで必ず作成される (探索開始前)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    export_from_ckpt = False
    if settings.restart and ckpt_path.exists():
        ckpt_path.unlink(missing_ok=True)
        vpath.unlink(missing_ok=True)
        _log(tr("restart_discard"))
    elif (not settings.params_from and not settings.quiet and ckpt_path.exists()):
        choice = _preload_resume_menu(ckpt_path, vpath, trials)
        if choice[0] == "quit":
            _log(tr("done"))
            return 0
        if choice[0] == "fresh":
            ckpt_path.unlink(missing_ok=True)
            vpath.unlink(missing_ok=True)
            _log(tr("restart_discard"))
        elif choice[0] == "extend":
            trials = int(choice[1])
        elif choice[0] == "export":
            export_from_ckpt = True

    # ---------------------------------------------------------- メモリ計画 (条件③)
    from .memguard import (MemoryGuard, model_disk_gb,
                           plan_load_strategy, ram_available_gb)
    plan = plan_load_strategy(model_dir, device, settings.eval_dtype)
    _log("\n" + tr("mem_section", disk=model_disk_gb(model_dir),
                   ram=ram_available_gb(),
                   vram=(tr("mem_vram_check") if device.startswith("cuda") else "")))
    _log(tr("mem_plan", plan=plan.describe()))
    guard = MemoryGuard(device=device)

    # ---------------------------------------------------------- 読み込み実行
    from .arch import ComponentRegistry
    use_quant = plan.quant
    load_info: dict[str, Any] = {"mode": plan.mode}
    if plan.mode == "streamed":
        from .modelio import load_model_streamed
        try:
            model, registry, load_info = load_model_streamed(
                model_dir, env, device, quant=plan.quant,
                components=settings.components)
        except Exception as e:  # noqa: BLE001
            _log(tr("stream_fail", etype=type(e).__name__, e=e))
            _log(tr("stream_fallback"))
            use_quant = "int4" if plan.quant == "int4" else plan.quant
            model = None
    else:
        model = None

    if model is None:
        from .modelio import load_model
        model = load_model(model_dir, env, plan.dtype, device)
        registry = ComponentRegistry(model)
        if use_quant == "int8":
            from .modelio import build_int8_eval_model
            _log(tr("int8_build_title"))
            try:
                model = build_int8_eval_model(model, registry)
            except Exception as e:  # noqa: BLE001
                _log(tr("int8_fail", etype=type(e).__name__))
                use_quant = "int4"
        if use_quant == "int4":
            from .modelio import build_int4_eval_model
            model = build_int4_eval_model(model, registry, device=device,
                                          components=settings.components)

    n_params = model_param_count(model)
    if n_params == 0:
        raise RuntimeError(tr("model_no_params"))
    # [v7] ロード直後にSWAP基準線を引き直す (ロード時の一時スワップアウトが
    # 残って「偽の危険状態」を出し続け、評価キャッシュを毎試行破棄させる
    # v6の問題を解消する・条件①)
    guard.tick(force=True)
    guard.rebaseline()

    # INT4重みキャッシュ予算の初期値 (条件③)
    # [v7] モデルの実フットプリント (非量子化パラメータ + INT4パック済み重み)
    # を差し引いた残りから、作業セットぶん (1.0GB: python基底+活性・生成の
    # 一時バッファ+基底分布+境界キャッシュ+安全マージン) を引いて決める。
    # RAMに余裕のある環境 (16GB級) では全レイヤがキャッシュされ、ヒット時は
    # 純bf16実行と同一速度になる。
    # これにより「モデル全体をキャッシュしてしまい他のフェーズがOOMする」
    # (低メモリ環境で実際に発生) ことを防ぐ。探索開始直前 (on_ready) に
    # 空きに応じて増方向へ再評価される。
    if use_quant == "int4":
        from .quant4 import PackedInt4Linear
        need_gb = PackedInt4Linear.total_dequant_bytes() / (1024 ** 3)
        if settings.dequant_cache_gb >= 0:
            cache_gb = settings.dequant_cache_gb
        else:
            from .memguard import ram_available_gb as _ram_gb
            avail_gb = _ram_gb()
            try:
                footprint_gb = (sum(p.numel() * p.element_size()
                                    for p in model.parameters())
                                + PackedInt4Linear.total_packed_bytes()) / (1024 ** 3)
            except Exception:
                footprint_gb = 0.0
            margin_gb = 1.0
            cache_gb = min(need_gb, max(0.0, avail_gb - footprint_gb - margin_gb))
        PackedInt4Linear.set_weight_cache_budget(int(cache_gb * (1024 ** 3)))
        PackedInt4Linear._WSTATE["target"] = int(need_gb * (1024 ** 3))
        PackedInt4Linear._WSTATE["need"] = int(need_gb * (1024 ** 3))

    def _finalize_int4_budget() -> None:
        """[v7] 探索開始直前 (評価構造確定後) にキャッシュ予算を最終化する。

        境界キャッシュ・基底分布などが割り当て済みの実空きRAMから決めるため、
        早期決定で起きていた二重計上 (OOM) が起きない。予算は増方向にのみ
        更新する (既にキャッシュ済みの重みを追い出さない)。
        """
        if use_quant != "int4" or settings.dequant_cache_gb >= 0:
            return
        try:
            from .quant4 import PackedInt4Linear
            from .memguard import ram_available_gb as _ram_gb2
            need_b = int(PackedInt4Linear._WSTATE.get("need", 0))
            if need_b <= 0:
                return
            avail_b = int(_ram_gb2() * (1024 ** 3))
            margin_b = int(0.9 * (1024 ** 3))   # 試行時の活性・安全マージン
            want = min(need_b, max(0, avail_b - margin_b))
            cur = int(PackedInt4Linear._WSTATE.get("budget", 0))
            if want > cur:
                PackedInt4Linear.set_weight_cache_budget(want)
                _log(tr("int4_exec_line", gb=want / (1024 ** 3), need=need_b / (1024 ** 3)))
        except Exception:
            pass

    from .modelio import load_tokenizer_quiet
    tokenizer = load_tokenizer_quiet(model_dir, verbose=not settings.quiet)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # [v3] transformers読み込み後にノイズフィルタをハンドラへ再接続
    # (遅延importでハンドラが構成された後でないと付けられないため)
    suppress_library_noise()
    for line in noise_summary_lines():
        _log(line)   # 読み込み時forwardで発生したカーネルフォールバックの要約

    # ---------------------------------------------------------- アーキテクチャ解析 (条件⑮)
    _log("\n" + tr("arch_section", n=registry.n_layers))
    for line in registry.summary():
        _log(line)

    # ---------------------------------------------------------- プローブ準備 (条件④)
    _log("\n" + tr("probes_section"))
    probes = load_probe_set(settings.harmful_file, settings.harmless_file,
                            settings.harmful_dataset, settings.harmless_dataset,
                            settings.cache_root)
    _log(probes.describe())

    # [v6] モデル規模別の自動プローブ数・検証候補数 (CPUの4B≈3時間目標)。
    # 明示指定 (--n-probe / --verify-topk) があればそちらを優先する。
    try:
        from .modelio import model_param_count as _mpc
        n_params_now = _mpc(model)
    except Exception:
        n_params_now = 0
    auto_probe = 48
    if n_params_now > 0:
        b = n_params_now / 1e9
        if device.startswith("cuda"):
            auto_probe = 64 if b <= 8 else 48
        else:
            auto_probe = 48 if b <= 2 else (40 if b <= 6 else (32 if b <= 12 else 24))
    n_probe = settings.n_probe if settings.n_probe > 0 else min(auto_probe, len(probes.harmful))
    harmful_explore = probes.harmful[:n_probe]
    harmless_explore = probes.harmless[:n_probe]
    n_verify = (settings.verify_probes if settings.verify_probes > 0
                else min(100, len(probes.harmful)))
    n_verify = max(8, n_verify)
    # [v7条件⑭] 検証用プロンプトをHuggingFaceデータセットから取得できる
    # (例: --verify-dataset mlabonne/harmful_behaviors → 先頭100件)。
    verify_prompts: list[str] = probes.harmful[:n_verify]
    if settings.verify_dataset:
        from .probes import load_verify_prompts
        ds_info: dict[str, Any] = {}
        ds_prompts = load_verify_prompts(settings.verify_dataset, n_verify,
                                         settings.cache_root, info=ds_info)
        if ds_prompts:
            verify_prompts = ds_prompts
            n_verify = max(8, min(n_verify, len(verify_prompts)))
            verify_prompts = verify_prompts[:n_verify]
            _log(tr("verify_dataset_note", d=settings.verify_dataset,
                    sp=ds_info.get("split", "?"), n=n_verify))
    verify_topk_eff = max(1, settings.verify_topk)
    if not device.startswith("cuda") and settings.verify_topk > 2:
        # [v7条件③] CPU実行時の全文生成検証は「元モデル + 上位2候補」の
        # 計3回ぶんに抑える (v6は元モデル+4候補の5回だった)。
        # さらに条件⑧のゲート (代理拒否率>0.4) が候補を絞るため、
        # 実際は3回以下になることが多い。
        verify_topk_eff = min(verify_topk_eff, 2)
        _log(tr("verify_topk_auto", k=verify_topk_eff, r=verify_topk_eff + 1))
    _log(tr("probes_subset", n=len(harmful_explore), v=n_verify))

    markers = refusal_marker_list()

    # [v6] プローブ多様性の表示 (オープナー分布の偏りの確認用)
    try:
        _openers = {p.split()[0].lower() for p in probes.harmful if p.strip()}
        _log(tr("probes_diversity", nh=len(probes.harmful),
                ng=len(probes.harmless), oh=len(_openers)))
    except Exception:
        pass

    from .surrogate import FastEvaluator, detect_response_prefix
    response_prefix = detect_response_prefix(tokenizer, settings.system_prompt)
    if response_prefix:
        _log(tr("thinking_prefix", p=response_prefix))

    # ---------------------------------------------------------- [v8条件⑤] 有害プローブの締め付け
    # 残差方向の計算と探索評価に使う有害プローブを「基底モデルが実際に
    # 拒否するもの」へ絞り込む。拒否傾向の弱いプローブは (1) 残差方向の
    # 平均を薄め、(2) 代理指標の拒否率を甘くするため。最終検証 (全文生成)
    # には影響しない (検証プロンプトは絞り込み前の全集合/データセット)。
    if settings.probe_hardening and len(harmful_explore) >= 8:
        from .surrogate import baseline_first_token_refusals
        _log("\n" + tr("probe_filter_title"))
        scores = baseline_first_token_refusals(
            model, tokenizer, harmful_explore, settings.system_prompt,
            response_prefix, device, env.batch_size)
        keep_min = max(12, len(harmful_explore) // 3)
        kept_idx = [i for i, s in enumerate(scores) if s == s and s >= 0.30]
        if len(kept_idx) < keep_min:
            # しきい値で必要数に満たない場合: スコア上位で補充する
            # (方向の質を最大化しつつ最低数を確保。すべてNaNの場合は無効化)
            kept_set = set(kept_idx)
            rest = sorted((i for i in range(len(scores)) if i not in kept_set),
                          key=lambda i: (-(scores[i] if scores[i] == scores[i]
                                           else -1.0)))
            kept_idx = sorted(kept_idx + rest[:keep_min - len(kept_idx)])
        if 0 < len(kept_idx) < len(harmful_explore):
            harmful_explore = [harmful_explore[i] for i in kept_idx]
            _log(tr("probe_filter_done", k=len(kept_idx), t=len(scores),
                    d=len(scores) - len(kept_idx)))
        else:
            _log(tr("probe_filter_keepall", t=len(harmful_explore)))

    # ---------------------------------------------------------- 残差方向 (条件⑪: キャッシュ)
    from .directions import load_or_compute_directions
    _log("\n" + tr("dir_section"))
    directions, directions_map, salience, from_cache = load_or_compute_directions(
        model, tokenizer, device, model_dir, settings.cache_root,
        harmful_explore, harmless_explore, env.batch_size,
        settings.system_prompt, response_prefix, registry.n_layers,
        registry=registry, components=settings.components)
    peak = int(torch.argmax(salience[1:]).item()) + 1
    _log(tr("dir_peak", peak=peak, n=registry.n_layers))
    guard.tick(force=True)   # [v4] フェーズ境界のメモリ記録 (方向計算後)

    # ---------------------------------------------------------- 評価器とエンジン
    # [v9⑦] チェックポイントからの書き出しモードでは軽量評価器 (基底継続生成
    # を省略) を使う。追加試行が必要になった時点で自動的に完全評価器へ
    # Upgradeされる (surrogate.ensure_full)。
    evaluator = FastEvaluator(
        model, tokenizer, device, registry, settings.system_prompt,
        harmful_explore, harmless_explore,
        batch_size=env.batch_size, response_prefix=response_prefix,
        cache_enabled=True, verbose=not settings.quiet,
        kl_tokens=settings.kl_tokens, lightweight=export_from_ckpt)
    guard.register_reducer(evaluator.reduce_memory)

    eq_diff = evaluator.verify_partial_equivalence()
    if eq_diff == float("inf"):
        _log(tr("partial_check_error", etype="verification"))
        evaluator.cache_enabled = False
    elif eq_diff < 1.0:
        _log(tr("partial_check_ok", d=eq_diff) if evaluator.cache_enabled
             else tr("partial_check_ok_nocache", d=eq_diff))
    else:
        _log(tr("partial_check_fail", d=eq_diff))
        evaluator.cache_enabled = False

    from .engine import AblationParams, PatchEngine
    engine = PatchEngine(registry, directions_map if directions_map else directions,
                         norm_preserve=settings.norm_preserve)

    # [v5] ランク1フック方式の自己検証: 完全重みパッチとの出力差を實測し、
    # 許容できない場合は従来の重み書き換え (exact) へフォールバックする。
    try:
        _ids0, _am0 = evaluator.harm_batches[0]
        _x = model.get_input_embeddings()(_ids0[:2]).to(device)
    except Exception:
        _x = torch.randn(2, getattr(getattr(model, "config", None), "hidden_size", 64)).to(device)
    hook_diff = engine.verify_hook_equivalence(_x)
    if (not math.isfinite(hook_diff)) or hook_diff > 0.05:
        engine.mode = "exact"
        _log(tr("hook_check_fail", d=hook_diff))
    else:
        _log(tr("hook_check_ok", d=hook_diff))

    # ---------------------------------------------------------- 外部パラメータ (条件⑭)
    focus_init: AblationParams | None = None
    apply_only = False
    if settings.params_from:
        from .params_io import (card_to_params, check_architecture_compatibility,
                                parse_model_card)
        _log("\n" + tr("params_section", src=settings.params_from))
        card = parse_model_card(settings.params_from)
        _log(card.summary())
        with open(model_dir / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)
        warn = check_architecture_compatibility(card, cfg)
        if warn:
            _log(tr("warn_prefix", msg=warn))
        card_params = card_to_params(card, registry.n_layers)
        if settings.params_mode == "apply":
            apply_only = True
            focus_init = card_params
            _log(tr("params_mode_apply"))
        else:
            focus_init = card_params
            _log(tr("params_mode_focus"))

    # ---------------------------------------------------------- 探索実行 (条件⑬: チェックポイント)
    from .search import ScapeSearch
    time_budget_sec = (0.0 if export_from_ckpt else settings.time_budget_min * 60.0)
    search = ScapeSearch(
        engine=engine, evaluator=evaluator, salience=salience,
        checkpoint_path=ckpt_path, trials=trials,
        target_refusal=settings.target_refusal, penalty=settings.penalty,
        seed=settings.seed or 0, components=settings.components,
        focus_init=focus_init, use_cache=True, quiet=settings.quiet,
        monitor=guard.tick, time_budget_sec=time_budget_sec,
        on_ready=_finalize_int4_budget)

    t_search = time.perf_counter()
    search_seconds_total = None
    if apply_only:
        result_params = focus_init
        _log("\n" + tr("apply_section"))
        engine.apply(result_params)
        m = evaluator.evaluate(result_params, use_cache=False)
        _log(tr("apply_metrics", r=m["refusal_soft"], k=m["kl"]))
        engine.reset()
        result = {"best_params": result_params, "best_metrics": {
            "refusal_soft": m["refusal_soft"], "kl": m["kl"], "J": 0.0},
            "history": [], "attempts": 0}
    elif export_from_ckpt:
        # [v9⑦] 探索フェーズをスキップし、チェックポイントの記録から候補を
        # 選んで書き出す。評価器は軽量モード (全文生成検証のみに使用)。
        try:
            with open(ckpt_path, encoding="utf-8") as f:
                ck = json.load(f)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(tr("resume_fail", e=e)) from e
        best_params = (AblationParams.from_dict(ck["best_params"])
                       if ck.get("best_params") else None)
        if best_params is None:
            raise RuntimeError(tr("search_eval_fail"))
        result = {"best_params": best_params,
                  "best_metrics": ck.get("best_metrics"),
                  "best_J": ck.get("best_J"),
                  "history": ck.get("history", []),
                  "attempts": int(ck.get("attempt", 0) or 0),
                  "interrupted": False, "budget_stopped": False}
        result_params = best_params
        try:
            _log("\n" + tr("v9_export_from_ckpt", n=result["attempts"],
                            j=f"{float(ck.get('best_J')):.5f}"))
        except Exception:
            _log("\n" + tr("v9_export_from_ckpt", n=result["attempts"], j="n/a"))
    else:
        _log("\n" + tr("scape_section", n=trials))
        _log(tr("scape_checkpoint_path", path=ckpt_path))
        if time_budget_sec > 0:
            _log(tr("scape_time_budget", t=fmt_duration(time_budget_sec)))
        result = search.run(resume=True)
        result_params = result["best_params"]
        if result_params is None:
            raise RuntimeError(tr("search_eval_fail"))
        if result.get("interrupted"):
            _log("\n" + tr("interrupted"))
            _log(tr("resume_hint"))
            _write_reports(settings, result, None, None, None, [],
                           None, None, env, plan, guard, use_quant,
                           load_info, n_probe, n_verify, [], evaluator, model_dir,
                           search_seconds=search_seconds_total)
            return 130
        _log(tr("search_finished", a=result["attempts"],
                t=time.perf_counter() - t_search))
        bm = result.get("best_metrics") or {}
        _log(tr("search_best", r=bm.get("refusal_soft", float("nan")),
                k=bm.get("kl", float("nan")), j=bm.get("J", float("nan"))))
    t_search_total = time.perf_counter() - t_search
    search_seconds_total = t_search_total
    _log(tr("search_seconds_note", t=fmt_duration(t_search_total)))
    guard.tick(force=True)   # [v4] フェーズ境界のメモリ記録 (探索終了後)

    # ---------------------------------------------------------- 最終検証 (条件④: 100プロンプト全文生成)
    from .collapse import collapse_verdict, degeneration_stats
    from .select import Candidate, choose_candidate

    final_refusal = None
    baseline_refusal = None
    kl_final = None
    verified_points: list[tuple[int, float]] = []
    candidates: list[Candidate] = []
    verified_cache: dict[str, dict[str, Any]] = {}

    # 検証結果キャッシュ (長時間実行の中断再開に対応: 候補ごとに追記保存)
    if vpath.exists() and not settings.restart:
        try:
            with open(vpath, encoding="utf-8") as f:
                vdata = json.load(f)
            baseline_refusal = vdata.get("baseline_refusal")
            for key, d in vdata.get("items", {}).items():
                d["params"] = AblationParams.from_dict(d["params"])
                verified_cache[key] = d
            if verified_cache or baseline_refusal is not None:
                _log(tr("verify_cache_reuse", n=len(verified_cache)))
        except Exception:
            verified_cache = {}

    def _save_verified() -> None:
        items = {k: {**{kk: vv for kk, vv in d.items() if kk != "params"},
                     "params": d["params"].to_dict()}
                 for k, d in verified_cache.items()}
        data = {"baseline_refusal": baseline_refusal, "items": items}
        tmp = vpath.with_suffix(".tmp")
        vpath.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        tmp.replace(vpath)

    def _collect_candidate_metrics(Jv: float, att_no: int, cand_params: AblationParams,
                                   verify_full: bool,
                                   hist: dict[str, Any] | None = None) -> Candidate:
        """1候補を全文生成検証+崩壊チェックしてCandidate化する (検証済みなら再利用)。

        [v7条件⑤] 前回の検証で測定した1プロンプトあたり秒数を
        exact_refusal_rate へ prior_rate として渡し、最初のバッチ完了前から
        残り時間を表示できるようにする。
        [v9⑦] hist (探索履歴レコード) が渡された場合、KL等の代理指標は記録値
        を使う (チェックポイント書き出しモードでは評価器が軽量なため
        再計算しない)。
        """
        key = json.dumps(cand_params.to_dict(), sort_keys=True)
        if key in verified_cache:
            d = verified_cache[key]
        else:
            # [v5] 最終検証ではスクリーニングを必ず無効化 (探索用の棄却判定を
            # 検証指標に混入させない)。run()のfinallyでも解除済み。
            evaluator.screen_refusal_fn = None
            evaluator.screen_kl_fn = None
            engine.reset()
            engine.apply(cand_params)
            d: dict[str, Any] = {}
            if verify_full:
                refusal_full, responses = evaluator.exact_refusal_rate(
                    verify_prompts,
                    max_new_tokens=settings.max_new_tokens, markers=markers,
                    prior_rate=evaluator.last_verify_rate)
                d["refusal"] = refusal_full
                d["refusal_n"] = int(round(refusal_full * min(n_verify, len(verify_prompts))))
                d["degen"] = degeneration_stats(responses)
            if hist is not None:
                # [v9⑦] チェックポイント記録の代理指標をそのまま使う
                d["kl"] = _num(hist.get("kl"))
                d["kl_first"] = _num_or_none(hist.get("kl_first"))
                d["nll_ratio"] = _num_or_none(hist.get("nll_ratio"))
                d["refusal_soft"] = _num_or_none(hist.get("refusal_soft"))
                d["refusal_first"] = _num_or_none(hist.get("refusal_first"))
            else:
                # [v5] 境界キャッシュは数値等価性を起動時に検証済みなので、
                # 最終候補の評価でも使用する (v4のuse_cache=Falseより大幅に高速)。
                m = evaluator.evaluate(cand_params, use_cache=True)
                d["kl"] = m["kl"]
                d["kl_first"] = m.get("kl_first")
                d["nll_ratio"] = m.get("nll_ratio")
                d["refusal_soft"] = m.get("refusal_soft")
                d["refusal_first"] = m.get("refusal_first")
            engine.reset()
            d["J"] = Jv
            d["params"] = cand_params
            d["attempt"] = att_no
            coll, reason = collapse_verdict(
                d.get("degen"), d.get("nll_ratio"),
                d.get("kl"), d.get("kl_first"))
            d["collapsed"] = coll
            d["collapse_reason"] = reason
            verified_cache[key] = d
            _save_verified()
            guard.tick(force=True)   # [v4] 候補ごとのメモリ記録
            if verify_full:
                _log(tr("verify_candidate", a=att_no, j=Jv, r=d["refusal"],
                        rn=d["refusal_n"], rt=n_verify, kl=d["kl"],
                        collapse=(("  [collapsed] " + reason) if coll else "")))
        return Candidate(
            no=0, attempt=d["attempt"],
            refusal=(d.get("refusal") if verify_full else m_get(d, "refusal_soft", 0.0)),
            refusal_n=d.get("refusal_n", 0),
            refusal_total=(n_verify if verify_full else 0),
            verified=bool(verify_full and d.get("refusal") is not None),
            refusal_first=_num_or_none(d.get("refusal_first")),
            kl=d["kl"], kl_first=d.get("kl_first"), nll_ratio=d.get("nll_ratio"),
            degen=d.get("degen"), collapsed=d.get("collapsed", False),
            collapse_reason=d.get("collapse_reason", ""), J=d.get("J", float("inf")),
            params=d["params"])

    def m_get(d: dict[str, Any], k: str, default: float = 0.0) -> float:
        v = d.get(k, default)
        return float(v) if isinstance(v, (int, float)) else default

    def _num(v: Any) -> float:
        return float(v) if isinstance(v, (int, float)) and v == v else float("nan")

    def _num_or_none(v: Any) -> float | None:
        return float(v) if isinstance(v, (int, float)) and v == v else None

    def _fmt(v: Any, spec: str = ".3f") -> str:
        return (format(float(v), spec)
                if isinstance(v, (int, float)) and v == v else "-")

    def _ranked_candidates(topk: int) -> list[tuple[float, int, AblationParams, dict[str, Any]]]:
        """履歴から上位候補 (重複除外) を取り出す。

        [v5] スクリーニング棄却済み (J=None) の履歴は常に最下位扱い。
        [v9②] ゲートは床のない初頭トークン比 (refusal_first) で判定する
        (--verify-gate で変更可)。質量加重比 (refusal_soft) には文法床
        (~0.3-0.4) があり完全に非検閲化されたモデルでも下回れないため、
        mass側でゲートすると「全候補スキップ」が起こり得た (4B実測)。
        """
        out: list[tuple[float, int, AblationParams, dict[str, Any]]] = []
        seen: set[str] = set()
        gated = 0
        gate = float(settings.verify_gate)

        def _j_of(h: dict[str, Any]) -> float:
            j = h.get("J")
            return float(j) if isinstance(j, (int, float)) else float("inf")

        def _rf_of(h: dict[str, Any]) -> float:
            rf = h.get("refusal_first", h.get("refusal_soft"))
            return float(rf) if isinstance(rf, (int, float)) else 1.0

        ranked = sorted(result.get("history", []), key=_j_of)
        for h in ranked:
            if not h.get("params"):
                continue
            key = json.dumps(h.get("params", {}), sort_keys=True)
            if key in seen:
                continue
            if _rf_of(h) > gate:
                gated += 1
                continue
            seen.add(key)
            out.append((_j_of(h), h.get("attempt", 0),
                        AblationParams.from_dict(h["params"]), h))
            if len(out) >= topk:
                break
        if gated > 0:
            _log(tr("verify_gate_skip", n=gated, r=gate))
        return out

    if settings.verify != "none":
        _log("\n" + tr("verify_section", n=n_verify, t=settings.max_new_tokens))
        if baseline_refusal is None:
            engine.reset()
            baseline_refusal, _ = evaluator.exact_refusal_rate(
                verify_prompts,
                max_new_tokens=settings.max_new_tokens, markers=markers)
            _save_verified()
        _log(tr("verify_baseline", r=baseline_refusal))
        guard.tick(force=True)

        topk = verify_topk_eff if settings.verify == "auto" else 1
        if not apply_only and result.get("history"):
            cand_list = _ranked_candidates(topk)
        elif result_params is not None:
            cand_list = [(0.0, 0, result_params, None)]
        else:
            cand_list = []
        candidates = [_collect_candidate_metrics(Jv, att, cp, True, hist=h)
                      for Jv, att, cp, h in cand_list]
        for i, c in enumerate(sorted(candidates, key=lambda c: c.sort_key()), start=1):
            c.no = i
            verified_points.append((c.attempt, c.refusal))
    elif result_params is not None:
        # 検証なしモード: 探索指標のみで候補化 (v9③: 選択は必ず経由させる)
        bm = result.get("best_metrics") or {}
        c = Candidate(no=1, attempt=(bm.get("attempt") or result.get("attempts", 0)),
                      refusal=m_get(bm, "refusal_soft", 0.0),
                      refusal_first=_num_or_none(bm.get("refusal_first")),
                      refusal_n=0, refusal_total=0, verified=False,
                      kl=m_get(bm, "kl", float("nan")), params=result_params)
        candidates = [c]

    # ---------------------------------------------------------- 候補選択 (条件⑥) / ゲート代替 (v9②③)
    chosen: Candidate | None = None

    def _best_history() -> dict[str, Any] | None:
        ok = [h for h in result.get("history", [])
              if isinstance(h.get("J"), (int, float)) and h["J"] == h["J"]
              and h.get("params")]
        return min(ok, key=lambda h: float(h["J"])) if ok else None

    def _trial_lookup(tno: int) -> Candidate | None:
        """[v9②③] 試行番号から候補を構成する (代理指標のみ・全文生成検証なし)。"""
        for h in result.get("history", []):
            if h.get("attempt") == tno and h.get("params"):
                return Candidate(
                    no=0, attempt=tno,
                    refusal=m_get(h, "refusal_soft", 0.0),
                    refusal_first=_num_or_none(h.get("refusal_first")),
                    refusal_n=0, refusal_total=0, verified=False,
                    kl=_num(h.get("kl")),
                    J=(float(h["J"]) if isinstance(h.get("J"), (int, float))
                       else float("inf")),
                    params=AblationParams.from_dict(h["params"]))
        return None

    def _on_more_trials(n_extra: int) -> list[Candidate]:
        search.trials = search.attempt + n_extra
        nonlocal result
        _log(tr("extra_trials", a=search.attempt, n=search.trials))
        result = search.run(resume=True)
        topk = verify_topk_eff if settings.verify == "auto" else 1
        cand_list = _ranked_candidates(topk + len(verified_cache))
        new_c = [_collect_candidate_metrics(Jv, att, cp, True, hist=h)
                 for Jv, att, cp, h in cand_list]
        for i, c in enumerate(sorted(new_c, key=lambda c: c.sort_key()), start=1):
            c.no = i
            if settings.verify != "none":
                verified_points.append((c.attempt, c.refusal))
        return sorted(new_c, key=lambda c: c.sort_key())

    def _gate_fallback() -> Any:
        """[v9②] 全候補が代理指標ゲートを通らなかった場合のユーザー判断。

        TTY: 1) 試行を追加して探索を続行 2) 終了 (書き出さない)
             3) 試行を指定して全文生成検証→書き出し 4) 試行を指定して検証なしで書き出し
        非TTY + --select auto : 最良J試行を検証なしで自動書き出し (明示的な自動指定)
        非TTY + 既定 (interactive): モデルを書き出さず終了 (無断書き出しの防止)
        戻り値: Candidate / None (書き出さない) / "retry" (候補再構築後に再度選択)
        """
        history = [h for h in result.get("history", []) if h.get("params")]
        if not history:
            return None
        gate = float(settings.verify_gate)
        best = _best_history()
        _log("\n" + tr("v9_gate_none_title", g=gate, n=len(history)))
        if best is not None:
            _log(tr("v9_gate_best_line", a=best.get("attempt", 0),
                    rs=_fmt(best.get("refusal_soft")),
                    rf=_fmt(best.get("refusal_first")),
                    kl=_fmt(best.get("kl"), ".4f")))
        is_tty = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
        if not is_tty:
            if settings.select == "auto":
                if best is None:
                    return None
                _log(tr("v9_gate_noexport_auto", a=best.get("attempt", 0)))
                return _trial_lookup(int(best.get("attempt", 0)))
            _log(tr("v9_gate_noexport_pipe"))
            return None
        _log(tr("v9_gate_opt1"))
        _log(tr("v9_gate_opt2"))
        _log(tr("v9_gate_opt3"))
        _log(tr("v9_gate_opt4"))
        while True:
            try:
                raw = input(tr("v9_gate_prompt")).strip().lower()
            except (EOFError, KeyboardInterrupt):
                return None
            if raw in ("", "1"):
                if apply_only:
                    _log(tr("cand_add_disabled"))
                    continue
                # 試行を追加して探索を続行 → 終了後に候補一覧から選び直す
                try:
                    nstr = input(tr("cand_add_n_prompt", n=50)).strip()
                except (EOFError, KeyboardInterrupt):
                    nstr = ""
                try:
                    n_extra = int(nstr) if nstr else 50
                except ValueError:
                    _log(tr("cand_need_number"))
                    continue
                if n_extra <= 0:
                    _log(tr("cand_le_zero"))
                    continue
                _log(tr("cand_add_running", n=n_extra))
                try:
                    new_c = _on_more_trials(n_extra)
                except Exception as e:  # noqa: BLE001
                    _log(tr("cand_add_fail", e=e))
                    continue
                if new_c:
                    nonlocal candidates
                    candidates = new_c
                    return "retry"
                continue
            if raw == "2":
                return None
            if raw in ("3", "4"):
                ranked = sorted(
                    history,
                    key=lambda h: (float(h["J"]) if isinstance(h.get("J"), (int, float))
                                   and h["J"] == h["J"] else float("inf")))
                _log(tr("v9_gate_trials_header"))
                for h in ranked[:10]:
                    _log(tr("v9_gate_trial_row", a=h.get("attempt", 0),
                            rs=_fmt(h.get("refusal_soft")), rf=_fmt(h.get("refusal_first")),
                            kl=_fmt(h.get("kl"), ".4f"),
                            j=_fmt(h.get("J"), ".4f")))
                try:
                    tno_raw = input(tr("v9_gate_trial_prompt")).strip().lstrip("#")
                except (EOFError, KeyboardInterrupt):
                    return None
                if not tno_raw.isdigit():
                    _log(tr("v9_gate_bad"))
                    continue
                tno = int(tno_raw)
                h = next((x for x in history if x.get("attempt") == tno), None)
                if h is None:
                    _log(tr("v9_gate_trial_bad", i=tno))
                    continue
                cand_params = AblationParams.from_dict(h["params"])
                if raw == "3":
                    _log(tr("v9_gate_verify_running", i=tno, n=n_verify))
                    cand = _collect_candidate_metrics(
                        float(h.get("J", 0.0) or 0.0), tno, cand_params, True)
                    _log(tr("v9_gate_verify_done", i=tno,
                            r=_fmt(cand.refusal, ".1%") if cand.verified
                            else _fmt(cand.refusal), k=cand.kl))
                    try:
                        confirm = input(tr("v9_gate_write_confirm", i=tno)).strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        confirm = ""
                    if confirm in ("y", "yes"):
                        return cand
                    continue
                try:
                    confirm = input(tr("v9_gate_write_confirm", i=tno)).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    confirm = ""
                if confirm in ("y", "yes"):
                    return _trial_lookup(tno)
                continue
            _log(tr("v9_gate_bad"))

    # [v9③] --select auto にしない限り、モデルの書き出しは必ず候補選択を経由する
    # (検証なしモードも含む)。全候補がゲートを通らない場合は ② の判断メニュー
    # を出し、無断でモデルを書き出さない。
    while True:
        if candidates:
            chosen = choose_candidate(
                candidates,
                interactive=(settings.select == "interactive"),
                on_more_trials=(_on_more_trials if not apply_only else None),
                trial_lookup=_trial_lookup)
            break
        if settings.verify != "none" and result.get("history"):
            act = _gate_fallback()
            if act == "retry":
                continue   # 追加試行で候補が再構築された → 選択へ戻る
            chosen = act
            break
        break

    if chosen is None:
        _log(tr("cand_saved_none"))
        _write_reports(settings, result, None, None, None, verified_points,
                       baseline_refusal, None, env, plan, guard, use_quant,
                       load_info, n_probe, n_verify, candidates, evaluator, model_dir,
                       search_seconds=search_seconds_total)
        _log(tr("cand_done_none"))
        return 0
    result_params = chosen.params
    if settings.verify != "none" and chosen.verified:
        final_refusal = chosen.refusal
    kl_final = chosen.kl
    _log("\n" + tr("final_choice", a=chosen.attempt,
                   r=(final_refusal if final_refusal is not None else chosen.refusal),
                   k=kl_final))

    # ---------------------------------------------------------- 出力 (条件⑰: 元モデル保護)
    output_dir = settings.output_dir
    if not settings.no_export and not settings.dry_run:
        from .modelio import OutputDirError, export_model, export_params_card
        key_map = _safetensors_key_map(model, model_dir, registry)
        _log(tr("export_keys", n=len(key_map)))
        # [v8条件③] 出力先に問題がある場合もエラーで終了しない。探索・検証に
        # 長時間をかけた成果 (非検閲化モデル) を失わせないため、パスの再入力
        # (対話時) または自動修正 (非対話時) を行って再試行する。
        # なお出力先が「存在するが空」のフォルダの場合は modelio 側で
        # エラーにせずそのまま使用する。
        exported = False
        for _repair in range(8):
            _log("\n" + tr("export_section", dir=output_dir))
            try:
                export_model(model_dir, output_dir, key_map, result_params, engine,
                             overwrite=settings.overwrite_output)
                exported = True
                break
            except OutputDirError as e:
                fixed = _repair_output_path(output_dir, e)
                if fixed is None:
                    _log(tr("export_skipped"))
                    break
                output_dir = fixed
        if exported:
            params_record = {
                "parameters": {
                    "direction_mode": result_params.direction_mode,
                    "direction_index": (round(result_params.direction_index, 2)
                                        if result_params.direction_mode == "global" else "per layer"),
                    **{f"{comp}.{k}": v for comp, kp in result_params.kernels.items()
                       for k, v in [("center", round(kp.center, 2)), ("radius", round(kp.radius, 2)),
                                    ("strength", round(kp.strength, 3)),
                                    ("shape", getattr(kp, "shape", "cosine"))]},
                },
                "metrics": {
                    "Refusals (full generation)": (
                        f"{chosen.refusal_n}/{n_verify}" if chosen and chosen.refusal_total else "n/a"),
                    "KL divergence": (f"{kl_final:.4f}" if kl_final is not None else "n/a"),
                    "Collapse check": (format_collapse_text(chosen) if chosen else "n/a"),
                },
            }
            # HuggingFace Hub形式のモデルカード (frontmatterのtagsに
            # Tema_Q-Abliteratedを追加、GitHubリンク・使い方・速度を英語で記載)
            export_params_card(output_dir, params_record,
                               model_dir=model_dir, base_model_ref=base_model_ref,
                               version=__version__)
            _log(tr("export_done", dir=output_dir))

            # HuggingFace Hubへのアップロード (対話: リポジトリ名+APIキー / 非対話: フラグ)
            # 失敗してもパイプラインは続行する (ローカル出力は残っている)。
            try:
                from .hfupload import maybe_upload
                maybe_upload(settings, output_dir)
            except Exception as e:  # noqa: BLE001
                _log(tr("hf_upload_fail", err=f"{type(e).__name__}: {e}", d=output_dir))

    # ---------------------------------------------------------- レポート (条件⑥)
    _write_reports(settings, result, result_params, final_refusal, kl_final,
                   verified_points, baseline_refusal, chosen, env, plan, guard,
                   use_quant, load_info, n_probe, n_verify, candidates, evaluator, model_dir,
                   search_seconds=search_seconds_total)

    _log("\n" + tr("done"))
    return 0


def format_collapse_text(chosen) -> str:
    if chosen is None:
        return "n/a"
    if chosen.collapsed:
        return f"DEGENERATED ({chosen.collapse_reason})"
    return "OK"


def _preload_resume_menu(ckpt_path: Path, vpath: Path, trials: int) -> tuple[str, int]:
    """[v9⑦] チェックポイントが残っているモデルの再実行方法を、モデルの
    ロード前に選ばせる (v8はロード後だったため、書き出しだけのつもりでも
    数十分のロード+評価器構築を待たされていた)。

    戻り値: (choice, n)
      - ("resume", trials)  : 続きから再開 (既定・非対話環境)
      - ("extend", 新試行数) : 試行を追加して延長
      - ("export", ckpt試行数): 探索をスキップし前回の候補から書き出す
      - ("fresh",  trials)  : チェックポイントを破棄して最初から
      - ("quit",  0)        : 何もせず終了
    """
    is_tty = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
    try:
        with open(ckpt_path, encoding="utf-8") as f:
            ck = json.load(f)
        attempts = int(ck.get("attempt", 0) or 0)
    except Exception:
        return ("resume", trials)
    if attempts <= 0:
        return ("resume", trials)
    if not is_tty:
        return ("resume", trials)

    best_j = ck.get("best_J")
    try:
        best_j_txt = f"{float(best_j):.5f}" if best_j is not None else "n/a"
    except Exception:
        best_j_txt = "n/a"
    saved_at = ck.get("saved_at")
    try:
        age_txt = fmt_duration(max(0.0, time.time() - float(saved_at)))
    except Exception:
        age_txt = "?"
    n_verified = 0
    if vpath.exists():
        try:
            with open(vpath, encoding="utf-8") as f:
                n_verified = len(json.load(f).get("items", {}))
        except Exception:
            n_verified = 0

    _log("\n" + tr("v9_preload_menu_title", a=attempts, j=best_j_txt,
                    t=age_txt, v=n_verified))
    _log(tr("v9_preload_before_load"))
    if attempts >= trials:
        _log(tr("v9_preload_menu_done"))
    _log(tr("v9_preload_menu_opt1", n=trials))
    _log(tr("v9_preload_menu_opt2", a=attempts))
    _log(tr("v9_preload_menu_opt3"))
    _log(tr("v9_preload_menu_opt4"))
    _log(tr("v9_preload_menu_opt5"))

    default_extra = 50
    while True:
        try:
            raw = input(tr("v9_preload_menu_prompt")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return ("resume", trials)
        if raw == "" or raw == "1":
            _log(tr("resume_menu_go_resume", n=trials))
            return ("resume", trials)
        if raw == "2":
            try:
                nstr = input(tr("v9_preload_menu_extra_n", n=default_extra)).strip()
            except (EOFError, KeyboardInterrupt):
                nstr = ""
            try:
                extra = int(nstr) if nstr else default_extra
            except ValueError:
                _log(tr("v9_preload_menu_need_number"))
                continue
            if extra <= 0:
                _log(tr("v9_preload_menu_need_number"))
                continue
            total = attempts + extra
            _log(tr("resume_menu_go_extend", n=extra, a=total))
            return ("extend", total)
        if raw == "3":
            _log(tr("resume_menu_go_export"))
            return ("export", attempts)
        if raw == "4":
            return ("fresh", trials)
        if raw == "5":
            return ("quit", 0)
        _log(tr("v9_preload_menu_bad"))


def _repair_output_path(bad_dir: Path, err: Exception) -> Path | None:
    """[v8条件③] 出力先パスの問題を修復する新しいパスを返す。

    対話環境 (TTY) ではユーザーに新しいパスを入力させる (空入力 = 同じ
    パスで再試行。フォルダを手動で空にした場合などを想定)。
    非対話環境では「-2」「-3」…の連番サフィックスで自動修正する。
    修復を諦めた場合は None を返す (呼び出し側は出力をスキップ)。
    """
    _log(tr("export_path_problem", msg=err))
    is_tty = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
    if is_tty:
        try:
            raw = input(tr("export_repair_prompt")).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw.lower() in ("q", "quit", "exit", "n", "no"):
            return None
        return Path(raw) if raw else bad_dir
    for i in range(2, 30):
        cand = bad_dir.with_name(f"{bad_dir.name}-{i}")
        if not cand.exists():
            _log(tr("export_autosuffix", d=cand))
            return cand
    return None


def _int4_stats_safe(model: nn.Module | None = None) -> dict[str, Any] | None:
    """[v7条件⑪] INT4実行の実績 (パック量・キャッシュ命中率・フットプリント) を集計する。

    E2Eテストはこの値を使って「INT4がbf16より遅く・メモリも多い」という
    v6の退化が再発していないかを機械的に検証する。modelを渡すと
    非量子化パラメータ (bf16維持分) のバイト数も記録し、重みフットプリント
    (kept + packed) の比較を可能にする。
    """
    try:
        from .quant4 import PackedInt4Linear
        stats = PackedInt4Linear.stats()
        if stats.get("modules", 0) <= 0:
            return None
        out = {
            "modules": stats["modules"],
            "packed_bytes": PackedInt4Linear.total_packed_bytes(),
            "dequant_total_bytes": PackedInt4Linear.total_dequant_bytes(),
            "cache_hits": stats["hits"],
            "cache_misses": stats["misses"],
            "cache_used_bytes": stats["cached_bytes"],
            "cache_budget_bytes": stats["budget_bytes"],
        }
        if model is not None:
            try:
                out["kept_bf16_param_bytes"] = int(
                    sum(p.numel() * p.element_size() for p in model.parameters()))
            except Exception:
                pass
        return out
    except Exception:
        return None


def _write_reports(settings: Settings, result: dict[str, Any], result_params,
                   final_refusal, kl_final, verified_points, baseline_refusal,
                   chosen, env: EnvReport, plan, guard, use_quant, load_info,
                   n_probe: int, n_verify: int, candidates, evaluator, model_dir: Path,
                   search_seconds: float | None = None) -> None:
    """レポートPNGと実行記録JSONを書き出す。"""
    if settings.no_report:
        return
    from .report import memory_timeline_png, refusal_curve_png
    report_dir = (Path(settings.report_file).parent if settings.report_file
                  else Path(settings.workdir) / "report")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(settings.report_file) if settings.report_file \
        else report_dir / "refusal_curve.png"
    refusal_curve_png(
        result.get("history", []), report_path,
        title=tr("report_title_refusal", model=model_dir.name),
        verified_points=verified_points,
        baseline_rate=baseline_refusal, final_rate=final_refusal)
    _log("\n" + tr("report_section", p=report_path))

    mem_path = report_dir / "memory_timeline.png"
    try:
        memory_timeline_png(guard.history, mem_path,
                            title=tr("report_title_memory", model=model_dir.name))
        _log(tr("report_section", p=mem_path))
    except Exception as e:  # noqa: BLE001
        _log(tr("report_skip_memory", etype=type(e).__name__))

    # 実行記録 (JSON)
    summary = {
        "version": __version__,
        "model": settings.model, "model_dir": str(model_dir),
        "language": settings.language,
        "trials_requested": settings.trials or "auto",
        "time_budget_min": settings.time_budget_min,
        "attempts": result.get("attempts"),
        "search_seconds": (round(search_seconds, 1) if search_seconds else None),
        "interrupted": bool(result.get("interrupted")),
        "budget_stopped": bool(result.get("budget_stopped")),
        "best_params": (result_params.to_dict() if result_params is not None else None),
        "best_metrics": result.get("best_metrics"),
        "final_refusal_full_generation": final_refusal,
        "baseline_refusal_full_generation": baseline_refusal,
        "kl_final": kl_final,
        "kl_tokens": settings.kl_tokens,
        "n_probe_explore": n_probe,
        "n_verify_prompts": n_verify,
        "verify_prompts_source": (settings.verify_dataset or "builtin"),
        "eval_quant": use_quant,
        "load_plan": {"mode": plan.mode, "dtype": plan.dtype, "quant": plan.quant,
                      "est_model_gb": round(plan.est_model_gb, 2), "reason": plan.reason},
        "load_info": load_info,
        "memory": guard.summary(),
        "environment": env.as_dict(),
        "cache_rebuilds": evaluator.cache_rebuilds,
        "partial_forward": bool(evaluator.manual_ok),
        "int4_stats": _int4_stats_safe(getattr(evaluator, "model", None)),
        "candidates": [
            {"no": c.no, "attempt": c.attempt,
             "refusal": (round(c.refusal, 4) if c.refusal == c.refusal else None),
             "refusal_n": c.refusal_n, "kl": (round(c.kl, 6) if c.kl == c.kl else None),
             "kl_first": (round(c.kl_first, 6) if isinstance(c.kl_first, float) else None),
             "nll_ratio": (round(c.nll_ratio, 4) if isinstance(c.nll_ratio, float) else None),
             "collapsed": c.collapsed, "collapse_reason": c.collapse_reason,
             "selected": (chosen is not None and c.no == chosen.no)}
            for c in candidates
        ],
    }
    spath = Path(settings.workdir) / "report" / "run_summary.json"
    with open(spath, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    _log(tr("report_summary_path", p=spath))


def main(argv: list[str] | None = None) -> int:
    """エントリポイント。"""
    _ensure_utf8_stdio()
    try:
        settings, _ = load_settings(argv)
        return run_pipeline(settings)
    except KeyboardInterrupt:
        _log("\n" + tr("interrupted"))
        _log(tr("resume_hint"))
        return 130
    except Exception as e:  # noqa: BLE001
        import traceback
        _log("\n" + tr("error", etype=type(e).__name__, e=e))
        if "--debug" in (sys.argv or []):
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
