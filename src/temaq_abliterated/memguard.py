# -*- coding: utf-8 -*-
"""メモリガード (v2新設): ロード計画・使用量監視・自動削減。

条件③「Colab・CPUのどちらでも4bitでロードでき、メモリ超過による停止や
SWAP発生を自動で防ぐ」のための3機能を提供する。

1. ロード計画 (plan_load_strategy)
   モデルのsafetensors合計サイズと空きメモリから、
   (a) そのまま読めるか  (b) 読んでからint8/int4へ縮小すべきか
   (c) そもそも一括読みが不可能でストリーミングint4が必要か
   を自動判定する。

2. 使用量監視 (MemoryGuard)
   RAM/VRAM/SWAP/プロセスRSSを定期監視し、残量低下やSWAP増加を検知したら
   警告表示と自動削減 (ガベージコレクション・キャッシュ解放・重みキャッシュ
   縮小・バッチ縮小) を行う。

3. OOM耐性 (oom_retry)
   評価・生成中のメモリ不足例外を捕捉し、削減処理のうえ縮小バッチで再試行する。
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from .msg import is_jp, tr


def _safe_print(text: str) -> None:
    """[v5] 進捗live行 (ui.LINES) と衝突しないprint。循環import回避のため
    遅延importし、失敗時は通常printへフォールバックする。"""
    try:
        from .ui import print_line
        print_line(text)
    except Exception:
        try:
            print(text, flush=True)
        except Exception:
            pass


# ---------------------------------------------------------------- 基本計測


def _read_meminfo() -> dict[str, float]:
    """Linux /proc/meminfo からMemAvailable/Swapを読む (psutil無しでも動作)。"""
    out = {}
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                parts = line.split()
                if parts and parts[0] in ("MemAvailable:", "MemTotal:", "SwapFree:", "SwapTotal:"):
                    out[parts[0][:-1]] = int(parts[1]) / (1024 ** 2)  # GB
    except OSError:
        pass
    return out


def malloc_trim() -> None:
    """glibcのアロケータが確保したままの解放済みメモリをOSへ返す (Linux)。"""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def ram_available_gb() -> float:
    """空きRAM (GB)。psutil優先、なければ/proc/meminfo、最終手段は0 (不明)。"""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        return _read_meminfo().get("MemAvailable", 0.0)


def ram_total_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return _read_meminfo().get("MemTotal", 0.0)


def rss_gb() -> float:
    """本プロセスの物理メモリ使用量 (GB)。"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 3)
    except Exception:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
        except Exception:
            return 0.0


def swap_used_gb() -> float:
    try:
        import psutil
        return psutil.swap_memory().used / (1024 ** 3)
    except Exception:
        mi = _read_meminfo()
        if "SwapTotal" in mi and "SwapFree" in mi:
            return max(0.0, mi["SwapTotal"] - mi["SwapFree"])
        return 0.0


def vram_free_gb(device: str = "cuda") -> tuple[float, float]:
    """(空きVRAM, 総VRAM) GB。CUDA以外は (0,0)。"""
    if device.startswith("cuda") and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return free / (1024 ** 3), total / (1024 ** 3)
    return 0.0, 0.0


def model_disk_gb(model_dir: Path) -> float:
    """モデルのsafetensors合計サイズ (GB) — 必要メモリの推定基準。"""
    total = 0
    for f in sorted(os.listdir(model_dir)):
        if f.endswith(".safetensors"):
            total += os.path.getsize(model_dir / f)
    return total / (1024 ** 3)


# ---------------------------------------------------------------- ロード計画


@dataclass
class LoadPlan:
    """モデル読み込み戦略。"""

    mode: str = "standard"       # "standard" (一括読み) / "streamed" (テンソル毎逐次)
    dtype: str = "bfloat16"      # 評価の基本dtype
    quant: str = "none"          # "none" / "int8" / "int4" (探索・評価モデルの量子化)
    dequant_cache_gb: float = 0.0  # INT4重みキャッシュ予算
    est_model_gb: float = 0.0    # 読み込み後の推定モデルメモリ
    budget_gb: float = 0.0       # 判断に使ったメモリ予算
    reason: str = ""

    def describe(self) -> str:
        q = tr("plan_quant_none" if self.quant == "none" else
               ("plan_quant_int8" if self.quant == "int8" else "plan_quant_int4"))
        mode = tr("plan_mode_standard" if self.mode == "standard" else "plan_mode_streamed")
        return (f"{mode} / {self.dtype}" + (f" / {q}" if self.quant != "none" else "")
                + f" (est {self.est_model_gb:.1f}GB / budget {self.budget_gb:.1f}GB) - {self.reason}")


def plan_load_strategy(model_dir: Path, device: str, eval_dtype: str = "auto",
                       override_ram_gb: float | None = None,
                       override_vram_gb: float | None = None) -> LoadPlan:
    """モデルサイズと空きメモリから読み込み戦略を決定する。

    自動 (eval_dtype="auto") の判断は CPU/GPU共通で:
      1. bf16のまま予算内に収まる → 量子化なし
      2. 一括読みは可能だが予算超過 → int8 (CPUのみ) または int4 へ縮小
      3. 一括読み自体が不可能 → ストリーミングINT4 (摂動対象のみbf16のまま)
    """
    size_gb = model_disk_gb(model_dir)
    avail_ram = override_ram_gb if override_ram_gb is not None else ram_available_gb()
    vram_free, _ = vram_free_gb(device) if override_vram_gb is None else (override_vram_gb, 0)

    if device.startswith("cuda"):
        budget = vram_free * 0.82          # VRAM予算 (活性化・断片化の余裕を持たせる)
        ram_gate = avail_ram               # 一括読みにはCPU RAMも一時必要
        full_need = size_gb + 1.2          # 活性化・KV・作業バッファ込み
    else:
        budget = avail_ram * 0.72          # RAM予算 (OS・キャッシュ分の余裕)
        ram_gate = avail_ram * 0.88
        full_need = size_gb + 1.8

    # 推定メモリ (bf16モデルを想定: 量子化対象は非対象部のみ)
    est_full = full_need
    est_int8 = size_gb * 0.62 + 0.7
    est_int4 = size_gb * 0.30 + 0.35 + 0.6  # 非対象INT4 + 対象/埋め込みbf16 + 活性化

    explicit = eval_dtype in ("int4", "int8", "bfloat16", "float16", "float32")

    if explicit:
        if eval_dtype == "int4":
            # [v7] 明示指定のINT4は常にストリーミング読み込みを使う。
            # 従来は「bf16が一括読みできるなら標準ロード→後から変換」だったが、
            # この経路は (a) bf16全体+INT4の二重メモリでRSS高水位を残し、
            # (b) ロード後の空きRAMを潰して重みキャッシュ予算が小さくなり、
            # LRUスラッシング (毎forwardの再復号) を起こしていた。
            return LoadPlan("streamed", "bfloat16", "int4", 0.0, est_int4, budget,
                            tr("plan_explicit_int4_stream"))
        if eval_dtype == "int8":
            if device.startswith("cuda"):
                return LoadPlan("streamed", "bfloat16", "int4", 0.0, est_int4, budget,
                                tr("plan_int8_gpu_switch"))
            return LoadPlan("standard", "bfloat16", "int8", 0.0, est_int8, budget,
                            tr("plan_explicit_int8"))
        # fp16/bf16/fp32 明示指定
        dt_est = est_full if eval_dtype != "float32" else size_gb * 2 + 0.9
        return LoadPlan("standard", eval_dtype, "none", 0.0, dt_est, budget,
                        tr("plan_explicit_dtype", dt=eval_dtype))

    # ---- 自動 ----
    if est_full <= budget:
        return LoadPlan("standard", "bfloat16", "none", 0.0, est_full, budget,
                        tr("plan_roomy"))
    if size_gb + 0.4 <= ram_gate:
        # 一括読み可能 → 変換で縮小 (int8は標準ロードから変換。
        # int4は二重メモリを避けるためストリーミング)
        if not device.startswith("cuda") and est_int8 <= budget:
            return LoadPlan("standard", "bfloat16", "int8", 0.0, est_int8, budget,
                            tr("plan_shrink_int8"))
        if est_int4 <= budget:
            return LoadPlan("streamed", "bfloat16", "int4", 0.0, est_int4, budget,
                            tr("plan_shrink_int4"))
    if est_int4 <= budget * 1.15:
        return LoadPlan("streamed", "bfloat16", "int4", 0.0, est_int4, budget,
                        tr("plan_stream_int4"))
    return LoadPlan("streamed", "bfloat16", "int4", 0.0, est_int4, budget,
                    tr("plan_danger"))


def decide_dequant_cache_gb(plan: LoadPlan, device: str) -> float:
    """INT4使用時、空きメモリから重みキャッシュ予算を決める (無理はしない)。"""
    if plan.quant != "int4":
        return 0.0
    if device.startswith("cuda"):
        free, _ = vram_free_gb(device)
        return max(0.0, min(free * 0.30, 2.0))
    avail = ram_available_gb() - plan.est_model_gb
    if avail <= 1.2:
        return 0.0
    return max(0.0, min((avail - 1.0) * 0.5, 2.0))


# ---------------------------------------------------------------- 監視と削減


@dataclass
class MemSnapshot:
    t: float
    ram_avail: float
    rss: float
    swap: float
    vram_free: float


class MemoryGuard:
    """実行中のメモリ監視と自動削減。

    tick() を探索ループなどから定期的に呼ぶと、状態を記録し
    危険水準で自動削減を行う。履歴は summary() / png出力に使える。

    [v7]:
      - rebaseline(): ストリーミング読み込み直後にSWAP基準線を引き直す。
        v6は「プロセス起動時」のSWAP量を基準にしていたため、ロード時の
        一時的なスワップアウトが残ったまま警告を発し続け、多境界キャッシュ
        を毎試行破棄させていた (6分/試行の再構築ループ)。
      - ログ出力は「本当に危険なとき」だけに間引き (条件⑥: 探索・検証中の
        [メモリ] 行の連続表示を廃止)。
      - 危険時の削減はINT4重みキャッシュの「段階的縮小」を優先し、
        評価構造 (境界キャッシュ) はOOT時にのみ失うようにする。
    """

    def __init__(self, device: str = "cpu", log_interval_s: float = 15.0):
        self.device = device
        self.log_interval = log_interval_s
        self._t0 = time.perf_counter()
        self._swap0 = swap_used_gb()
        self._last_log = 0.0
        self._last_trim = 0.0
        self._reducers: list[Callable[[bool], None]] = []
        self.history: list[MemSnapshot] = []
        self.warnings: list[str] = []
        self.mitigations = 0
        self.oom_events = 0

    def periodic_trim(self, force: bool = False) -> None:
        """[v7] 定期的なガベージコレクション + ヒープのOS返却。

        評価・生成の一時テンソル (サイズがフェーズごとに変わる) はglibcの
        アリーナ断片化としてRSSに残り、「使っていないのにメモリを押さえた
        まま」の状態を作る (低メモリ環境でのOOM要因)。30秒ごとに
        malloc_trimでOSへ返す。重い処理の直後 (force=True) にも呼べる。
        """
        now = time.perf_counter()
        if not force and now - self._last_trim < 30.0:
            return
        self._last_trim = now
        import gc as _gc
        _gc.collect()
        malloc_trim()

    def rebaseline(self) -> None:
        """[v7] SWAP基準線を現在地点で引き直す (モデル読み込み直後に呼ぶ)。

        読み込みピークで一旦スワップアウトされたページはOSが自発的に
        巻き戻さない限り残るため、それを起因とした「偽の危険状態」を
        防ぐ (v6の毎試行キャッシュ破棄ループの主要原因)。
        """
        self._swap0 = swap_used_gb()
        self.snapshot()
        self.periodic_trim(force=True)

    # ---------------------------------------------------------------- 状態

    def snapshot(self) -> MemSnapshot:
        vfree, _ = vram_free_gb(self.device)
        s = MemSnapshot(time.perf_counter() - self._t0, ram_available_gb(),
                        rss_gb(), swap_used_gb(), vfree)
        self.history.append(s)
        return s

    def tick(self, tag: str = "", force: bool = False) -> str:
        """現状を確認し "ok" / "warn" / "critical" を返す。ログは間引き表示。"""
        s = self.snapshot()
        status = "ok"
        reasons: list[str] = []
        if s.ram_avail < 0.5:
            status, reasons = "critical", [tr("mem_reason_avail", gb=s.ram_avail)]
        elif s.ram_avail < 1.2:
            status, reasons = "warn", [tr("mem_reason_avail", gb=s.ram_avail)]
        swap_delta = s.swap - self._swap0
        if swap_delta > 0.8 and not self.device.startswith("cuda"):
            status = "critical"
            reasons.append(tr("mem_reason_swap", gb=swap_delta))
        elif swap_delta > 0.3:
            if status == "ok":
                status = "warn"
            reasons.append(tr("mem_reason_swap", gb=swap_delta))
        if self.device.startswith("cuda") and s.vram_free and s.vram_free < 0.4:
            status = "critical"
            reasons.append(tr("mem_reason_vram", gb=s.vram_free))

        now = time.perf_counter()
        # [v7条件⑥] 表示は「危険 (critical)」なときだけ (警告はsummaryにのみ記録)。
        # force=True (フェーズ境界の記録) では表示しない。
        if status == "critical" and now - self._last_log > self.log_interval:
            self._last_log = now
            msg = tr("mem_guard_line", rss=s.rss, ram=s.ram_avail,
                     vram=(f" VRAM={s.vram_free:.1f}GB" if self.device.startswith("cuda") else ""),
                     swap=s.swap,
                     reasons=((" <- " if not is_jp else " ← ") + ", ".join(reasons)) if reasons else "")
            _safe_print("  " + msg)
        if reasons and reasons[0] not in self.warnings:
            self.warnings.extend(reasons)
        if status == "critical":
            # [v7] 危険時: INT4重みキャッシュを段階的に縮小する。境界キャッシュは
            # OOM再試行時のみ失う (評価構造の再構築は高コストのため・条件①)。
            self.mitigate(hard=False)
            try:
                from .quant4 import PackedInt4Linear
                if PackedInt4Linear._WSTATE.get("budget", 0) > 0:
                    PackedInt4Linear.shrink_weight_cache(0.5)
            except Exception:
                pass
        elif status == "ok":
            # [v5] 空きRAMが回復したらINT4重みキャッシュ予算を段階的に引き上げる
            # (初期が部分キャッシュだった場合、徐々に全レイヤbf16化へ近づける)
            self._maybe_grow_int4_budget(s)
            # [v7] 定期的にヒープをOSへ返す (アリーナ断片化の抑制)
            self.periodic_trim()
        return status

    def _maybe_grow_int4_budget(self, s: MemSnapshot) -> None:
        try:
            from .quant4 import PackedInt4Linear
            state = PackedInt4Linear._WSTATE
            budget = int(state.get("budget", 0))
            target = int(state.get("target", 0))
            if budget <= 0 or target <= budget:
                return
            step = 1 << 30   # +1GBずつ
            # 成長の条件: 現在の空きRAM - 安全マージン > 予算+step
            if s.ram_avail - 2.0 > (budget + step) / (1024 ** 3):
                new_budget = min(target, budget + step)
                PackedInt4Linear.set_weight_cache_budget(new_budget)
                _safe_print(tr("memguard_budget_grow", gb=new_budget / (1024 ** 3)))
        except Exception:
            pass

    # ---------------------------------------------------------------- 削減

    def register_reducer(self, fn: Callable[[bool], None]) -> None:
        """hard=Trueのときだけ重い削減 (キャッシュ破棄等) を行う関数を登録する。"""
        self._reducers.append(fn)

    def mitigate(self, hard: bool = True) -> None:
        """メモリ削減を実行する。hard=Trueならキャッシュ類も解放する。"""
        self.mitigations += 1
        gc.collect()
        malloc_trim()
        try:
            if hasattr(torch, "cpu") and hasattr(torch.cpu, "empty_cache"):
                torch.cpu.empty_cache()
        except Exception:
            pass
        if self.device.startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        if hard:
            try:
                from .quant4 import PackedInt4Linear
                PackedInt4Linear.clear_weight_cache()
            except Exception:
                pass
        for fn in self._reducers:
            try:
                fn(hard)
            except Exception:
                pass

    def is_oom_error(self, e: BaseException) -> bool:
        if isinstance(e, (MemoryError,)):
            return True
        if isinstance(e, torch.cuda.OutOfMemoryError):
            return True
        if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
            return True
        return False

    def oom_retry(self, fn: Callable[[], Any], retries: int = 2, desc: str = "") -> Any:
        """メモリ不足で失敗した場合に削減して再試行するラッパ。"""
        last: BaseException | None = None
        for i in range(retries + 1):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                if not self.is_oom_error(e):
                    raise
                self.oom_events += 1
                last = e
                _safe_print(tr("mem_oom_retry", desc=(f" ({desc})" if desc else ""),
                               i=i + 1, n=retries))
                self.mitigate(hard=True)
        raise RuntimeError(tr("mem_oom_unresolved", last=last)) from last

    # ---------------------------------------------------------------- 集計

    def summary(self) -> dict[str, Any]:
        peak_rss = max((s.rss for s in self.history), default=0.0)
        min_avail = min((s.ram_avail for s in self.history), default=0.0)
        min_vram = min((s.vram_free for s in self.history if s.vram_free > 0), default=0.0)
        swap_delta = max((s.swap for s in self.history), default=0.0) - self._swap0
        return {
            "peak_rss_gb": round(peak_rss, 2),
            "min_available_ram_gb": round(min_avail, 2),
            "min_free_vram_gb": round(min_vram, 2),
            "swap_increase_gb": round(max(0.0, swap_delta), 2),
            "mitigations": self.mitigations,
            "oom_events": self.oom_events,
            "warnings": self.warnings[:8],
            "n_snapshots": len(self.history),
        }
