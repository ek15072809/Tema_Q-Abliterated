# -*- coding: utf-8 -*-
"""実行環境の検出と最適設定の自動決定 (条件⑦) + ライブラリ警告ノイズの抑制 (v3)。

CPUコア数・RAM・GPU種別・OSを検出し、評価dtype・スレッド数・
バッチサイズ・プローブ数などの実行パラメータを自動決定する。

v3: transformers/torch が大量に出す「動作には問題ないが繰り返し表示される」
警告 (最適化カーネル未導入のフォールバック通知、_pytreeのEnum非推奨等) を
ログフィルタで抑制し、要点だけを1行にまとめて表示する。
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any

from .msg import tr, tr_text


def _ensure_utf8_stdio() -> None:
    """Windowsコンソールで日本語出力が文字化けしないようUTF-8へ切替える。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                enc = (getattr(stream, "encoding", "") or "").lower()
                if enc not in ("utf-8", "utf8"):
                    stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


# ---------------------------------------------------------------- 警告ノイズ抑制 (v3)

# 「正しいが遅い」だけで繰り返し表示される既知メッセージのパターン
_KERNEL_NOISE_SUBSTRINGS = (
    "is falling back to its reference PyTorch implementation",
    "This is correct but much slower",
)
_MISC_NOISE_SUBSTRINGS = (
    # torch.utils._pytree: Enum定数のregister_constant非推奨 (W0927系)
    "register_constant",
    "torch.utils._pytree",
)


class _NoiseFilter(logging.Filter):
    """既知の無害な警告レコードだけを落とすログフィルタ。"""

    def __init__(self, stats: dict):
        super().__init__()
        self.stats = stats

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        for pat in _KERNEL_NOISE_SUBSTRINGS:
            if pat in msg:
                self.stats["kernel_fallback"] += 1
                return False
        for pat in _MISC_NOISE_SUBSTRINGS:
            if pat in msg and record.name.startswith(("transformers", "torch")):
                self.stats["misc"] += 1
                return False
        return True


NOISE_STATS: dict[str, int] = {"kernel_fallback": 0, "misc": 0}
_installed_handler_ids: set[int] = set()
_shared_filter: "_NoiseFilter | None" = None


def suppress_library_noise() -> None:
    """transformers / torch の繰り返し警告ノイズを抑制する (何度呼んでも安全)。

    - カーネルフォールバック通知 (causal_conv1d_fn / chunk_gated_delta_rule /
      fused_recurrent_gated_delta_rule 等が最適化カーネル未導入のため
      リファレンス実装で動く旨の「正しいが遅い」警告) → 抑制し要約1行を出す
    - torch._pytree の Enum register_constant 非推奨警告 → 抑制

    transformers は遅延importされるため、ハンドラは import 前後で変わる。
    この関数はハンドラ単位で冪等であり、モデル読み込み後にも呼ぶこと。
    """
    global _shared_filter
    if _shared_filter is None:
        _shared_filter = _NoiseFilter(NOISE_STATS)
        # warningsモジュール経由のもの (冪等)
        warnings.filterwarnings("ignore", message=".*register_constant.*")
        warnings.filterwarnings("ignore", message=".*_pytree.*",
                                category=DeprecationWarning)
        warnings.filterwarnings(
            "ignore",
            message=".*is falling back to its reference PyTorch implementation.*")
    f = _shared_filter

    # logging経由のもの: 子ロガー (transformers.integrations.hub_kernels 等) の
    # レコードは "transformers" ルートロガーのハンドラで処理されるため、
    # ロガーと「ハンドラ」の両方にフィルタを付ける。
    for name in ("transformers", "torch", ""):
        lg = logging.getLogger(name) if name else logging.getLogger()
        lg.addFilter(f)
        for h in lg.handlers:
            if id(h) not in _installed_handler_ids:
                h.addFilter(f)
                _installed_handler_ids.add(id(h))


def noise_summary_lines() -> list[str]:
    """抑制した警告の要約 (呼び出し側で表示する)。"""
    lines = []
    if NOISE_STATS["kernel_fallback"] > 0:
        lines.append(tr("kernel_fallback_note"))
    return lines


@dataclass
class EnvReport:
    """検出された環境情報と、そこから導かれる実行設定。"""

    os_name: str = ""
    python_version: str = ""
    torch_version: str = ""
    device: str = "cpu"
    device_name: str = ""
    vram_gb: float = 0.0
    cpu_name: str = ""
    cores_physical: int = 1
    cores_logical: int = 1
    ram_gb: float = 0.0
    # 自動決定される実行設定
    eval_dtype: str = "bfloat16"       # 評価モデルのdtype
    use_int8_eval: bool = False        # 探索フェーズをint8量子化で行うか
    torch_threads: int = 0             # 0なら変更しない
    batch_size: int = 8                # 評価バッチサイズ
    max_batch_size: int = 32
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            tr("env_os", v=self.os_name),
            tr("env_python", v=self.python_version),
            tr("env_torch", v=self.torch_version),
            tr("env_device", v=self.device) + (f" ({self.device_name})" if self.device_name else ""),
        ]
        if self.device.startswith("cuda"):
            lines.append(tr("env_vram", v=f"{self.vram_gb:.1f}"))
        lines += [
            tr("env_cpu", v=self.cpu_name, p=self.cores_physical, l=self.cores_logical),
            tr("env_ram", v=f"{self.ram_gb:.1f}"),
            tr("env_batch", v=self.batch_size),
        ]
        for n in self.notes:
            lines.append(tr("note_prefix", msg=tr_text(n)))
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "os": self.os_name, "python": self.python_version, "torch": self.torch_version,
            "device": self.device, "device_name": self.device_name, "vram_gb": self.vram_gb,
            "cpu": self.cpu_name, "cores_physical": self.cores_physical,
            "cores_logical": self.cores_logical, "ram_gb": self.ram_gb,
            "eval_dtype": self.eval_dtype, "use_int8_eval": self.use_int8_eval,
            "batch_size": self.batch_size,
        }


def detect_environment(auto_device: bool = True, prefer_device: str | None = None) -> EnvReport:
    """環境を検出し、実行設定を決定する。"""
    rep = EnvReport()
    rep.os_name = f"{platform.system()} {platform.release()}"
    rep.python_version = platform.python_version()

    try:
        import torch
        rep.torch_version = torch.__version__
    except ImportError as e:
        raise RuntimeError(tr("torch_missing")) from e

    # デバイス判定
    if prefer_device:
        rep.device = prefer_device
    elif auto_device:
        if torch.cuda.is_available():
            rep.device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            rep.device = "mps"
        else:
            rep.device = "cpu"
    else:
        rep.device = "cpu"

    if rep.device == "cuda":
        rep.device_name = torch.cuda.get_device_name(0)
        rep.vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        rep.eval_dtype = "bfloat16"
        rep.use_int8_eval = False
        rep.batch_size = 32
        rep.max_batch_size = 64
    elif rep.device == "mps":
        rep.device_name = "Apple Silicon"
        rep.eval_dtype = "float16"
        rep.batch_size = 16
    else:
        # CPU環境
        try:
            import psutil
            rep.ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            rep.cores_physical = psutil.cpu_count(logical=False) or 1
            rep.cores_logical = psutil.cpu_count(logical=True) or 1
            if hasattr(platform, "machine"):
                rep.cpu_name = platform.machine()
        except ImportError:
            rep.ram_gb = 0.0
            rep.cores_logical = os.cpu_count() or 1
            rep.cores_physical = max(1, rep.cores_logical // 2)

        # CPU名の取得 (py-cpuinfoがあれば)
        try:
            import cpuinfo  # type: ignore
            info = cpuinfo.get_cpu_info()
            rep.cpu_name = info.get("brand_raw", rep.cpu_name)
        except Exception:
            pass

        rep.eval_dtype = "bfloat16"
        # RAM 16GB以下のWindows CPU環境では大きいモデル用にint8を候補にする
        # (実際の適用はモデル規模を見て modelio 側で決定)
        rep.use_int8_eval = rep.ram_gb > 0 and rep.ram_gb <= 20.0
        rep.batch_size = max(2, min(16, rep.cores_physical * 4))
        rep.notes.append("env_cpu_slow")

    # スレッド数: CPU時は物理コア数に合わせる (論理コアだと遅くなることが多い)
    if rep.device == "cpu":
        rep.torch_threads = rep.cores_physical

    return rep


def apply_runtime_settings(rep: EnvReport) -> None:
    """検出結果をランタイムへ反映する。"""
    import torch

    if rep.torch_threads and rep.torch_threads > 0:
        try:
            torch.set_num_threads(rep.torch_threads)
        except Exception:
            pass

    # [v7訂正] glibcのチューニング (mallopt) は行わない。
    # 一時テンソルの大半は1〜8MBで、glibcの既定 (動的mmapしきい値) では
    # 大きい割り当てがmmap→解放即返却になる。しきい値を8MBへ引き上げると
    # これらがアリーナに溜まり、サイズの異なる割り当ての交互出現で
    # 断片化が急増する (実測: 低メモリ環境でOOM)。既定のまま+周期的な
    # malloc_trim (memguard/surrogate) が最も安定する。
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.mallopt(-1, 1024 * 1024)   # M_TRIM_THRESHOLD
        libc.mallopt(-3, 8 * 1024 * 1024)  # M_MMAP_THRESHOLD
    except Exception:
        pass
