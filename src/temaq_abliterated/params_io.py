# -*- coding: utf-8 -*-
"""外部モデルカード (README.md) の Abliteration parameters 読み込み (条件⑭)。

公開されている非検閲化モデルのモデルカードには、生成に使われた
パラメータ一覧が Markdown表で記録されている。本モジュールは次の2形式を解釈する:

形式A (単一方向):
    | **direction_index** | per layer または数値 |
    | **attn.o_proj.max_weight** | 1.49 |
    | **attn.o_proj.max_weight_position** | 24.57 |
    | **attn.o_proj.min_weight** | 0.92 |
    | **attn.o_proj.min_weight_distance** | 15.70 |
    | **mlp.down_proj.*** | ... |

形式B (多重方向):
    | **attn.o_proj.max_weights.0** | 0: 1.04 |
    | **attn.o_proj.max_weights.1** | 1: 1.49 |
    | ... (min_weights.N も同様)

読み込んだ値は次の2通りに使える:
  - apply :  そのまま適用して探索せずに非検閲化する
  - focus:  値の周辺に探索範囲を絞って探索する
"""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .msg import tr
from typing import Any

from .engine import AblationParams, KernelParams

# 外部カードのコンポーネント名 -> 本ツールの内部コンポーネント名
COMPONENT_MAP = {
    "attn.o_proj": "attn_out",
    "mlp.down_proj": "mlp_down",
}


@dataclass
class ParsedCard:
    """モデルカードから解析したパラメータ。"""

    direction_index: float | None = None      # None = "per layer"
    single: dict[str, dict[str, float]] = field(default_factory=dict)  # 形式A: comp -> {max_weight,...}
    multi: dict[str, dict[str, list[float]]] = field(default_factory=dict)  # 形式B: comp -> {max:[..], min:[..]}
    raw: dict[str, str] = field(default_factory=dict)
    source: str = ""
    format: str = ""

    @property
    def is_multi(self) -> bool:
        return bool(self.multi)

    def summary(self) -> str:
        lines = [tr("params_card_src", s=self.source), tr("params_card_format", f=self.format)]
        di = "per layer" if self.direction_index is None else f"{self.direction_index:.2f}"
        lines.append(f"  direction_index: {di}")
        for comp, kv in self.single.items():
            lines.append(f"  {comp}: max_weight={kv.get('max_weight')}, "
                         f"pos={kv.get('max_weight_position')}, "
                         f"min={kv.get('min_weight')}, dist={kv.get('min_weight_distance')}")
        for comp, kv in self.multi.items():
            lines.append(f"  {comp}(multi): max={kv.get('max')}, min={kv.get('min')}")
        return "\n".join(lines)


def _fetch_text(source: str) -> str:
    """パスまたはURLからテキストを取得する。"""
    if re.match(r"^https?://", source, re.I):
        # HuggingFaceのblob URLをrawへ置換
        url = source.replace("/blob/", "/raw/")
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read().decode("utf-8", errors="replace")
    p = Path(source)
    return p.read_text(encoding="utf-8", errors="replace")


def _parse_table(text: str) -> dict[str, str]:
    """Markdown表から | key | value | を抽出する。"""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 2:
            continue
        key = cells[0].strip("*").strip()
        val = cells[1].strip("*").strip()
        # 表ヘッダー・区切り行は無視
        if not key or key.startswith(":") or val.startswith(":"):
            continue
        if key in ("Parameter", "Metric"):
            continue
        out[key] = val
    return out


def parse_model_card(source: str) -> ParsedCard:
    """モデルカード (README.mdパス/URL/本文テキスト) を解析する。"""
    text = _fetch_text(source) if (re.match(r"^https?://", source, re.I) or Path(source).exists()) else source
    card = ParsedCard(source=source)

    # Abliteration parameters セクションを探す (無ければ全体から)
    m = re.search(r"#+\s*Abliteration parameters(.*?)(?=\n#|\Z)", text, re.S | re.I)
    section = m.group(1) if m else text
    table = _parse_table(section)
    card.raw = table

    # direction_index
    if "direction_index" in table:
        v = table["direction_index"].lower()
        if "per layer" in v:
            card.direction_index = None
        else:
            try:
                card.direction_index = float(v.split(":")[-1])
            except ValueError:
                card.direction_index = None

    # 形式A / 形式Bの判定と読み込み
    for key, val in table.items():
        mm = re.match(r"^(attn\.o_proj|mlp\.down_proj)\.(max_weights|min_weights)\.(\d+)$", key)
        if mm:
            comp, kind, idx = mm.group(1), mm.group(2), int(mm.group(3))
            card.format = "multi-direction"
            try:
                num = float(val.split(":")[-1])
            except ValueError:
                continue
            entry = card.multi.setdefault(comp, {"max": [], "min": []})
            kindkey = "max" if kind == "max_weights" else "min"
            while len(entry[kindkey]) <= idx:
                entry[kindkey].append(0.0)
            entry[kindkey][idx] = num
            continue
        mm = re.match(r"^(attn\.o_proj|mlp\.down_proj)\.(max_weight|max_weight_position|min_weight|min_weight_distance)$", key)
        if mm:
            comp, name = mm.group(1), mm.group(2)
            card.format = card.format or "single"
            try:
                num = float(val.split(":")[-1])
            except ValueError:
                continue
            card.single.setdefault(comp, {})[name] = num

    if not card.single and not card.multi:
        raise ValueError(
            tr("params_not_detected"))
    return card


def _hat_kernel_to_params(max_weight: float, max_weight_position: float,
                          min_weight: float, min_weight_distance: float) -> KernelParams:
    """外部形式 (線形ハットカーネル) を本ツール形式 (コサインバンプ) へ変換する。

    対応関係: 中心=位置 / 半径=距離 / 強度=最大重み。
    外部形式はバンド端でも min_weight>0 の値を持つが、本ツールのカーネルは
    端で0に到達する。平均的な効き具合を合わせるため、強度を
    (max_weight + min_weight)/2 で丸めた中心強度として近似する。
    """
    center = float(max_weight_position)
    radius = max(1.0, float(min_weight_distance))
    strength = max(0.0, float(max_weight))
    return KernelParams(center=center, radius=radius, strength=strength)


def card_to_params(card: ParsedCard, n_layers: int) -> AblationParams:
    """解析済みカードを本ツールのAblationParamsへ変換する。"""
    kernels: dict[str, KernelParams] = {}
    if card.is_multi:
        # 形式B: 各方向の重みの平均を1つの代表強度へ集約し、方向はdirection_indexを使う
        for comp, entry in card.multi.items():
            mx = entry.get("max", [])
            mn = entry.get("min", [])
            s = sum(mx) / len(mx) if mx else 0.0
            pos = None
            kernels[COMPONENT_MAP.get(comp, "attn_out")] = KernelParams(
                center=float(n_layers * 0.7), radius=float(max(2.0, n_layers * 0.3)), strength=float(s))
    for comp, kv in card.single.items():
        internal = COMPONENT_MAP.get(comp, comp)
        if internal in kernels and not card.is_multi:
            pass
        kernels[internal] = _hat_kernel_to_params(
            kv.get("max_weight", 0.0), kv.get("max_weight_position", n_layers * 0.7),
            kv.get("min_weight", 0.0), kv.get("min_weight_distance", n_layers * 0.3))

    mode = "per_layer" if card.direction_index is None else "global"
    di = 0.0 if card.direction_index is None else card.direction_index
    return AblationParams(direction_mode=mode, direction_index=di, kernels=kernels)


def focus_bounds_adjustment(params: AblationParams, frac: float = 0.15) -> dict[str, tuple[float, float]]:
    """focusモード用: 各座標の探索範囲を初期値周辺へ狭める。戻り値: 座標->(lo,hi)"""
    out = {}
    for comp, kp in params.kernels.items():
        out[f"{comp}.center"] = (kp.center * (1 - frac), kp.center * (1 + frac))
        out[f"{comp}.radius"] = (max(1.0, kp.radius * (1 - frac)), kp.radius * (1 + frac))
        out[f"{comp}.strength"] = (max(0.0, kp.strength * (1 - frac * 2)), kp.strength * (1 + frac * 2))
    return out


def check_architecture_compatibility(card: ParsedCard, model_config: dict[str, Any]) -> str | None:
    """外部パラメータの適用前にアーキテクチャ類似性を確認する (警告文字列を返す)。"""
    archs = model_config.get("architectures", [])
    n_layers = model_config.get("num_hidden_layers", 0)
    # カードの max_weight_position が明らかに層数を超えていないか
    for comp, kv in card.single.items():
        pos = kv.get("max_weight_position", 0)
        if n_layers and pos > n_layers * 1.5:
            return tr("params_arch_warn", pos=pos, n=n_layers)
    return None
