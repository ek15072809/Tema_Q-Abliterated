# -*- coding: utf-8 -*-
"""アーキテクチャ抽象化: 各種トランスフォーマー実装から
「アブレーション可能な射影モジュール」を自動発見する (条件⑮: 広範なアーキテクチャ対応)。

対応方針は属性プロービング: 各レイヤから次のモジュールを探索する。
  - attention出力射影: self_attn.o_proj / self_attn.out_proj /
                       linear_attn.out_proj / linear_attention.out_proj / conv.out_proj
  - MLP下流射影:       mlp.down_proj / mlp.shared_expert.down_proj /
                       mlp.experts[].down_proj / block_sparse_moe.experts[].w2 /
                       feed_forward.w2 / feed_forward.experts[].w2 /
                       shared_mlp.output_linear / moe.experts[].output_linear

[v6] 新アーキテクチャ対応:
  - **融合3D専門家** (Qwen4Exp / Gemma4-MoE): 専門家重みが nn.Linear ではなく
    ``nn.Parameter(num_experts, out, in)`` (Qwen4ExpTextExperts / Gemma4TextExperts)
    として格納される。これらは「 expertsコンテナ + weight属性名 」のペアで
    登録し、engineは3D摂動 (専門家スライスごとの直交化) を適用する。
  - **共有専門家** (Qwen4Exp: mlp.shared_expert, 標準MLPと同じ構造)
  - ``layer.experts`` (Gemma4-MoEはmlpの外側に experts を持つ)

対象モジュールは TargetModule で表現する:
  module     = 体重を持つモジュール (nn.Linear または専門家コンテナ)
  weight_attr= "weight" (nn.Linear) / "down_proj" (融合専門家)
  fused      = True なら3D融合専門家 (rank-1フック不可 → exactモードで摂動)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .msg import tr
from torch import nn


@dataclass
class TargetModule:
    """摂動対象1モジュール (重みの在処を含む)。"""

    module: nn.Module
    weight_attr: str = "weight"     # nn.Linear互換は "weight"
    fused: bool = False             # 3D融合専門家 (E, out, in)
    n_experts: int = 0

    def weight(self) -> torch.Tensor:
        w = getattr(self.module, self.weight_attr, None)
        if w is None:
            raise RuntimeError(tr("module_no_weight", t=type(self.module).__name__))
        return w

    def out_features(self) -> int:
        w = self.weight()
        if self.fused:
            return int(w.shape[1])
        return int(w.shape[0])


def get_text_model(model: nn.Module) -> nn.Module:
    """埋め込みとデコーダレイヤを保持する「テキスト本体」モジュールを取得する。

    通常のForCausalLMでは model.model、マルチモーダル (vision言語) モデルでは
    model.model.language_model など構造が変わるため、embed_tokensとlayersを
    両方持つモジュールを幅優先で探索して解決する (v2: Qwen3.5等のVLM対応)。
    """
    m = model
    if hasattr(m, "get_base_model"):
        m = m.get_base_model()
    queue = [m]
    seen: set[int] = {id(m)}
    while queue:
        cur = queue.pop(0)
        if getattr(cur, "embed_tokens", None) is not None and getattr(cur, "layers", None) is not None:
            return cur
        for child in cur.children():
            if id(child) not in seen:
                seen.add(id(child))
                queue.append(child)
    return getattr(m, "model", m)


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """モデルからデコーダレイヤのリストを取得する (マルチモーダル対応)。"""
    # マルチモーダル/通常とも: テキスト本体を解決してからlayersを返す
    core = get_text_model(model)
    if hasattr(core, "layers"):
        return core.layers
    m = model
    if hasattr(m, "get_base_model"):
        m = m.get_base_model()
    inner = getattr(m, "model", m)
    if hasattr(inner, "layers"):
        return inner.layers
    raise RuntimeError(tr("layers_unrecognized", t=type(m).__name__))


def _try_add(modules: dict[str, list[TargetModule]], comp: str, module: Any,
             weight_attr: str = "weight", fused: bool = False) -> None:
    """対象モジュールを登録する (融合専門家は weight_shape から次元を確認)。"""
    if not isinstance(module, nn.Module):
        return
    w = getattr(module, weight_attr, None)
    if w is None or not torch.is_tensor(w):
        return
    if fused:
        # 3D (E, out, in) であること・要素数が正であることを確認
        if w.dim() != 3 or w.shape[0] < 1 or w.shape[1] < 1 or w.shape[2] < 1:
            return
    else:
        if w.dim() != 2:
            return
    modules.setdefault(comp, []).append(
        TargetModule(module=module, weight_attr=weight_attr, fused=fused,
                     n_experts=int(w.shape[0]) if fused else 0))


def _detect_fused_experts(container: Any) -> str | None:
    """モジュールが「融合3D専門家」コンテナか判定する。
    戻り値: 下流射影の属性名 ("down_proj") / None (非対象)。"""
    if not isinstance(container, nn.Module):
        return None
    dp = getattr(container, "down_proj", None)
    if torch.is_tensor(dp) and dp.dim() == 3 and getattr(container, "num_experts", None):
        return "down_proj"
    return None


def get_layer_components(layer: nn.Module) -> dict[str, list[TargetModule]]:
    """1レイヤから摂動対象モジュールを探す。戻り値: {コンポーネント名: [TargetModule]}"""
    modules: dict[str, list[TargetModule]] = {}

    # --- attention出力射影 ---
    sa = getattr(layer, "self_attn", None)
    if sa is not None:
        _try_add(modules, "attn_out", getattr(sa, "o_proj", None))
        _try_add(modules, "attn_out", getattr(sa, "out_proj", None))
    # ハイブリッド線形注意 (Qwen3.5/Qwen4Exp系): linear_attn / linear_attention の out_proj
    for la_name in ("linear_attn", "linear_attention"):
        la = getattr(layer, la_name, None)
        if la is not None:
            _try_add(modules, "attn_out", getattr(la, "out_proj", None))
    cv = getattr(layer, "conv", None)
    if cv is not None:
        _try_add(modules, "attn_out", getattr(cv, "out_proj", None))

    # --- MLP下流射影 ---
    mlp = getattr(layer, "mlp", None)
    if mlp is not None:
        _try_add(modules, "mlp_down", getattr(mlp, "down_proj", None))
        # 共有専門家 (Qwen4Exp: 標準MLP構造の shared_expert)
        _try_add(modules, "mlp_down", getattr(getattr(mlp, "shared_expert", None), "down_proj", None))
        experts = getattr(mlp, "experts", None)
        if experts is not None:
            fused_attr = _detect_fused_experts(experts)
            if fused_attr is not None:
                # 融合3D専門家 (Qwen4ExpTextExperts / Gemma4TextExperts)
                _try_add(modules, "mlp_down", experts, weight_attr=fused_attr, fused=True)
            else:
                try:
                    for expert in experts:
                        _try_add(modules, "mlp_down", getattr(expert, "down_proj", None))
                        _try_add(modules, "mlp_down", getattr(expert, "w2", None))
                        _try_add(modules, "mlp_down", getattr(expert, "output_linear", None))
                except TypeError:
                    pass
    # Gemma4-MoE: 専門家は layer 直下 (mlp と並列) に置かれる
    if getattr(layer, "enable_moe_block", False) or hasattr(layer, "experts"):
        experts = getattr(layer, "experts", None)
        if experts is not None:
            fused_attr = _detect_fused_experts(experts)
            if fused_attr is not None:
                _try_add(modules, "mlp_down", experts, weight_attr=fused_attr, fused=True)
    bsm = getattr(layer, "block_sparse_moe", None)
    if bsm is not None:
        try:
            for expert in getattr(bsm, "experts", []):
                _try_add(modules, "mlp_down", getattr(expert, "w2", None))
                _try_add(modules, "mlp_down", getattr(expert, "down_proj", None))
        except TypeError:
            pass
    ff = getattr(layer, "feed_forward", None)
    if ff is not None:
        _try_add(modules, "mlp_down", getattr(ff, "w2", None))
        try:
            for expert in getattr(ff, "experts", []):
                _try_add(modules, "mlp_down", getattr(expert, "w2", None))
        except TypeError:
            pass
    sm = getattr(layer, "shared_mlp", None)
    if sm is not None:
        _try_add(modules, "mlp_down", getattr(sm, "output_linear", None))
    moe = getattr(layer, "moe", None)
    if moe is not None:
        try:
            for expert in getattr(moe, "experts", []):
                _try_add(modules, "mlp_down", getattr(expert, "output_linear", None))
                _try_add(modules, "mlp_down", getattr(expert, "w2", None))
        except TypeError:
            pass

    # 重複除去 (同一オブジェクトが複数経路で登録された場合)
    for k in list(modules.keys()):
        seen: list[int] = []
        uniq = []
        for m in modules[k]:
            if id(m.module) not in seen:
                seen.append(id(m.module))
                uniq.append(m)
        modules[k] = uniq
    return modules


class ComponentRegistry:
    """モデル全体の摂動対象モジュールを保持するレジストリ。"""

    def __init__(self, model: nn.Module):
        self.layers = get_decoder_layers(model)
        self.n_layers = len(self.layers)
        # comp -> レイヤ番号 -> [TargetModule]
        self.map: dict[str, dict[int, list[TargetModule]]] = {}
        for li in range(self.n_layers):
            for comp, mods in get_layer_components(self.layers[li]).items():
                self.map.setdefault(comp, {})[li] = mods
        if not self.map:
            raise RuntimeError(tr("no_perturb_targets"))

    @property
    def components(self) -> list[str]:
        return sorted(self.map.keys())

    def modules_of(self, comp: str, layer: int) -> list[nn.Module]:
        """互換API: 指定レイヤの対象モジュール本体のリスト。"""
        return [tm.module for tm in self.map.get(comp, {}).get(layer, [])]

    def targets_of(self, comp: str, layer: int) -> list[TargetModule]:
        return self.map.get(comp, {}).get(layer, [])

    def touched_layers(self, comp: str) -> list[int]:
        return sorted(self.map.get(comp, {}).keys())

    def has_fused(self, comp: str) -> bool:
        return any(tm.fused for layers in self.map.get(comp, {}).values()
                   for tm in layers)

    def summary(self) -> list[str]:
        """アーキテクチャ解析の要約行 (cli表示用)。"""
        lines = []
        for comp in self.components:
            n_mod = sum(len(v) for v in self.map[comp].values())
            n_fused = sum(1 for v in self.map[comp].values() for tm in v if tm.fused)
            txt = tr("arch_component", comp=comp, n=n_mod)
            if n_fused:
                txt += "  " + tr("arch_fused_note", n=n_fused)
            lines.append(txt)
        return lines


def module_weight(module: nn.Module | TargetModule) -> torch.Tensor:
    """モジュール (またはTargetModule) から重みテンソルを取得する (ラップ対応)。"""
    if isinstance(module, TargetModule):
        return module.weight()
    base = module
    while hasattr(base, "base_layer"):
        base = base.base_layer
    w = getattr(base, "weight", None)
    if w is None:
        raise RuntimeError(tr("module_no_weight", t=type(module).__name__))
    return w


def module_weight_is_float(module: nn.Module | TargetModule) -> bool:
    w = module_weight(module)
    return w.dtype in (torch.float32, torch.float64)
