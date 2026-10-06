# -*- coding: utf-8 -*-
"""摂動エンジン: 方向指向削除 (directional ablation) の重みパッチ適用。

v5: ランク1フック方式 (従来の「重み行列を丸ごと書き換え」を廃止)。

従来 (v4以前) の apply は触れたモジュールごとに
  clone → fp32化 → 外積で全要素書き換え → bf16へ戻す
を行い、1モジュールあたり数百MBのメモリ往復が発生していた。
しかし摂動自体は数学的にランク1である:
    W' = W - w * v v^T W  (v: 単位方向ベクトル, w: 強度)
       = W - w * outer(v, u)   ただし u = W^T v
よって forward出力は
    y' = W'x = Wx - w * (u·x) * v
を「重みを書き換えずに」出力フックで計算できる。さらにノルム保存
(行ノルムの補正 s = rn/ra) も
    ra^2 = rn^2 - 2w·v·(W@u) + w^2·v^2·|u|^2
を2本のmatvecで事前計算すれば、行列実体化は不要になる。

これにより:
  - 適用/リセットがフックの付け外しだけになり、1試行あたり数秒の
    重み書き換えコストとGB級のメモリチャーンが消える
  - INT4量子化されたモジュールでも (出力を後段補正するため) 同一の
    摂動を適用可能
  - エクスポート時は従来どおり fp32 の完全パッチ (ablated_weight_for)
    を元重みから計算するため、出力モデルの精度は変わらない

数値的な差: 探索中の代理評価はフック補正 (bf16行列積 + fp32補正) に
よる微小丸め差を含む。起動時に verify_hook_equivalence() で
完全パッチとの出力差を實測検証し、許容不能な場合は exact モード
(従来の重み書き換え) へ自動フォールバックする。

v6: 融合3D専門家 (Qwen4Exp / Gemma4-MoE) に対応。
  専門家重みが nn.Parameter(E, out, in) として格納されるモデルでは
  出力フックで摂動を分解できない (フォワードがルーティングを内部で
  行うため) ので、該当モジュールだけ **exactモード** (専門家スライス
  ごとに fp32 で書き換え) を併用する。2Dモジュールは引き続きフック。
  また方向ベクトルがコンポーネントごとに異なる次元を持てる
  (directions_map: ハイパーコネクション対応)。
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field, asdict
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .msg import tr

from .arch import ComponentRegistry, TargetModule, module_weight


@dataclass
class KernelParams:
    """1コンポーネント分のカーネルパラメータ。

    shape (v9):
      "cosine" : 旧来の平滑コサインカーネル (中心から半径内をコサインで減衰)。
                 狭い帯を_peak周辺に当てる局所探索向け。
      "flat"   : 帯内一様強度 (Arditi et al. の全層一様直交化に対応)。
                 帯を全域 (center=L/2, radius=L) にすれば
                 「全レイヤ同じ強度で方向を除去する」参照候補になる。
    """

    center: float = 0.7    # カーネル中心レイヤ位置 p
    radius: float = 0.25   # カーネル半径 r (レイヤ数)
    strength: float = 0.0  # 中心強度 s (0で無効)
    shape: str = "cosine"  # "cosine" | "flat" (v9)

    def weight_at(self, layer: int) -> float:
        d = abs(layer - self.center)
        if d > self.radius or self.strength <= 0:
            return 0.0
        if self.shape == "flat":
            return float(self.strength)
        return float(self.strength * 0.5 * (1.0 + math.cos(math.pi * d / self.radius)))


@dataclass
class AblationParams:
    """探索対象となる非検閲化パラメータ群。"""

    direction_mode: str = "per_layer"   # "per_layer" or "global"
    direction_index: float = 0.0        # global時に使う方向索引 (0=第1レイヤ方向)
    kernels: dict[str, KernelParams] = field(default_factory=lambda: {
        "attn_out": KernelParams(),
        "mlp_down": KernelParams(),
    })

    def band_start(self, n_layers: int) -> int:
        """摂動が及ぶ最も早いレイヤ (キャッシュ境界の決定に使用)。"""
        starts = []
        for kp in self.kernels.values():
            if kp.strength > 0:
                starts.append(max(0, math.floor(kp.center - kp.radius)))
        return min(starts) if starts else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction_mode": self.direction_mode,
            "direction_index": round(float(self.direction_index), 3),
            "kernels": {k: {"center": round(v.center, 2), "radius": round(v.radius, 2),
                            "strength": round(v.strength, 3),
                            "shape": (v.shape if v.shape != "cosine" else "cosine")}
                       for k, v in self.kernels.items()},
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "AblationParams":
        k = d.get("kernels", {})
        return AblationParams(
            direction_mode=d.get("direction_mode", "per_layer"),
            direction_index=float(d.get("direction_index", 0.0)),
            kernels={name: KernelParams(center=float(v["center"]), radius=float(v["radius"]),
                                        strength=float(v["strength"]),
                                        shape=(v["shape"] if v.get("shape") in ("cosine", "flat")
                                               else "cosine"))
                     for name, v in k.items()} if k else {"attn_out": KernelParams(),
                                                          "mlp_down": KernelParams()},
        )


def direction_vector(directions: torch.Tensor, params: AblationParams, layer: int) -> torch.Tensor:
    """指定レイヤで使う方向ベクトルを返す (per_layer or global補間)。"""
    if params.direction_mode == "global":
        di = float(params.direction_index)
        # 索引は「レイヤ方向リスト」への実数索引: 0.0=レイヤ0方向, 補間あり
        fi, i = math.modf(di)
        i = int(i)
        i = max(0, min(i, directions.shape[0] - 2))
        v = directions[i].lerp(directions[i + 1], float(fi))
        return F.normalize(v, p=2, dim=0)
    i = max(0, min(layer + 1, directions.shape[0] - 1))  # +1: 先頭は埋め込み
    return directions[i]


def compute_ablated_weight(W: torch.Tensor, v: torch.Tensor, w: float,
                           norm_preserve: bool = True) -> torch.Tensor:
    """重み行列Wに強度wの方向直交化を適用した新しい重みを返す (fp32計算)。

    W: (out, in) または (E, out, in) [融合専門家] / v: (out,) 単位ベクトル fp32
    非有限値 (NaN/Inf) が生じた場合はValueErrorを送出する (崩壊候補の除外)。

    [v6] 3D融合専門家: 各専門家スライス W[e] に対して同じ直交化を適用する
    (メモリ峰值を抑えるためスライスごとに逐次計算)。
    """
    if W.dim() == 3:
        out = torch.empty_like(W)
        for e in range(W.shape[0]):
            out[e] = _ablate_2d(W[e], v, w, norm_preserve)
        if not torch.isfinite(out).all():
            raise ValueError(tr("orth_nonfinite"))
        return out
    return _ablate_2d(W, v, w, norm_preserve)


def _ablate_2d(W: torch.Tensor, v: torch.Tensor, w: float,
               norm_preserve: bool) -> torch.Tensor:
    Wa = W - w * torch.outer(v, v @ W)
    if norm_preserve:
        rn = W.norm(dim=1, keepdim=True)
        ra = Wa.norm(dim=1, keepdim=True)
        Wa = Wa * (rn / (ra + 1e-8))
    if not torch.isfinite(Wa).all():
        raise ValueError(tr("orth_nonfinite"))
    return Wa


def compute_ablated_weight_multi(W: torch.Tensor, vs: list[torch.Tensor], ws: list[float],
                                 norm_preserve: bool = True) -> torch.Tensor:
    """複数方向 (直交化済み) に対する多重直交化。外部パラメータ適用 (v1.2形式) で使用。"""
    Wa = W
    for v, w in zip(vs, ws):
        if w <= 0:
            continue
        Wa = Wa - w * torch.outer(v, v @ Wa)
    if norm_preserve:
        rn = W.norm(dim=1, keepdim=True)
        ra = Wa.norm(dim=1, keepdim=True)
        Wa = Wa * (rn / (ra + 1e-8))
    return Wa


def orthonormalize(vs: list[torch.Tensor]) -> list[torch.Tensor]:
    """方向ベクトル集合をグラム・シュミットで直交化する。"""
    out: list[torch.Tensor] = []
    for v in vs:
        u = v.clone().float()
        for prev in out:
            u = u - (u @ prev) * prev
        n = u.norm()
        if n > 1e-6:
            out.append(u / n)
    return out


# ---------------------------------------------------------------- ランク1フック

@dataclass
class _Rank1Spec:
    """1モジュール分の摂動スペック (フックで使う事前計算値)。"""
    v: torch.Tensor        # (out,) fp32
    u: torch.Tensor        # (in,)  fp32  = W^T v
    w: float               # 強度
    s: torch.Tensor | None # (out,) fp32 行ノルム補正 (norm_preserve=off なら None)


def _make_spec(W: torch.Tensor, v: torch.Tensor, w: float,
               norm_preserve: bool) -> _Rank1Spec:
    """モジュール重みW (bf16等) からフック用の事前計算を行う (fp32精度)。"""
    Wf = W.detach().to(torch.float32)
    u = torch.matmul(v, Wf)                      # (in,)
    if not torch.isfinite(u).all():
        raise ValueError(tr("orth_nonfinite"))
    s = None
    if norm_preserve:
        Wu = torch.matmul(Wf, u)                 # (out,)
        rn = Wf.norm(dim=1)
        u2 = float(u.dot(u))
        # 行ごと: |W'_i|^2 = |W_i|^2 - 2w·v_i·(W_i·u) + w^2·v_i^2·|u|^2
        ra2 = rn * rn - 2.0 * w * v * Wu + (w * w) * (v * v) * u2
        ra = torch.sqrt(torch.clamp(ra2, min=0.0))
        s = rn / (ra + 1e-8)
        if not torch.isfinite(s).all():
            raise ValueError(tr("orth_nonfinite"))
    return _Rank1Spec(v=v, u=u, w=float(w), s=s)


def _make_hook(spec: _Rank1Spec):
    """出力をランク1補正するforwardフック (クロージャ)。"""
    v = spec.v
    u = spec.u
    w = spec.w
    s = spec.s

    def hook(module, args, kwargs, output):
        x = args[0] if args else kwargs.get("input", None)
        if x is None:
            return output
        y = output
        if not isinstance(y, torch.Tensor):
            return output
        with torch.no_grad():
            alpha = torch.matmul(x.detach().to(torch.float32), u)   # (...,)
            corr = alpha.unsqueeze(-1) * (w * v)                     # (..., out)
            out = y.detach().to(torch.float32) - corr
            if s is not None:
                out = out * s
            return out.to(y.dtype)
    return hook


class PatchEngine:
    """ランク1フック方式の摂動エンジン (v5) + 融合専門家exactモード (v6)。

    mode = "hook"  : 出力フックで摂動を適用 (既定, 高速・低メモリ)。
                     融合3D専門家のみ個別に exact (重み書き換え) になる。
    mode = "exact" : 従来どおり重み行列を fp32 で書き換える (フォールバック)。

    directions: (L+1, D) テンソル (残差方向) または
                {comp: (L+1, D)} (コンポーネント別方向, v6)。
    """

    def __init__(self, registry: ComponentRegistry, directions,
                 norm_preserve: bool = True):
        self.registry = registry
        self.norm_preserve = norm_preserve
        self.mode = "hook"
        if isinstance(directions, dict):
            self.directions = None
            self.dir_map: dict[str, torch.Tensor] = {
                c: t.to(torch.float32) for c, t in directions.items()}
        else:
            self.directions = directions.to(torch.float32)
            self.dir_map = {}
        self._hooks: dict[int, Any] = {}          # id(module) -> RemovableHandle
        self._specs: dict[int, _Rank1Spec] = {}
        self._orig: dict[int, torch.Tensor] = {}  # exactモード用 (同dtype)
        self._applied_params: AblationParams | None = None

    # ---------------------------------------------------------------- 方向

    def _directions_for(self, comp: str) -> torch.Tensor:
        if self.dir_map and comp in self.dir_map:
            return self.dir_map[comp]
        return self.directions

    def _direction_vector(self, comp: str, params: AblationParams, layer: int) -> torch.Tensor:
        stack = self._directions_for(comp)
        if stack is None:
            raise RuntimeError(tr("dir_missing_comp", comp=comp))
        return direction_vector(stack, params, layer)

    # ---------------------------------------------------------------- 状態

    def current_touched_module_count(self) -> int:
        return len(self._hooks) + len(self._orig)

    # ---------------------------------------------------------------- リセット

    def reset(self) -> None:
        """すべての摂動を解除する (フック除去 / 重み復元)。"""
        for h in self._hooks.values():
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()
        self._specs.clear()
        if self._orig:
            for comp in self.registry.components:
                for li, tms in self.registry.map[comp].items():
                    for tm in tms:
                        mid = id(tm.module)
                        if mid in self._orig:
                            w = tm.weight()
                            w.data = self._orig[mid].to(w.dtype)
            self._orig.clear()
        self._applied_params = None

    @contextlib.contextmanager
    def suspend_patches(self):
        """摂動を一時的に解除するコンテキスト (キャッシュ再構築等で
        「無摂動」のforwardが必要なときに使う)。退出時に自動復元する。"""
        params = self._applied_params
        self.reset()
        try:
            yield
        finally:
            if params is not None:
                self.apply(params)

    # ---------------------------------------------------------------- 適用

    def apply(self, params: AblationParams) -> int:
        """パラメータに従って摂動を適用する。戻り値: 触れたモジュール数。

        2D Linear → ランク1フック。融合3D専門家 → exact (スライス毎書き換え)。
        """
        self.reset()
        self._applied_params = params
        touched = 0
        for comp, kp in params.kernels.items():
            if comp not in self.registry.map or kp.strength <= 0:
                continue
            for li in self.registry.map[comp].keys():
                w = kp.weight_at(li)
                if w <= 0:
                    continue
                v = self._direction_vector(comp, params, li).cpu()
                for tm in self.registry.targets_of(comp, li):
                    mid = id(tm.module)
                    use_hook = (self.mode == "hook") and (not tm.fused)
                    if use_hook:
                        try:
                            spec = _make_spec(tm.weight().data, v, w, self.norm_preserve)
                        except ValueError:
                            raise
                        h = tm.module.register_forward_hook(_make_hook(spec), with_kwargs=True)
                        self._hooks[mid] = h
                        self._specs[mid] = spec
                    else:
                        # exact: 融合専門家 or フォールバック
                        wt = tm.weight()
                        if mid not in self._orig:
                            self._orig[mid] = wt.data.clone()
                        W = self._orig[mid].to(torch.float32).cpu()
                        Wa = compute_ablated_weight(W, v, w, self.norm_preserve)
                        wt.data = Wa.to(wt.dtype).to(wt.device)
                    touched += 1
        return touched

    # ---------------------------------------------------------------- 検証

    def verify_hook_equivalence(self, probe_input: torch.Tensor | None = None) -> float:
        """フック方式と完全重みパッチ方式の出力差 (相対値) を實測する。

        probe_input: 呼び出し側が実入力例を渡せる。ただしモジュールごとに
        in_featuresが異なる (線形注意のout_projはヘッド連結長など) ため、
        形状が合わないモジュールでは大きさを合わせた合成入力を使う。
        検証対象は「同一W・同一入力に対するフック補正と完全パッチの一致」であり、
        実際の活性分布は不要 (相対差で評価する)。
        戻り値: 両方式の出力の相対差。infは検証失敗 (フック無効化)。
        """
        if self.mode != "hook":
            return 0.0
        diffs: list[float] = []
        for comp in self.registry.components:
            # 方向が未取得のコンポーネント (--components attn で専門家をINT4化した
            # 場合など) はこの実行では摂動しないため検証対象から除外する
            if self.dir_map and comp not in self.dir_map:
                continue
            if not self.dir_map and self.directions is None:
                continue
            layers = self.registry.map.get(comp, {})
            if not layers:
                continue
            li = sorted(layers.keys())[len(layers) // 2]
            tm = self.registry.targets_of(comp, li)[0]
            if tm.fused:
                continue   # 融合専門家は常にexact方式のため検証不要
            W = tm.weight().data
            v = self._direction_vector(comp, self._test_params(), li).cpu()
            w = 1.5
            try:
                spec = _make_spec(W, v, w, self.norm_preserve)
                # 完全パッチ (fp32)
                Wa = compute_ablated_weight(W.to(torch.float32).cpu(), v, w,
                                            self.norm_preserve).to(W.dtype)
                if (probe_input is not None and probe_input.dim() == 2
                        and probe_input.shape[-1] == W.shape[1]):
                    x = probe_input.to(W.dtype)
                else:
                    # 大きさを実重みのスケールに合わせた合成入力 (形状は必ず一致)
                    scale = float(W.float().std()) * float(W.shape[1]) ** 0.5
                    x = (torch.randn(4, W.shape[1]) * max(scale, 1e-6)).to(W.dtype)
                y_exact = torch.nn.functional.linear(x, Wa).float()
                y_base = torch.nn.functional.linear(x, W).float()
                alpha = torch.matmul(x.float(), spec.u)
                corr = alpha.unsqueeze(-1) * (spec.w * spec.v)
                y_hook = y_base - corr
                if spec.s is not None:
                    y_hook = y_hook * spec.s
                diff = (y_exact - y_hook).abs().max().item()
                denom = y_exact.abs().mean().item() + 1.0
                diffs.append(diff / denom)
            except Exception:
                return float("inf")
        if not diffs:
            return 0.0
        return max(diffs)

    def _test_params(self) -> AblationParams:
        """検証用のダミーパラメータ (全コンポーネント中央層に強度を与える)。"""
        L = self.registry.n_layers
        mid = max(1, L // 2)
        kp = KernelParams(center=float(mid), radius=2.0, strength=1.5)
        return AblationParams(direction_mode="per_layer", direction_index=0.0,
                              kernels={"attn_out": kp, "mlp_down": kp})

    # ---------------------------------------------------------------- エクスポート

    def export_deltas(self, params: AblationParams, safetensors_keys: dict[str, str],
                      ) -> dict[str, torch.Tensor]:
        """エクスポート用に、摂動適用後の重み一覧を計算して返す (fp32完全パッチ)。

        safetensors_keys: id(module) -> safetensorsキー の対応表。
        v5: フックは重みを書き換えないため、現在の (無摂動) 重みから
        直接 ablated_weight_for と同じ計算を行う。
        """
        self.reset()
        result: dict[str, torch.Tensor] = {}
        for comp in self.registry.components:
            for li, tms in self.registry.map[comp].items():
                for tm in tms:
                    mid = id(tm.module)
                    if mid not in safetensors_keys:
                        continue
                    W = tm.weight().data
                    new_w = self.ablated_weight_for(comp, li, params, W)
                    if new_w is not None:
                        result[safetensors_keys[mid]] = new_w
        return result

    def ablated_weight_for(self, comp: str, layer: int, params: AblationParams,
                           W: torch.Tensor) -> torch.Tensor | None:
        """元の重みテンソルWに対して摂動適用後の重みを計算して返す (エクスポート用)。

        Wはsafetensorsシャードから読んだ直のオリジナル重み。コンポーネントが
        このレイヤで摂動を受けない場合はNoneを返す。
        [v6] Wが3D (E, out, in) なら融合専門家として専門家スライス毎に適用する。
        """
        kp = params.kernels.get(comp)
        if kp is None:
            return None
        w = kp.weight_at(layer)
        if w <= 0:
            return None
        v = self._direction_vector(comp, params, layer)
        orig_dtype = W.dtype
        Wf = W.to(torch.float32)
        Wa = compute_ablated_weight(Wf, v, w, self.norm_preserve)
        return Wa.to(orig_dtype)
