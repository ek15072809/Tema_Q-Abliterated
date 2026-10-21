# -*- coding: utf-8 -*-
"""INT4重み量子化 (v2新設, v7実行経路全面改訂)。

グループ対称INT4 (WQ4G: weight-only int4, group-wise symmetric) を実装する。

  - 1つのuint8に2つのint4をパックして保存し、重みあたり約0.55バイト
    (スケール込み) で保持する。bf16比でおよそ1/3.6のメモリ使用量。
  - グループ化: in_features軸をgroup_size(既定128)ごとに分割し、
    グループごとに対称スケール scale = amax/7 を持つ。
  - 復号: ニブル値n (0..15) から符号付き値 q = (n XOR 8) - 8 (−8..7) を復元し
    W ≈ q × scale で脱量子化してから行列積する (重みのみ量子化)。
  - 摂動対象 (o_proj/down_proj)・lm_head・埋め込みは量子化しないため、
    パッチ適用・出力精度への影響を排除している。

[v7] 実行経路の改訂 (v6の「INT4の方が遅く・メモリも多い」問題の解消):

  1. **分割ニブル行列積 (split-nibble GEMM)**: キャッシュミス時は重み全体を
     bf16へ展開しない。重みを偶数列・奇数列の2つの半分行列 (out, in/2) の
     まま復号し、入力も x[...,0::2] / x[...,1::2] に分割して2回のmatmulで
     計算する。中間テンソルは従来の「完全展開 + int16経由 + strided copy」の
     約1/3になり、メモリ峰值・トラフィックともに大幅に減少する。
  2. **行チャンク逐次パック**: pack_int4_chunked() が重みを行ブロック単位で
     逐次パックする (ストリーミング読み込みと併用)。変換ピークは
     チャンク1つ分 (~16MB) に抑えられ、RSS高水位を残さない。
  3. **実LRU重みキャッシュ**: 予算は「ロード後の実空きRAM」から決定され、
     空きに応じて弾的に拡大する (memguard連携)。ヒット時は単一の
     bf16行列積 (純bf16実行と同一カーネル) になる。

この方式はCPU/GPU両方で動作する (bitwise演算のみを使用)。
"""

from __future__ import annotations

import contextlib
import random
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import nn

# 量子化対象とする最小の in_features (これ未満はオーバーヘッドが割に合わない)
MIN_IN_FEATURES = 16
DEFAULT_GROUP_SIZE = 128
_GROUP_CANDIDATES = (128, 64, 32, 16)
# 行チャンクパックの目安 (ソーステンソルの要素数)
_PACK_CHUNK_ELEMS = 8_000_000

# dtypeごとの要素バイト数 (forwardホットパスで使う。dictは起動時に確定)
_DTYPE_SIZES: dict[torch.dtype, int] = {}


def _dtype_size(dt: torch.dtype) -> int:
    """dtypeの要素バイト数 (結果はキャッシュ)。"""
    n = _DTYPE_SIZES.get(dt)
    if n is None:
        n = torch.empty(1, dtype=dt).element_size()
        _DTYPE_SIZES[dt] = n
    return n


def _pick_group(n_elements: int, desired: int = DEFAULT_GROUP_SIZE) -> int:
    """要素数を割り切れる偶数グループサイズを選ぶ (パック単位は2列のため偶数必須)。"""
    for g in (desired, 128, 64, 32, 16, 8, 4, 2):
        if g % 2 == 0 and g <= n_elements and n_elements % g == 0:
            return g
    return n_elements if n_elements % 2 == 0 else max(2, n_elements - 1)


def pack_int4(weight: torch.Tensor, group_size: int = DEFAULT_GROUP_SIZE
              ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """fp重みをグループ対称INT4へパックする。

    符号化はオフセット方式 (格納ニブル = 量子値 + 8 ∈ [0,15]) を使う。
    戻り値: (packed[uint8, (out, ceil(in/2))], scale[fp32, (out, in/g)], g, pad)
      pad は in_features が奇数だった場合に足したダミー列の数。
    """
    w = weight.detach()
    out_f, in_f = w.shape
    pad = in_f % 2
    if pad:
        w = F.pad(w, (0, 1))  # 奇数列はダミー0で偶数化
    total = in_f + pad
    g = _pick_group(total, group_size)
    ng = total // g

    grouped = w.reshape(out_f, ng, g)
    # スケールはfp32で計算・保存 (量子化自体は元dtypeのまま行い、ピークメモリを抑える)
    amax = grouped.abs().amax(dim=-1).to(torch.float32)
    scale = (amax / 7.0).clamp(min=1e-12)
    q = torch.round(grouped / scale.to(w.dtype).unsqueeze(-1)).clamp_(-8, 7)

    n = (q + 8).to(torch.uint8).reshape(out_f, total)  # 0..15
    packed = n[:, 0::2] | (n[:, 1::2] << 4)
    return packed.contiguous(), scale.contiguous(), g, pad


def pack_int4_chunked(weight: torch.Tensor, group_size: int = DEFAULT_GROUP_SIZE,
                      chunk_elems: int = _PACK_CHUNK_ELEMS
                      ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """重みを行ブロック単位に分割して逐次パックする (ピークメモリ抑制)。

    戻り値は pack_int4 と同一 (packed/scale/g/pad)。スケールは行ごと
    (グループごと) に決まるため、チャンク分割は数値に影響しない。
    [v7] 標準ロード後の一括変換でもこちらを使い、変換ピークを行
    チャンク1つ分 (目安16MB) に抑える。
    """
    w = weight.detach()
    out_f, in_f = w.shape
    if out_f * in_f <= chunk_elems:
        return pack_int4(w, group_size)
    pad = in_f % 2
    total = in_f + pad
    g = _pick_group(total, group_size)
    ng = total // g
    P = total // 2
    packed = torch.empty(out_f, P, dtype=torch.uint8)
    scale = torch.empty(out_f, ng, dtype=torch.float32)
    rows = max(1, chunk_elems // max(1, in_f))
    for r0 in range(0, out_f, rows):
        r1 = min(out_f, r0 + rows)
        p_c, s_c, g_c, _pad = pack_int4(w[r0:r1], g)
        packed[r0:r1] = p_c
        scale[r0:r1] = s_c
    return packed, scale, g, pad


def pack_into_buffers(packed: torch.Tensor, scale: torch.Tensor,
                       weight_rows: torch.Tensor, row_offset: int,
                       group_size: int) -> None:
    """ストリーミング読み込み用: 行チャンクを事前確保バッファへ書き込む。

    packed/scale は全体ぶんを事前確護したバッファ (out, P)/(out, ng)。
    weight_rows は (n_rows, in_features) のチャンク。group_size は
    全体で共通の値 (_pick_group(total) を呼び出し側が決定する)。
    """
    p_c, s_c, _g, _pad = pack_int4(weight_rows, group_size)
    n = weight_rows.shape[0]
    packed[row_offset:row_offset + n] = p_c
    scale[row_offset:row_offset + n] = s_c


def dequant_halves(packed: torch.Tensor, scale: torch.Tensor, group_size: int,
                   dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """[v7] パック済みINT4を偶数列・奇数列の半分行列へ復号する。

    返り値: (W_lo, W_hi) = (out, P) × 2。W_lo[:, k] が入力の (2k) 番列、
    W_hi[:, k] が (2k+1) 番列に対応する (奇数in_featuresのダミー列は
    W_hiの末尾に付く — 呼び出し側でスライスして使う)。
    中間テンソルは半分行列2つだけで、完全展開よりもトラフィックが少ない。
    ニブル抽出はuint8空間で行い (floatへのシフトは不可)、その後1回だけ
    計算dtypeへ変換する。
    """
    out_f, P = packed.shape
    lo = (packed & 0x0F).to(dtype).sub_(8)      # 0..15 → -8..7
    hi = (packed >> 4).to(dtype).sub_(8)
    sc = scale.to(dtype)
    ng = sc.shape[1]
    if ng > 1:
        # 1グループ = g入力列 = g/2パック列 (各半分行列内)
        lo.view(out_f, ng, -1).mul_(sc.reshape(out_f, ng, 1))
        hi.view(out_f, ng, -1).mul_(sc.reshape(out_f, ng, 1))
    else:
        lo.mul_(sc.reshape(out_f, 1))
        hi.mul_(sc.reshape(out_f, 1))
    return lo, hi


def dequant_int4(packed: torch.Tensor, scale: torch.Tensor, group_size: int,
                 out_features: int, in_features: int, dtype: torch.dtype,
                 pad: int = 0) -> torch.Tensor:
    """パック済みINT4を指定dtypeの重み行列へ復号する。

    v5: 出力バッファを先行確保し、(低位, 高位) ニブルをビューへの
    strided copy_ で直接書き込む (中間テンソルとstackの丸ごとコピーを削減)。
    dtype変換はcopy_時にimplicitに行われる。
    """
    out_f = packed.shape[0]
    P = packed.shape[1]
    total_in = in_features + pad
    n = packed.to(torch.int16)          # (out, P)
    lo = (n & 0x0F).sub_(8)             # -8..7  (in-place)
    hi = ((n >> 4) & 0x0F).sub_(8)      # -8..7
    w = torch.empty((out_f, total_in), dtype=dtype)
    pair = w.view(out_f, P, 2)          # pair[..., 0] = 偶数列, pair[..., 1] = 奇数列
    pair[..., 0].copy_(lo)
    pair[..., 1].copy_(hi)
    sc = scale.to(dtype)
    if group_size >= total_in:
        w.mul_(sc.reshape(out_f, 1))
    else:
        g = group_size
        ng = total_in // g
        w.view(out_f, ng, g).mul_(sc.reshape(out_f, ng, 1))
    return w if not pad else w[:, :in_features]


class PackedInt4Linear(nn.Module):
    """グループ対称INT4で重みを保持するLinear (重みのみ量子化)。

    [v7] forwardは3経路:
      1. LRUキャッシュヒット → キャッシュ済みbf16重みで1回の行列積
         (純bf16実行と同一カーネル・同一速度)。
      2. キャッシュミス + 予算あり → 完全展開してキャッシュへ登録し、
         1回の行列積 (次回からはヒット)。
      3. キャッシュミス + 予算なし → **分割ニブル行列積**: 重みを
         (out, in/2) の半分行列2つへ復号し、入力も偶奇で分割して
         2回の行列積で計算。重み全体のbf16展開が発生しないため、
         メモリ峰值は低く抑えられる (速度はバッチサイズで弾化)。

    摂動対象 (o_proj/down_proj)・lm_head・埋め込みは量子化しないため、
    パッチ適用・出力精度への影響を排除している。
    """

    # 重みキャッシュのグローバル状態 (メモリ予算と登録モジュール)
    _WSTATE: dict[str, Any] = {"budget": 0, "used": 0, "mods": []}
    # 全インスタンスの弱参照 (統計・一括操作用, __init__時に登録)
    _ALL: list = []
    _CLOCK = 0

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None,
                 group_size: int = DEFAULT_GROUP_SIZE, device: str | None = None):
        super().__init__()
        packed, scale, g, pad = pack_int4_chunked(weight.detach().cpu(), group_size)
        dev = device or weight.device
        self.out_features = int(weight.shape[0])
        self.in_features = int(weight.shape[1])
        self.group_size = g
        self.pad = pad
        # [v11.1] 元重みのdtype (互換用 .weight メタテンソルのdtypeとして使用。
        # リモートコードの forward には hidden_states.to(module.weight.dtype)
        # のような参照があり、量子化で .weight が無くなると動作しなくなる)
        self._weight_dtype = weight.dtype
        self.register_buffer("qweight", packed.to(dev), persistent=False)
        self.register_buffer("scale", scale.to(dev), persistent=False)
        if bias is not None:
            self.register_buffer("bias", bias.detach().to(dev).clone(), persistent=False)
        else:
            self.bias = None
        self._wcache: torch.Tensor | None = None
        self._wcache_bytes = 0
        self._last_use = 0
        self._hits = 0
        self._misses = 0
        self._demand = 0
        import weakref
        PackedInt4Linear._ALL.append(weakref.ref(self))

    @classmethod
    def from_packed(cls, packed: torch.Tensor, scale: torch.Tensor, group_size: int,
                    pad: int, in_features: int, out_features: int | None = None,
                    bias: torch.Tensor | None = None, device: str | None = None,
                    weight_dtype: torch.dtype | None = None):
        """[v7] ストリーミング読み込み用: パック済みバッファから直接構築する。

        pack_into_buffers() で行チャンク単位に埋めた packed/scale を渡す。
        [v11.1] weight_dtype: 元チェックポイントテンソルのdtype (互換用
        .weight メタテンソルに使う。省略時はbfloat16)。
        """
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        dev = device or packed.device
        obj.out_features = int(out_features if out_features is not None else packed.shape[0])
        obj.in_features = int(in_features)
        obj.group_size = int(group_size)
        obj.pad = int(pad)
        obj._weight_dtype = weight_dtype or torch.bfloat16
        obj.register_buffer("qweight", packed.to(dev), persistent=False)
        obj.register_buffer("scale", scale.to(dev), persistent=False)
        if bias is not None:
            obj.register_buffer("bias", bias.detach().to(dev).clone(), persistent=False)
        else:
            obj.bias = None
        obj._wcache = None
        obj._wcache_bytes = 0
        obj._last_use = 0
        obj._hits = 0
        obj._misses = 0
        obj._demand = 0
        import weakref
        PackedInt4Linear._ALL.append(weakref.ref(obj))
        return obj

    @property
    def weight(self) -> torch.Tensor:
        """[v11.1] 量子化後も ``module.weight.dtype`` / ``.shape`` 等の属性参照が
        壊れないようにするための軽量なメタテンソル。

        リモートコード (auto_map) のモデル実装には、forward中に
        ``hidden_states.to(self.mlp.gate_proj.weight.dtype)`` のような
        参照があり、量子化で .weight が無くなると動作しなくなる
        (XHToken/Spark-X2.6-4B で実際に発生)。このメタテンソルは
        dtype/形状情報のみを持ち、メモリ消費と演算コストはゼロ。
        実際の演算は forward (qweight/scale) で行われるため、値は持たない。
        """
        w = self.__dict__.get("_weight_meta")
        if w is None:
            w = torch.empty((self.out_features, self.in_features),
                            dtype=self._weight_dtype, device="meta")
            self.__dict__["_weight_meta"] = w
        return w

    # ------------------------------------------------------------ 統計 (v5)

    @classmethod
    def total_packed_bytes(cls) -> int:
        """全INT4モジュール (融合専門家含む) のパック済み重みの合計バイト数。"""
        total = 0
        for ref in cls._ALL:
            m = ref()
            if m is not None and hasattr(m, "memory_bytes"):
                total += m.memory_bytes()
        return total

    @classmethod
    def total_dequant_bytes(cls) -> int:
        """全INT4モジュールを脱量子化した場合のbf16バイト数 (材料化判定用)。"""
        total = 0
        for ref in cls._ALL:
            m = ref()
            if m is None:
                continue
            db = getattr(m, "dequant_bytes", None)
            if db is not None:
                total += db()
            else:
                total += m.out_features * m.in_features * 2
        return total

    @classmethod
    def stats(cls) -> dict[str, Any]:
        hits = misses = 0
        n = 0
        for ref in cls._ALL:
            m = ref()
            if m is not None:
                n += 1
                hits += getattr(m, "_hits", 0)
                misses += getattr(m, "_misses", 0)
        return {"modules": n, "hits": hits, "misses": misses,
                "cached_bytes": cls._WSTATE["used"], "budget_bytes": cls._WSTATE["budget"]}

    # ------------------------------------------------------------ 重みキャッシュ

    @classmethod
    def set_weight_cache_budget(cls, nbytes: int) -> None:
        """脱量子化重みのLRUキャッシュ予算 (バイト)。0で無効。"""
        cls._WSTATE["budget"] = int(nbytes)
        if nbytes <= 0:
            cls.clear_weight_cache()

    @classmethod
    def weight_cache_used_bytes(cls) -> int:
        return int(cls._WSTATE["used"])

    @classmethod
    def clear_weight_cache(cls) -> None:
        for ref in list(cls._WSTATE["mods"]):
            mod = ref()
            if mod is not None:
                mod._drop_cache()
        cls._WSTATE["mods"] = []
        cls._WSTATE["used"] = 0

    @classmethod
    def shrink_weight_cache(cls, ratio: float = 0.5) -> None:
        """[v7] 予算をratio倍に縮小し、あふれ分をランダム退避する (メモリ圧対応)。

        MemoryGuardが「危険」判定したときに呼び出される。キャッシュを
        完全に捨てると再 materialize コストが跳ね返るため、段階的な
        縮小にとどめる (完全解放は clear_weight_cache)。
        """
        try:
            state = cls._WSTATE
            new_budget = max(0, int(state["budget"] * ratio))
            state["budget"] = new_budget
            while state["used"] > new_budget and state["mods"]:
                live = [m for m in (r() for r in state["mods"]) if m is not None]
                if not live:
                    break
                victim = live[random.randrange(len(live))]
                victim._drop_cache()
                state["mods"] = [r for r in state["mods"]
                                 if r() is not None and r() is not victim]
            if new_budget <= 0:
                cls.clear_weight_cache()
        except Exception:
            pass

    def _drop_cache(self) -> None:
        if self._wcache is not None:
            self._wcache = None
            PackedInt4Linear._WSTATE["used"] -= self._wcache_bytes
            self._wcache_bytes = 0

    def _cache_insert(self, W: torch.Tensor) -> None:
        import weakref
        state = PackedInt4Linear._WSTATE
        nbytes = W.numel() * W.element_size()
        if state["budget"] <= 0 or nbytes > state["budget"]:
            return
        # [v7] 需要度ゲート: 予算が一杯のとき、1回しかアクセスされていない
        # モジュール (巡回するだけの全レイヤパス等) は既存のキャッシュを
        # 追い出さない。繰り返しアクセスされるモジュール (探索評価が触れる
        # 深い層) だけが追い出し権を持つため、常駐集合は「反復アクセス層」へ
        # 収束し、退避→再挿入の無限チャーン (断片化と速度低下の原因) が止まる。
        if state["used"] + nbytes > state["budget"] and self._demand < 2:
            return
        # 予算確保 (退避: 無作為選択)
        # [v7] LRUではなくランダム退避にする。Transformerのforwardは
        # レイヤ0→Lの「順序アクセス」であり、被覆率100%未満のLRUは
        # 「次に必要なモジュールをちょうど退避させる」最悪ケースになって
        # ヒット率が厳密に0%になる (実測)。ランダム退避ならヒット率≈被覆率
        # になり、部分キャッシュでも速度が線形に効く。
        while state["used"] + nbytes > state["budget"]:
            live = [m for m in (r() for r in state["mods"]) if m is not None]
            if not live:
                break
            victim = live[random.randrange(len(live))]
            if victim is self:
                if len(live) <= 1:
                    break
                continue
            victim._drop_cache()
            victim._demand = 0
            state["mods"] = [r for r in state["mods"]
                             if r() is not None and r() is not victim]
        # 二重登録を避れてから追加
        state["mods"] = [r for r in state["mods"] if r() is not None and r() is not self]
        self._wcache = W
        self._wcache_bytes = nbytes
        self._demand = 0
        state["used"] += nbytes
        state["mods"].append(weakref.ref(self))

    def dequantized_weight(self, dtype: torch.dtype | None = None,
                           use_cache: bool = True) -> torch.Tensor:
        """完全な (out, in) 重みを返す (キャッシュ経路・エクスポート互換)。"""
        dt = dtype or self.scale.dtype
        if use_cache and self._wcache is not None \
                and self._wcache.dtype == dt and self._wcache.device == self.qweight.device:
            PackedInt4Linear._CLOCK += 1
            self._last_use = PackedInt4Linear._CLOCK
            self._hits += 1
            return self._wcache
        self._misses += 1
        self._demand = getattr(self, "_demand", 0) + 1
        W = dequant_int4(self.qweight, self.scale, self.group_size,
                         self.out_features, self.in_features, dt, self.pad)
        if use_cache:
            self._drop_cache()
            self._cache_insert(W)
        return W

    def _cached_weight(self, dtype: torch.dtype) -> torch.Tensor | None:
        """キャッシュ済み重み (ヒット時)。未ヒット時は None (フォワードは分割経路)。"""
        if self._wcache is not None and self._wcache.dtype == dtype \
                and self._wcache.device == self.qweight.device:
            PackedInt4Linear._CLOCK += 1
            self._last_use = PackedInt4Linear._CLOCK
            self._hits += 1
            return self._wcache
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) キャッシュヒット: 単一行列積 (bf16実行と同一カーネル)
        W = self._cached_weight(x.dtype)
        if W is not None:
            b = self.bias.to(x.dtype) if self.bias is not None else None
            return F.linear(x, W, b)
        # 2) ミス + 予算あり: 完全展開してキャッシュ登録 (次回からヒット)
        state = PackedInt4Linear._WSTATE
        nbytes = self.out_features * (self.in_features + self.pad) \
            * _dtype_size(x.dtype)
        if state["budget"] > 0 and nbytes <= state["budget"]:
            self._demand = getattr(self, "_demand", 0) + 1
            W = self.dequantized_weight(x.dtype, use_cache=True)
            b = self.bias.to(x.dtype) if self.bias is not None else None
            return F.linear(x, W, b)
        # 3) ミス + 予算なし/超大型: 分割ニブル行列積 (重み全体を展開しない)
        self._misses += 1
        self._demand = getattr(self, "_demand", 0) + 1
        return self._forward_split(x)

    def _forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """[v7] 分割ニブル行列積: xの偶数列・奇数列を別々に掛けて足す。

        重みを行チャンク (固定サイズ) ごとに復号しながら部分行列積を
        足し込む。これにより:
        - 一時メモリは「チャンク1つ分」(目安2MB) で一定
        - アロケータは同一サイズの割り当てを再利用するため、
          サイズの異なるテンソルが交互に来ることで起きるヒープ断片化
          (低メモリ環境のOOM要因・実測) が起きない
        - 重み全体のbf16展開が発生しないためメモリ峰值も低い
        """
        out_f = self.out_features
        # [v7] 偶奇スライスは非連続ビューになる。一部のCPU/BLAS環境では
        # 「非連続入力×大行列」のGEMMがメモリ爆発・極端な低速化を起こす
        # (実測: lm_headで無限に消費) ため、明示的に連続化してから渡す。
        x0 = x[..., 0::2].contiguous()
        x1 = x[..., 1::2].contiguous()
        n_odd = x1.shape[-1]
        y = x.new_empty(*x.shape[:-1], out_f)
        # 行チャンク: 半分行列1つぶんが目安 2MB になる行数
        in_half = max(1, (self.in_features + self.pad) // 2)
        rows = max(1, (2 * 1024 * 1024) // max(1, in_half * 2))
        sc = self.scale.to(x.dtype)
        ng = sc.shape[1]
        sc = sc.reshape(out_f, ng, 1) if ng > 1 else sc.reshape(out_f, 1)
        for r0 in range(0, out_f, rows):
            r1 = min(out_f, r0 + rows)
            n = self.qweight[r0:r1]
            lo = (n & 0x0F).to(x.dtype).sub_(8)
            hi = (n >> 4).to(x.dtype).sub_(8)
            if ng > 1:
                m = r1 - r0
                lo.view(m, ng, -1).mul_(sc[r0:r1].reshape(m, ng, 1))
                hi.view(m, ng, -1).mul_(sc[r0:r1].reshape(m, ng, 1))
            else:
                lo.mul_(sc[r0:r1])
                hi.mul_(sc[r0:r1])
            if hi.shape[1] > n_odd:
                hi = hi[:, :n_odd]   # 奇数in_featuresのダミー列を除去
            y[..., r0:r1] = F.linear(x0, lo, None)
            y[..., r0:r1].add_(F.linear(x1, hi, None))
        if self.bias is not None:
            y = y + self.bias.to(x.dtype)
        return y

    def verify_split_equivalence(self, dtype: torch.dtype = torch.bfloat16) -> float:
        """[v7] 分割経路と完全展開経路の出力差 (相対値) を實測する (E2E検証用)。"""
        scale = float(self.scale.float().abs().mean().item()) or 1e-6
        x = (torch.randn(8, self.in_features, dtype=torch.float32) * scale).to(dtype)
        with torch.no_grad():
            y_full = F.linear(x, self.dequantized_weight(dtype, use_cache=False))
            self._drop_cache()
            y_split = self._forward_split(x)
        denom = y_full.abs().mean().item() + 1e-9
        return float((y_full - y_split).abs().max().item() / denom)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, qint4_group={self.group_size}")

    def memory_bytes(self) -> int:
        return self.qweight.numel() + self.scale.numel() * 4


# ---------------------------------------------------------------- 構築ヘルパ


@contextlib.contextmanager
def _params_on_meta():
    """構築時にパラメータだけをmetaデバイスへ載せるコンテキスト。

    バッファ (RoPEのinv_freqなど) は実CPUテンソルとして作られるため、
    チェックポイントに存在しない非永続バッファも正しい値を持つ。
    これはストリーミングロード (モデル全体を一度にメモリへ展開しない) の要。
    """
    orig = nn.Module.register_parameter

    def _register(self: nn.Module, name: str, param: nn.Parameter | None) -> None:
        if param is not None and not param.is_meta:
            param = nn.Parameter(torch.empty(param.shape, device="meta"),
                                 requires_grad=param.requires_grad)
        orig(self, name, param)

    nn.Module.register_parameter = _register  # type: ignore[method-assign]
    try:
        yield
    finally:
        nn.Module.register_parameter = orig  # type: ignore[method-assign]


def _set_tensor(model: nn.Module, key: str, value: torch.Tensor, device: str,
                dtype: torch.dtype | None = None) -> None:
    """チェックポイントのテンソルをモジュールへ直接設定する。

    対応するモジュールが存在しないキー (MTP層など未使用モジュール) は
    何もせず無視する。metaパラメータ/バッファは実デバイスへ割り当て直す。

    [v7] dtypeを指定するとParameterをそのdtypeへ鋳直す。ストリーミング
    ロードの骨組み (from_config) はtorch既定のfp32で構築されるため、
    指定しないと評価モデル全体がfp32で動いてしまい (計算量・メモリ2倍、
    脱量子化キャッシュ必要量も2倍でLRUスラッシング)、bf16比で大幅に
    遅くなる。チェックポイントの実dtype (多くはbf16) と揃えるのが正しい。
    """
    parts = key.split(".")
    parent: nn.Module | None = model
    for p in parts[:-1]:
        if parent is None:
            return
        if p.isdigit():
            try:
                parent = parent[int(p)]
            except (IndexError, TypeError):
                return
        else:
            parent = getattr(parent, p, None)
    if parent is None:
        return
    leaf = parts[-1]
    cur = getattr(parent, leaf, None)
    if isinstance(cur, torch.nn.Parameter):
        tgt = dtype or cur.dtype
        # [v7] .clone()で独立コピーにする: safetensorsのget_tensorはmmapビューを
        # 返すことがあり、そのまま重みにすると「ファイルページキャッシュが
        # プロセスのメモリ上限 (cgroup) に加算され続ける」状態になる
        # (実測: 1.6GB残留で低メモリ環境がOOM)。コピー後にfadviseで
        # ページキャッシュを返却できる。
        src = value.to(device=device, dtype=tgt)
        if cur.is_meta:
            setattr(parent, leaf, torch.nn.Parameter(
                src.clone(), requires_grad=False))
        else:
            cur.data = src.clone()
        return
    if isinstance(cur, torch.Tensor):
        if cur.is_meta or cur.device.type == "meta":
            setattr(parent, leaf, value.to(device=device, dtype=cur.dtype))
        else:
            setattr(parent, leaf, value.to(device=cur.device, dtype=cur.dtype))
        return
    # 既存が無い名前: そのまま設定してみる (失敗は呼び出し側で握りつぶす)
    try:
        setattr(parent, leaf, value.to(device))
    except Exception:
        pass


def swap_linears_with_int4(model: nn.Module, keep_ids: set[int],
                           group_size: int = DEFAULT_GROUP_SIZE,
                           device: str | None = None) -> tuple[int, int]:
    """keep_ids (摂動対象等) 以外のnn.LinearをPackedInt4Linearへ置換する。

    標準ロード後に in-place で呼ぶ場合にも使う (1モジュールずつ変換して
    元重みを即時解放するため、メモリ峰值の増加は1テンソル分で済む)。
    戻り値: (置換モジュール数, 確保メモリ[バイト])
    """
    n_swap, n_bytes = 0, 0
    entries = list(model.named_modules())
    for path, mod in entries:
        if not isinstance(mod, nn.Linear) or id(mod) in keep_ids:
            continue
        if mod.in_features < MIN_IN_FEATURES:
            continue
        parts = path.split(".")
        parent = model
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        try:
            new_mod = PackedInt4Linear(mod.weight.data, mod.bias, group_size=group_size,
                                       device=device)
            setattr(parent, parts[-1], new_mod)
            n_swap += 1
            n_bytes += new_mod.memory_bytes()
        except Exception:
            continue  # 変換できないモジュールは元のまま (fpで動作)
    return n_swap, n_bytes


# ---------------------------------------------------------------- 融合3D専門家INT4 (v6)


class PackedInt4FusedExperts(nn.Module):
    """融合3D専門家重み (E, out, in) をINT4で保持する置換モジュール。

    Qwen4ExpTextExperts / Gemma4TextExperts は専門家ごとに nn.Linear を
    持たず、``gate_up_proj`` / ``down_proj`` が 3D Parameter として格納
    される。本モジュールはこれらを (E*out, in) に平坦化してINT4パックし、
    参照実装と同一の forward(hidden_states, top_k_index, top_k_weights)
    を提供する (ルーティング数値は完全一致)。

    脱量子化は専門家スライス単位で行い、PackedInt4Linearと共通のLRU
    予算 (_WSTATE) に乗せる。摂動対象 (mlp_down) に指定された専門家は
    置換されないため、摂動経路がINT4の影響を受けることはない。
    """

    def __init__(self, gate_up: torch.Tensor, down: torch.Tensor, act_fn,
                 group_size: int = DEFAULT_GROUP_SIZE, device: str | None = None):
        super().__init__()
        self.num_experts = int(gate_up.shape[0])
        self.act_fn = act_fn
        self.gu_out = int(gate_up.shape[1])          # 2 * intermediate
        self.gu_in = int(gate_up.shape[2])           # hidden
        self.dn_out = int(down.shape[1])             # hidden
        self.dn_in = int(down.shape[2])              # intermediate
        dev = device or gate_up.device

        # (E, out, in) → (E*out, in) に平坦化してパック (スケールも行単位)
        # [v7] 行チャンク逐次パック (中間体のピークを抑制)
        gu_p, gu_s, gu_g, gu_pad = pack_int4_chunked(
            gate_up.detach().cpu().reshape(-1, self.gu_in), group_size)
        dn_p, dn_s, dn_g, dn_pad = pack_int4_chunked(
            down.detach().cpu().reshape(-1, self.dn_in), group_size)
        self.gu_packed, self.gu_scale = gu_p.to(dev), gu_s.to(dev)
        self.gu_group, self.gu_pad = gu_g, gu_pad
        self.dn_packed, self.dn_scale = dn_p.to(dev), dn_s.to(dev)
        self.dn_group, self.dn_pad = dn_g, dn_pad
        # 専門家スライスのLRUキャッシュ (PackedInt4Linear._WSTATEと予算を共有)
        self._slice_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._slice_bytes = 0
        self._last_use = 0
        import weakref
        self._self_ref = weakref.ref(self)
        PackedInt4Linear._ALL.append(self._self_ref)

    # ------------------------------------------------------------ キャッシュ

    def _drop_cache(self) -> None:
        for t in self._slice_cache.values():
            try:
                PackedInt4Linear._WSTATE["used"] -= t.numel() * t.element_size()
            except Exception:
                pass
        self._slice_cache = {}
        self._slice_bytes = 0
        try:
            PackedInt4Linear._WSTATE["mods"] = [
                r for r in PackedInt4Linear._WSTATE["mods"] if r() is not self]
        except Exception:
            pass

    def _cache_put(self, key: tuple[str, int], w: torch.Tensor) -> None:
        state = PackedInt4Linear._WSTATE
        nbytes = w.numel() * w.element_size()
        if state["budget"] <= 0 or nbytes > state["budget"]:
            return
        while state["used"] + nbytes > state["budget"] and self._slice_cache:
            k0 = next(iter(self._slice_cache))
            t0 = self._slice_cache.pop(k0)
            state["used"] -= t0.numel() * t0.element_size()
        state["mods"] = [r for r in state["mods"]
                         if r() is not None and r() is not self]
        self._slice_cache[key] = w
        state["used"] += nbytes
        state["mods"].append(self._self_ref)

    def _slice(self, which: str, e: int, dtype: torch.dtype | None = None) -> torch.Tensor:
        key = (which, e)
        w = self._slice_cache.get(key)
        if w is not None:
            PackedInt4Linear._CLOCK += 1
            self._last_use = PackedInt4Linear._CLOCK
            return w
        if which == "gu":
            out_f, packed, scale, g, pad, in_f = (
                self.gu_out, self.gu_packed, self.gu_scale,
                self.gu_group, self.gu_pad, self.gu_in)
            rows = packed[e * out_f:(e + 1) * out_f]
            sc = scale[e * out_f:(e + 1) * out_f]
        else:
            out_f, packed, scale, g, pad, in_f = (
                self.dn_out, self.dn_packed, self.dn_scale,
                self.dn_group, self.dn_pad, self.dn_in)
            rows = packed[e * out_f:(e + 1) * out_f]
            sc = scale[e * out_f:(e + 1) * out_f]
        dt = dtype or self.gu_scale.dtype
        w = dequant_int4(rows.contiguous(), sc, g, out_f, in_f, dt, pad)
        self._cache_put(key, w)
        return w

    # ------------------------------------------------------------ forward

    def memory_bytes(self) -> int:
        return (self.gu_packed.numel() + self.gu_scale.numel() * 4
                + self.dn_packed.numel() + self.dn_scale.numel() * 4)

    def dequant_bytes(self) -> int:
        """全専門家をbf16化した場合のバイト数。"""
        return ((self.gu_out * self.gu_in + self.dn_out * self.dn_in)
                * 2 * self.num_experts)

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = int(expert_idx[0])
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            dt = current_state.dtype
            wgu = self._slice("gu", expert_idx, dt)
            wdn = self._slice("dn", expert_idx, dt)
            gate, up = torch.nn.functional.linear(current_state, wgu).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = torch.nn.functional.linear(current_hidden_states, wdn)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


def swap_fused_experts_with_int4(model: nn.Module, keep_ids: set[int],
                                 device: str | None = None) -> tuple[int, int]:
    """keep_ids 以外の融合3D専門家コンテナをPackedInt4FusedExpertsへ置換する。

    戻り値: (置換モジュール数, 確保メモリ[バイト])
    """
    n_swap, n_bytes = 0, 0
    entries = list(model.named_modules())
    for path, mod in entries:
        gu = getattr(mod, "gate_up_proj", None)
        dn = getattr(mod, "down_proj", None)
        if (not torch.is_tensor(gu) or gu.dim() != 3
                or not torch.is_tensor(dn) or dn.dim() != 3):
            continue
        if id(mod) in keep_ids:
            continue
        parts = path.split(".")
        parent = model
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        try:
            new_mod = PackedInt4FusedExperts(gu.data, dn.data, mod.act_fn, device=device)
            setattr(parent, parts[-1], new_mod)
            n_swap += 1
            n_bytes += new_mod.memory_bytes()
        except Exception:
            continue
    return n_swap, n_bytes


def int4_coverage_bytes(model: nn.Module) -> dict[str, float]:
    """モデル重みのうちINT4化されたバイト数と総バイト数を集計する。"""
    total = 0.0
    quant = 0.0
    for _path, mod in model.named_modules():
        if isinstance(mod, PackedInt4Linear):
            total += mod.out_features * mod.in_features * 2      # bf16換算
            quant += mod.memory_bytes()
        elif isinstance(mod, PackedInt4FusedExperts):
            total += (mod.gu_out * mod.gu_packed.shape[1] * 2 * mod.num_experts
                      + mod.down_out * mod.dn_packed.shape[1] * 2 * mod.num_experts)
            quant += mod.memory_bytes()
    for _name, p in model.named_parameters():
        if p.is_meta:
            continue
        w = p.data
        if w.dim() >= 2:
            total += w.numel() * w.element_size()
    return {"int4_bytes": quant, "total_bytes": total}
