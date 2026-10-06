# -*- coding: utf-8 -*-
"""崩壊ガード (v2新設): KLが小さくてもモデルが崩壊していないかを確認する。

条件⑦「KLが正しく測定できているか (KLが小さいのに崩壊することがないか)」への
2重の回答:

1. 出力側検査 (degeneration_stats)
   100プロンプト全文生成の応答を使い、空応答・同一n-gramの異常反復・
   語彙多様性の消失 (トークン繰り返し率) を検出する。
   拒否率測定で既に生成した応答を再利用するため追加コストはゼロ。

2. 統計側検査 (nll_ratio / 多位置KL)
   教師強制で基底モデルの継続トークン列を評価し、
   平均負対数尤度 (NLL) の基底比が閾値を超えたら「分布の崩壊」と判定する。
   最初のトークンだけのKLでは捉えられない、生成途中からの破綻を検出できる。

判定は collapse_verdict() に集約され、候補選択画面に表示される。
"""

from __future__ import annotations

import re
from typing import Any

from .msg import tr

# 崩壊判定の既定閾値
DEFAULT_NLL_RATIO_MAX = 2.5     # 基底比NLLがこの倍率を超えたら崩壊疑い
DEFAULT_REPEAT_RATE_MAX = 0.40  # 反復応答の割合
DEFAULT_EMPTY_RATE_MAX = 0.30   # 空応答の割合
DEFAULT_UNIQUE_RATIO_MIN = 0.30  # 語彙多様性の下限


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _is_repetitive(text: str) -> bool:
    """同一3-gramの異常反復、または多様性の消失を検出する。"""
    words = _words(text)
    if len(words) < 8:
        return False
    grams: dict[str, int] = {}
    for i in range(len(words) - 2):
        g = " ".join(words[i:i + 3])
        grams[g] = grams.get(g, 0) + 1
    if grams and max(grams.values()) >= 4:
        return True
    uniq = len(set(words)) / len(words)
    return uniq < DEFAULT_UNIQUE_RATIO_MIN


def degeneration_stats(responses: list[str]) -> dict[str, Any]:
    """応答列の崩壊 (degeneration) 統計。"""
    n = max(1, len(responses))
    empty = sum(1 for r in responses if not r.strip())
    repetitive = sum(1 for r in responses if _is_repetitive(r))
    uniq_ratios = []
    for r in responses:
        words = _words(r)
        if len(words) >= 4:
            uniq_ratios.append(len(set(words)) / len(words))
    mean_unique = sum(uniq_ratios) / len(uniq_ratios) if uniq_ratios else 1.0
    score = empty / n + repetitive / n
    return {
        "n": len(responses),
        "empty_rate": round(empty / n, 4),
        "repeat_rate": round(repetitive / n, 4),
        "mean_unique_ratio": round(mean_unique, 4),
        "degeneration_score": round(score, 4),
    }


def collapse_verdict(degen: dict[str, Any] | None = None,
                     nll_ratio: float | None = None,
                     kl_multi: float | None = None,
                     kl_first: float | None = None,
                     nll_ratio_max: float = DEFAULT_NLL_RATIO_MAX) -> tuple[bool, str]:
    """崩壊判定。戻り値: (崩壊か, 理由文字列)。

    - NLL比 > nll_ratio_max        → 「NLLが基底比X倍 (分布崩壊の疑い)」
    - 反復率・空応答率が異常       → 「応答の反復/空白が異常」
    - 多位置KLが最初トークンKLより
      極端に大きい (生成途中で破綻) → 警告として表示
    """
    reasons: list[str] = []
    if nll_ratio is not None and nll_ratio > nll_ratio_max:
        reasons.append(tr("coll_nll", r=nll_ratio, m=nll_ratio_max))
    if degen:
        if degen.get("repeat_rate", 0.0) > DEFAULT_REPEAT_RATE_MAX:
            reasons.append(tr("coll_repeat", r=degen['repeat_rate']))
        if degen.get("empty_rate", 0.0) > DEFAULT_EMPTY_RATE_MAX:
            reasons.append(tr("coll_empty", r=degen['empty_rate']))
        if degen.get("mean_unique_ratio", 1.0) < DEFAULT_UNIQUE_RATIO_MIN:
            reasons.append(tr("coll_diversity", r=degen["mean_unique_ratio"]))
    if reasons:
        return True, " / ".join(reasons)
    note = ""
    if (kl_multi is not None and kl_first is not None
            and kl_first < 0.02 and kl_multi > 8 * max(kl_first, 1e-4) and kl_multi > 0.05):
        note = tr("coll_note_kl", first=kl_first, multi=kl_multi)
    return False, note


def format_collapse(collapsed: bool, reason: str) -> str:
    if collapsed:
        return tr("coll_flag_bad") + reason + ")"
    if reason[:1] in ("注", "!") or reason.startswith("Note"):
        return tr("coll_flag_warn") + reason
    return "OK"
