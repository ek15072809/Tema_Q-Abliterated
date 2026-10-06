# -*- coding: utf-8 -*-
"""レポート生成 (条件⑥): 横軸=試行回数・縦軸=拒否率の折れ線グラフPNGと、
実験比較グラフ群の出力。日本語フォントを自動選択する。"""

from __future__ import annotations

from pathlib import Path

from .msg import is_jp
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt  # noqa: E402

# 日本語フォントの候補 (Linux/macOS/Windows)
_FONT_CANDIDATES = [
    "Noto Sans CJK JP", "Noto Sans CJK SC", "Noto Sans SC", "Noto Sans JP",
    "Yu Gothic UI", "Yu Gothic", "MS Gothic", "Meiryo", "Hiragino Sans",
    "IPAexGothic", "IPAGothic", "WenQuanYi Zen Hei",
]


def _register_extra_fonts() -> None:
    """配布されがちなCJKフォントパスを明示的に登録する (見つかる場合)。"""
    extra_paths = [
        "/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/YuGothM.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
        "C:/Windows/Fonts/meiryo.ttc",
    ]
    for p in extra_paths:
        try:
            if Path(p).exists():
                fm.fontManager.addfont(p)
        except Exception:
            pass


def setup_japanese_font() -> str | None:
    """日本語表示可能なフォントを探して設定する。戻り値: フォント名 (なければNone)。"""
    _register_extra_fonts()
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _FONT_CANDIDATES:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    matplotlib.rcParams["axes.unicode_minus"] = False
    return None


def refusal_curve_png(history: list[dict[str, Any]], out_path: Path,
                       title: str = "探索過程: 拒否率の収束",
                       verified_points: list[tuple[int, float]] | None = None,
                       baseline_rate: float | None = None,
                       final_rate: float | None = None,
                       surrogate_corr: float | None = None) -> Path:
    """横軸=試行回数・縦軸=拒否率の折れ線グラフを出力する。

    [v6] surrogate_corr (代理指標と全文生成拒否のプローブ単位相関) が
    与えられれば図中に注記する。verified_points は全文生成で検証した
    候補の実測拒否率で、ソフト拒否率曲線との対応関係を直接示す。
    """
    jp = is_jp()
    font = setup_japanese_font() if jp else None

    attempts = [h["attempt"] for h in history]
    raw = [h.get("refusal_soft") for h in history]

    # その時点までの最良値 (拒否率の改善曲線)
    best: list[float] = []
    cur = float("inf")
    for v in raw:
        if v is not None and v < cur:
            cur = v
        best.append(cur if cur != float("inf") else float("nan"))

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(attempts, raw, color="#9ec5e8", alpha=0.55, lw=1.0,
            label="各試行のソフト拒否率" if jp else "soft refusal per trial")
    ax.plot(attempts, best, color="#1f5fa8", lw=2.0,
            label="その時点までの最良値" if jp else "best so far")

    if baseline_rate is not None:
        ax.axhline(baseline_rate, color="#c0392b", ls="--", lw=1.4,
                   label=(f"元モデル (全文生成) {baseline_rate:.2f}" if jp
                          else f"baseline (full gen) {baseline_rate:.2f}"))
    if final_rate is not None:
        ax.scatter([attempts[-1] if attempts else 0], [final_rate], color="#117a3b", zorder=5, s=46,
                   label=(f"最終モデル (全文生成) {final_rate:.2f}" if jp
                          else f"final (full gen) {final_rate:.2f}"))
    if verified_points:
        xs, ys = zip(*verified_points)
        ax.scatter(xs, ys, color="#8e44ad", zorder=5, s=34,
                   label="検証候補 (全文生成)" if jp else "verified candidates")

    ax.set_xlabel("試行回数 (attempts)" if jp else "Attempts")
    ax.set_ylabel("拒否率 (refusal rate)" if jp else "Refusal rate")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    if surrogate_corr is not None and surrogate_corr == surrogate_corr:
        ax.text(0.02, 0.02,
                (f"代理指標と全文生成拒否の相関: {surrogate_corr:.2f} (n=プローブ)"
                 if jp else f"surrogate vs full-generation correlation: {surrogate_corr:.2f}"),
                transform=ax.transAxes, fontsize=8.5, color="#333333", va="bottom")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def memory_timeline_png(snapshots: list, out_path: Path,
                         title: str = "メモリ使用量の推移") -> Path:
    """メモリガードの記録からRAM/RSS/SWAPの推移グラフを出力する (v2)。

    v4修正: スナップショットが少ない場合 (短い実行・フェーズ境界のみの記録) でも
    点が必ず見えるようにマーカーを描き、軸範囲をデータに合わせる。
    従来は1〜2点の折れ線が描画されず「空のグラフ」に見える問題があった。
    """
    jp = is_jp()
    if jp:
        setup_japanese_font()
    if not snapshots:
        return out_path
    ts = [s.t for s in snapshots]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    series = [
        ("プロセスRSS" if jp else "process RSS", [s.rss for s in snapshots],
         "#1f5fa8", 2.0, "-"),
        ("空きRAM" if jp else "available RAM", [s.ram_avail for s in snapshots],
         "#117a3b", 1.8, "-"),
    ]
    swaps = [s.swap for s in snapshots]
    if any(v > 0.01 for v in swaps):
        series.append(("SWAP使用量" if jp else "swap used", swaps, "#c0392b", 1.6, "--"))
    vr = [s.vram_free for s in snapshots]
    if any(v > 0 for v in vr):
        series.append(("空きVRAM" if jp else "free VRAM", vr, "#7d3c98", 1.6, ":"))

    for name, ys, color, lw, ls in series:
        # マーカー付き: 点数が少なくても可視化される (v4)
        marker = "o" if len(ts) <= 4 else ("." if len(ts) <= 40 else None)
        ax.plot(ts, ys, color=color, lw=lw, ls=ls, label=name,
                marker=marker, ms=5, markevery=max(1, len(ts) // 40))

    span = (max(ts) - min(ts)) if len(ts) > 1 else 0.0
    if span < 5.0 and len(ts) >= 2:
        pad = max(0.5, span * 0.1)
        ax.set_xlim(min(ts) - pad, max(ts) + pad)
    all_y = [v for _, ys, *_ in series for v in ys]
    if all_y:
        lo, hi = min(all_y), max(all_y)
        if hi - lo < 0.3:
            mid = (lo + hi) / 2
            ax.set_ylim(mid - 0.25, mid + 0.25)
    ax.set_xlabel("経過時間 (秒)" if jp else "elapsed (s)")
    ax.set_ylabel("メモリ (GB)" if jp else "memory (GB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    if len(ts) < 3:
        ax.text(0.02, 0.02, f"{len(ts)} snapshots",
                transform=ax.transAxes, fontsize=8, color="#666666", va="bottom")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def comparison_curves_png(curves: dict[str, list[float]], out_path: Path,
                          title: str = "探索手法比較: 拒否率の収束",
                          ylabel: str = "拒否率 (その時点までの最良)") -> Path:
    """複数手法の収束曲線を同一軸に描く比較グラフ。"""
    jp = is_jp()
    if jp:
        setup_japanese_font()
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    colors = ["#1f5fa8", "#c0392b", "#7d3c98", "#117a3b", "#b9770e"]
    for i, (name, ys) in enumerate(curves.items()):
        xs = list(range(1, len(ys) + 1))
        ax.plot(xs, ys, color=colors[i % len(colors)], lw=2.0, label=name)
    ax.set_xlabel("試行回数 (attempts)" if jp else "Attempts")
    ax.set_ylabel(ylabel if jp else "best-so-far refusal rate")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def comparison_curves_multi_png(histories: dict[str, list[dict[str, Any]]],
                                out_path: Path,
                                title: str = "探索手法比較",
                                include_time_panel: bool = True) -> Path:
    """v4: 探索手法の多面的比較グラフ (拒否率 / 目的関数J / 実時間)。

    histories: {手法名: 試行履歴のリスト (search の history 形式)}。

    3パネル構成:
      (a) その時点までの最良ソフト拒否率 vs 試行回数
      (b) その時点までの最良目的関数 J vs 試行回数   ← ツールが実際に最小化する基準
      (c) 最良 J vs 経過時間 (1試行あたり評価コスト込みの anytime 性)
    図下部に最終到達値 (拒否率・KL・J・所要時間) を併記する。

    従来版 (comparison_curves_png) は拒否率の1軸だけだったため、
    「ランダム探索のほうが拒否率が低い」という表面的な比較になりやすかった。
    制約付き目的関数 J = KL + λ·max(0, R̂−r*)² を見れば、目標達成後の
    拒否率の追加低減はJを改善せず、品質 (KL) の悪化と引き換えになることが
    読み取れる。このグラフはその3面 (拒否率・J・時間) を1枚にまとめる。
    """
    from .search import best_so_far_curve  # 循環import回避のため遅延 import

    jp = is_jp()
    if jp:
        setup_japanese_font()
    colors = ["#1f5fa8", "#c0392b", "#7d3c98", "#117a3b", "#b9770e"]
    panels = 3 if include_time_panel else 2
    fig, axes = plt.subplots(1, panels, figsize=(4.6 * panels, 4.6),
                             constrained_layout=True)
    if panels == 2:
        axes = list(axes)

    summaries: dict[str, dict[str, float]] = {}
    for i, (name, hist) in enumerate(histories.items()):
        if not hist:
            continue
        refusal = best_so_far_curve(hist, "refusal_soft")
        Js = best_so_far_curve(hist, "J")
        t_axis: list[float] = []
        acc = 0.0
        for h in hist:
            acc += float(h.get("eval_seconds") or 0.0)
            t_axis.append(acc)
        # [v5] J/kl はスクリーニング棄却履歴でNoneになり得る (棄却=最下位扱い)
        def _j_of(h: dict[str, Any]) -> float:
            v = h.get("J")
            return float(v) if isinstance(v, (int, float)) and v == v else float("inf")
        best = min(hist, key=_j_of)
        summaries[name] = {"refusal": float(best.get("refusal_soft", float("nan"))),
                           "kl": (float(best["kl"]) if isinstance(best.get("kl"), (int, float))
                                  else float("nan")),
                           "J": _j_of(best),
                           "seconds": acc}
        c = colors[i % len(colors)]
        xs = list(range(1, len(refusal) + 1))
        axes[0].plot(xs, refusal, color=c, lw=2.0, label=name)
        axes[1].plot(xs, Js, color=c, lw=2.0, label=name)
        if panels == 3:
            axes[2].plot(t_axis, Js, color=c, lw=2.0, label=name)

    axes[0].set_xlabel("試行回数" if jp else "attempts")
    axes[0].set_ylabel("最良ソフト拒否率" if jp else "best-so-far refusal rate")
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[1].set_xlabel("試行回数" if jp else "attempts")
    axes[1].set_ylabel("最良目的関数 J" if jp else "best-so-far objective J")
    if panels == 3:
        axes[2].set_xlabel("経過時間 (秒)" if jp else "elapsed (s)")
        axes[2].set_ylabel("最良目的関数 J" if jp else "best-so-far objective J")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle(title, fontsize=12)
    parts = []
    for name, s in summaries.items():
        parts.append(f"{name}: R={s['refusal']:.3f} KL={s['kl']:.4f} "
                     f"J={s['J']:.4f} ({s['seconds']:.0f}s)")
    if parts:
        fig.text(0.5, 0.012, " | ".join(parts), ha="center", fontsize=8.5,
                 color="#333333")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def speedup_bar_png(labels: list[str], values: list[float], out_path: Path,
                    title: str = "1候補あたり評価時間の比較", unit: str = "秒") -> Path:
    """高速化比較の棒グラフ。"""
    jp = is_jp()
    if jp:
        setup_japanese_font()
    if not jp and unit == "秒":
        unit = "s"   # 英語表示時はDejaVu Sansで描ける単位へ置換
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    xs = range(len(labels))
    ax.bar(xs, values, color=["#c0392b", "#e67e22", "#1f5fa8"][:len(labels)], alpha=0.85)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3g}{unit}", ha="center", va="bottom")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"時間 ({unit})" if jp else f"time ({unit})")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
