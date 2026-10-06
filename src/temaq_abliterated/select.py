# -*- coding: utf-8 -*-
"""最終候補の選択UI (v2新設, 条件⑥。v3でi18n対応。v9で拡張)。

探索終了後、主要な候補を (番号・拒否率・KL) 付きで一覧表示し、
ユーザーがどの版の非検閲化モデルを採用するか選べる。
ここから追加の試行 (トライ数増加) を依頼することもできる。

[v9②③] --select auto を指定しない限り、モデルの書き出しは必ずこの
選択UIを経由する。全文生成検証済み候補に加えて「試行番号による直接指定」
(t30 のような入力) も可能 — 検証をスキップされた候補から書き出す場合に使う。

非対話環境 (パイプ/CI) では自動的に最良候補が選ばれるため、
バッチ処理やColabでのノートブック実行も妨げない。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from .collapse import format_collapse
from .msg import tr


@dataclass
class Candidate:
    """選択画面に表示される1候補。"""

    no: int                          # 表示番号 (1始まり)
    attempt: int                     # 探索試行番号
    refusal: float                   # 全文生成による拒否率 (代理指標のみの場合もある)
    refusal_n: int = 0               # 拒否応答数 (分母は評価プロンプト数)
    refusal_total: int = 100
    kl: float = float("nan")         # 多位置KL
    kl_first: float | None = None    # 初頭トークンKL (参考)
    nll_ratio: float | None = None
    degen: dict[str, Any] | None = None
    collapsed: bool = False
    collapse_reason: str = ""
    J: float = float("inf")
    params: Any = None               # engine.AblationParams
    refusal_first: float | None = None   # [v9] 初頭トークン比 (代理・床なし)
    verified: bool = True                # [v9] 全文生成検証済みか

    def sort_key(self) -> tuple[float, float]:
        return (self.refusal if self.refusal == self.refusal else 9.0, self.kl)

    def row(self, show_collapse: bool = True) -> str:
        if self.verified and self.refusal_total:
            rf = f"{self.refusal:>5.1%} ({self.refusal_n}/{self.refusal_total})"
        else:
            # [v9] 全文生成検証なしの候補 (試行直接指定): 代理指標を明示する
            rf = (f"{self.refusal:>5.3f} (surrogate"
                  + (f", first {self.refusal_first:.3f}" if self.refusal_first is not None else "")
                  + ")")
        cells = [f" {self.no:>2} ", f"#{self.attempt:<4}",
                 rf,
                 f"{self.kl:8.4f}"]
        if show_collapse:
            cells.append(format_collapse(self.collapsed, self.collapse_reason))
        return " | ".join(cells)


def _default_pick(cands: list[Candidate]) -> Candidate | None:
    """崩壊していない候補から (拒否率, KL) 最小のものを選ぶ。

    [v9] 全文生成検証済み候補を優先する (未検証候補は代理指標しかないため、
    同一スコアでも検証済みを上に置く)。
    """
    ok = [c for c in cands if not c.collapsed]
    pool = ok if ok else cands
    if not pool:
        return None
    return min(pool, key=lambda c: (not c.verified,) + c.sort_key())


def choose_candidate(
    cands: list[Candidate],
    *,
    interactive: bool,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    on_more_trials: Callable[[int], list[Candidate]] | None = None,
    default_extra_trials: int = 50,
    trial_lookup: Callable[[int], Candidate | None] | None = None,
) -> Candidate | None:
    """候補一覧を表示して選択させる。

    interactive=True かつ stdinがTTYなら対話選択。それ以外は自動最良選択。
    入力: 番号 (採用) / Enter (既定=推奨候補) / a または a<N> (N試行追加) /
          t<N> (試行番号Nを直接採用・全文生成検証なし) / q (保存せず終了)
    on_more_trials(n) は追加試行後に更新された全候補リストを返すこと。
    trial_lookup(n) は試行番号nの候補 (代理指標のみ) を返す。無い場合はNone。
    """
    if not cands:
        return None
    # カスタム入力関数 (テスト/組み込み) が渡された場合は対話可能とみなす。
    # 既定のinput()を使う場合のみ、実際にTTYかどうかを確認する
    # (パイプ実行やCIでブロックしないため)。
    uses_default_input = input_fn is input
    is_tty = (not uses_default_input) or (
        hasattr(sys.stdin, "isatty") and sys.stdin.isatty())

    def _show() -> None:
        print_fn("")
        print_fn(tr("cand_list_title"))
        print_fn(tr("cand_header"))
        for c in sorted(cands, key=lambda c: c.no):
            print_fn(c.row())
        if trial_lookup is not None:
            print_fn(tr("cand_trial_hint"))
        print_fn("")

    if not interactive or not is_tty:
        best = _default_pick(cands)
        if best is not None:
            if not is_tty and interactive:
                print_fn(tr("cand_auto_pick_pipe", no=best.no,
                            r=best.refusal, k=best.kl))
            else:
                print_fn(tr("cand_auto_pick", no=best.no, r=best.refusal, k=best.kl))
        return best

    _show()
    while True:
        try:
            raw = input_fn(tr("cand_prompt")).strip()
        except (EOFError, KeyboardInterrupt):
            print_fn(tr("cand_input_abort"))
            return _default_pick(cands)
        low = raw.lower()
        if low in ("q", "quit", "exit"):
            print_fn(tr("cand_quit"))
            return None
        if raw == "" or low in ("d", "y", "yes", "best"):
            best = _default_pick(cands)
            if best is not None:
                print_fn(tr("cand_recommended", no=best.no, r=best.refusal, k=best.kl))
            return best
        add_match = (low == "a"
                     or (low.startswith("a") and low[1:].lstrip("+").isdigit())
                     or (low.startswith("+") and low[1:].isdigit()))
        if add_match:
            if on_more_trials is None:
                print_fn(tr("cand_add_disabled"))
                continue
            nstr = raw[1:].strip().lstrip("+")
            if not nstr:
                try:
                    nstr = input_fn(tr("cand_add_n_prompt", n=default_extra_trials)).strip()
                except (EOFError, KeyboardInterrupt):
                    nstr = ""
            try:
                n = int(nstr) if nstr else default_extra_trials
            except ValueError:
                print_fn(tr("cand_need_number"))
                continue
            if n <= 0:
                print_fn(tr("cand_le_zero"))
                continue
            print_fn(tr("cand_add_running", n=n))
            try:
                new_cands = on_more_trials(n)
            except Exception as e:  # noqa: BLE001
                print_fn(tr("cand_add_fail", e=e))
                continue
            if new_cands:
                cands = new_cands
            _show()
            continue
        # [v9②③] 試行番号による直接指定: t30 / T30 / #30
        tnum = None
        if low.startswith("t") and low[1:].strip().lstrip("#").isdigit():
            tnum = int(low[1:].strip().lstrip("#"))
        elif raw.startswith("#") and raw[1:].isdigit():
            tnum = int(raw[1:])
        if tnum is not None:
            if trial_lookup is None:
                print_fn(tr("cand_trial_disabled"))
                continue
            hit = trial_lookup(tnum)
            if hit is None:
                print_fn(tr("cand_trial_not_found", i=tnum))
                continue
            print_fn(tr("cand_trial_confirm", i=tnum, r=hit.refusal, k=hit.kl))
            try:
                confirm = input_fn(tr("cand_trial_confirm_yn")).strip().lower()
            except (EOFError, KeyboardInterrupt):
                confirm = ""
            if confirm not in ("y", "yes"):
                continue
            print_fn(tr("cand_adopted_trial", i=tnum))
            return hit
        if raw.isdigit():
            idx = int(raw)
            hit = next((c for c in cands if c.no == idx), None)
            if hit is None:
                print_fn(tr("cand_not_found", i=idx, m=max(c.no for c in cands)))
                continue
            if hit.collapsed:
                confirm = input_fn(tr("cand_collapsed_confirm",
                                      i=idx, reason=hit.collapse_reason)).strip().lower()
                if confirm not in ("y", "yes"):
                    continue
            print_fn(tr("cand_adopted", i=idx))
            return hit
        print_fn(tr("cand_bad_input"))
