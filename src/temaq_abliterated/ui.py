# -*- coding: utf-8 -*-
"""進捗表示UI (v5全面改訂)。

v4までの問題点とv5の対応:
  1. \\r で短い行を上書きした際に旧表示の残骸 (「ETA 0s 10s」や
     「評価中: KL バッチ 1/4   チ 8/8 (残り 0s)」) が残る
     → LineManager: 常に「前回の表示幅」を記憶し、空白パディングで
       行末まで消去してから書き込む (TTY)。
  2. '#' のインジケーターが見づらい
     → ブロック文字 (█░) を使い、エンコーディング不可の環境では
       自動で ASCII (=.) へフォールバック。パーセント・n/N を併記。
  3. 残り時間がstep境界でしか更新されない
     → 専用ティッカースレッドが1秒ごとに行を再描画し、
       「前回step完了時点のETA − 経過秒数」のカウントダウンを表示。
       ETA自体はstep完了ごとに実測 (直近step時間の移動平均) で再計算する
       (表示だけ毎秒、計算はstepごと — 単調処理では十分正確)。
  4. メモリログ等の print が進捗行に混入する
     → LineManager経由の print_line() は現在のライブ行を退避して
       独立した行に出力し、その後ライブ行を再描画する。

非TTY (リダイレクト) 時は \\r 更新を行わず、時間間引き (既定10秒) の
通常行として出力する。--quiet 指定時は何も表示しない。
"""

from __future__ import annotations

import sys
import threading
import time

from .msg import tr

# ---------------------------------------------------------------- 基本ユーティリティ


def fmt_duration(seconds: float) -> str:
    """秒数を "1h 23m" / "4m 32s" / "8s" 形式の人間可読文字列へ変換する。"""
    seconds = max(0, int(round(seconds)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def fmt_duration_short(seconds: float) -> str:
    """残り時間表示用: 秒未満を切り捨て "2h 05m" / "12s" 形式。"""
    return fmt_duration(seconds)


def _is_tty(stream=None) -> bool:
    try:
        stream = stream or sys.stdout
        return bool(stream.isatty())
    except Exception:
        return False


def _out(text: str, newline: bool = True) -> None:
    end = "\n" if newline else ""
    try:
        sys.stdout.write(text + end)
        sys.stdout.flush()
    except Exception:
        pass


# インジケーター文字: stdoutのエンコーディングで実際に書けるか起動時に判定
def _pick_bar_chars() -> tuple[str, str]:
    for filled, empty in (("█", "░"), ("■", "·"), ("=", "."), ("#", "-")):
        try:
            enc = getattr(sys.stdout, "encoding", None) or "utf-8"
            (filled + empty).encode(enc)
            return filled, empty
        except Exception:
            continue
    return "=", "."


_BAR_CHARS: tuple[str, str] = ("█", "░")
try:
    _BAR_CHARS = _pick_bar_chars()
except Exception:
    _BAR_CHARS = ("=", ".")


def render_bar(frac: float, width: int = 20) -> str:
    """進捗バーを描画 (frac: 0..1)。ブロック文字は環境で自動フォールバック。"""
    frac = min(1.0, max(0.0, frac))
    filled, empty = _BAR_CHARS
    n = int(round(width * frac))
    return filled * n + empty * (width - n)


# ---------------------------------------------------------------- 行管理 (残骸ゼロ出力)


class LineManager:
    """単一行ライブ表示と通常printの排他制御。

    - show(text): ライブ行を \\r + パディングで更新 (残骸を出さない)。
    - print_line(text): ライブ行を退避し、独立した行に出力して再描画。
    - ロックでティッカー (別スレッド) との競合を防ぐ。
    """

    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.tty = _is_tty(self.stream)
        self._lock = threading.RLock()
        self._live_len = 0
        self._live_text = ""

    # ---- 内部 ----
    def _write(self, s: str) -> None:
        try:
            self.stream.write(s)
            self.stream.flush()
        except Exception:
            pass

    def _pad_to(self, text: str, prev_len: int) -> str:
        """前回表示幅より短い場合、末尾を空白で埋めて残骸を消す。"""
        pad = prev_len - _visible_len(text)
        if pad > 0:
            text = text + " " * pad
        return text

    # ---- 公開API ----
    def show(self, text: str) -> None:
        """ライブ行を更新 (TTYのみ。非TTYでは何もしない)。"""
        if not self.tty:
            return
        with self._lock:
            clipped = text[:200]
            self._write("\r" + self._pad_to(clipped, self._live_len))
            self._live_len = _visible_len(clipped)
            self._live_text = clipped

    def hide(self) -> None:
        """ライブ行を空白で消去 (次にprint_lineするときの一時退避)。"""
        if not self.tty or self._live_len == 0:
            return
        with self._lock:
            self._write("\r" + " " * self._live_len + "\r")
            # _live_len/_live_textは再描画のために保持

    def unhide(self) -> None:
        if not self.tty or not self._live_text:
            return
        with self._lock:
            self._write(self._live_text)

    def print_line(self, text: str) -> None:
        """[v8条件②] 確定行をライブ行の物理行に直接上書きして出力する。

        v7は「\r + 空白×live_len + \r」でライブ行を消去してから行を書き、
        ライブ行を次の物理行に再描画していた。TTY上の表示は正しいが、
        生ストリームをログに取る環境 (ConPTYのtee・ターミナル転写・
        スクロールバックのコピー) では「消去用の空白の連なり」がそのまま
        残り、Trial行などの確定行の前に大量の不自然な空白が出ていた
        (v8改善要望②)。
        v8は確定行をライブ行の物理位置に直接上書きして改行で確定する。
        追加の空白は一切出力されず、どんなキャプチャ方式でも確定行だけが
        残る。ライブ行は次のティッカー描画 (最長0.5秒後) に現在行へ
        再表示されるため、画面上の見た目も v7 と同じである。
        """
        with self._lock:
            if self.tty and self._live_len > 0:
                clipped = text[:200]
                self._write("\r" + self._pad_to(clipped, self._live_len) + "\n")
                self._live_len = 0
                self._live_text = ""
            else:
                self._write(text + "\n")

    def newline_close(self) -> None:
        """ライブ行を確定させる (改行で閉じる)。"""
        with self._lock:
            if self.tty and self._live_len > 0:
                self._write("\n")
            self._live_len = 0
            self._live_text = ""


def _visible_len(s: str) -> int:
    """表示幅 (ターミナルの桁数) の近似。全角文字を2扱いする。"""
    n = 0
    for ch in s:
        n += 2 if ord(ch) > 0x2E80 else 1
    return n


def pad_display(s: str, width: int, right: bool = True) -> str:
    """[v7] 表示幅がwidth桁になるようスペースを詰める (CJK=2桁換算)。

    進捗バーのラベルや試行番号を固定幅に揃えるために使う
    (条件⑥: インデント・桁位置の整列)。
    """
    pad = width - _visible_len(s)
    if pad <= 0:
        return s
    return s + " " * pad if right else " " * pad + s


# グローバルな行マネージャ (memguard等のどこからでも利用可能)
LINES = LineManager()


# ---------------------------------------------------------------- ライブ行の所有権 (v5)

_LIVE_OWNER: dict[str, Any] = {"obj": None}


def _claim_live(obj: Any) -> None:
    """ライブ行の所有者をobjに移す (直近の能動的な書き込み者が唯一の描画者)。"""
    _LIVE_OWNER["obj"] = obj


def _is_live_owner(obj: Any) -> bool:
    """objが現在のライブ行所有者か (ティッカーの描画競合防止)。"""
    return _LIVE_OWNER["obj"] is obj


def print_line(text: str) -> None:
    """進捗表示と衝突しない通常行出力 (グローバル)。"""
    LINES.print_line(text)


# ---------------------------------------------------------------- 進捗 (v5中核)


class _StepStats:
    """step時間の統計 (ウォームアップ除外つき移動平均)。"""

    def __init__(self, window: int = 8):
        self.window = window
        self.times: list[float] = []
        self.t_last_step: float = time.perf_counter()

    def observe(self, dt: float) -> None:
        # 初回stepは各種ウォームアップを含むため平均から除外
        self.times.append(dt)
        if len(self.times) > self.window:
            self.times.pop(0)

    def rate(self) -> float | None:
        """1stepあたりの平均秒数 (信頼できない場合はNone)。"""
        if len(self.times) < 1:
            return None
        return sum(self.times) / len(self.times)

    def touch(self) -> None:
        self.t_last_step = time.perf_counter()


class Progress:
    """バッチ/プロンプト単位の進捗表示 (n/N + 経過 + 残りカウントダウン)。

    v7の動作 (条件④⑤⑦):
      - **n/Nは「完了件数」**: 測定が始まった瞬間に 1/N とは表示せず、
        1件完了で初めて 1/N になる (update_to(i) = i件完了)。
      - **残り時間は前回の実績からも推定**: prior_rate (1件あたり秒数) を
        与えると、最初の1件が終わる前から実在の残り時間を表示する。
        データがない場合は 0 秒ではなく "--" を表示する (条件⑤)。
      - **後処理をバーに含める**: extend_total() で最終ステップ後の
        後処理 (基底分布計算等) を進捗に加算でき、"6/6 残り0s" のまま
        長時間待つことがなくなる (条件⑦)。
      - TTY: 専用ティッカースレッドが0.5秒周期で行を再描画し、
        「残り」を毎秒カウントダウンする。ETA自体はstep完了時に
        直近step時間の移動平均から再計算する。
      - 非TTY: interval秒ごと (既定10秒) に通常行として出力。
      - quiet: 何も表示しない。

    ラベルは pad_display で固定幅に揃えられる (条件⑥)。
    """

    LABEL_WIDTH = 24   # ラベルの表示桁数 (全角12文字分・EN長ラベルにも収まる)

    def __init__(self, total: int, prefix: str = "", quiet: bool = False,
                 interval: float = 10.0, enabled: bool = True, unit: str = "",
                 prior_rate: float | None = None):
        self.total = max(1, int(total))
        self.prefix = prefix
        self.quiet = quiet
        self.enabled = enabled and not quiet
        self.interval = max(1.0, interval)
        self.unit = unit
        self.n = 0
        self.t0 = time.perf_counter()
        self._tty = _is_tty()
        self._stats = _StepStats()
        self._eta_at_step: float | None = None   # step完了時点の残り秒
        self._t_step: float = time.perf_counter()
        self._note = ""
        self._done_text = ""
        self._last_nontty = 0.0
        self._prior_rate = (float(prior_rate) if prior_rate and prior_rate > 0 else None)
        self._stop = threading.Event()
        self._render_lock = threading.Lock()
        if self.enabled and self._tty:
            self._ticker = threading.Thread(target=self._tick_loop, daemon=True)
            self._ticker.start()
            _claim_live(self)   # [v5] ライブ行の所有権を取得
        else:
            self._ticker = None

    # ---------------------------------------------------------------- API (v7)

    def extend_total(self, extra: int, note: str = "") -> None:
        """[v7] 末尾の後処理ぶんだけステップ総数を延長する (条件⑦)。

        例: 6バッチの生成のあとに3バッチの基底分布計算が続く処理で、
        生成6ステップ完了時に extend_total(3) を呼ぶと、バーは 6/9
        「後処理」となり、残り時間に後処理ぶんが含まれる。
        """
        if not self.enabled or extra <= 0:
            return
        self.total = self.total + int(extra)
        if note:
            self._note = note
        if self._tty:
            _claim_live(self)
            with self._render_lock:
                self._render()

    # ---------------------------------------------------------------- 内部
    def _tick_loop(self) -> None:
        while not self._stop.wait(0.5):
            try:
                if _is_live_owner(self):   # [v5] 所有権を失ったら描画しない
                    with self._render_lock:
                        self._render()
            except Exception:
                pass

    def _remaining(self) -> float | None:
        """現在表示すべき残り秒。推定不能な場合は None (表示は "--")。

        v7: stepデータがないときは prior_rate (前回の同種処理の実績)
        から見積もる。両方なければ None (根拠のない「0秒」を表示しない)。
        """
        if self._eta_at_step is not None:
            elapsed_since_step = time.perf_counter() - self._t_step
            return max(0.0, self._eta_at_step - elapsed_since_step)
        left = self.total - self.n
        if left <= 0:
            return 0.0
        rate = self._stats.rate()
        if rate:
            return rate * left
        if self._prior_rate:
            return self._prior_rate * left
        return None

    def _render(self) -> None:
        i = self.n
        elapsed = time.perf_counter() - self.t0
        remaining = self._remaining()
        pct = 100.0 * i / self.total
        note = f"  {self._note}" if self._note else ""
        rem_txt = fmt_duration(remaining) if remaining is not None else "--"
        label = pad_display(self.prefix, self.LABEL_WIDTH)
        line = (f"{label} {render_bar(i / self.total)} {pct:3.0f}%  "
                f"{i}/{self.total}{note}  "
                f"{tr('ui_elapsed')} {fmt_duration(elapsed)}  "
                f"{tr('ui_remaining')} {rem_txt}")
        if self._tty:
            LINES.show(line)
        else:
            LINES.print_line("  " + line)

    # ---------------------------------------------------------------- API
    def update(self, step: int = 1) -> None:
        if not self.enabled:
            return
        self.update_to(self.n + step)

    def update_to(self, i: int) -> None:
        """i件完了したときに呼ぶ (v7: 件数は「完了済み」を表す・条件④)。"""
        if not self.enabled:
            return
        now = time.perf_counter()
        dt = now - self._t_step
        if i > self.n:
            self._stats.observe(dt / (i - self.n))
        self.n = min(self.total, int(i))
        self._t_step = now
        # step完了ごとに残り時間を再計算して「基準ETA」を更新
        rate = self._stats.rate()
        if rate is not None:
            self._eta_at_step = rate * (self.total - self.n)
        else:
            base = self._prior_rate or 0.0
            elapsed = now - self.t0
            est = (elapsed / max(1, self.n)) if self.n else base
            self._eta_at_step = est * (self.total - self.n) if est > 0 else None
        if self._tty:
            _claim_live(self)   # [v5] 能動的な更新は所有権を取り戻す
            with self._render_lock:
                self._render()
        else:
            now2 = time.perf_counter()
            # v7: 完了時の二重表示を防ぐ (最終行はclose()が出す)
            if self.n == 1 or (self.n < self.total and now2 - self._last_nontty >= self.interval):
                self._last_nontty = now2
                with self._render_lock:
                    self._render()

    def set_note(self, note: str) -> None:
        """ライブ行に補足情報 (例: 評価フェーズ名) を表示する。"""
        if not self.enabled:
            return
        if note != self._note:
            self._note = note
            if self._tty:
                with self._render_lock:
                    self._render()

    def close(self, done_text: str = "") -> None:
        """表示を確定する (ティッカー停止 + 行を閉じる)。"""
        if not self.enabled:
            return
        self._stop.set()
        if self._ticker is not None:
            self._ticker.join(timeout=1.0)
        # v7: 完了時は「残り 0s」ではなく経過時間で確定させる (条件⑦)。
        # 中断等で未完了のまま閉じる場合は実勢の n/N を表示する。
        self._note = ""
        elapsed = time.perf_counter() - self.t0
        label = pad_display(self.prefix, self.LABEL_WIDTH)
        frac = min(1.0, self.n / self.total)
        done = done_text or (f"{label} {render_bar(frac)} {100 * frac:3.0f}%  "
                             f"{self.n}/{self.total}  "
                             f"{tr('ui_elapsed')} {fmt_duration(elapsed)}")
        if self._tty:
            with self._render_lock:
                LINES.show(done)
                LINES.newline_close()
        else:
            with self._render_lock:
                LINES.print_line("  " + done)
        self.enabled = False


# ---------------------------------------------------------------- 試行単位の進捗

class TrialProgress:
    """SCAPE探索の試行ごと統計行 + 評価中ライブ行。

    - trial(): 1試行の統計行 (n/N, 所要, 指標, best, 経過, ETA)。
      v7: 「Trial n/N」「所要秒」などの各欄を固定幅化し、行が
      そのまま縦に揃って読めるようにした (条件⑥)。
    - seed_rates(): チェックポイント再開時に過去試行の実績秒数を与え、
      最初の試行から正確な ETA を出す (条件⑤)。
    - live():  評価中のライブ行 (KL バッチ n/N など)。
      「評価の残り」はバッチstepごとに再計算、「探索全体の残り」は
      試行ごとに再計算し、どちらも毎秒のカウントダウン表示となる。
    """

    def __init__(self, total: int, quiet: bool = False, every_n: int = 10, enabled: bool = True):
        self.total = max(0, int(total))
        self.quiet = quiet
        self.enabled = enabled
        self.every_n = max(1, every_n)
        self.t0 = time.perf_counter()
        self.completed = 0
        self._tty = _is_tty()
        self._step_stats = _StepStats()
        self._eval_eta: float | None = None    # 評価1件の残り (step境界で更新)
        self._eval_t_step: float = time.perf_counter()
        self._search_eta: float | None = None  # 探索全体の残り (試行ごとに更新)
        self._live_note = ""
        self._stop = threading.Event()
        if self.enabled and not self.quiet and self._tty:
            self._ticker = threading.Thread(target=self._tick_loop, daemon=True)
            self._ticker.start()
        else:
            self._ticker = None

    # ---- v7: 再開時の実績シード ----
    def seed_rates(self, seconds: list[float]) -> None:
        """チェックポイント履歴の試行秒数を統計に先投入する (条件⑤)。

        再開直後 (まだ1試行も完了していない) でも ETA が過去実績から
        出るため、「残り 0秒」や無根拠な楽観値を表示しない。
        """
        for s in seconds[-8:]:
            if s and s > 0:
                self._step_stats.observe(float(s))
        rate = self._step_stats.rate()
        if rate is not None and self.total > 0:
            self._search_eta = rate * self.total

    # ---- 毎秒カウントダウン描画 ----
    def _tick_loop(self) -> None:
        while not self._stop.wait(0.5):
            try:
                if self._live_note and _is_live_owner(self):   # [v5] 所有権チェック
                    self._render_live()
            except Exception:
                pass

    def _render_live(self) -> None:
        if not self._live_note:
            return
        parts = [self._live_note]
        if self._eval_eta is not None:
            countdown = max(0.0, self._eval_eta - (time.perf_counter() - self._eval_t_step))
            parts.append(f"({tr('ui_eval_remaining')} {fmt_duration(countdown)})")
        else:
            parts.append(f"({tr('ui_eval_remaining')} --)")
        if self._search_eta is not None:
            parts.append(f"| {tr('ui_search_remaining')} {fmt_duration(self._search_eta)}")
        LINES.show("    " + "  ".join(parts))

    # ---- 評価中のライブ行 ----
    def live(self, text: str) -> None:
        if not self.enabled or self.quiet or not self._tty:
            return
        _claim_live(self)   # [v5] 評価中はこの行がライブ行の所有者
        self._live_note = text
        self._render_live()

    def live_step(self, text: str, eta: float | None = None) -> None:
        """評価バッチなどのstep境界でのlive行更新。

        [v7] eta (この評価の残り秒の推定値) を渡すと毎秒カウントダウン表示
        される。Noneなら "--" (推定不能) を表示する (条件⑤)。
        """
        if not self.enabled or self.quiet or not self._tty:
            return
        _claim_live(self)   # [v5]
        now = time.perf_counter()
        self._eval_t_step = now
        self._eval_eta = eta
        self._live_note = text
        self._render_live()

    def clear_live(self) -> None:
        """評価中ライブ行の表示を止める (試行統計行の出力前など)。

        [v8条件②] v7はここで「\r + 空白×live_len + 改行」を書いてライブ行
        を消去していたため、生ストリームのログキャプチャに空白のみの行が
        1試行ごとに残っていた。v8は何も書かずにライブ行の描画だけを止め、
        直後に print_line される確定行 (試行統計行) が \r でライブ行の
        物理行を直接上書きする。空白行は一切出なくなった。
        """
        if not self.enabled or self.quiet or not self._tty or not self._live_note:
            return
        self._live_note = ""
        self._eval_eta = None

    # ---- 試行完了行 ----
    def trial(self, attempt: int, eval_seconds: float, refusal: float, kl: float, J: float,
              best_J: float, best_attempt: int, cached: bool = False,
              screened: bool = False, refusal_first: float | None = None) -> None:
        """1試行の統計行を出力する (v7: 各欄を固定幅で整列・条件⑥)。

        [v9①] 初頭トークン比 (first) を並記する — 質量加重多位置比には
        文法床 (~0.3-0.4) があり、床付近の値だけでは実際の拒否率の
        判別ができないため (床のない first が実態をよく追跡する)。
        """
        if not self.enabled:
            return
        self.completed = attempt
        self.clear_live()
        # 試行時間の統計 → 探索全体のETA
        if not cached and eval_seconds > 0:
            self._step_stats.observe(eval_seconds)
        elapsed = time.perf_counter() - self.t0
        if self.quiet:
            if attempt % self.every_n != 0 and attempt != self.total:
                return
        remaining = self.total - attempt
        rate = self._step_stats.rate()
        if rate is not None and remaining > 0:
            self._search_eta = rate * remaining
        elif attempt > 0 and remaining > 0:
            self._search_eta = (elapsed / attempt) * remaining
        elif remaining > 0:
            self._search_eta = None
        else:
            self._search_eta = 0.0
        # 桁揃え: 試行番号は "999/999" 幅、所要は "9999.9s" 幅
        w = max(3, len(str(self.total)))
        if self.total and attempt <= self.total:
            trial_txt = f"Trial {attempt:>{w}}/{self.total}"
        else:
            # 初期化フェーズ等で規定試行数を超過した評価の表示
            trial_txt = f"Trial {attempt:>{w}}"
        parts = [pad_display(trial_txt, 18),
                 f"{eval_seconds:7.1f}s" + (" (cached)" if cached else ""),
                 f"refusal {refusal:.3f}"
                 + (f" first {refusal_first:.3f}"
                    if refusal_first is not None and refusal_first == refusal_first else "")
                 + "  "
                 + (f"KL {kl:.4f}" if kl == kl else "KL -")
                 + (f"  J {J:.4f}" if J == J and J < float("inf") else "  J -")]
        best_txt = ("best J " + (f"{best_J:.4f} @ #{best_attempt}" if best_J == best_J else "n/a"))
        parts.append(best_txt)
        if screened:
            parts.append(tr("ui_screened"))
        if not self.quiet or attempt % self.every_n == 0 or attempt == self.total:
            eta_txt = fmt_duration(self._search_eta) if self._search_eta else "--"
            parts.append(f"elapsed {fmt_duration(elapsed)}  ETA {eta_txt}")
        print_line("  " + " | ".join(parts))

    def start(self) -> None:
        """(再)実行開始時: ティッカーを起動し直す (finishで止めた後の再利用)。

        run()が再度呼ばれたとき (対話モードの追加試行など) 、1秒カウントダウン
        表示が復活するようにする。ティッカー生存時は何もしない。
        """
        if not self.enabled or self.quiet or not self._tty:
            return
        if self._ticker is not None and self._ticker.is_alive():
            return
        self._stop = threading.Event()
        self._ticker = threading.Thread(target=self._tick_loop, daemon=True)
        self._ticker.start()

    def finish(self, text: str = "") -> None:
        self._stop.set()
        if self._ticker is not None:
            self._ticker.join(timeout=1.0)
        self.clear_live()
        if text:
            print_line(text)
