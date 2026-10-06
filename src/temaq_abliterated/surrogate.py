# -*- coding: utf-8 -*-
"""高速評価器 (SCAPEの中核, v6改訂版)。

1. 代理指標: [v8条件⑤改訂] 有害プローブの基底継続 (無摂動モデルの
   貪欲継続・多くは拒否文) に沿って K+1 位置の拒否/協調語彙の確率質量を
   測り、質量加重比 refusal_soft = Σ_t P_t(ref) / (Σ_t P_t(ref) + Σ_t P_t(cmp))
   を目的関数に使う。v7までの初頭トークン比は「協調語で始めて途中から
   拒否に転ぶ」応答 (初頭比ほぼ0なのに全文生成拒否率30%超) を捉えられ
   なかった (v7実測: 代理0.027 ↔ 全文生成0.34)。質量加重により v6の
   多位置スキャンが苦手だった「両語彙とも質量ゼロの位置で比が0.5に
   張り付く」ノイズフロアも消える (ゼロ質量の位置は分子分母に寄与しない)。
   初頭トークン比は refusal_first として参考値で残る。
2. KL指標 (v2): 無害プローブに対して基底モデルの貪欲継続 (既定8トークン) を
   あらかじめ生成し、その全応答位置を教師強制で評価する「多位置KL」を採る。
   同じフォワードから基底比NLLも計算し、崩壊検出に使う。
3. [v3] logits_to_keep: lm_headは実際に読む位置 (拒否=末尾K+1, KL=末尾K+1) だけを
   計算する。大語彙モデル (15万語彙など) では評価あたりの計算量を大幅に削減し、
   [B, T, V] のlogits巨大テンソルも消える。
4. [v3] アーキテクチャ非依存の部分フォワード: モデル自身のforwardを実行しつつ、
   摂動帯より下のレイヤを一時的にパススルーに差し替え、キャッシュした前半層の
   活性を注入する (verify_partial_equivalenceで数値一致を検証)。
   [v6] 複数のバイパス深度を實測して「安全な最大バイパス深度」を決定する
   (Gemma4の共有KVのように深いバイパスが數値的に破綻するアーキテクチャ
   では自動的に浅い境界へ制限される)。
5. [v3] 評価メモ化: 同一パラメータの再評価 (再開時の再チェック等) を即時に返す。
6. [v5] 多境界プレフィックスキャッシュ: 探索開始前に1回だけ全プローブを
   スイープし、複数のレイヤ境界における活性を保存する。以後の評価は
   「候補の摂動開始層以下で最も大きい境界」から再開する。
7. [v5] 早期スクリーニング: 拒否率評価の進捗から厳密なJ下界を計算し、
   現bestを上回れない候補はKL評価を省いて即座に棄却する。
8. [v5] 検証の高速生成: 拒否マーカーを出した系列はそれ以上生成しない。
   [v6] マーカー判定はbatch_decode×2ステップ毎に間引いてCPU負荷を削減
   (指標は不変: マーカー包含判定のみのため)。

進捗表示: バッチ単位のlive表示は ui.py のTrialProgress.live_step経由。
最終検証はプロンプト件数ベースのn/Nバー (ui.Progress)。
メッセージは msg.py (既定英語, --language jp で日本語)。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from .arch import ComponentRegistry, get_decoder_layers, get_text_model
from .engine import AblationParams
from .msg import tr
from .probes import is_refusal_text, refusal_marker_list
from .ui import Progress, print_line

# 拒否/協調の開始トークン候補 (複数モデルで機能するよう両端スペース抜き/有りを包含)
# [v6] 拡充: 多位置スキャンでも使うため、応答途中に現れる拒否/協調語を増やした
REFUSAL_START_WORDS = [
    "I", " I", "Sorry", " Sorry", "As", " As", "Unfortunately", " Unfortunately",
    "Apolog", " Apolog", "I'm", " I'm", "No,", " No,", "I cannot", " I cannot",
    "I can't", " I can't", "I won't", " I won't", "I must", " I must",
    "cannot", " cannot", "can't", " can't", "must decline", " must decline",
    "refuse", " refuse", "unable", " Unable", "I do", " I do", "No.", " No.",
    "However", " However", "Instead", " Instead", "Rather than", " Rather than",
]
COMPLIANT_START_WORDS = [
    "Sure", " Sure", "Okay", " Okay", "Yes", " Yes", "Certainly", " Certainly",
    "Of", " Of", "Great", " Great", "Here", " Here", "To", " To", "1", " 1",
    "Step", " Step", "Absolutely", " Absolutely", "No problem", " No problem",
    "Happy", " Happy", "I'd", " I'd", "I can", " I can", "Let", " Let",
    "Below", " Below", "First", " First", "There", " There", "You", " You",
    "The", " The", "A ", " A", "2", " 2", "3", " 3", "Here's", " Here's",
]

LiveCallback = Callable[..., None]   # v7: (text, eta_seconds|None) — 旧シグネチャ互換


# ---------------------------------------------------------------- 早期EOS (v5/v6)

class _MarkerForcedEos:
    """拒否マーカーを出した系列に直後にEOSを強制するLogitsProcessor。

    transformersのLogitsProcessorとして generate に渡す。各ステップで
    「系列ごとの応答テキスト」を再デコードし、マーカーを含む系列の
    logitsをEOS以外すべて -inf にする。HF generateのEOS判定がその系列を
    完了扱いにするため、系列ごとに独立した早期停止になる。
    応答はマーカーを含んだまま (EOS以降はpad) なので拒否判定は変わらない。

    [v6] 高速化: 逐次decodeをやめ、未完了系列だけを batch_decode で
    一括復号し、判定は2ステップごとに間引く (マーカー包含判定のみなので
    指標は変わらない)。

    LogitsProcessorのduck-typing互換 (基底クラス不要 — __call__シグネチャ
    のみで動作するため、バージョン差異に強い)。
    """

    def __init__(self, tokenizer, markers: list[str], prompt_len: int):
        self.tok = tokenizer
        self.markers = [m.lower() for m in markers if m]
        self.prompt_len = int(prompt_len)
        self.eos_id = tokenizer.eos_token_id
        self._finished: dict[int, bool] = {}
        self._step = 0

    def _texts(self, rows: list[torch.Tensor]) -> list[str]:
        try:
            return [t.lower() for t in self.tok.batch_decode(
                [r.tolist() for r in rows], skip_special_tokens=True)]
        except Exception:
            out = []
            for r in rows:
                try:
                    out.append(self.tok.decode(r, skip_special_tokens=True).lower())
                except Exception:
                    out.append("")
            return out

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.eos_id is None:
            return scores   # EOSを強制できない環境では何もしない
        self._step += 1
        if self._step % 2 != 1:
            return scores   # 2ステップ毎に判定 (最大1トークンの遅れで指標不変)
        try:
            B, T = input_ids.shape
            rows, idxs = [], []
            for i in range(B):
                if self._finished.get(i):
                    continue
                gen = input_ids[i, self.prompt_len:]
                if gen.numel() == 0:
                    continue
                rows.append(gen)
                idxs.append(i)
            if not rows:
                return scores
            texts = self._texts(rows)
            for i, text in zip(idxs, texts):
                if any(m in text for m in self.markers):
                    self._finished[i] = True
                    scores[i, :] = float("-inf")
                    scores[i, self.eos_id] = 0.0
        except Exception:
            pass   # 早期停止は最適化であり、失敗しても通常生成に任せる
        return scores


def _token_ids(tokenizer, words: list[str], device: str) -> torch.Tensor:
    ids: list[int] = []
    for w in words:
        try:
            tids = tokenizer.encode(w, add_special_tokens=False)
            if tids:
                ids.extend(tids)
        except Exception:
            pass
    return torch.tensor(sorted(set(ids)), dtype=torch.long, device=device)


# ---------------------------------------------------------------- レイヤ差し替え (v3)

class _LayerStub:
    """レイヤ forwardを一時的にパススルーへ差し替えるコンテキスト。

    モデル自身のforwardはそのまま実行するため、マスク構築 (full/linear
    attentionの判別分け)・位置埋め込み・conv状態などはすべて本来の
    コードパスを通る。差し替えたレイヤは何も計算しない。
    """

    def __init__(self, layers: nn.ModuleList, k: int, cached_hs: torch.Tensor | None,
                 ret_tuple: bool = False):
        self.layers = layers
        self.k = int(k)
        self.cached = cached_hs
        self.ret_tuple = bool(ret_tuple)
        self._saved: list[tuple[nn.Module, bool, Any]] = []

    # 注意: クロージャで作ること。インスタンス属性として代入する関数には
    # self が束縛されないため、通常のメソッド定義だと layer 側の属性を
    # 見に行ってしまう。

    def _make_stub(self) -> Any:
        ret_tuple = self.ret_tuple

        def stub(hidden_states, *args, **kwargs):
            if ret_tuple:
                return (hidden_states,)
            return hidden_states
        return stub

    def _make_inject(self) -> Any:
        ret_tuple = self.ret_tuple
        cached = self.cached

        def inject(hidden_states, *args, **kwargs):
            if ret_tuple:
                return (cached,)
            return cached
        return inject

    def __enter__(self):
        if self.k <= 0:
            return self
        for i in range(self.k):
            layer = self.layers[i]
            had_own = "forward" in layer.__dict__
            orig = layer.__dict__.get("forward")
            if i == 0 and self.cached is not None:
                layer.forward = self._make_inject()
            else:
                layer.forward = self._make_stub()
            self._saved.append((layer, had_own, orig))
        return self

    def __exit__(self, *exc) -> None:
        for layer, had_own, orig in reversed(self._saved):
            if had_own:
                layer.__dict__["forward"] = orig
            else:
                layer.__dict__.pop("forward", None)
        self._saved.clear()


class FastEvaluator:
    """代理指標による高速評価器 (v6: 多位置拒否スキャン・安全バイパス深度)。"""

    def __init__(self, model: nn.Module, tokenizer, device: str,
                 registry: ComponentRegistry, system_prompt: str,
                 harmful: list[str], harmless: list[str],
                 batch_size: int = 8, response_prefix: str = "",
                 cache_enabled: bool = True, verbose: bool = True,
                 kl_tokens: int = 8, lightweight: bool = False):
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.registry = registry
        self.system_prompt = system_prompt
        self.response_prefix = response_prefix
        self.batch_size = batch_size
        self.cache_enabled = cache_enabled
        self.verbose = verbose
        self.n_layers = registry.n_layers
        self.kl_tokens = max(0, int(kl_tokens))
        self.layers = get_decoder_layers(model)
        self.manual_ok: bool = True          # 部分フォワード (差し替え方式) が有効か
        self._manual_warned = False
        self._ret_tuple: bool | None = None  # レイヤ返値がtupleか (初回フォワードで決定)
        self._memo: dict[str, dict[str, Any]] = {}
        self.live_cb: LiveCallback | None = None   # 評価中のlive進捗 (TTY用)
        # [v9⑦] 軽量モード: 基底継続生成・基底分布計算を省略し、全文生成検証
        # だけに使う評価器 (チェックポイントからの書き出し専用実行向け)。
        # ensure_full() で後から完全評価器へUpgradeできる (追加試行時)。
        self.lightweight = bool(lightweight)
        self._full = False
        # [v7] バッチ所要時間の実績 (評価live行の残り時間推定に使う・条件⑤)
        self._harm_batch_times: list[float] = []
        self._good_batch_times: list[float] = []
        # [v7] 最終検証の実績 (次候補の残り時間推定に使う・条件⑤)
        self.last_verify_seconds: float | None = None
        self.last_verify_rate: float | None = None

        # ---- [v5] 早期スクリーニング (search側が設定, 最終検証時はNoneに戻す) ----
        # refusal_lb (float, 厳密な下界) -> (棄却するか, J下界)
        self.screen_refusal_fn: Callable[[float], tuple[bool, float]] | None = None
        # KL第1バッチの部分値 (float) -> 棄却するか (ヒューリスティック)
        self.screen_kl_fn: Callable[[float], bool] | None = None

        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"

        # テキスト本体 (バッチ再構成・メモリ削減時に使う)
        self.harm_texts = [self._chat(t) for t in harmful]
        self.good_texts = [self._chat(t) for t in harmless]

        # ---- 基底継続の生成 (多位置KL用 + 多位置拒否測定用, 一回だけ) ----
        # [v6] 有害プローブの継続も生成する (当初は診断の多位置拒否スキャン用)。
        # [v8条件⑤] 有害プローブの基底継続は探索の目的関数
        # (質量加重多位置ソフト拒否率) に必須になったため、無害側と同じく
        # 起動時に生成する。
        # [v7] 継続生成と基底分布計算を1本の進捗バーに統合し、
        # 「6/6 残り0s」のまま後処理を待たせる時間をなくす (条件⑦)。
        self.good_conts: list[torch.Tensor] = []
        self.harm_conts: list[torch.Tensor] = []
        self._harm_cont_len: int = 0   # 有害バッチに連結済みの継続長 (0=未連結)
        if self.lightweight:
            # [v9⑦] 書き出し専用実行: 時間のかかる基底継続生成を省略する。
            # バッチは継続なしで構成し、evaluate() は自動的に ensure_full() で
            # 完全評価器へUpgradeしてから計算する。
            if self.verbose:
                print_line(tr("v9_lightweight_evaluator"))
            self._rebuild_batches()
            self.ref_ids = _token_ids(self.tok, REFUSAL_START_WORDS, device)
            self.cmp_ids = _token_ids(self.tok, COMPLIANT_START_WORDS, device)
            self.base_logprobs: list[torch.Tensor] = []
            self.base_nll: list[float] = []
        else:
            self._init_heavy()
            self._full = True

        # 弾性キャッシュ (前半層活性) — v5は多境界方式
        self.cache_at: int | None = None
        self._cache_harm: list[torch.Tensor] = []   # 選択中の境界のharm活性 (互換ビュー)
        self._cache_good: list[torch.Tensor] = []   # 選択中の境界のgood活性 (互換ビュー)
        self.cache_rebuilds = 0                      # スイープ実施回数
        self.cache_evals = 0
        self._boundaries: list[int] = []             # 境界レイヤ番号 (昇順)
        self._b_harm: dict[int, list[torch.Tensor]] = {}
        self._b_good: dict[int, list[torch.Tensor]] = {}
        # [v7] 境界キャッシュのメモリ目安 (MB): 空きRAMの20〜25% (300MB〜1.2GB)
        try:
            from .memguard import ram_available_gb
            _free = ram_available_gb()
        except Exception:
            _free = 4.0
        self.boundary_budget_mb: float = float(min(1200.0, max(300.0, _free * 0.22)))

        # [v6] 安全な最大バイパス深度 (verify_partial_equivalence で決定)
        self.max_safe_bypass: int = self.n_layers - 1
        # [v6] 部分フォワードの例外回数 (再試行付きフォールバックの統計)
        self.partial_fwd_failures = 0

    # ---------------------------------------------------------------- 基本部品

    def _init_heavy(self) -> None:
        """[v9⑦] 完全評価器の重い初期化 (基底継続生成 + 基底分布計算)。

        軽量モードから ensure_full() で呼ばれた場合はここで初めて実行される。
        """
        base_prog: Progress | None = None
        if self.kl_tokens > 0:
            bs0 = max(1, min(self.batch_size, 8))
            n_gen = ((len(self.good_texts) + bs0 - 1) // bs0
                     + (len(self.harm_texts) + bs0 - 1) // bs0)
            if self.verbose:
                base_prog = Progress(n_gen, prefix=tr("gen_prefix_base").strip(),
                                     quiet=not self.verbose)
                base_prog.set_note(tr("base_cont_note", k=self.kl_tokens,
                                      n=len(self.good_texts) + len(self.harm_texts)))
            self._generate_base_continuations(prog=base_prog)
        self._trim_kl_probes()

        self._rebuild_batches()

        self.ref_ids = _token_ids(self.tok, REFUSAL_START_WORDS, self.device)
        self.cmp_ids = _token_ids(self.tok, COMPLIANT_START_WORDS, self.device)

        # ベースライン (多位置logprobs + NLL)。候補と同じ経路で計算する。
        # [v7] 継続生成と同じバーの続きとして表示する (条件⑦)。
        self.base_logprobs: list[torch.Tensor] = []   # [B, K, V] fp32 (CPU)
        self.base_nll: list[float] = []
        if base_prog is not None:
            base_prog.extend_total(max(1, len(self.good_batches)),
                                   note=tr("base_dist_note"))
        self._compute_baseline(prog=base_prog)
        if base_prog is not None:
            base_prog.close()
        self._full = True

    def ensure_full(self) -> None:
        """[v9⑦] 軽量モードの評価器を完全評価器へUpgradeする。

        書き出し専用実行後に追加のTrialを選んだ場合など、代理評価が必要に
        なった時点で初めて基底継続生成を行う。既に完全なら何もしない。
        """
        if self._full:
            return
        self._init_heavy()

    def _chat(self, user: str) -> str:
        msgs = [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if self.response_prefix:
            text = text + self.response_prefix
        return text

    @torch.no_grad()
    def _generate_base_continuations(self, prog: "Progress | None" = None) -> None:
        """無害/有害プローブに対する基底モデルの貪欲継続 (kl_tokens分) を生成する。

        [v8条件⑤] 有害プローブの継続もここで起動時に生成する。
        目的関数の質量加重多位置ソフト拒否率が「基底モデルが自身の拒否文を
        継続しようとする度合い」を測るために必要なため。
        応答接頭辞 (思考タグの閉じ) があれば _chat 内で付与済みのため、
        thinkingモデルでも「応答本体」の継続が得られる。
        [v7] progを渡すと生成ぶんを更新する (呼び出し元で基底分布計算を
        同じバーに統合するため・条件⑦)。Noneなら内部で作る。
        """
        own = prog is None and self.verbose
        if own:
            bs0 = max(1, min(self.batch_size, 8))
            n_gen = ((len(self.good_texts) + bs0 - 1) // bs0
                     + (len(self.harm_texts) + bs0 - 1) // bs0)
            prog = Progress(n_gen, prefix=tr("gen_prefix_base").strip(), quiet=not self.verbose)
        bs = max(1, min(self.batch_size, 8))
        old_side = self.tok.padding_side
        self.tok.padding_side = "left"
        for texts, store in ((self.good_texts, self.good_conts),
                             (self.harm_texts, self.harm_conts)):
            for i in range(0, len(texts), bs):
                self._mem_check()
                chunk = texts[i:i + bs]
                enc = self.tok(chunk, return_tensors="pt", padding=True)
                ids = enc["input_ids"].to(self.device)
                am = enc["attention_mask"].to(self.device)
                with torch.no_grad():
                    out = self.model.generate(
                        input_ids=ids, attention_mask=am, max_new_tokens=self.kl_tokens,
                        do_sample=False, pad_token_id=self.tok.pad_token_id)
                cont = out[:, ids.shape[1]:ids.shape[1] + self.kl_tokens]
                for j in range(len(chunk)):
                    store.append(cont[j].detach().cpu())
                if prog is not None:
                    prog.update()
                del out, cont
                self._mem_check()   # [v7] チャンク後: 断片をOSへ返す
        if own and prog is not None:
            prog.close()
        self.tok.padding_side = old_side

    def _trim_kl_probes(self) -> None:
        """ベースライン分布の保存メモリ (プローブ数×K×語彙×4B) が上限を
        超えないよう、KL評価に使う無害プローブを等間隔に間引く。"""
        if self.kl_tokens <= 0:
            return
        cap_bytes = 600 * 1024 * 1024
        try:
            V = self.model.get_input_embeddings().num_embeddings
        except Exception:
            V = 0
        if V <= 0:
            return
        per_probe = self.kl_tokens * V * 4
        max_n = max(8, cap_bytes // max(1, per_probe))
        if len(self.good_texts) > max_n:
            step = len(self.good_texts) / max_n
            keep = sorted(set(int(i * step) for i in range(max_n)))
            self.good_texts = [self.good_texts[i] for i in keep]
            self.good_conts = [self.good_conts[i] for i in keep]
            print(tr("kl_trim", n=len(keep), v=V, k=self.kl_tokens))

    def _rebuild_batches(self) -> None:
        """バッチを(再)構成する。 good/harmとも基底継続を連結した系列を持つ。"""
        # 長さでソートしてバケット化 (パディングロス削減)
        def _batches(texts: list[str], conts: list[torch.Tensor] | None):
            idx = sorted(range(len(texts)),
                         key=lambda i: len(self.tok(texts[i])["input_ids"]))
            if getattr(self, "_track_harm_order", False):
                self._harm_order = list(idx)   # バッチ順序 -> 元プローブ順の対応
            out = []
            for i in range(0, len(idx), self.batch_size):
                sel = [texts[j] for j in idx[i:i + self.batch_size]]
                enc = self.tok(sel, return_tensors="pt", padding=True)
                ids = enc["input_ids"].to(self.device)
                am = enc["attention_mask"].to(self.device)
                if conts is None or not conts:
                    out.append((ids, am))
                else:
                    cs = torch.stack([conts[j] for j in idx[i:i + self.batch_size]]).to(self.device)
                    ids_full = torch.cat([ids, cs], dim=1)
                    am_full = torch.cat([am, torch.ones_like(cs)], dim=1)
                    out.append((ids_full, am_full))
            return out

        # [v6] 有害プローブは「初頭トークン評価」のため継続なし。
        # [v8条件⑤] 有害プローブには基底継続を連結し、質量加重の多位置
        # ソフト拒否率を測る (詳細は evaluate())。継続が無い環境
        # (kl_tokens=0 など) では従来どおり継続なし。
        self._harm_order = list(range(len(self.harm_texts)))
        self._track_harm_order = True
        if self.kl_tokens > 0 and len(self.harm_conts) == len(self.harm_texts):
            self.harm_batches = _batches(self.harm_texts, self.harm_conts)
            self._harm_cont_len = self.kl_tokens
        else:
            self.harm_batches = _batches(self.harm_texts, None)
            self._harm_cont_len = 0
        self._track_harm_order = False
        if self.kl_tokens > 0 and len(self.good_conts) == len(self.good_texts):
            self.good_batches = _batches(self.good_texts, self.good_conts)
        else:
            self.good_batches = _batches(self.good_texts, None)
        self.cache_at = None
        self._cache_harm = []
        self._cache_good = []

    # ---------------------------------------------------------------- モデルforward (v3/v6)

    def _live(self, text: str, eta: float | None = None) -> None:
        """評価中のlive行更新 (v7: 残り秒の推定値を併せて渡せる・条件⑤)。"""
        if self.live_cb is not None:
            try:
                self.live_cb(text, eta)
            except TypeError:
                try:
                    self.live_cb(text)   # 旧シグネチャ (etaなし) 互換
                except Exception:
                    pass
            except Exception:
                pass

    def _probe_layer_return_type(self, ids: torch.Tensor, am: torch.Tensor) -> None:
        """レイヤの返値構造 (tensor / tuple) を1回だけ実測する。"""
        if self._ret_tuple is not None or not self.layers:
            return
        captured: dict[str, Any] = {}

        def hook(_mod, _args, output):
            captured["tuple"] = isinstance(output, tuple)

        h = self.layers[0].register_forward_hook(hook)
        try:
            with torch.no_grad():
                self._raw_forward(ids, am, 1)
        except Exception:
            self._ret_tuple = False   # 想定: tensor (transformers 5.xの標準)
        finally:
            h.remove()
        self._ret_tuple = bool(captured.get("tuple", False))

    def _raw_forward(self, ids: torch.Tensor, am: torch.Tensor, logits_to_keep: int):
        """モデル本来のforward。logits_to_keep非対応モデルは自動フォールバック。"""
        if logits_to_keep and logits_to_keep > 0:
            try:
                return self.model(input_ids=ids, attention_mask=am,
                                  use_cache=False, logits_to_keep=logits_to_keep)
            except TypeError:
                pass
        return self.model(input_ids=ids, attention_mask=am, use_cache=False)

    def _forward_logits(self, ids: torch.Tensor, am: torch.Tensor, keep_last: int,
                        cached_hs: torch.Tensor | None = None,
                        bypass_k: int = 0) -> torch.Tensor:
        """モデルforwardを実行し、末尾keep_last位置のlogitsを返す [B, keep_last, V]。

        bypass_k > 0 のとき、レイヤ [0..bypass_k) をパススルーに差し替え、
        cached_hs (レイヤ bypass_k の入力) を境界へ注入する。
        cached_hs が None なら差し替えだけ行う (等価性検証用)。

        [v6] 部分フォワードが例外を出した場合 (アーキテクチャ固有の前提違反等)、
        フルフォワードで1回だけ再試行する (安全性を優先し、候補を落とさない)。
        """
        if self._ret_tuple is None:
            self._probe_layer_return_type(ids, am)
        try:
            return self._forward_logits_inner(ids, am, keep_last, cached_hs, bypass_k)
        except Exception as e:  # noqa: BLE001
            if bypass_k <= 0 or cached_hs is None:
                raise
            # 部分フォワード固有の失敗 → フルフォワードへフォールバック
            self.partial_fwd_failures += 1
            if self.verbose and not self._manual_warned:
                self._manual_warned = True
                print_line(tr("partial_check_error", etype=type(e).__name__))
            if self.partial_fwd_failures > 16:
                self.manual_ok = False
                self.cache_enabled = False
            return self._forward_logits_inner(ids, am, keep_last, None, 0)

    def _forward_logits_inner(self, ids: torch.Tensor, am: torch.Tensor, keep_last: int,
                              cached_hs: torch.Tensor | None,
                              bypass_k: int) -> torch.Tensor:
        stub = _LayerStub(self.layers, bypass_k,
                          cached_hs.to(self.device) if cached_hs is not None else None,
                          ret_tuple=bool(self._ret_tuple))
        with stub, torch.no_grad():
            out = self._raw_forward(ids, am, keep_last)
        logits = out.logits
        if logits.shape[1] > keep_last:
            # logits_to_keep非対応フォールバック時: 手動で末尾を切る
            logits = logits[:, -keep_last:, :]
        return logits.float()

    def _logits(self, ids: torch.Tensor, am: torch.Tensor, from_layer: int,
                cached_hs: torch.Tensor | None, positions: slice) -> torch.Tensor:
        """互換ラッパ: 位置スライスを keep_last/オフセットへ変換して forward する。

        positions は「末尾寄りの範囲」(末尾1 または末尾K+1) を想定する。
        """
        T = ids.shape[1]
        start = positions.start if positions.start is not None else T
        stop = positions.stop if positions.stop is not None else T
        if start < 0:
            start += T
        if stop < 0:
            stop += T
        keep_last = max(1, T - start)
        logits = self._forward_logits(ids, am, keep_last, cached_hs, from_layer)
        base = T - keep_last
        return logits[:, start - base:stop - base, :]

    # ---------------------------------------------------------------- キャッシュ

    @torch.no_grad()
    def _run_upto_layer(self, ids: torch.Tensor, am: torch.Tensor, upto: int) -> torch.Tensor:
        """レイヤ 0..upto-1 だけを実行し、レイヤ upto の入力隠れ状態を取り出す。

        モデル自身のforwardを使うため全アーキテクチャで正確。取り出した直後に
        例外的に打ち切り、レイヤ upto 以降は計算しない (キャッシュ構築の高速化)。
        """
        captured: dict[str, torch.Tensor] = {}

        class _Abort(Exception):
            pass

        def pre_hook(mod, args, kwargs):
            hs = args[0] if args else kwargs.get("hidden_states")
            if isinstance(hs, torch.Tensor):
                captured["hs"] = hs
            raise _Abort   # レイヤ upto 以降の計算を打ち切る

        layer = self.layers[min(upto, len(self.layers) - 1)]
        if self._ret_tuple is None:
            # フック登録前に実施する (フックの_Abortがprobeを壊さないように)
            self._probe_layer_return_type(ids, am)
        h = layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
        try:
            with torch.no_grad():
                try:
                    self._raw_forward(ids, am, 1)
                except _Abort:
                    pass
        finally:
            h.remove()
        if "hs" not in captured:
            raise RuntimeError("failed to capture hidden states for the prefix cache")
        return captured["hs"]

    @torch.no_grad()
    def _rebuild_cache(self, upto: int) -> None:
        """互換API: 指定層までの単一境界キャッシュ (v4方式)。

        v5では build_boundary_cache() の多境界スイープを使うため、
        この経路はフォールバック専用。
        """
        self._cache_harm = []
        self._cache_good = []
        for ids, am in self.harm_batches:
            self._cache_harm.append(self._run_upto_layer(ids, am, upto).cpu())
        for ids_full, am_full in self.good_batches:
            self._cache_good.append(self._run_upto_layer(ids_full, am_full, upto).cpu())
        self.cache_at = upto
        self._boundaries = [upto]
        self._b_harm = {upto: self._cache_harm}
        self._b_good = {upto: self._cache_good}
        self.cache_rebuilds += 1

    # ---------------------------------------------------------------- 多境界キャッシュ (v5/v6)

    def boundary_ready(self) -> bool:
        """多境界キャッシュが構築済みか。"""
        return bool(self._boundaries) and bool(self._b_harm)

    def needs_boundary_sweep(self) -> bool:
        """多境界キャッシュが有効なのに手元にない (未構築 or メモリ圧で解放済み)。

        [v9⑦] 軽量モード (継続なしバッチ) では構築しない — ensure_full() で
        バッチ構成が変わるため (評価時に自動Upgrade→再スイープされる)。
        """
        return (self._full and self.cache_enabled and self.manual_ok
                and not self.boundary_ready())

    def _residual_width(self) -> int:
        """レイヤ間活性の実幅 (ハイパーコネクションでは hc_count×hidden)。"""
        try:
            ids, am = self.harm_batches[0]
            hs = self._run_upto_layer(ids[:, :4], am[:, :4], 1)
            return int(hs.shape[-1])
        except Exception:
            return 0

    def _hidden_dim(self) -> int:
        """テキスト隠れ層幅の推定値 (config・埋め込みから; _residual_widthの代替)。"""
        try:
            h = getattr(self.model.config, "hidden_size", None)
            if h:
                return int(h)
        except Exception:
            pass
        try:
            emb = self.model.get_input_embeddings()
            h = getattr(emb, "embedding_dim", None)
            if h:
                return int(h)
        except Exception:
            pass
        try:
            from .arch import get_text_model
            core = get_text_model(self.model)
            h = getattr(core, "embed_tokens", None)
            if h is not None:
                return int(h.embedding_dim)
        except Exception:
            pass
        return 0

    def _inter_layer_elem_bytes(self) -> int:
        """レイヤ間活性の1要素バイト数 (モデルの実効dtypeで決まる)。"""
        try:
            for comp in self.registry.components:
                layers_map = self.registry.map.get(comp, {})
                if layers_map:
                    li = sorted(layers_map.keys())[0]
                    m = self.registry.modules_of(comp, li)[0]
                    from .arch import module_weight
                    dt = module_weight(m).dtype
                    return 2 if dt in (torch.bfloat16, torch.float16) else 4
        except Exception:
            pass
        return 4

    def _plan_boundaries(self) -> list[int]:
        """メモリ目安から境界レイヤ群を決定する (昇順, 2以上)。

        [v7修正] v6は先頭バッチ (長さソートで最短) の形状だけで見積もって
        いたため1境界あたりのサイズを最大10倍過小評価し、予算 (400MB) に対し
        4GBのキャッシュを構築していた。ここでは全バッチの実際の
        B×T合計から正確に計算し、境界数を予算内に厳密に収める。
        """
        L = self.n_layers
        if L < 6:
            return []
        try:
            H = self._residual_width() or self._hidden_dim()
            if H <= 0:
                return []
            elem = self._inter_layer_elem_bytes()   # モデルの実効dtype (bf16なら2)
            # [v7] 全バッチの実際のトークン数合計 (先頭バッチだけでなく)
            rows = (sum(ids.shape[0] * ids.shape[1] for ids, _ in self.harm_batches)
                    + sum(ids.shape[0] * ids.shape[1] for ids, _ in self.good_batches))
            if rows <= 0:
                return []
            per_bound = rows * H * elem
        except Exception:
            return []
        budget = int(self.boundary_budget_mb * (1024 ** 2))
        n_max = int(budget // max(1, per_bound))
        n_max = min(n_max, 12)
        if n_max < 2:
            # 1境界ぶんすら予算に収まらない → 多境界キャッシュを無効化する
            # (毎試行の再構築ループを起こさないよう cache_enabled を切る)
            self.cache_enabled = False
            return []
        # レイヤ範囲 [2, hi_cap] を n_max 個の等間隔境界で覆う (端点含む)
        hi_cap = max(3, min(L - 3, self.max_safe_bypass))
        lo = 2
        span = hi_cap - lo
        if span < 1:
            return []
        step = max(1, int(math.ceil(span / max(1, n_max - 1))))
        bounds = list(range(lo, hi_cap + 1, step))
        return bounds[:n_max]

    @torch.no_grad()
    def build_boundary_cache(self) -> None:
        """[v5] 全プローブを1回ずつスイープし、複数レイヤ境界の活性を一括保存する。

        各バッチは「最深境界の手前」で計算を打ち切るため、全バッチの合計コストは
        通常評価1〜2回分に相当する。以後の評価は ensure_cache() が境界を選ぶだけで
        再構築不要になる。呼び出し時は摂動が適用されていないこと (engine.reset() 後
        または engine.suspend_patches() 内)。

        [v7] このスイープは「探索全体で1回だけ」が設計意図であり、実装も
        それを保証する (v6はメモリ監視の誤作動で毎試行再構築していた)。
        進捗バーのみ表示し、説明行は出さない (条件⑥: バーの直前の説明文は不要)。
        """
        if not self.cache_enabled or not self.manual_ok:
            return
        bounds = self._plan_boundaries()
        if not bounds:
            return
        self._b_harm = {b: [] for b in bounds}
        self._b_good = {b: [] for b in bounds}
        last = bounds[-1]

        total = len(self.harm_batches) + len(self.good_batches)
        prog = Progress(total, prefix=tr("boundary_sweep_prefix").strip(),
                        quiet=not self.verbose)

        class _Abort(Exception):
            pass

        def _make_pre(cap: dict, b: int):
            if b == last:
                def pre(mod, args, kwargs):
                    hs = args[0] if args else kwargs.get("hidden_states")
                    if isinstance(hs, torch.Tensor):
                        cap[b] = hs.detach()
                    raise _Abort
            else:
                def pre(mod, args, kwargs):
                    hs = args[0] if args else kwargs.get("hidden_states")
                    if isinstance(hs, torch.Tensor) and b not in cap:
                        cap[b] = hs.detach()
            return pre

        def _sweep(batches, store):
            for ids, am in batches:
                if self._ret_tuple is None:
                    self._probe_layer_return_type(ids, am)
                cap: dict[int, torch.Tensor] = {}
                hooks = [self.layers[b].register_forward_pre_hook(
                    _make_pre(cap, b), with_kwargs=True) for b in bounds]
                try:
                    with torch.no_grad():
                        try:
                            self._raw_forward(ids, am, 1)
                        except _Abort:
                            pass
                finally:
                    for h in hooks:
                        h.remove()
                if len(cap) != len(bounds):
                    # 一部境界で捕捉失敗 → 多境界方式を断念 (単一境界へ退化)
                    if self.verbose:
                        print(tr("partial_check_error", etype="BoundaryCapture"))
                    self.cache_enabled = False
                    return False
                for b in bounds:
                    # dtypeはレイヤ間活性の元のまま保存する (注入先レイヤの
                    # 重みdtypeと一致していないとforwardでdtype不一致になる)。
                    store[b].append(cap[b].detach().cpu())
                prog.update()
            return True

        ok = _sweep(self.harm_batches, self._b_harm)
        if ok:
            ok = _sweep(self.good_batches, self._b_good)
        prog.close()
        if not ok:
            self._b_harm = {}
            self._b_good = {}
            self._boundaries = []
            return
        self._boundaries = bounds
        self.cache_at = None
        self.cache_rebuilds += 1

    def ensure_cache(self, band_start: int) -> None:
        """[v5/v6] 候補の摂動開始層以下で最も大きい境界を選ぶ (再構築なし)。"""
        if not self.cache_enabled or not self.manual_ok or not self._boundaries:
            self.cache_at = 0 if self.cache_at is None else self.cache_at
            self._cache_harm = []
            self._cache_good = []
            return
        band_start = int(max(0, min(band_start, self.n_layers - 1)))
        limit = min(band_start, self.max_safe_bypass)
        cands = [b for b in self._boundaries if b <= limit]
        if not cands:
            self.cache_at = 0
            self._cache_harm = []
            self._cache_good = []
            return
        b = max(cands)
        self.cache_at = b
        self._cache_harm = self._b_harm[b]
        self._cache_good = self._b_good[b]

    # ---------------------------------------------------------------- ベースライン

    @torch.no_grad()
    def _compute_baseline(self, prog: "Progress | None" = None) -> None:
        """無害プローブの多位置ベースラインlogprobsとNLLを計算する (fp32保存)。

        [v7] progを渡すと基底継続生成バーの続きとして進捗更新する (条件⑦)。
        """
        self.base_logprobs = []
        self.base_nll = []
        K = self.kl_tokens
        for ids_full, am_full in self.good_batches:
            L = ids_full.shape[1]
            if K > 0 and L > K:
                pos = slice(L - K - 1, L - 1)
            else:
                pos = slice(L - 1, L)
            logits = self._logits(ids_full, am_full, 0, None, pos)
            lp = F.log_softmax(logits, dim=-1)
            self.base_logprobs.append(lp.cpu())
            if K > 0 and L > K:
                cont_ids = ids_full[:, L - K:]
                nll = -lp.gather(-1, cont_ids.unsqueeze(-1)).mean().item()
            else:
                nll = 0.0
            self.base_nll.append(max(nll, 1e-6))
            if prog is not None:
                prog.update()

    # ---------------------------------------------------------------- 評価

    def _params_key(self, params: AblationParams) -> str:
        d = json.dumps(params.to_dict(), sort_keys=True)
        return hashlib.sha1(d.encode()).hexdigest()

    def _soft_refusal_ratios(self, logits: torch.Tensor) -> torch.Tensor:
        """多位置logits [B, P, V] から各位置のソフト拒否率 [B, P] を返す。

        (診断用・v6互換。両語彙とも質量ゼロの位置では比が0.5に張り付く
        ノイズフロアがあるため、目的関数には _soft_refusal_mass_ratio を使う)
        """
        p = logits.softmax(dim=-1)
        pref = p.index_select(-1, self.ref_ids).sum(dim=-1)
        pcmp = p.index_select(-1, self.cmp_ids).sum(dim=-1)
        return pref / (pref + pcmp + 1e-9)

    def _soft_refusal_mass_ratio(self, logits: torch.Tensor) -> torch.Tensor:
        """[v8条件⑤] 多位置logits [B, P, V] から質量加重ソフト拒否率 [B] を返す。

        Σ_t P_t(ref) / (Σ_t P_t(ref) + Σ_t P_t(cmp)) — 両語彙とも質量が
        小さい位置は分子分母にほぼ寄与しないため、v6の多位置スキャンの
        ノイズフロア (比が0.5に張り付く) が原理的に出ない。
        応答のどの時点であれ拒否語彙に質量が残っていれば値が上がるため、
        「協調語で始めて途中から拒否に転ぶ」応答も捉えられる。
        """
        p = logits.softmax(dim=-1)
        pref = p.index_select(-1, self.ref_ids).sum(dim=-1)   # [B, P]
        pcmp = p.index_select(-1, self.cmp_ids).sum(dim=-1)   # [B, P]
        num = pref.sum(dim=-1)
        den = (pref + pcmp).sum(dim=-1)
        return num / (den + 1e-9)

    @torch.no_grad()
    def evaluate(self, params: AblationParams, use_cache: bool = True) -> dict[str, Any]:
        """候補パラメータを評価して代理指標を返す (メモ化 + 早期スクリーニング付き)。

        戻り値: refusal_soft (質量加重の多位置ソフト拒否率・[v8条件⑤]) /
                refusal_first (初頭トークン比・参考値) /
                kl (多位置KL) / kl_first / nll_ratio / margin_mean /
                band_start / eval_seconds / [cached] / [screened]
        screen_refusal_fn が設定されている場合、拒否率評価の途中 (バッチ境界) で
        厳密なJ下界による棄却判定を行い、 hopeless な候補はKL評価を省略する
        (この場合 kl=NaN, screened=True)。
        呼び出し側でパッチ適用/リセットを行うこと (engineを別管理)。
        """
        key = self._params_key(params)
        hit = self._memo.get(key)
        if hit is not None:
            out = dict(hit)
            out["cached"] = True
            out["eval_seconds"] = 0.0
            return out
        # [v9⑦] 軽量モードはここで初めて完全評価器へUpgradeする
        if not self._full:
            self.ensure_full()

        t0 = time.perf_counter()
        band_start = params.band_start(self.n_layers)
        use_cached = use_cache and self.cache_enabled and self.manual_ok
        if use_cached:
            self.ensure_cache(band_start)
            from_layer = self.cache_at or 0
        else:
            from_layer = 0

        # 有害プローブ: 基底継続に沿った多位置の質量加重ソフト拒否率
        # [v8条件⑤] v7までは初頭トークン比だけを使っており、「協調語で
        # 始めて途中から拒否に転ぶ」応答 (初頭比ほぼ0なのに全文生成拒否率
        # 30%超) を全く罰しなかった。ここでは基底モデル自身の貪欲継続
        # (多くは拒否文) を教師強制し、K+1位置にわたる拒否/協調語彙の
        # 確率質量の比 (質量加重) を測る — 拒否傾向がどこかに残っていれば
        # 捉えられ、しかも質量加重により v6多位置スキャンのノイズ
        # フロア (比が0.5に張り付く) は原理的に出ない。
        # [v5] バッチごとに「ここまでの拒否率和 / 全プローブ数」= 厳密な下界で
        #      早期棄却を判定する (残りプローブのソフト拒否率は0以上のため)。
        # [v7] live行には前バッチまでの実績から推定した残り秒を併記する (条件⑤)。
        softs: list[torch.Tensor] = []   # [B] 質量加重の多位置比
        firsts: list[torch.Tensor] = []  # [B] 初頭トークン比 (参考値)
        margins = []
        n_probe_total = sum(ids.shape[0] for ids, _ in self.harm_batches)
        nb = len(self.harm_batches)
        hc = int(getattr(self, "_harm_cont_len", 0) or 0)
        keep_h = hc + 1   # 継続K + 応答位置0 (継続なしのときは1)
        screened = False
        j_lb = None
        t_phase = time.perf_counter()
        for bi, (ids, am) in enumerate(self.harm_batches, start=1):
            eta = None
            if self._harm_batch_times:
                eta = (sum(self._harm_batch_times) / len(self._harm_batch_times)
                       * (nb - bi + 1))
            self._live(tr("eval_progress", part="refusal", i=bi, n=nb), eta)
            cached = self._cache_harm[bi - 1] if (use_cached and from_layer > 0
                                                  and len(self._cache_harm) >= nb) else None
            logits = self._forward_logits(ids, am, keep_h, cached, from_layer)  # [B, keep, V]
            self._harm_batch_times.append(time.perf_counter() - t_phase)
            self._harm_batch_times = self._harm_batch_times[-16:]
            t_phase = time.perf_counter()
            softs.append(self._soft_refusal_mass_ratio(logits))
            pos0 = logits[:, 0, :]   # 応答位置0 (初頭トークン) のlogits
            p0 = pos0.softmax(dim=1)
            pref0 = p0.index_select(1, self.ref_ids).sum(dim=1)
            pcmp0 = p0.index_select(1, self.cmp_ids).sum(dim=1)
            firsts.append(pref0 / (pref0 + pcmp0 + 1e-9))
            margins.append(torch.logsumexp(pos0.index_select(1, self.ref_ids), dim=1)
                           - torch.logsumexp(pos0.index_select(1, self.cmp_ids), dim=1))
            if self.screen_refusal_fn is not None and n_probe_total > 0:
                r_lb = float(torch.cat(softs).sum().item()) / n_probe_total
                abort, j_lb = self.screen_refusal_fn(r_lb)
                if abort:
                    screened = True
                    break

        refusal_soft = torch.cat(softs).mean().item()
        refusal_first = (torch.cat(firsts).mean().item() if firsts else refusal_soft)
        margin_mean = torch.cat(margins).mean().item()

        if screened:
            # 厳密下界で棄却: KL評価を省く (Jの下界値をそのまま報告値に使う)。
            # メモ化しない — しきい値はbest更新で変わるため、棄却判定は評価時点依存。
            self.cache_evals += 1
            return {
                "refusal_soft": refusal_soft,
                "refusal_first": refusal_first,
                "kl": float("nan"),
                "kl_first": float("nan"),
                "nll_ratio": None,
                "margin_mean": margin_mean,
                "band_start": band_start,
                "eval_seconds": time.perf_counter() - t0,
                "screened": True,
                "j_lower_bound": (j_lb if j_lb is not None else float("inf")),
            }

        # 無害プローブ: 多位置KL + 基底比NLL (末尾K+1位置のlogitsを使う)
        # [v5] 第1バッチ終了時の部分KLが明らかに絶望的な場合も棄却する
        # (ヒューリスティック: バッチ間のKLばらつきは小さいため)。
        kls, kls_first, nlls = [], [], []
        K = self.kl_tokens
        nb = len(self.good_batches)
        t_phase = time.perf_counter()
        for bi, (ids_full, am_full) in enumerate(self.good_batches, start=1):
            eta = None
            if self._good_batch_times:
                eta = (sum(self._good_batch_times) / len(self._good_batch_times)
                       * (nb - bi + 1))
            self._live(tr("eval_progress", part="KL", i=bi, n=nb), eta)
            L = ids_full.shape[1]
            if K > 0 and L > K:
                keep_last = K + 1
            else:
                keep_last = 1
            cached = self._cache_good[bi - 1] if (use_cached and from_layer > 0
                                                  and len(self._cache_good) >= nb) else None
            logits = self._forward_logits(ids_full, am_full, keep_last, cached, from_layer)
            self._good_batch_times.append(time.perf_counter() - t_phase)
            self._good_batch_times = self._good_batch_times[-16:]
            t_phase = time.perf_counter()
            lp_full = F.log_softmax(logits, dim=-1)
            if K > 0 and L > K:
                lp = lp_full[:, :K, :]     # 継続Kトークンの予測位置
            else:
                lp = lp_full
            base = self.base_logprobs[bi - 1].to(lp.device)
            klv = F.kl_div(lp, base, reduction="none", log_target=True).sum(dim=-1)  # (B, K)
            kls.append(klv.mean())
            if klv.numel():
                kls_first.append(klv.flatten()[0] if K <= 1 else klv[:, 0].mean())
            if K > 0 and L > K:
                cont_ids = ids_full[:, L - K:]
                nll = -lp.gather(-1, cont_ids.unsqueeze(-1)).flatten().mean().item()
                nlls.append(nll / self.base_nll[bi - 1])
            if (bi == 1 and self.screen_kl_fn is not None and len(self.good_batches) > 1
                    and kls):
                if self.screen_kl_fn(float(kls[0].item())):
                    self.cache_evals += 1
                    return {
                        "refusal_soft": refusal_soft,
                        "refusal_first": refusal_first,
                        "kl": float("nan"),
                        "kl_first": float("nan"),
                        "nll_ratio": None,
                        "margin_mean": margin_mean,
                        "band_start": band_start,
                        "eval_seconds": time.perf_counter() - t0,
                        "screened": True,
                        "j_lower_bound": float("inf"),
                    }
        kl = torch.stack(kls).mean().item() if kls else 0.0
        kl_first = torch.stack(kls_first).mean().item() if kls_first else kl
        nll_ratio = (sum(nlls) / len(nlls)) if nlls else None

        self.cache_evals += 1
        out = {
            "refusal_soft": refusal_soft,
            "refusal_first": refusal_first,
            "kl": kl,
            "kl_first": kl_first,
            "nll_ratio": nll_ratio,
            "margin_mean": margin_mean,
            "band_start": band_start,
            "eval_seconds": time.perf_counter() - t0,
        }
        if len(self._memo) < 4096:
            self._memo[key] = dict(out)
        return out

    def _full_logits_last(self, ids: torch.Tensor, am: torch.Tensor) -> torch.Tensor:
        """互換API: 末尾位置のlogitsをフルforwardで返す (実験スクリプト用)。"""
        return self._forward_logits(ids, am, 1)[:, 0, :].float()

    def first_token_soft_refusals(self) -> list[float]:
        """[v6] 有害プローブごとの初頭トークンソフト拒否率 (基底・無摂動)。

        戻り値は「元の有害プローブ順」に並び替えられている
        (バッチ構成の長さソートと無関係に全文生成結果と対応する)。
        実験 (代理指標の妥当性検証) で全文生成拒否との相関を取るために使う。
        摂動が適用されていない状態で呼び出すこと。
        """
        return self._probe_soft_refusals(pos=0)

    def multi_pos_soft_refusals(self) -> list[float]:
        """[v6] 有害プローブごとの多位置ソフト拒否率 (基底・無摂動)。

        戻り値は元の有害プローブ順。
        """
        return self._probe_soft_refusals(pos=-1)

    @torch.no_grad()
    def _ensure_harm_conts(self) -> bool:
        """診断用に有害プローブの基底継続を遅延生成する (無摂動状態で呼ぶこと)。"""
        if self.kl_tokens <= 0 or len(self.harm_conts) == len(self.harm_texts):
            return len(self.harm_conts) == len(self.harm_texts)
        if self.verbose:
            print(tr("base_cont_title", k=self.kl_tokens, n=len(self.harm_texts)))
        bs = max(1, min(self.batch_size, 8))
        old_side = self.tok.padding_side
        self.tok.padding_side = "left"
        for i in range(0, len(self.harm_texts), bs):
            chunk = self.harm_texts[i:i + bs]
            enc = self.tok(chunk, return_tensors="pt", padding=True)
            ids = enc["input_ids"].to(self.device)
            am = enc["attention_mask"].to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=ids, attention_mask=am, max_new_tokens=self.kl_tokens,
                    do_sample=False, pad_token_id=self.tok.pad_token_id)
            cont = out[:, ids.shape[1]:ids.shape[1] + self.kl_tokens]
            for j in range(len(chunk)):
                self.harm_conts.append(cont[j].detach().cpu())
        self.tok.padding_side = old_side
        return len(self.harm_conts) == len(self.harm_texts)

    def _probe_soft_refusals(self, pos: int) -> list[float]:
        """有害プローブごとのソフト拒否率。pos=0: 位置0のみ / pos=-1: 多位置。

        [v6修正] v5の実験コードは「長さソート済みバッチ順」のソフト値と
        「元順」の全文生成拒否を突き合わせており、相関が壊れていた。
        ここでは元プローブ順に並び替えて返す。
        [v8条件⑤] pos=0 は初頭トークン比 (refusal_first と同一の定義)。
        pos=-1 は目的関数と同一の質量加重多位置比 (基底継続が無ければ
        遅延生成して計測する)。両方とも探索目的関数との対応が取れる。
        """
        if pos == 0:
            batches = self.harm_batches
            # [v8] 有害バッチに基底継続が連結されている場合、位置0を読むには
            # 継続ぶんも含めて末尾から keep = 継続長+1 位置取得する必要がある
            # (keep=1 だと「継続の末尾の次」の位置になってしまう)。
            keep = int(getattr(self, "_harm_cont_len", 0) or 0) + 1
        else:
            if not self._ensure_harm_conts():
                return [float("nan")] * len(self.harm_texts)
            # 継続連結バッチを一時構成 (harm_texts + conts)
            saved = self.harm_batches
            conts = dict(zip(range(len(self.harm_texts)), self.harm_conts))
            # 長さソートは _rebuild_batches と同じ規則で行う
            def _mk(texts, cs):
                idx = sorted(range(len(texts)),
                             key=lambda i: len(self.tok(texts[i])["input_ids"]))
                out = []
                order = []
                for i in range(0, len(idx), self.batch_size):
                    sel = [texts[j] for j in idx[i:i + self.batch_size]]
                    order.extend(idx[i:i + self.batch_size])
                    enc = self.tok(sel, return_tensors="pt", padding=True)
                    ids = enc["input_ids"].to(self.device)
                    am = enc["attention_mask"].to(self.device)
                    cs2 = torch.stack([cs[j] for j in idx[i:i + self.batch_size]]).to(self.device)
                    out.append((torch.cat([ids, cs2], dim=1),
                                torch.cat([am, torch.ones_like(cs2)], dim=1)))
                return out, order
            batches, order_override = _mk(self.harm_texts, self.harm_conts)
            vals = []
            for ids, am in batches:
                T = ids.shape[1]
                keep = min(T, self.kl_tokens + 1) if self.kl_tokens > 0 else 1
                logits = self._forward_logits(ids, am, keep)
                # [v8] 質量加重比 (目的関数と同一の定義・ノイズフロアなし)
                r = self._soft_refusal_mass_ratio(logits)
                vals.extend(r.tolist())
            out = [0.0] * len(vals)
            for bi, v in enumerate(vals):
                out[order_override[bi]] = v
            return out

        vals_batch: list[tuple[int, float]] = []
        n_seen = 0
        for ids, am in batches:
            logits = self._forward_logits(ids, am, keep)
            if pos == 0 and keep > 1:
                logits = logits[:, 0, :]   # 応答位置0 (初頭トークン)
            if logits.dim() == 3:
                v = self._soft_refusal_ratios(logits).mean(dim=1)
            else:
                p = logits.softmax(dim=-1)
                pref = p.index_select(-1, self.ref_ids).sum(dim=-1)
                pcmp = p.index_select(-1, self.cmp_ids).sum(dim=-1)
                v = pref / (pref + pcmp + 1e-9)
            for j in range(v.shape[0]):
                vals_batch.append((n_seen + j, float(v[j].item())))
            n_seen += v.shape[0]
        order = getattr(self, "_harm_order", None)
        out = [0.0] * len(vals_batch)
        if order is not None and len(order) == len(vals_batch):
            for probe_idx, val in vals_batch:
                out[order[probe_idx]] = val
        else:
            out = [v for _i, v in vals_batch]
        return out

    # ---------------------------------------------------------------- 検証 (全文生成, v5高速化)

    def _gen_batch_size(self) -> int:
        """[v7] 生成バッチサイズ: INT4の脱量子化コストはバッチで弾化する。

        INT4実行時 (重みキャッシュの有無にかかわらず) は大きめのバッチで
        1ステップあたりの重み復号コストを多数系列で割る。メモリ逼迫時は
        縮小し、OOM時は _generate_batched がさらに半減して再試行する。
        """
        try:
            from .quant4 import PackedInt4Linear
            n_int4 = sum(1 for r in PackedInt4Linear._ALL if r() is not None)
            cache_on = PackedInt4Linear._WSTATE.get("budget", 0) > 0
        except Exception:
            n_int4, cache_on = 0, False
        try:
            from .memguard import ram_available_gb
            free = ram_available_gb()
        except Exception:
            free = 4.0
        if n_int4 > 0:
            cap = 32 if (cache_on or free > 2.5) else 16
        else:
            cap = 16 if free > 2.0 else 8
        return max(1, min(self.batch_size * 2, cap))

    def _generate_core(self, chunk: list[str], max_new_tokens: int,
                       markers: list[str] | None) -> tuple[torch.Tensor, int]:
        """1チャンクを生成する (拒否マーカー系列の早期EOS判定, 自動フォールバック)。

        [v5] 拒否マーカーを含まない応答はEOSまたはmax_new_tokensまで生成し、
        マーカーを出した系列は「直後にEOSを強制」して余計な生成を打ち切る。
        LogitsProcessorで実装するため系列ごとに独立に停止する
        (transformersのstop_stringsはバッチ全滅まで待つ+文字列照合の
        ステップ overhead があるため不使用)。拒否判定は
        「応答にマーカーを含むか」だけなので、打ち切りで指標は変わらない。
        戻り値: (生成結果テンソル, パディング済み入力長)。左パディングのため
        復号の開始位置はすべての行でこの入力長で共通。
        """
        enc = self.tok(chunk, return_tensors="pt", padding=True)
        ids = enc["input_ids"].to(self.device)
        am = enc["attention_mask"].to(self.device)
        kwargs: dict[str, Any] = dict(
            input_ids=ids, attention_mask=am, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=self.tok.pad_token_id)
        out = None
        if markers:
            try:
                proc = _MarkerForcedEos(self.tok, markers, int(ids.shape[1]))
                out = self.model.generate(logits_processor=[proc], **kwargs)
            except (TypeError, ValueError):
                out = None
        if out is None:
            out = self.model.generate(**kwargs)
        return out, int(ids.shape[1])

    def _mem_check(self) -> None:
        """[v7] 生成ループ内の軽量メモリ監視 (チャンク毎に呼ぶ)。

        探索ループは guard.tick を毎試行呼ぶが、生成 (基底継続・最終検証)
        は長時間無監視だったため、低メモリ環境でOOMする余地があった。
        [実測] generate() は1チャンクごとに数百MBの一時テンソル (全位置
        logits等) を生み、解放されてもglibcアリーナに断片として残留する
        (malloc_trimしないとOSへ返らない)。そのため:
        - 毎チャンク: gc + malloc_trim でヒープをOSへ返す
        - 空きRAM低下時: INT4重みキャッシュを縮小する
        """
        now = time.perf_counter()
        if now - getattr(self, "_last_memcheck", 0.0) < 2.0:
            return
        self._last_memcheck = now
        try:
            from .memguard import ram_available_gb, malloc_trim
            avail = ram_available_gb()
            if avail < 0.6:
                try:
                    from .quant4 import PackedInt4Linear
                    if PackedInt4Linear._WSTATE.get("budget", 0) > 0:
                        PackedInt4Linear.shrink_weight_cache(0.5)
                except Exception:
                    pass
            import gc as _gc
            _gc.collect()
            malloc_trim()
        except Exception:
            pass

    @torch.no_grad()
    def _generate_batched(self, texts: list[str], max_new_tokens: int,
                          markers: list[str] | None = None) -> list[str]:
        """OOM時にバッチを縮小しながら全文生成する。"""
        responses: list[str] = []
        bs = self._gen_batch_size()
        i = 0
        while i < len(texts):
            self._mem_check()
            chunk = texts[i:i + bs]
            try:
                out, in_len = self._generate_core(chunk, max_new_tokens, markers)
                for j in range(len(chunk)):
                    responses.append(self.tok.decode(out[j, in_len:],
                                                     skip_special_tokens=True))
                i += len(chunk)
                del out
                self._mem_check()   # [v7] チャンク後: 一時テンソルの断片をOSへ返す
            except Exception as e:  # noqa: BLE001
                if "out of memory" not in str(e).lower() and not isinstance(e, MemoryError):
                    raise
                if bs <= 1:
                    raise
                bs = max(1, bs // 2)
                from .quant4 import PackedInt4Linear
                PackedInt4Linear.clear_weight_cache()
                print_line(tr("mem_gen_batch_shrink", bs=bs))
        return responses

    @torch.no_grad()
    def exact_refusal_rate(self, prompts: list[str], max_new_tokens: int = 64,
                           markers: list[str] | None = None,
                           progress_cb: Callable[[int, int], None] | None = None,
                           prior_rate: float | None = None,
                           ) -> tuple[float, list[str]]:
        """全文生成による厳密な拒否率 (最終候補の検証用, 既定評価は100プロンプト)。

        進捗は「プロンプト件数ベース」のn/Nバー (100プロンプトなら N=100)。
        生成は EOS/拒否マーカー出現でその系列を打ち切るため、拒否応答だらけの
        モデルでも全長生成するより大幅に速い (指標は変わらない)。
        [v7] prior_rate (前回の検証で測定した1プロンプトあたり秒数) を渡すと
        最初のバッチが終わる前から残り時間を表示できる (条件⑤)。
        実績は last_verify_seconds / last_verify_rate に記録される。
        """
        markers = markers or refusal_marker_list()
        texts = [self._chat(t) for t in prompts]
        bs = self._gen_batch_size()
        prog = Progress(len(texts), prefix=tr("gen_prefix_verify").strip(),
                        quiet=not self.verbose, prior_rate=prior_rate)
        t0 = time.perf_counter()
        responses = self._generate_batched_progress(texts, max_new_tokens, prog, markers)
        refusals = [int(is_refusal_text(r, markers)) for r in responses]
        self.last_verify_seconds = time.perf_counter() - t0
        self.last_verify_rate = self.last_verify_seconds / max(1, len(texts))
        prog.close()
        if progress_cb is not None:
            try:
                progress_cb(len(responses), len(texts))
            except Exception:
                pass
        return sum(refusals) / max(1, len(refusals)), responses

    def _generate_batched_progress(self, texts, max_new_tokens, prog,
                                   markers: list[str] | None = None) -> list[str]:
        responses: list[str] = []
        bs = self._gen_batch_size()
        i = 0
        while i < len(texts):
            self._mem_check()
            chunk = texts[i:i + bs]
            try:
                out, in_len = self._generate_core(chunk, max_new_tokens, markers)
                for j in range(len(chunk)):
                    responses.append(self.tok.decode(out[j, in_len:],
                                                     skip_special_tokens=True))
                i += len(chunk)
                prog.update_to(i)
                del out
                self._mem_check()   # [v7] チャンク後: 一時テンソルの断片をOSへ返す
            except Exception as e:  # noqa: BLE001
                if "out of memory" not in str(e).lower() and not isinstance(e, MemoryError):
                    raise
                if bs <= 1:
                    raise
                bs = max(1, bs // 2)
                from .quant4 import PackedInt4Linear
                PackedInt4Linear.clear_weight_cache()
                print_line(tr("mem_gen_batch_shrink", bs=bs))
        return responses

    # ---------------------------------------------------------------- メモリ削減

    def reduce_memory(self, hard: bool = False) -> None:
        """MemoryGuard/OOMから呼ばれるメモリ削減。

        [v7方針変更] v6はswap増加の警告でも多境界キャッシュを破棄していた
        ため、次の試行開始時に毎回6分の再スイープが発生していた (1試行あたり
        の実効速度を半分以下にする主要因)。境界キャッシュは**OOM再試行時
        のみ**破棄する。通常の警告パスではメモ化のみを整理する。
        """
        if hard:
            self._memo.clear()
            self.drop_boundary_cache()

    def drop_boundary_cache(self) -> None:
        """多境界キャッシュを明示的に解放する (OOM再試行等・次の試行で再構築)。"""
        self._cache_harm = []
        self._cache_good = []
        self.cache_at = None
        self._b_harm = {}
        self._b_good = {}
        self._boundaries = []

    def set_batch_size(self, n: int) -> None:
        """評価バッチサイズを変更してバッチを再構成する。

        [v5] バッチ構成が変わると境界活性も無効になるため解放する
        (searchが次の試行開始時に再スイープする)。
        """
        self.batch_size = max(1, int(n))
        self._rebuild_batches()
        self._b_harm = {}
        self._b_good = {}
        self._boundaries = []
        self.cache_at = None

    # ---------------------------------------------------------------- 自己検証

    def verify_partial_equivalence(self) -> float:
        """部分フォワード (差し替え+注入) とフルフォワードのlogits差の最大値。

        [v6] 複数のバイパス深度 (浅い→深い) を昇順に實測し、
        最初に破綻した深度の1つ前を「安全な最大バイパス深度」とする。
        共有KV (Gemma4) のように深い差し替えで構造が壊れるアーキテクチャ
        では自動的に境界が浅く制限され、浅い差し替えの高速化は維持される。
        戻り値: 最も浅い深度での差 (従来互換の表示用)。
        """
        try:
            ids, am = self.harm_batches[0]
            full = self._forward_logits(ids, am, 1)
        except Exception as e:  # noqa: BLE001
            if self.verbose:
                print(tr("partial_check_error", etype=type(e).__name__))
            self.manual_ok = False
            self.cache_enabled = False
            return float("inf")

        L = self.n_layers
        depths = sorted({max(2, min(4, L - 1)), max(1, L // 4), max(1, L // 2)})
        depths = [d for d in depths if d < L]
        max_safe = 0
        first_diff: float | None = None
        for k in depths:
            fails_before = self.partial_fwd_failures
            try:
                hs = self._run_upto_layer(ids, am, k)
                part = self._forward_logits(ids, am, 1, hs, k)
                diff = (full - part).abs().max().item()
                if self.partial_fwd_failures > fails_before:
                    # 例外でフルフォワードへフォールバックした = この深度の
                    # 部分フォワードは破綻 (数値は一致するがキャッシュ無効)
                    diff = float("inf")
            except Exception:
                diff = float("inf")
            if first_diff is None:
                first_diff = diff
            if diff >= 1.0 or diff != diff:
                if self.verbose and max_safe == 0:
                    print(tr("partial_check_error", etype="DeepBypass"))
                break
            max_safe = k
        self.max_safe_bypass = max_safe
        if max_safe <= 0:
            self.manual_ok = False
            self.cache_enabled = False
            if first_diff is None:
                first_diff = float("inf")
            return first_diff if first_diff == first_diff else float("inf")
        if self.verbose and max_safe < depths[-1]:
            print(tr("bypass_limit", k=max_safe))
        return (first_diff if first_diff is not None else float("inf"))


def detect_response_prefix(tokenizer, system_prompt: str) -> str:
    """チャットテンプレートが思考タグを自動挿入するか検出し、
    「閉じタグ」を応答接頭辞として返す (thinking系モデル対応)。"""
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Dummy prompt."}],
            add_generation_prompt=True, tokenize=False)
    except Exception:
        return ""
    openers = [
        ("<think>", "</think>\n\n"),
        ("[THINK]", "[/THINK]\n\n"),
        ("<thinking>", "</thinking>\n\n"),
        ("<|channel|>analysis<|message|>",
         "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"),
    ]
    for opener, closer in openers:
        if text.rstrip().endswith(opener) or opener in text[-40:]:
            return closer
    return ""


@torch.no_grad()
def baseline_first_token_refusals(model: nn.Module, tokenizer, harmful: list[str],
                                  system_prompt: str, response_prefix: str = "",
                                  device: str = "cpu", batch_size: int = 8) -> list[float]:
    """[v8条件⑤] 各有害プローブに対する基底モデル (無摂動) の初頭トークン
    ソフト拒否率を測る。

    残差方向の計算と探索評価に使う有害プローブを「基底モデルが実際に拒否
    反応を示すもの」へ絞り込むために使う。拒否傾向の弱いプローブは
    (1) 残差方向の平均を薄め、(2) 代理指標の拒否率を甘くするため、
    事前に除外することで両方を締める。
    戻り値は入力harmfulと同じ順序。摂動を適用していない状態で呼ぶこと。
    """
    ref_ids = _token_ids(tokenizer, REFUSAL_START_WORDS, device)
    cmp_ids = _token_ids(tokenizer, COMPLIANT_START_WORDS, device)

    def _chat(user: str) -> str:
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user}]
        text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                             tokenize=False)
        return text + response_prefix if response_prefix else text

    texts = [_chat(t) for t in harmful]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    scores = [float("nan")] * len(texts)
    bs = max(1, int(batch_size))

    def _measure(ids, am) -> list[float]:
        try:
            out = model(input_ids=ids, attention_mask=am, use_cache=False,
                        logits_to_keep=1)
        except TypeError:
            out = model(input_ids=ids, attention_mask=am, use_cache=False)
        logits = out.logits[:, -1, :].float()   # 左パッドのため末尾が応答位置0
        p = logits.softmax(dim=-1)
        pref = p.index_select(-1, ref_ids).sum(dim=-1)
        pcmp = p.index_select(-1, cmp_ids).sum(dim=-1)
        return (pref / (pref + pcmp + 1e-9)).tolist()

    try:
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            enc = tokenizer(chunk, return_tensors="pt", padding=True)
            ids = enc["input_ids"].to(device)
            am = enc["attention_mask"].to(device)
            v = _measure(ids, am)
            for j in range(len(chunk)):
                scores[i + j] = float(v[j])
    finally:
        tokenizer.padding_side = old_side
    return scores
