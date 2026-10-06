# -*- coding: utf-8 -*-
"""ユーザー向けメッセージの国際化 (i18n)。

既定では英語で出力し、``--language jp`` 指定時のみ日本語になる
(仕様: 出力ログの言語は --language jp としない限り英語)。

使い方:
    from .msg import set_language, tr
    set_language("jp")            # or "en"
    print(tr("scape_start", trials=80))
"""

from __future__ import annotations

LANGUAGES = ("en", "jp")

# 既定言語 (英語)
_lang: str = "en"

# メッセージカタログ: key -> {lang: text}
# {name} 形式のプレースホルダは tr() のキーワード引数で埋める。
_CATALOG: dict[str, dict[str, str]] = {
    # ---------------------------------------------------------------- banner / generic
    "banner_title": {"en": "  Tema_Q-Abliterated  v{version}",
                     "jp": "  Tema_Q-Abliterated  v{version}"},
    "banner_sub": {"en": "  Exploratory weight uncensoring tool (SCAPE algorithm)",
                   "jp": "  探索的ウェイト非検閲化ツール (SCAPE アルゴリズム搭載)"},
    "done": {"en": "Done.", "jp": "完了しました。"},
    "error": {"en": "Error: {etype}: {e}", "jp": "エラー: {etype}: {e}"},
    "interrupted": {"en": "\nInterrupted (checkpoint saved; re-run to resume).",
                    "jp": "\n中断されました (チェックポイントは保存済み。再実行すると再開できます)。"},
    "resume_hint": {"en": "  Re-run the same command to resume from the checkpoint "
                          "(progress is saved after every trial).",
                    "jp": "  同じコマンドを再実行するとチェックポイントから再開します "
                          "(進捗は1試行ごとに保存されています)。"},
    "note_prefix": {"en": "  Note: {msg}", "jp": "  注意: {msg}"},
    "warn_prefix": {"en": "  Warning: {msg}", "jp": "  警告: {msg}"},

    # ---------------------------------------------------------------- env
    "env_section": {"en": "[Environment]", "jp": "[環境検出]"},
    "workspace_info": {"en": "Workspace: {dir}",
                       "jp": "ワークスペース: {dir}"},
    "env_os": {"en": "OS          : {v}", "jp": "OS          : {v}"},
    "env_python": {"en": "Python      : {v}", "jp": "Python      : {v}"},
    "env_torch": {"en": "PyTorch     : {v}", "jp": "PyTorch     : {v}"},
    "env_device": {"en": "Device      : {v}", "jp": "デバイス    : {v}"},
    "env_vram": {"en": "VRAM        : {v} GB", "jp": "VRAM        : {v} GB"},
    "env_cpu": {"en": "CPU         : {v} ({p} physical / {l} logical cores)",
                "jp": "CPU         : {v} ({p}物理/{l}論理コア)"},
    "env_ram": {"en": "RAM         : {v} GB", "jp": "RAM         : {v} GB"},
    "env_batch": {"en": "Batch size  : {v}", "jp": "バッチサイズ: {v}"},
    "env_cpu_slow": {
        "en": "Running on CPU - slower than GPU. SCAPE surrogate evaluation keeps per-trial cost low.",
        "jp": "CPU実行のためGPU比で低速です。SCAPEの代理評価により1試行あたりの負荷は大幅に抑えられます。"},

    # ---------------------------------------------------------------- model load
    "model_section": {"en": "[Model loading] {model}", "jp": "[モデル読み込み] {model}"},
    "model_local_path": {"en": "  Local path: {path}", "jp": "  ローカルパス: {path}"},
    "model_hf_fetch": {"en": "  Downloading model from HuggingFace: {model}", "jp": "  HuggingFaceからモデルを取得します: {model}"},
    "model_hf_cache": {"en": "  Cache destination: {path}", "jp": "  キャッシュ先: {path}"},
    "model_loaded_dtype": {"en": "  Model loaded in {dt} (parameters: {n}B)",
                           "jp": "  モデルを {dt} で読み込みました (パラメータ数: {n}B)"},
    "model_load_fail_dtype": {"en": "  Loading in {dt} failed: {etype}: {e}",
                              "jp": "  {dt} での読み込みに失敗: {etype}: {e}"},
    "model_load_all_fail": {"en": "Model loading failed for all dtype candidates: {last}",
                            "jp": "すべてのdtype候補でモデル読み込みに失敗しました: {last}"},
    "torch_missing": {"en": "PyTorch is not installed. Run: pip install torch",
                      "jp": "PyTorchがインストールされていません。pip install torch を実行してください。"},
    "model_nan_logits": {"en": "logits contain NaN after loading",
                         "jp": "読み込み後のlogitsがNaNになりました"},
    "model_nan_logits_stream": {"en": "logits contain NaN after streaming load",
                                "jp": "ストリーミング読み込み後のlogitsがNaNになりました"},
    "model_no_params": {"en": "Model parameter count is 0 - loading failed.",
                        "jp": "モデルのパラメータ数が0です。読み込みに失敗しています。"},
    "stream_title": {"en": "  Streaming load (per-tensor, meta-device skeleton)...",
                     "jp": "  ストリーミング読み込み (テンソル単位, メタデバイス構築)..."},
    "stream_shard": {"en": "    Loaded {f} (INT4-converted modules so far: {n})",
                     "jp": "    {f} を読み込みました (INT4変換 {n}モジュールまで)"},
    "stream_done": {"en": "    Streaming load complete: INT4 {n} modules ({gb:.2f}GB) / perturbation targets & lm_head kept in bf16",
                    "jp": "    ストリーミング読み込み完了: INT4 {n}モジュール ({gb:.2f}GB) / 摂動対象・lm_headはbf16のまま"},
    "stream_fail": {"en": "  Streaming load failed ({etype}: {e})",
                    "jp": "  ストリーミング読み込みに失敗 ({etype}: {e})"},
    "stream_fallback": {"en": "  Falling back to standard loading (watch memory)",
                        "jp": "  一括読み込みへフォールバックします (メモリに注意してください)"},
    "stream_meta_left": {"en": "{n} parameters still unassigned after streaming load (e.g. {ex}) - falling back to standard load.",
                         "jp": "ストリーミング読み込み後も {n} 個のパラメータが未割当です (例: {ex})。一括読み込みへフォールバックします。"},
    "int4_build": {"en": "    Built INT4 evaluation model (quantized {n} modules, weights {gb:.2f}GB; targets & lm_head keep full precision)",
                   "jp": "    INT4評価モデルを構築しました (量子化 {n}モジュール, 重み {gb:.2f}GB, 対象・lm_headは元精度)"},
    "int8_build": {"en": "    Built int8 dynamic-quantization evaluation model (module-wise: {n} modules; targets & lm_head keep full precision)",
                   "jp": "    int8評価モデルを構築しました (モジュール単位量子化 {n}件 / 対象・lm_headは元精度)"},
    "int8_build_title": {"en": "  Building int8 dynamic-quantization model for search (target projections stay full precision)",
                         "jp": "  探索用にint8動的量子化モデルを構築します (対象射影は元精度のまま)"},
    "int8_fail": {"en": "  int8 conversion failed ({etype}) - switching to INT4",
                  "jp": "  int8化に失敗 ({etype}) → より省メモリなINT4へ切り替えます"},
    "int4_cache_budget": {"en": "  INT4 weight cache budget: {gb:.2f}GB ({detail})",
                          "jp": "  INT4重みキャッシュ予算: {gb:.2f}GB ({detail})"},
    "int4_cache_lru": {"en": "LRU, auto-evict on overflow", "jp": "LRU, 予算超過時は自動退避"},
    "int4_cache_off": {"en": "disabled - not enough free RAM, weights are dequantized on the fly",
                       "jp": "無効 — 空きRAM不足のため每次脱量子化します"},
    "no_safetensors": {"en": "No safetensors files found in {d}.", "jp": "{d} にsafetensorsファイルが見つかりません。"},

    # ---------------------------------------------------------------- tokenizer / warnings
    "tok_regex_fixed": {"en": "  Tokenizer: buggy Mistral pre-tokenizer regex detected and fixed (fix_mistral_regex=True).",
                        "jp": "  トークナイザ: 誤ったMistralプリトークナイザ正規表現を検出し修正しました (fix_mistral_regex=True)。"},
    "tok_regex_false_positive": {
        "en": "  Tokenizer: the transformers regex warning is a false positive for this tokenizer "
              "(pre-tokenizer is not the buggy Mistral Split); warning suppressed.",
        "jp": "  トークナイザ: transformersの正規表現警告は本トークナイザには無関係の誤検知のため抑制しました。"},
    "kernel_fallback_note": {
        "en": "  Note: optimized kernels (flash-linear-attention / causal-conv1d) not installed - "
              "reference PyTorch kernels are used (correct but slower). Hide-able noise suppressed.",
        "jp": "  注意: 最適化カーネル (flash-linear-attention / causal-conv1d) 未導入のため "
              "リファレンス実装で動作します (正確だが低速)。反復する注意ログは抑制済みです。"},

    # ---------------------------------------------------------------- memory plan
    "plan_mode_standard": {"en": "bulk load", "jp": "一括読み込み"},
    "plan_mode_streamed": {"en": "streaming load", "jp": "ストリーミング読み込み"},
    "plan_quant_none": {"en": "no quantization", "jp": "量子化なし"},
    "plan_quant_int8": {"en": "INT8 dynamic quantization", "jp": "INT8動的量子化"},
    "plan_quant_int4": {"en": "INT4 weight quantization", "jp": "INT4重み量子化"},
    "plan_explicit_int4": {"en": "explicit request: convert to INT4 after loading",
                           "jp": "明示指定: 読み込み後INT4へ変換"},
    "plan_explicit_int4_stream": {"en": "explicit request: streaming INT4 (too large for bulk load)",
                                  "jp": "明示指定: 一括読みが不要なためストリーミングINT4"},
    "plan_int8_gpu_switch": {"en": "INT8 is not supported on GPU - switched to INT4",
                             "jp": "INT8はGPU非対応のためINT4へ切り替え"},
    "plan_explicit_int8": {"en": "explicit request: convert to INT8 after loading",
                           "jp": "明示指定: 読み込み後INT8へ変換"},
    "plan_explicit_dtype": {"en": "explicit request: evaluate in {dt}", "jp": "明示指定: {dt}で評価"},
    "plan_roomy": {"en": "memory has headroom (no quantization needed)",
                   "jp": "メモリに余裕あり (量子化不要)"},
    "plan_shrink_int8": {"en": "shrinking to INT8 dynamic quantization to save RAM",
                         "jp": "RAM節約のためINT8動的量子化へ縮小"},
    "plan_shrink_int4": {"en": "shrinking to INT4 weight quantization to save RAM",
                         "jp": "RAM節約のためINT4重み量子化へ縮小"},
    "plan_stream_int4": {"en": "bulk load not feasible - streaming INT4 (avoids SWAP)",
                         "jp": "一括読み不可のためストリーミングINT4 (SWAP回避)"},
    "plan_danger": {"en": "warning: model is very large for the free memory; INT4 is still risky",
                    "jp": "警告: 空きメモリに対してモデルが大きく、INT4でも非常に危険です"},
    "mem_section": {"en": "[Memory plan] model {disk:.2f}GB / free RAM {ram:.1f}GB{vram}",
                    "jp": "[メモリ計画] モデル {disk:.2f}GB / 空きRAM {ram:.1f}GB{vram}"},
    "mem_vram_check": {"en": " / checking free VRAM", "jp": " / 空きVRAMを確認"},
    "mem_plan": {"en": "  Plan: {plan}", "jp": "  計画: {plan}"},
    "mem_guard_line": {"en": "  [Memory] RSS={rss:.1f}GB free RAM={ram:.1f}GB{vram} SWAP={swap:.1f}GB{reasons}",
                       "jp": "  [メモリ] RSS={rss:.1f}GB 空きRAM={ram:.1f}GB{vram} SWAP={swap:.1f}GB{reasons}"},
    "mem_oom_retry": {"en": "  [Memory] OOM detected{desc} -> reducing memory and retrying ({i}/{n})",
                      "jp": "  [メモリ] メモリ不足を検出{desc} → 削減して再試行 ({i}/{n})"},
    "mem_oom_unresolved": {"en": "Could not resolve out-of-memory: {last}",
                           "jp": "メモリ不足が解消できませんでした: {last}"},
    "mem_gen_batch_shrink": {"en": "  [Memory] OOM during generation -> retrying with batch size {bs}",
                             "jp": "  [メモリ] 生成中にメモリ不足 → バッチサイズを{bs}へ縮小して再試行"},
    "mem_reason_avail": {"en": "free RAM {gb:.2f}GB", "jp": "空きRAM {gb:.2f}GB"},
    "mem_reason_swap": {"en": "SWAP increase {gb:.2f}GB", "jp": "SWAP増加 {gb:.2f}GB"},
    "mem_reason_vram": {"en": "free VRAM {gb:.2f}GB", "jp": "空きVRAM {gb:.2f}GB"},

    # ---------------------------------------------------------------- architecture / probes
    "arch_section": {"en": "[Architecture analysis] layers: {n}", "jp": "[アーキテクチャ解析] レイヤ数: {n}"},
    "arch_component": {"en": "  Perturbation target component {comp}: {n} modules",
                       "jp": "  摂動対象コンポーネント {comp}: {n} モジュール"},
    "probes_section": {"en": "[Probes]", "jp": "[プローブ準備]"},
    "probes_desc": {"en": "  {nh} harmful / {ng} harmless probes ({src})",
                    "jp": "  有害プローブ {nh}件 / 無害プローブ {ng}件 ({src})"},
    "probes_builtin": {"en": "built-in", "jp": "内蔵"},
    "probes_file": {"en": "file: {f}", "jp": "ファイル: {f}"},
    "probes_dataset": {"en": "dataset: {d}", "jp": "データセット: {d}"},
    "probes_subset": {"en": "  Exploration subset: {n} probes / final verification: {v} prompts (full generation)",
                      "jp": "  探索用サブセット: {n}プローブ / 最終検証: {v}プロンプト (全文生成)"},
    "thinking_prefix": {"en": "  Thinking-style model detected - response prefix set: {p!r}",
                        "jp": "  思考モデルと判定されたため応答接頭辞を設定します: {p!r}"},
    "probes_empty": {"en": "Harmful/harmless probe sets are empty. Check your input files.",
                     "jp": "有害・無害プローブが空です。ファイル内容を確認してください。"},
    "datasets_missing": {"en": "The datasets library is required to load dataset {d} "
                               "(pip install datasets), or pass --harmful-file with a text file.",
                         "jp": "データセット {d} を読み込むには datasets ライブラリが必要です "
                               "(pip install datasets)。または --harmful-file 等でテキストファイルを指定してください。"},
    "hub_missing": {"en": "huggingface_hub is not installed (pip install huggingface_hub). "
                          "You can also pass a local model directory path.",
                    "jp": "huggingface_hubが未導入です (pip install huggingface_hub)。"
                          "ローカルのモデルディレクトリを直接指定することでも実行できます。"},
    "dir_no_config": {"en": "Directory {d} has no config.json.", "jp": "ディレクトリ {d} にconfig.jsonが見つかりません。"},

    # ---------------------------------------------------------------- directions
    "dir_section": {"en": "[Residual directions]", "jp": "[残差方向の計算]"},
    "dir_cached": {"en": "  Residual directions loaded from cache: {name}",
                   "jp": "  残差方向をキャッシュから読み込みました: {name}"},
    "dir_computing": {"en": "  Computing residual directions (mean difference of harmful/harmless probes)...",
                      "jp": "  残差方向を計算しています (有害/無害プローブの平均差)..."},
    "dir_progress": {"en": "  Hidden states: {kind} batch {i}/{n}",
                     "jp": "  隠れ状態計算: {kind} バッチ {i}/{n}"},
    "dir_prog_prefix": {"en": "  Hidden states ({kind})", "jp": "  隠れ状態計算 ({kind})"},
    "dir_kind_harmful": {"en": "harmful", "jp": "有害"},
    "dir_kind_harmless": {"en": "harmless", "jp": "無害"},
    "dir_peak": {"en": "  Salience peak: layer {peak} / {n}", "jp": "  顕著度ピーク: レイヤ {peak} / {n}"},
    "base_cont_title": {"en": "  Generating base continuations for multi-position KL ({k} tokens x {n} probes)...",
                        "jp": "  多位置KL用の基底継続を生成しています ({k}トークン × {n}プローブ)..."},
    "base_cont_progress": {"en": "  Base continuations: batch {i}/{n}", "jp": "  基底継続生成: バッチ {i}/{n}"},
    "gen_prefix_base": {"en": "  Base continuations", "jp": "  基底継続生成"},
    "kl_trim": {"en": "  KL probe set thinned to {n} probes (memory cap for storing base distributions, vocab {v:,} x {k} positions)",
                "jp": "  KL評価プローブをメモリ上限のため {n}件に間引きました (語彙数{v:,} × {k}位置の基底分布を保存)"},

    # ---------------------------------------------------------------- partial forward
    "partial_check_ok": {"en": "  Partial-forward self-check: max logit diff {d:.3g} (OK) - prefix activation cache enabled",
                         "jp": "  部分フォワード自己検証: 最大logit差 {d:.3g} (OK) — 前半層キャッシュを有効化しました"},
    "partial_check_ok_nocache": {"en": "  Partial-forward self-check: max logit diff {d:.3g} (OK)",
                                 "jp": "  部分フォワード自己検証: 最大logit差 {d:.3g} (OK)"},
    "partial_check_fail": {"en": "  Partial-forward check mismatch (max logit diff {d:.3g}) - prefix cache disabled, using full forward",
                           "jp": "  部分フォワード検証が不一致 (最大logit差 {d:.3g}) — 前半層キャッシュを無効化しフルフォワードで評価します"},
    "partial_check_error": {"en": "  Partial-forward self-check failed ({etype}) - using standard model forward",
                            "jp": "  部分フォワード検証が失敗 ({etype}) → モデル標準forwardで評価します"},
    "cache_progress": {"en": "Prefix cache: {kind} batch {i}/{n} (ETA {eta})",
                       "jp": "前半層キャッシュ構築: {kind} バッチ {i}/{n} (残り {eta})"},
    "eval_progress": {"en": "Evaluating: {part} batch {i}/{n}",
                      "jp": "評価中: {part} バッチ {i}/{n}"},

    # ---------------------------------------------------------------- SCAPE search
    "scape_section": {"en": "[SCAPE search] trials: {n}", "jp": "[SCAPE探索] 試行回数: {n}"},
    "scape_checkpoint_path": {"en": "  Checkpoint: {path} (saved after every trial, auto-resume on re-run)",
                              "jp": "  チェックポイント: {path} (1試行ごとに保存、再実行で自動再開)"},
    "scape_time_budget": {"en": "  Time budget: {t} (search stops gracefully at the best-so-far point)",
                          "jp": "  時間予算: {t} (到達時点の最良解で探索を打ち切り最終検証へ進みます)"},
    "resume_found": {"en": "  Resuming from checkpoint (existing trials: {a}/{n})", "jp": "  チェックポイントから再開します (既存試行: {a}/{n})"},
    "resume_done": {"en": "  Already reached the requested trial count - returning best solution.",
                    "jp": "  既に規定試行回数に到達しています。最良解を返します。"},
    "resume_fail": {"en": "  Checkpoint load failed ({e}) - starting fresh.", "jp": "  チェックポイントの読み込みに失敗しました ({e})。最初から開始します。"},
    "restart_discard": {"en": "\n[Checkpoint discarded] starting a fresh search.", "jp": "\n[チェックポイント破棄] 最初から探索します"},
    "search_finished": {"en": "  Search finished: {a} trials / {t:.0f}s", "jp": "  探索完了: {a}試行 / 所要 {t:.0f}s"},
    "search_budget_stop": {"en": "  Time budget reached - stopping search at trial {a} (best-so-far kept, checkpoint saved)",
                           "jp": "  時間予算に到達したため試行 {a} で探索を打ち切ります (最良解とチェックポイントは保存済み)"},
    "search_best": {"en": "  Best: soft refusal={r:.3f}  KL={k:.4f}  J={j:.4f}",
                    "jp": "  最良: ソフト拒否率={r:.3f}  KL={k:.4f}  J={j:.4f}"},
    "search_eval_fail": {"en": "  Search could not run a single evaluation.", "jp": "探索が1度も評価を実行できませんでした。"},
    "eval_failed_why": {"en": "  Trial {a}: evaluation failed ({t}: {m}) - treated as penalty (J=inf)",
                        "jp": "  試行{a}: 評価が失敗しました ({t}: {m}) — ペナルティ扱い (J=inf) にします"},
    "search_est_total": {"en": "  Estimated total search time: {t} ({r:.0f}s/trial) - Ctrl+C anytime; progress is checkpointed",
                         "jp": "  探索全体の推定所要時間: {t} (1試行あたり約{r:.0f}秒) — Ctrl+Cでいつでも中断可能です (進捗は保存されます)"},
    "apply_section": {"en": "\n[Apply] applying external parameters to the model", "jp": "\n[適用] 外部パラメータをモデルへ適用します"},
    "apply_metrics": {"en": "  soft refusal={r:.3f}  KL={k:.4f}", "jp": "  ソフト拒否率={r:.3f}  KL={k:.4f}"},
    "extra_trials": {"en": "  Running extra trials ({a} -> {n})...", "jp": "  追加探索を実行します ({a} → {n}試行)"},

    # ---------------------------------------------------------------- verification
    "verify_section": {"en": "\n[Final verification: refusal rate by full generation] {n} prompts x {t} tokens",
                       "jp": "\n[最終検証: 全文生成による拒否率] {n}プロンプト × {t}トークン生成"},
    "verify_progress": {"en": "  Generating: batch {i}/{n} ({done}/{total} prompts)", "jp": "  生成中: バッチ {i}/{n} ({done}/{total}プロンプト)"},
    "gen_prefix_verify": {"en": "  Full generation", "jp": "  全文生成"},
    "verify_baseline": {"en": "  Baseline model refusal rate (full generation): {r:.3f}", "jp": "  元モデルの拒否率 (全文生成): {r:.3f}"},
    "verify_candidate": {"en": "  Candidate (trial #{a}, J={j:.4f}): full-generation refusal={r:.3f} ({rn}/{rt}) KL={kl:.4f}{collapse}",
                         "jp": "  候補 (試行#{a}, J={j:.4f}): 全文生成拒否率={r:.3f} ({rn}/{rt}) KL={kl:.4f}{collapse}"},
    "verify_cache_reuse": {"en": "  Verification cache: {n} candidates (reusing)", "jp": "  検証キャッシュ: {n}候補 (再利用します)"},
    "final_choice": {"en": "\n  Final selection: trial #{a} refusal {r:.3f} / KL {k:.4f}",
                     "jp": "\n  最終採用: 試行#{a} 拒否率 {r:.3f} / KL {k:.4f}"},

    # ---------------------------------------------------------------- collapse
    "coll_nll": {"en": "NLL/base {r:.1f}x (> {m:.1f})", "jp": "基底比NLL {r:.1f}倍 (> {m:.1f})"},
    "coll_repeat": {"en": "repetitive answers {r:.0%}", "jp": "反復応答 {r:.0%}"},
    "coll_empty": {"en": "empty answers {r:.0%}", "jp": "空応答 {r:.0%}"},
    "coll_diversity": {"en": "vocabulary diversity {r:.2f}", "jp": "語彙多様性 {r:.2f}"},
    "coll_note_kl": {"en": "Note: multi-position KL {multi:.4f} is large vs first-token KL {first:.4f}",
                     "jp": "注意: 初頭KL {first:.4f} に対し多位置KL {multi:.4f} が大きい"},
    "coll_flag_bad": {"en": "DEGENERATED (", "jp": "⚠ 崩壊 ("},
    "coll_flag_warn": {"en": "! ", "jp": "△ "},

    # ---------------------------------------------------------------- selection UI
    "cand_list_title": {"en": "Candidate list (verified by full generation):", "jp": "候補一覧 (全文生成による検証済み):"},
    "cand_header": {"en": "  #  | trial  | refusal (full gen)      | KL       | collapse check",
                    "jp": "  #  | 試行  | 拒否率 (全文生成)      | KL       | 崩壊チェック"},
    "cand_prompt": {"en": "Enter candidate number to adopt (Enter=recommended / a[N]=add N trials / q=quit without saving): ",
                    "jp": "採用する候補の番号を入力 (Enter=推奨 / a[N]=N試行追加 / q=保存せず終了): "},
    "cand_auto_pick": {"en": "  Auto-selected: candidate {no} (refusal {r:.1%} / KL {k:.4f})",
                       "jp": "  自動選択: 候補 {no} (拒否率 {r:.1%} / KL {k:.4f})"},
    "cand_auto_pick_pipe": {"en": "  (non-interactive environment - auto-selected: candidate {no} refusal {r:.1%} / KL {k:.4f})",
                            "jp": "  (非対話環境のため自動選択: 候補 {no} 拒否率 {r:.1%} / KL {k:.4f})"},
    "cand_recommended": {"en": "  Adopting recommended candidate {no} (refusal {r:.1%} / KL {k:.4f})",
                         "jp": "  推奨候補 {no} を採用します (拒否率 {r:.1%} / KL {k:.4f})"},
    "cand_input_abort": {"en": "  (input aborted - auto-selecting recommended candidate)", "jp": "  (入力中断 → 推奨候補を自動選択します)"},
    "cand_quit": {"en": "  Quitting without saving.", "jp": "  保存せず終了します。"},
    "cand_saved_none": {"en": "\nExiting without saving (checkpoint remains).", "jp": "\n保存せず終了します (チェックポイントは残っています)。"},
    "cand_done_none": {"en": "\nDone (model not saved).", "jp": "\n完了しました (モデルは保存していません)。"},
    "cand_add_disabled": {"en": "  Extra trials are not available in this mode.", "jp": "  このモードでは追加試行できません。"},
    "cand_add_n_prompt": {"en": "  Enter number of extra trials (default {n}): ", "jp": "  追加する試行数を入力 (既定 {n}): "},
    "cand_add_running": {"en": "  Adding {n} trials and continuing search...", "jp": "  {n}試行を追加して探索を続行します..."},
    "cand_add_fail": {"en": "  Extra trials failed: {e}", "jp": "  追加試行に失敗しました: {e}"},
    "cand_need_number": {"en": "  Please enter a number.", "jp": "  数値を入力してください。"},
    "cand_le_zero": {"en": "  Values <= 0 are ignored.", "jp": "  0以下は無視します。"},
    "cand_not_found": {"en": "  Candidate {i} does not exist (1-{m}).", "jp": "  候補 {i} は存在しません (1〜{m})。"},
    "cand_collapsed_confirm": {"en": "  Candidate {i} may have collapsed ({reason}). Really adopt it? [y/N]: ",
                               "jp": "  候補 {i} は崩壊の疑いがあります ({reason})。 本当に採用しますか? [y/N]: "},
    "cand_adopted": {"en": "  Adopting candidate {i}.", "jp": "  候補 {i} を採用します。"},
    "cand_bad_input": {"en": "  Could not parse input. Enter a number / Enter / a[N] / q.", "jp": "  解釈できませんでした。番号 / Enter / a[N] / q のいずれかを入力してください。"},

    # ---------------------------------------------------------------- export / report
    "export_section": {"en": "\n[Model export] {dir}", "jp": "\n[モデル出力] {dir}"},
    "export_keys": {"en": "  Mapped keys: {n}", "jp": "  対応キー数: {n}"},
    "export_done": {"en": "  Export complete: {dir}", "jp": "  出力完了: {dir}"},
    "export_shard": {"en": "    Written: {f}{extra}", "jp": "    書き出し: {f}{extra}"},
    "export_changed": {"en": " (perturbed {n} tensors)", "jp": " (摂動適用 {n}テナソル)"},
    "export_sidecar": {"en": "    Copied {n} sidecar files", "jp": "    サイドカーファイル {n}件をコピーしました"},
    "export_same_dir": {"en": "Output directory equals the source model directory.",
                        "jp": "出力先が元モデルのディレクトリと同一です。"},
    "export_inside_dir": {"en": "Output directory is inside the source model directory.",
                          "jp": "出力先が元モデルのディレクトリ内部です。"},
    "export_exists": {"en": "Output directory {d} already exists and is not empty. Pass --overwrite-output to overwrite it.",
                      "jp": "出力先 {d} は既に存在し、空ではありません。上書きするには --overwrite-output を指定してください。"},
    "export_path_file": {"en": "Output path {d} exists but is a file, not a directory.",
                         "jp": "出力パス {d} はディレクトリではなくファイルです。"},
    "export_dir_empty_ok": {"en": "  Output directory exists and is empty - using it as-is.",
                            "jp": "  出力先ディレクトリは存在しますが空のため、そのまま使用します。"},
    "export_path_problem": {"en": "  Output path problem: {msg}",
                            "jp": "  出力パスに問題があります: {msg}"},
    "export_repair_prompt": {"en": "  Enter a new output directory (empty = same path retry, q = skip export): ",
                             "jp": "  新しい出力先ディレクトリを入力してください (空=同じパスで再試行, q=出力をスキップ): "},
    "export_autosuffix": {"en": "  Non-interactive run: exporting to alternative directory {d} instead.",
                          "jp": "  非対話実行のため、代替ディレクトリ {d} に出力します。"},
    "export_skipped": {"en": "  Export skipped. The search/verification results are kept in the report.",
                       "jp": "  出力をスキップしました。探索・検証結果はレポートに保存されています。"},

    # ---------------------------------------------------------------- [HuggingFace upload]
    "hf_ask": {"en": "\nUpload the exported model to the HuggingFace Hub? [y/N]: ",
               "jp": "\n書き出したモデルをHuggingFace Hubにアップロードしますか? [y/N]: "},
    "hf_repo_prompt": {"en": "Repository name (the suffix '-Tema_Q-Abliterated' is added automatically): ",
                       "jp": "リポジトリ名 (接尾辞 '-Tema_Q-Abliterated' は自動で付与されます): "},
    "hf_token_prompt": {"en": "HuggingFace API token with write access (https://huggingface.co/settings/tokens): ",
                        "jp": "HuggingFace APIキー (書き込み権限・https://huggingface.co/settings/tokens で作成): "},
    "hf_repo_resolved": {"en": "  Target repository: {r}",
                         "jp": "  アップロード先リポジトリ: {r}"},
    "hf_uploading": {"en": "  Uploading to the HuggingFace Hub (input: {r})...",
                     "jp": "  HuggingFace Hubへアップロードしています (入力: {r})..."},
    "hf_upload_done": {"en": "  Uploaded: https://huggingface.co/{r}",
                       "jp": "  アップロード完了: https://huggingface.co/{r}"},
    "hf_upload_fail": {"en": "  Upload failed: {err}\n  The exported model is still available locally at {d}",
                       "jp": "  アップロードに失敗しました: {err}\n  書き出したモデルはローカルの {d} に残っています"},
    "hf_skip_declined": {"en": "  Upload skipped.",
                         "jp": "  アップロードをスキップしました。"},
    "hf_skip_noninteractive": {"en": "  HuggingFace upload skipped (non-interactive run). "
                                     "Use --hf-repo and --hf-token to upload without prompts.",
                               "jp": "  非対話実行のためHuggingFaceアップロードをスキップしました。"
                                     "プロンプトなしでアップロードするには --hf-repo と --hf-token を指定してください。"},
    "hf_repo_empty": {"en": "repository name is empty", "jp": "リポジトリ名が空です"},
    "hf_repo_invalid": {"en": "invalid repository name: {r}", "jp": "リポジトリ名が不正です: {r}"},
    "hf_repo_need_ns": {"en": "could not determine the username namespace for the repository "
                              "(log in with the API token first)",
                        "jp": "リポジトリのユーザー名 (名前空間) を特定できませんでした"
                              "(先にAPIキーでログインしてください)"},

    # ---------------------------------------------------------------- [v8④] 再実行メニュー
    "resume_menu_title": {"en": "  [Resume] Checkpoint found: {a}/{n} trials done (best J {j}, {t} ago, {v} candidates verified)",
                          "jp": "  [再開] チェックポイントがあります: {a}/{n} 試行完了 (best J {j}, {t}前, 検証済み候補 {v}件)"},
    "resume_menu_done": {"en": "    (the requested number of trials is already complete)",
                        "jp": "    (規定試行数はすでに完了しています)"},
    "resume_menu_opt1": {"en": "    1 = resume the search (continue to trial {n})  [default]",
                         "jp": "    1 = 探索を続きから再開 (試行{n}まで)  [既定]"},
    "resume_menu_opt2": {"en": "    2 = add more trials (extend the search beyond {a})",
                         "jp": "    2 = 試行を追加 ({a}を超えて延長)"},
    "resume_menu_opt3": {"en": "    3 = export from previous candidates (skip the search phase)",
                         "jp": "    3 = 前回の候補から書き出し (探索をスキップ)"},
    "resume_menu_opt4": {"en": "    4 = start over (discard the checkpoint)",
                         "jp": "    4 = 最初からやり直す (チェックポイントを破棄)"},
    "resume_menu_prompt": {"en": "  Choose [1-4, Enter=1]: ",
                            "jp": "  選択 [1-4, Enter=1]: "},
    "resume_menu_extra_n": {"en": "  How many additional trials? [{n}]: ",
                             "jp": "  追加する試行数は? [{n}]: "},
    "resume_menu_bad": {"en": "  Enter a number between 1 and 4 (or press Enter for 1).",
                        "jp": "  1〜4の番号 (またはEnter) を入力してください。"},
    "resume_menu_need_number": {"en": "  Enter a positive number.",
                                "jp": "  正の数を入力してください。"},
    "resume_menu_go_resume": {"en": "  Resuming the search (to trial {n}).",
                              "jp": "  探索を再開します (試行{n}まで)。"},
    "resume_menu_go_extend": {"en": "  Extending the search by +{n} trials (to {a}).",
                              "jp": "  探索を +{n}試行 延長します (試行{a}まで)。"},
    "resume_menu_go_export": {"en": "  Skipping the search phase - verifying and exporting previous candidates.",
                              "jp": "  探索フェーズをスキップし、前回の候補を検証して書き出します。"},
    "report_section": {"en": "\n[Report] {p}", "jp": "\n[レポート] {p}"},
    "report_title_refusal": {"en": "Search progress: refusal rate convergence ({model})",
                             "jp": "探索過程: 拒否率の収束 ({model})"},
    "report_title_memory": {"en": "Memory usage timeline ({model})", "jp": "メモリ使用量の推移 ({model})"},
    "report_skip_memory": {"en": "  Skipped memory timeline chart ({etype})", "jp": "  メモリ推移グラフの生成をスキップしました ({etype})"},
    "report_summary_path": {"en": "[Run record] {p}", "jp": "[実行記録] {p}"},

    # ---------------------------------------------------------------- params card
    "params_section": {"en": "\n[External model card] {src}", "jp": "\n[外部モデルカード読み込み] {src}"},
    "params_mode_apply": {"en": "  Mode: applying loaded parameters as-is (no search)",
                          "jp": "  モード: 読み込んだパラメータをそのまま適用します (探索なし)"},
    "params_mode_focus": {"en": "  Mode: narrowing the search around the loaded parameters",
                          "jp": "  モード: 読み込んだパラメータ周辺に探索範囲を絞ります"},
    "params_card_src": {"en": "  Source: {s}", "jp": "  読み込み元: {s}"},
    "params_card_format": {"en": "  Format: {f}", "jp": "  形式: {f}"},
    "params_arch_warn": {"en": "  Card max_weight_position={pos} is far from the target layer count {n} - parameters may be for a different model scale.",
                         "jp": "カードの max_weight_position={pos} が対象モデルの層数 {n} と大きく乖離しています。異なる規模のモデル向けパラメータの可能性があります。"},
    "params_not_detected": {"en": "No Abliteration parameters found in the model card. "
                                  "Check that a Markdown table parameter record exists in the README.",
                            "jp": "モデルカードから Abliteration parameters を検出できませんでした。"
                                  "READMEにMarkdown表形式のパラメータ記録があるか確認してください。"},
    "params_key_missed": {"en": "  Note: {n} modules could not be mapped to safetensors keys (excluded from output)",
                          "jp": "  注意: {n}個のモジュールがsafetensorsキーと対応できませんでした (出力から除外)"},

    # ---------------------------------------------------------------- misc errors
    "model_required": {"en": "A target model is required (local path or HuggingFace repo id).",
                       "jp": "対象モデルを指定してください (ロカールパスまたはHuggingFaceリポジトリID)。"},
    "layers_unrecognized": {"en": "Could not recognize the layer structure of this model: {t}",
                            "jp": "このモデルのレイヤ構造を認識できませんでした: {t}"},
    "no_perturb_targets": {"en": "No perturbation-target projection modules were detected.",
                           "jp": "摂動対象となる射影モジュールを検出できませんでした。"},
    "module_no_weight": {"en": "Module {t} has no weight tensor", "jp": "モジュール {t} に重みがありません"},
    "orth_nonfinite": {"en": "Non-finite values appeared after orthogonalization (strength too high)",
                       "jp": "直交化後の重みに非有限値が生じました (強度が大きすぎます)"},

    # ---------------------------------------------------------------- v5 UI / 進捗
    "ui_elapsed": {"en": "elapsed", "jp": "経過"},
    "ui_remaining": {"en": "remaining", "jp": "残り"},
    "ui_eval_remaining": {"en": "this eval:", "jp": "この評価"},
    "ui_search_remaining": {"en": "search total remaining", "jp": "探索全体 残り"},
    "ui_screened": {"en": "screened", "jp": "早期除外"},

    # ---------------------------------------------------------------- v5 速度改善
    "hook_check_title": {"en": "  Rank-1 patch self-check (hook vs exact weight patch)...",
                         "jp": "  ランク1パッチ自己検証 (フック方式 vs 完全重みパッチ)..."},
    "hook_check_ok": {"en": "  Rank-1 patch self-check: max output diff {d:.2e} (OK) - hook-based patching enabled",
                      "jp": "  ランク1パッチ自己検証: 最大出力差 {d:.2e} (OK) — フック方式の摂動適用を有効化しました"},
    "hook_check_fail": {"en": "  Rank-1 patch self-check mismatch (max diff {d:.2e}) - falling back to exact weight patching",
                        "jp": "  ランク1パッチ検証が不一致 (最大差 {d:.2e}) — 完全重みパッチ方式へフォールバックします"},
    "boundary_sweep_title": {"en": "  Building multi-boundary prefix cache (one full sweep, no rebuilds during search)...",
                             "jp": "  多境界プレフィックスキャッシュを構築します (探索開始前に1回だけ実施、探索中の再構築なし)..."},
    "boundary_sweep_prefix": {"en": "  Prefix sweep", "jp": "  境界スイープ"},
    "boundary_sweep_done": {"en": "  Prefix cache ready: {n} boundaries (every {s} layers), {mb:.0f}MB - evaluations resume from the nearest boundary",
                            "jp": "  プレフィックスキャッシュ完成: 境界{n}箇所 ({s}レイヤ間隔, {mb:.0f}MB) — 評価は最寄り境界から再開します"},
    "screen_reject": {"en": "  Early screen: candidate rejected after partial evaluation (J lower bound {j:.4f} > threshold {t:.4f})",
                      "jp": "  早期スクリーニング: 部分評価で候補を除外しました (J下界 {j:.4f} > しきい値 {t:.4f})"},
    "gen_fastpath_on": {"en": "  Generation fast path: enough free RAM to cache dequantized weights ({gb:.1f}GB) for verification",
                        "jp": "  生成高速経路: 空きRAMが十分なため検証中は脱量子化重みをキャッシュします ({gb:.1f}GB)"},
    "gen_fastpath_off": {"en": "  Generation path: keeping INT4 weights (free RAM {free:.1f}GB < needed {need:.1f}GB) - batch size {b} amortizes dequantization",
                         "jp": "  生成経路: INT4のまま生成します (空きRAM {free:.1f}GB < 必要 {need:.1f}GB) — バッチサイズ{b}で脱量子化を償却します"},
    "gen_decided_stop": {"en": "  Early-stop rule: a sequence stops when it emits EOS or a refusal marker (metric unchanged, saves time)",
                         "jp": "  早期停止: EOSまたは拒否マーカーが出たらその系列を打ち切り判定します (指標は従来どおり、時間を節約)"},
    "eval_batch_note": {"en": "  Eval batch size: {b} (INT4 dequantization amortized over the batch)",
                        "jp": "  評価バッチサイズ: {b} (INT4脱量子化をバッチで償却)"},
    "memguard_budget_grow": {"en": "  [Memory] free RAM recovered - INT4 weight cache budget raised to {gb:.1f}GB",
                             "jp": "  [メモリ] 空きRAMが回復 — INT4重みキャッシュ予算を {gb:.1f}GB ずつ引き上げます"},
    # ---------------------------------------------------------------- v6 additions
    "arch_fused_note": {"en": "({n} fused 3D expert weight tensors -> exact-mode perturbation)",
                        "jp": "(融合3D専門家重み {n}件 → exactモード摂動)"},
    "dir_map_used": {"en": "  Component-specific directions captured for {n} component(s) (residual width != projection out-dim)",
                     "jp": "  コンポーネント別の方向を {n}件取得しました (残差幅と射影出力次元が不一致のため)"},
    "dir_missing_comp": {"en": "Direction vector missing for component {comp}",
                         "jp": "コンポーネント {comp} の方向ベクトルが未定義です"},
    "bypass_limit": {"en": "  Deep-bypass verification failed -> prefix cache limited to boundaries below layer {k} (arch-specific constraint)",
                     "jp": "  深いバイパスの等価性検証に失敗したため前方キャッシュを層 {k} 未満の境界に制限します (アーキテクチャ固有の制約)"},
    "int4_fused_note": {"en": "  INT4 note: fused expert weights ({gb:.2f}GB) are perturbation targets and stay in bf16. Use --components attn to INT4-quantize them.",
                        "jp": "  INT4注記: 融合専門家の重み ({gb:.2f}GB) は摂動対象のためbf16のままです。--components attn 指定時はINT4化できます。"},
    "int4_coverage": {"en": "  INT4 coverage: {pct:.1f}% of model bytes quantized ({gb:.2f}GB as int4 of {tot:.2f}GB total)",
                      "jp": "  INT4適用率: モデル全体の {pct:.1f}% を量子化 ({tot:.2f}GB中 {gb:.2f}GB をint4化)"},
    "probes_diversity": {"en": "  Probe set: {nh} harmful / {ng} harmless prompts ({oh} distinct opening styles)",
                         "jp": "  プローブ: 有害 {nh}件 / 無害 {ng}件 (冒頭表現 {oh}種類)"},
    "probe_filter_title": {"en": "[Probe hardening] Measuring baseline refusal per harmful probe...",
                           "jp": "[プローブ締め付け] 有害プローブごとの基底拒否率を測定しています..."},
    "probe_filter_done": {"en": "  Kept {k}/{t} harmful probes (dropped {d} the baseline model does not refuse to; they dilute the direction and soften the metric)",
                          "jp": "  有害プローブ {t}件のうち {k}件を残しました (基底モデルが拒否しない {d}件を除外 — 方向の希釈と指標の甘さを防ぐ)"},
    "probe_filter_keepall": {"en": "  All {t} harmful probes show baseline refusal signal - keeping all",
                             "jp": "  有害プローブ {t}件すべてに基底拒否シグナルあり — 全件維持"},
    "search_seconds_note": {"en": "Search phase time: {t}",
                            "jp": "探索フェーズの所要時間: {t}"},
    "verify_topk_auto": {"en": "  CPU run: full-generation verification limited to baseline + top {k} candidates = {r} runs total (use --verify-topk to override)",
                         "jp": "  CPU実行: 全文生成検証は元モデル+上位{k}候補の計{r}回に制限します (--verify-topk で変更可能)"},
    "verify_gate_skip": {"en": "  Skipped full-generation verification for {n} candidates (surrogate refusal > {r}) - they cannot reach the target anyway",
                         "jp": "  代理指標の拒否率が {r} を超える候補 {n}件は全文生成検証をスキップしました (目標に到達し得ないため)"},
    "verify_dataset_note": {"en": "  Verification prompts: HuggingFace dataset {d} (split: {sp}, first {n} rows)",
                            "jp": "  検証プロンプト: HuggingFaceデータセット {d} (分割: {sp}, 先頭 {n}件)"},
    "verify_dataset_empty": {"en": "No prompts could be loaded from dataset {d}",
                             "jp": "データセット {d} からプロンプトを読み込めませんでした"},
    # ---------------------------------------------------------------- v9 additions
    "v9_seed_phase": {"en": "[Seed grid] Evaluating reference candidates that cover the strength x band-width spectrum (baseline / full-band flat kernels / single-component / peak-band) ...",
                      "jp": "[シードグリッド] 強度×帯幅スペクトルを覆う参照候補 (基底 / 全層一様flat / 単一コンポーネント / ピーク帯) を評価しています..."},
    "v9_floor_calibrated": {"en": "[Floor calibration] Measured surrogate floor = {f:.3f} (grammatical mass on refusal vocab that remains even on fully ablated models) -> effective target {t:.3f} (was {r:.3f}); J recomputed for the recorded trials",
                            "jp": "[床校正] 代理指標の床を実測: {f:.3f} (完全に直交化したモデルにも残る拒否語彙の文法的質量) → 実効目標 {t:.3f} (既定 {r:.3f})。記録済み試行のJも再計算しました"},
    "v9_lightweight_evaluator": {"en": "  [Lightweight evaluator] Base-continuation generation skipped (export-from-checkpoint mode); full evaluation will be built on demand if extra trials are requested",
                                 "jp": "  [軽量評価器] 基底継続生成を省略しました (チェックポイントから書き出しモード)。追加試行が必要になった時点で完全評価器を自動構築します"},
    "v9_export_from_ckpt": {"en": "[Export from checkpoint] Skipping the search phase - using the {n} recorded trials (best J {j})",
                            "jp": "[チェックポイントから書き出し] 探索フェーズをスキップします — 記録済みの {n}試行を使用します (最良 J {j})"},
    "v9_gate_none_title": {"en": "[Gate] No candidate passed the surrogate gate (first-token refusal <= {g}) - full-generation verification was skipped for all {n} candidates",
                           "jp": "[ゲート] 代理指標のしきい値 (初頭トークン比 <= {g})を下回る候補がなく、{n}件すべて全文生成検証をスキップしました"},
    "v9_gate_best_line": {"en": "  Best recorded trial: #{a}  refusal(mass) {rs} / refusal(first) {rf} / KL {kl}",
                          "jp": "  記録上の最良: 試行 #{a}  refusal(mass) {rs} / refusal(first) {rf} / KL {kl}"},
    "v9_gate_opt1": {"en": "    1 = add trials and continue the search  [default]",
                     "jp": "    1 = 試行を追加して探索を続行  [既定]"},
    "v9_gate_opt2": {"en": "    2 = end without writing the model",
                     "jp": "    2 = このまま終了 (モデルを書き出さない)"},
    "v9_gate_opt3": {"en": "    3 = pick a trial, verify it by full generation, then decide",
                     "jp": "    3 = 試行を指定して全文生成検証してから書き出す"},
    "v9_gate_opt4": {"en": "    4 = pick a trial and write the model without verification",
                     "jp": "    4 = 試行を指定して検証なしでモデルを書き出す"},
    "v9_gate_prompt": {"en": "  Choose [1-4, Enter=1]: ",
                       "jp": "  選択 [1-4, Enter=1]: "},
    "v9_gate_bad": {"en": "  Enter a number between 1 and 4 (or press Enter for 1).",
                    "jp": "  1〜4の数値を入力してください (Enterで1)。"},
    "v9_gate_trials_header": {"en": "  Top recorded trials (by J):",
                              "jp": "  記録済み試行の上位 (J順):"},
    "v9_gate_trial_row": {"en": "    #{a:<4} refusal(mass) {rs:<6} first {rf:<6} KL {kl:<8} J {j}",
                          "jp": "    #{a:<4} refusal(mass) {rs:<6} first {rf:<6} KL {kl:<8} J {j}"},
    "v9_gate_trial_prompt": {"en": "  Trial number to use: ",
                             "jp": "  使用する試行番号: "},
    "v9_gate_trial_bad": {"en": "  Trial {i} is not in the checkpoint history.",
                          "jp": "  試行 {i} はチェックポイントの履歴に存在しません。"},
    "v9_gate_verify_running": {"en": "  Verifying trial #{i} by full generation ({n} prompts) ...",
                               "jp": "  試行 #{i} を全文生成で検証しています ({n}プロンプト)..."},
    "v9_gate_verify_done": {"en": "  Trial #{i}: full-generation refusal {r} (KL {k:.4f})",
                            "jp": "  試行 #{i}: 全文生成拒否率 {r} (KL {k:.4f})"},
    "v9_gate_write_confirm": {"en": "  Write the model from trial #{i}? [y/N]: ",
                              "jp": "  試行 #{i} のモデルを書き出しますか? [y/N]: "},
    "v9_gate_noexport_auto": {"en": "  (--select auto) No candidate passed the gate - automatically writing the best-J trial #{a} WITHOUT full-generation verification",
                              "jp": "  (--select auto) ゲートを通過した候補がないため、最良Jの試行 #{a} を全文生成検証なしで自動書き出します"},
    "v9_gate_noexport_pipe": {"en": "  Non-interactive environment with interactive selection: the model was NOT written (no candidate passed the gate). Re-run in a terminal to choose an action, or use --select auto to write the best trial automatically.",
                              "jp": "  非対話環境のため対話選択できません — ゲートを通過した候補がないためモデルは書き出しませんでした。ターミナルで再実行して選択するか、--select auto で最良試行を自動書き出しできます。"},
    "v9_preload_menu_title": {"en": "  [Re-run detected] Checkpoint found for this model: {a} trials done / best J {j} ({t} ago, {v} candidates verified)",
                              "jp": "  [再実行検出] このモデルのチェックポイントがあります: {a}試行 完了 / 最良 J {j} ({t}前, 検証済み候補 {v}件)"},
    "v9_preload_menu_done": {"en": "    (the requested number of trials is already complete - option 1 will proceed to final verification)",
                             "jp": "    (指定試行数は完了済み — 1を選ぶと最終検証へ進みます)"},
    "v9_preload_menu_opt1": {"en": "    1 = continue the search (resume / proceed to verification)  [default]",
                             "jp": "    1 = 探索を続行 (再開 / 最終検証へ進む)  [既定]"},
    "v9_preload_menu_opt2": {"en": "    2 = add more trials (extend the search beyond {a})",
                            "jp": "    2 = 試行を追加 ({a} から先へ探索を延長)"},
    "v9_preload_menu_opt3": {"en": "    3 = write the model from the recorded candidates (skip the search phase)",
                            "jp": "    3 = 記録済み候補からモデルを書き出す (探索フェーズをスキップ)"},
    "v9_preload_menu_opt4": {"en": "    4 = start over (discard the checkpoint)",
                            "jp": "    4 = 最初からやり直す (チェックポイントを破棄)"},
    "v9_preload_menu_opt5": {"en": "    5 = quit (do nothing)",
                            "jp": "    5 = 終了 (何もしない)"},
    "v9_preload_menu_prompt": {"en": "  Choose [1-5, Enter=1]: ",
                               "jp": "  選択 [1-5, Enter=1]: "},
    "v9_preload_menu_bad": {"en": "  Enter a number between 1 and 5 (or press Enter for 1).",
                            "jp": "  1〜5の数値を入力してください (Enterで1)。"},
    "v9_preload_menu_extra_n": {"en": "  How many additional trials? [{n}]: ",
                                "jp": "  追加する試行数 [{n}]: "},
    "v9_preload_menu_need_number": {"en": "  Enter a positive number.",
                                     "jp": "  正の数値を入力してください。"},
    "v9_preload_before_load": {"en": "  (This menu is shown before model loading so no loading time is wasted)",
                               "jp": "  (このメニューはモデルのロード前に表示されます — ロード時間を無駄にしません)"},
    "cand_trial_hint": {"en": "  (t<N> adopts trial #N directly - surrogate metrics only, no full-generation verification)",
                        "jp": "  (t<N> で試行 #N を直接採用できます — 代理指標のみ・全文生成検証なし)"},
    "cand_trial_disabled": {"en": "  Trial-number adoption is not available in this mode.",
                            "jp": "  このモードでは試行番号による採用はできません。"},
    "cand_trial_not_found": {"en": "  Trial {i} is not in the search history.",
                             "jp": "  試行 {i} は探索履歴に存在しません。"},
    "cand_trial_confirm": {"en": "  Trial #{i}: refusal(surrogate) {r:.3f} / KL {k:.4f} - not verified by full generation",
                           "jp": "  試行 #{i}: refusal(代理) {r:.3f} / KL {k:.4f} — 全文生成検証はしていません"},
    "cand_trial_confirm_yn": {"en": "  Adopt this trial and write the model? [y/N]: ",
                              "jp": "  この試行を採用してモデルを書き出しますか? [y/N]: "},
    "cand_adopted_trial": {"en": "  Adopting trial #{i} (surrogate metrics).",
                           "jp": "  試行 #{i} を採用します (代理指標)。"},
    "int4_exec_line": {"en": "  INT4 execution: packed weights are ~1/3.6 of bf16, dequant cache budget {gb:.1f}GB (full materialization would take {need:.1f}GB; grows automatically with free RAM)",
                       "jp": "  INT4実行: 重みはbf16比 約1/3.6。脱量子化キャッシュ予算 {gb:.1f}GB (全展開 {need:.1f}GB・空きRAMに応じて自動拡大)"},
    "base_cont_note": {"en": "{k} tokens x {n} probes",
                        "jp": "{k}トークン×{n}プローブ"},
    "base_dist_note": {"en": "base distributions (KL reference)",
                       "jp": "基底分布 (KL参照)"},
    "dtype_unsupported": {"en": "Unsupported eval dtype: {dt}",
                          "jp": "サポートされていない評価dtypeです: {dt}"},
}


def set_language(lang: str) -> None:
    """出力言語を設定する ("en" / "jp")。"""
    global _lang
    lang = (lang or "en").lower()
    if lang in ("ja", "jp", "jpn", "japanese", "日本語"):
        _lang = "jp"
    elif lang in ("en", "eng", "english"):
        _lang = "en"
    else:
        _lang = "en"


def get_language() -> str:
    return _lang


def tr(key: str, **fmt) -> str:
    """キーに対応する現在の言語のメッセージを返す (フォーマット失敗時は英語→生キーへ退化)。"""
    entry = _CATALOG.get(key)
    if entry is None:
        return key
    text = entry.get(_lang) or entry.get("en") or key
    if fmt:
        try:
            return text.format(**fmt)
        except Exception:
            return text
    return text


def tr_text(text: str) -> str:
    """テキストがカタログキーなら翻訳、そうでなければそのまま返す (notes等)。"""
    entry = _CATALOG.get(text)
    if entry is None:
        return text
    return entry.get(_lang) or entry.get("en") or text


def is_jp() -> bool:
    return _lang == "jp"
