# -*- coding: utf-8 -*-
"""temaq_abliterated: LLMの方向指向削除(非検閲化)を探索的に実行するツール (v1.1.0)。

SCAPE (Structured Cached Abliteration with Prefix Evaluation) と呼ぶ
独自の効率化探索アルゴリズムによって、GPUがないCPU環境でも
(INT4量子化・メモリガード・多位置KL・候補選択UI搭載)、
実用的な速度でウェイト空間上の最適な非検閲化パラメータを探索する。

v1.1.0 の変更 (既存機能はそのまま維持):
  - 新規アーキテクチャ対応:
      * XHToken/Spark-X2.6-4B (auto_mapリモートコード: trust_remote_code読み込み、
        リモートコード (*.py) のダウンロードと出力モデルへの同梱)
      * deepseek-ai/DeepSeek-V4.1-Flash-1017 (分组低ランク出力射影 o_b_proj、
        複数形の共有専門家 mlp.shared_experts、mHC 4Dストリーム隠れ状態)
      * LiquidAI/LFM2.5-24B-A2B (feed_forward直下の融合3D専門家)
  - 出力モデルのモデルカード (README.md) の base_model は「元モデル」を
    常に指すよう変更 (元カードが別の上位モデルを指していても上書き)

v1.0.0 の主な機能:
  - シードグリッドによる構造化初期化: 強度×帯幅スペクトル (基底 /
    全層一様flatカーネル×3強度 / 単一コンポーネント / ピーク狭帯) を
    探索冒頭で評価し、Arditi et al. の全層直交化解を起点に含める
  - 代理指標の文法床の実測と実効目標の較正: 質量加重多位置ソフト拒否率は
    完全に非検閲化されたモデルでも ~0.3-0.4 の床を持つ (基底継続が拒否文
    である限り文法的に拒否語彙の質量が残るため)。床を実測して
    target_eff = max(r*, 床+0.03) とし J を再定義
  - KernelParams の flat (帯内一様) カーネル形状
  - 検証ゲートは床のない初頭トークン比 (refusal_first) (--verify-gate)。
    全候補スキップ時はモデルを無断で書き出さずユーザーの判断を仰ぐ
  - --select auto 指定時を除き、モデル書き出し前に必ず最終候補の選択を
    経由 (t<N> で試行番号の直接採用も可能)
  - 同一モデル再実行時はロード前に再開方法を選べる (チェックポイント)。
    書き出しモードは軽量評価器で基底継続生成を省略
  - ワークスペースは既定で ./tq_workspace/<モデル名>/ に構築され、
    キャッシュ・チェックポイント・レポート・出力が集約される
  - 書き出したモデルに HuggingFace Hub 形式のモデルカード (README.md、
    frontmatterのtagsにTema_Q-Abliteratedを追加) を添付し、
    対話的にHuggingFace Hubへアップロードできる (--hf-repo/--hf-token)
"""

__version__ = "1.1.0"
__all__ = ["cli"]
