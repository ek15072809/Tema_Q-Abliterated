# -*- coding: utf-8 -*-
"""プローブ (評価用プロンプト群) の読み込み。

内蔵プローブ (v2: 有害200/無害200件に拡充, 条件④)・テキストファイル・
HuggingFaceデータセットの3種類に対応し、データセットのダウンロードキャッシュも
すべてプロジェクトフォルダ内に置く (条件⑪)。

v2では「間引き」を行わない: 探索用サブセット (n_probe) と最終検証用
100プロンプトの選択は呼び出し側 (cli) で行う。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .msg import tr
from typing import Any

DATA_DIR = Path(__file__).parent / "data"


@dataclass
class ProbeSet:
    harmful: list[str]
    harmless: list[str]
    source_description: str = ""

    def describe(self) -> str:
        return tr("probes_desc", nh=len(self.harmful), ng=len(self.harmless),
                  src=self.source_description)


def _read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _load_builtin() -> tuple[list[str], list[str]]:
    harmful = _read_lines(DATA_DIR / "harmful_probes.txt")
    harmless = _read_lines(DATA_DIR / "harmless_probes.txt")
    return harmful, harmless


def _load_hf_dataset(spec: dict[str, Any], cache_root: Path) -> list[str]:
    """HuggingFaceデータセットからプロンプト列を読む (キャッシュはプロジェクト内)。"""
    dataset = spec["dataset"]
    split = spec.get("split")
    column = spec.get("column", "text")
    limit = int(spec.get("limit", 0) or 0)
    return _hf_dataset_texts(dataset, split, column, limit, cache_root)


# 検証プロンプト列の自動判定に使う列名 (この順で探す)
_HF_COLUMN_CANDIDATES = ("text", "prompt", "instruction", "question", "query")


def _hf_dataset_texts(dataset: str, split: str | None, column: str | None,
                      limit: int, cache_root: Path,
                      info: dict[str, Any] | None = None) -> list[str]:
    """HuggingFaceデータセットからテキスト列を先頭limit件だけ読む。

    [v7] `datasets` ライブラリが無い環境では parquet ファイルを直接
    ダウンロードして読むフォールバックも提供する (依存を最小化)。
    いずれの経路もキャッシュはプロジェクトフォルダ内に置く。
    [v8条件①] split未指定時は test 分割を優先する。実際に使った分割は
    info["split"] に書き戻す (呼び出し側の表示用)。
    """
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))
    try:
        from datasets import load_dataset
    except ImportError as e:
        texts = _hf_dataset_texts_parquet(dataset, split, column, limit, cache_root,
                                          info=info)
        if texts:
            return texts
        raise RuntimeError(tr("datasets_missing", d=dataset)) from e

    ds = load_dataset(dataset, split=split)
    try:
        from datasets import DatasetDict
    except ImportError:  # 古いバージョン互換
        DatasetDict = dict  # type: ignore[assignment,misc]
    if isinstance(ds, DatasetDict):   # split未指定 → 分割を選び直す
        try:
            from datasets import get_dataset_split_names
            sps = list(get_dataset_split_names(dataset))
        except Exception:
            sps = []
        # [v8条件①] split未指定のときは test 分割を優先する
        # (mlabonne/harmful_behaviors のように train/test が分かれた
        # データセットでは、検証用プロンプトとして test 側 (未学習データ)
        # を使う方が過大評価を避けられる)。
        split = ("test" if "test" in sps
                 else ("train" if "train" in sps else (sps[0] if sps else None)))
        ds = load_dataset(dataset, split=split)
    if info is not None:
        info["split"] = split or "(default)"
    cols = list(ds.column_names) if hasattr(ds, "column_names") else []
    use_col = column
    if use_col is None or use_col not in cols:
        for c in _HF_COLUMN_CANDIDATES:
            if c in cols:
                use_col = c
                break
        else:
            use_col = cols[0] if cols else None
    texts: list[str] = []
    if use_col is None:
        return texts
    n = 0
    for row in ds:
        val = row.get(use_col) if hasattr(row, "get") else None
        if val is None:
            val = next(iter(row.values()))
        if isinstance(val, str) and val.strip():
            texts.append(val.strip())
            n += 1
            if limit > 0 and n >= limit:
                break
    return texts


def _pick_split_parquet(files: list[str], split: str | None) -> list[str]:
    """[v8条件①] splitに対応するparquetファイル群を選ぶ (test優先)。

    split未指定の場合は test 分割のparquetを優先し、なければ train、
    どちらもなければ任意のparquetを使う (互換動作)。
    """
    def _matching(sp: str) -> list[str]:
        return [f for f in files if f.endswith(".parquet")
                and f"/{sp}" in f.replace("\\", "/")]

    if split:
        pq = _matching(split)
        if pq:
            return pq
        return [f for f in files if f.endswith(".parquet")]
    for sp in ("test", "train"):
        pq = _matching(sp)
        if pq:
            return pq
    return [f for f in files if f.endswith(".parquet")]


def _hf_dataset_texts_parquet(dataset: str, split: str | None, column: str | None,
                              limit: int, cache_root: Path,
                              info: dict[str, Any] | None = None) -> list[str]:
    """`datasets` 未導入環境向けフォールバック: parquetを直接ダウンロードする。

    huggingface_hub + pyarrow (pandas.read_parquet) が使える場合のみ動作する。
    失敗時は空リストを返す (呼び出し側が datasets の導入を促す)。
    [v8条件①] split未指定時は test 分割 (data/test-*.parquet) を優先する。
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        return []
    try:
        api = HfApi()
        files = api.list_repo_files(dataset, repo_type="dataset")
    except Exception:
        return []
    pq = _pick_split_parquet(files, split)
    if not pq:
        return []
    if info is not None:
        # 実際に選択されたparquetから分割名を類推して報告する
        fname = pq[0].replace("\\", "/")
        for sp in ("test", "train", "validation"):
            if f"/{sp}" in fname:
                info["split"] = sp
                break
        else:
            info["split"] = split or "(default)"
    try:
        local = hf_hub_download(dataset, pq[0], repo_type="dataset",
                                cache_dir=str(cache_root / "datasets"))
    except Exception:
        return []
    try:
        try:
            import pyarrow.parquet as pqd  # type: ignore
            tbl = pqd.read_table(local)
            cols = tbl.column_names
            use_col = column if column in cols else next(
                (c for c in _HF_COLUMN_CANDIDATES if c in cols), cols[0])
            col = tbl.column(use_col).to_pylist()
        except ImportError:
            import pandas as pd  # type: ignore
            df = pd.read_parquet(local)
            cols = list(df.columns)
            use_col = column if column in cols else next(
                (c for c in _HF_COLUMN_CANDIDATES if c in cols), cols[0])
            col = df[use_col].tolist()
    except Exception:
        return []
    texts = [str(v).strip() for v in col if isinstance(v, str) and str(v).strip()]
    if limit > 0:
        texts = texts[:limit]
    return texts


def load_verify_prompts(spec: str, limit: int, cache_root: Path,
                        info: dict[str, Any] | None = None) -> list[str]:
    """[v7条件⑭][v8条件①] 最終検証用プロンプトをHuggingFaceデータセットから読む。

    specの書式: "repo_id" または "repo_id:split" (例:
    "mlabonne/harmful_behaviors" / "mlabonne/harmful_behaviors:test")。
    splitを指定しなかった場合、データセットに test 分割があればそれを
    優先して使う (例: mlabonne/harmful_behaviors なら
    data/test-00000-of-00001.parquet の先頭から limit 件)。
    test がなく train 等しかないデータセットでは従来どおり最初の分割を使う。
    列は text/prompt/instruction/question/query の順に自動判定する。
    """
    spec = spec.strip()
    dataset, split = spec, None
    if spec.count(":") == 1:
        dataset, split = spec.split(":", 1)
    texts = _hf_dataset_texts(dataset, split, None, int(limit), cache_root, info=info)
    if not texts:
        raise RuntimeError(tr("verify_dataset_empty", d=spec))
    return texts[:limit]


def load_probe_set(
    harmful_file: str | None,
    harmless_file: str | None,
    harmful_ds: dict[str, Any],
    harmless_ds: dict[str, Any],
    cache_root: Path,
) -> ProbeSet:
    """設定に従ってプローブセットを構築する (v2: 間引きなし・全件を返す)。"""
    harmful_builtin, harmless_builtin = _load_builtin()
    sources: list[str] = [tr("probes_builtin")]

    harmful: list[str] = []
    harmless: list[str] = []

    if harmful_file:
        harmful = _read_lines(Path(harmful_file))
        sources.append(tr("probes_file", f=harmful_file))
    elif harmful_ds and harmful_ds.get("dataset"):
        harmful = _load_hf_dataset(harmful_ds, cache_root)
        sources.append(tr("probes_dataset", d=harmful_ds["dataset"]))
    else:
        harmful = harmful_builtin

    if harmless_file:
        harmless = _read_lines(Path(harmless_file))
        sources.append(tr("probes_file", f=harmless_file))
    elif harmless_ds and harmless_ds.get("dataset"):
        harmless = _load_hf_dataset(harmless_ds, cache_root)
        sources.append(tr("probes_dataset", d=harmless_ds["dataset"]))
    else:
        harmless = harmless_builtin

    if not harmful or not harmless:
        raise RuntimeError(tr("probes_empty"))

    return ProbeSet(harmful=harmful, harmless=harmless,
                    source_description=" / ".join(sources))


def refusal_marker_list() -> list[str]:
    """拒否判定マーカー (内蔵)。"""
    return _read_lines(DATA_DIR / "refusal_markers.txt")


def is_refusal_text(text: str, markers: list[str]) -> bool:
    """応答テキストが拒否かどうかをマーカー照合で判定する。"""
    t = text.lower().replace("*", "").replace("’", "'").replace("「", "").replace("」", "")
    t = " ".join(t.split())
    if not t.strip():
        return True  # 空応答は悪化とみなす
    return any(m in t for m in markers)
