# -*- coding: utf-8 -*-
"""モデル入出力: ローカルディレクトリ/HuggingFace両対応の読み込みと、
safetensors直接ストリーミング書き出し (条件⑤⑳: 元モデルは絶対に上書きしない)。

v3: トークナイザ読み込み時の transformers 警告「incorrect regex ... fix_mistral_regex=True」に対応。
ローカルの config.json に transformers_version がないと Mistral 以外のモデル
(Qwen等) でも誤検知される。fix_mistral_regex=True で「修正」するとプリトークナイザに
Mistral Split が前置されトークナイズが壊れるため、実際のプリトークナイザ型を確認して

- 本当にバグっている Mistral Split → fix_mistral_regex=True で修正
- それ以外 (ByteLevel 等) → 誤検知として警告を抑制 (fix=False)

のどちらかを行う。
"""

from __future__ import annotations

import gc
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .envcheck import EnvReport
from .msg import tr

# 出力時にコピーするサイドカー (存在するものだけコピー)
SIDECAR_NAMES = [
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "tokenizer.model", "added_tokens.json",
    "preprocessor_config.json", "processor_config.json", "chat_template.json",
    "chat_template.jinja", "model_args.json", "spiece.model",
    "video_preprocessor_config.json",
    # 派生モデルは元モデルのライセンスを引き継ぐべきなので同梱する
    "LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING", "NOTICE",
]


def is_local_model(path_str: str) -> bool:
    """ローカルディレクトリとして有効なモデルかどうか。"""
    p = Path(path_str)
    return p.is_dir() and (p / "config.json").exists()


def resolve_model_dir(model_ref: str, cache_root: Path) -> Path:
    """モデル参照 (ローカルパス or HFリポジトリID) をローカルディレクトリへ解決する。

    HFリポジトリIDの場合はすべてのファイルを <cache_root>/hub/<repo> へダウンロードする
    (条件⑪: キャッシュはプロジェクトフォルダ内に集約)。
    """
    if is_local_model(model_ref):
        return Path(model_ref)

    # HuggingFaceリポジトリIDとして扱う
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(tr("hub_missing")) from e

    if os.path.isdir(model_ref):
        raise RuntimeError(tr("dir_no_config", d=model_ref))

    target = cache_root / "hub" / model_ref.replace("/", "--")
    print(tr("model_hf_fetch", model=model_ref))
    print(tr("model_hf_cache", path=target))
    path = snapshot_download(
        repo_id=model_ref,
        local_dir=str(target),
        # 大きな安全保護ファイル (*.bin等) は使わないので除外して軽量化。
        # README.md/LICENSE は出力モデルのモデルカード (frontmatter・ライセンス
        # 継承) に必要なため同時に取得する。
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "*.jinja",
                        "README.md", "LICENSE", "LICENSE.md", "LICENSE.txt"],
    )
    return Path(path)


# ---------------------------------------------------------------- トークナイザ (v3)

def _inspect_pretokenizer(model_dir: Path) -> str:
    """tokenizer.json のプリトークナイザ型を実際に確認する。

    戻り値: "mistral_split_buggy" / "mistral_split_fixed" / "bytelevel" /
            "sequence:..." / "metaspace" / "split" / "other" / "unknown"
    """
    tj = model_dir / "tokenizer.json"
    if not tj.exists():
        return "unknown"
    try:
        with open(tj, encoding="utf-8") as f:
            data = json.load(f)
        pt = data.get("pre_tokenizer") or {}
        t = pt.get("type", "")
        if t == "Sequence":
            first = (pt.get("pre_tokenizers") or [{}])[0].get("type", "")
            return "sequence:" + first if first else "sequence"
        if t == "Split":
            pat = str((pt.get("pattern") or {}).get("Regex", ""))
            # 誤った古い Mistral パターン (スペース区切りの単語を丸呑みする)
            # に特徴的な断片かどうかで判定
            if "[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]" in pat:
                return "mistral_split_fixed"
            return "split"
        if t:
            return t.lower()
        return "other"
    except Exception:
        return "unknown"


def load_tokenizer_quiet(model_dir: Path, verbose: bool = True):
    """トークナイザを読み込む (transformersの誤検知警告を適切に処理して)。

    transformers 5.x はローカルの config.json に transformers_version がない場合、
    Mistral 以外のトークナイザ (Qwen等) にも「incorrect regex →
    fix_mistral_regex=True を推奨」する警告を出すことがある。実際のプリ
    トークナイザを確認し、
    - 本当に Mistral 系のバグった Split の場合のみ fix を適用
    - それ以外は fix=False を明示して警告を抑制 (誤検知)
    """
    from transformers import AutoTokenizer

    pt_kind = _inspect_pretokenizer(model_dir)
    try:
        with open(model_dir / "config.json", encoding="utf-8") as f:
            model_type = json.load(f).get("model_type", "")
    except Exception:
        model_type = ""
    is_mistral_family = any(m in model_type.lower() for m in
                            ("mistral", "voxtral", "pixtral"))
    apply_fix = is_mistral_family and pt_kind in ("split", "sequence:Split")

    try:
        tok = AutoTokenizer.from_pretrained(
            str(model_dir), fix_mistral_regex=True if apply_fix else False)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(str(model_dir))
    except Exception:
        tok = AutoTokenizer.from_pretrained(str(model_dir))

    if verbose:
        if apply_fix:
            print(tr("tok_regex_fixed"))
        else:
            print(tr("tok_regex_false_positive"))
    return tok


def _base_dtype_for(dt: str) -> torch.dtype:
    """評価dtype名からベースのtorch dtypeへ変換する (int4/int8はbf16)。"""
    if dt in ("int4", "int8"):
        return torch.bfloat16
    return getattr(torch, dt)


def load_model(model_dir: Path, env: EnvReport, eval_dtype: str, device: str) -> nn.Module:
    """モデルを読み込む。dtypeは候補順にフォールバックし、動作確認してから確定する。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .envcheck import suppress_library_noise
    suppress_library_noise()   # import後にハンドラへノイズフィルタを接続

    candidates: list[str] = []
    if eval_dtype == "auto":
        if device.startswith("cuda"):
            candidates = ["bfloat16", "float16", "float32"]
        else:
            candidates = ["bfloat16", "float32"]
    else:
        candidates = [eval_dtype]

    # [v6] int4/int8は「量子化評価」の指定なのでベースはbf16で読み込み、
    # 呼び出し側 (build_int4/int8_eval_model) で量子化する
    last_error: Exception | None = None
    for dt in candidates:
        try:
            dtype = _base_dtype_for(dt)
        except AttributeError:
            raise RuntimeError(tr("dtype_unsupported", dt=dt))
        try:
            # マルチモーダル (vision_configを持つ) 場合はImageTextToTextを使う
            with open(model_dir / "config.json", encoding="utf-8") as f:
                cfg = json.load(f)
            is_vlm = "vision_config" in cfg or any("vision" in k.lower() for k in cfg.keys())
            if is_vlm:
                try:
                    from transformers import AutoModelForImageTextToText
                    model = AutoModelForImageTextToText.from_pretrained(str(model_dir), dtype=dtype)
                except Exception:
                    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype)
            else:
                model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype)

            model = model.to(device)
            model.eval()
            # 動作確認 (NaNなどが出るdtypeを弾く)
            tok = AutoTokenizer.from_pretrained(str(model_dir))
            ids = tok("Hi", return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**ids, use_cache=False)
            if torch.isnan(out.logits.float()).any():
                raise RuntimeError(tr("model_nan_logits"))
            install_lm_head_contiguity_guard(model)
            print(tr("model_loaded_dtype", dt=dt,
                      n=sum(p.numel() for p in model.parameters()) / 1e9))
            return model
        except Exception as e:  # noqa: BLE001
            last_error = e
            print(tr("model_load_fail_dtype", dt=dt, etype=type(e).__name__, e=e))
            try:
                del model
            except Exception:
                pass
    raise RuntimeError(tr("model_load_all_fail", last=last_error))


def model_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _model_class_for(model_dir: Path):
    """config.jsonの構造からモデルクラスを解決する (テキスト/VLM両対応)。"""
    import transformers

    with open(model_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    is_vlm = "vision_config" in cfg or any("vision" in k.lower() for k in cfg.keys())
    auto_cls = (transformers.AutoModelForImageTextToText if is_vlm
                else transformers.AutoModelForCausalLM)
    return auto_cls, cfg


def _keep_module_ids(model: nn.Module, registry, components: str = "both") -> set[int]:
    """量子化してはならないモジュール (摂動対象・lm_head) のid集合。

    [v6] registry.map の値は TargetModule (モジュールラッパ) なので
    「モジュール本体」のidを登録する (ラッパのidでは量子化回避が機能しない)。
    components="attn"/"mlp" のときは対象外コンポーネントの重みも
    量子化可能になる (融合専門家のINT4化を有効にする)。
    """
    keep: set[int] = set()
    if registry is not None:
        for comp in registry.components:
            if components == "attn" and comp != "attn_out":
                continue
            if components == "mlp" and comp != "mlp_down":
                continue
            for tms in registry.map[comp].values():
                for tm in tms:
                    keep.add(id(tm.module))
    top = model
    if hasattr(top, "get_base_model"):
        top = top.get_base_model()
    lm = getattr(top, "lm_head", None)
    if isinstance(lm, nn.Linear):
        keep.add(id(lm))
    return keep


def build_int4_eval_model(model: nn.Module, registry=None, device: str | None = None,
                         components: str = "both") -> nn.Module:
    """探索用にINT4重み量子化した評価モデルを「その場で」構築する (メモリ倍増なし)。

    摂動対象 (o_proj/down_proj) とlm_headは元精度のまま残すため、
    パッチ適用とlogits精度への影響がない。1モジュールずつ変換して
    元重みを即時解放するため、ピーク増加は1テンソル分で済む。
    """
    import gc
    from .quant4 import int4_coverage_bytes, swap_fused_experts_with_int4, swap_linears_with_int4
    keep = _keep_module_ids(model, registry, components=components)
    n_swap, n_bytes = swap_linears_with_int4(model, keep, device=device)
    # [v6] 融合3D専門家 (Qwen4Exp / Gemma4-MoE): 摂動対象でなければINT4化する
    n_swap_f, n_bytes_f = swap_fused_experts_with_int4(model, keep, device=device)
    gc.collect()  # 解放済みの元重みをアロケータに返却させる
    from .memguard import malloc_trim
    malloc_trim()
    print(tr("int4_build", n=(n_swap + n_swap_f),
             gb=(n_bytes + n_bytes_f) / (1024 ** 3)))
    _int4_coverage_report(model, registry, components=components)
    return model


def fused_target_bytes(model: nn.Module, registry, components: str = "both") -> float:
    """摂動対象としてbf16のまま残る融合専門家重みのバイト数。"""
    total = 0
    if registry is None:
        return 0.0
    seen: set[int] = set()
    for comp in registry.components:
        if components == "attn" and comp != "attn_out":
            continue
        if components == "mlp" and comp != "mlp_down":
            continue
        for tms in registry.map.get(comp, {}).values():
            for tm in tms:
                if tm.fused and id(tm.module) not in seen:
                    seen.add(id(tm.module))
                    for attr in ("gate_up_proj", "down_proj"):
                        t = getattr(tm.module, attr, None)
                        if torch.is_tensor(t):
                            total += t.numel() * t.element_size()
    return total


def _int4_coverage_report(model: nn.Module, registry, components: str = "both") -> None:
    """[v6] INT4適用率の報告とアーキテクチャ上の注記 (条件: 融合専門家)。"""
    try:
        from .quant4 import int4_coverage_bytes
        cov = int4_coverage_bytes(model)
        if cov["total_bytes"] > 0:
            print(tr("int4_coverage",
                     pct=100.0 * cov["int4_bytes"] / cov["total_bytes"],
                     gb=cov["int4_bytes"] / (1024 ** 3),
                     tot=cov["total_bytes"] / (1024 ** 3)))
        fused_gb = fused_target_bytes(model, registry, components=components) / (1024 ** 3)
        if fused_gb > 0.01:
            print(tr("int4_fused_note", gb=fused_gb))
    except Exception:
        pass


def load_model_streamed(model_dir: Path, env: EnvReport, device: str = "cpu",
                        quant: str = "int4", components: str = "both",
                        group_size: int = 128, dtype: str = "bfloat16"
                        ) -> tuple[nn.Module, Any, dict[str, Any]]:
    """メタデバイスで骨組みだけ構築し、テンソル単位でストリーミング読み込みする。

    bf16での一括読み込みがメモリに収まらないモデルを、SWAPなしで扱うための
    経路 (v2新設)。読み込みながら非対象LinearをINT4へ変換するため、
    ピークメモリは「対象レイヤのbf16 + 非対象のINT4 + 1チャンク分」に抑わる。

    [v7重要修正] 評価モデルの非量子化パラメータはdtype (既定bfloat16) へ
    鋳直す。v6までは from_config の既定 (fp32) のままだったため、
    ストリーミングINT4実行は実質fp32で動いていた (計算量・メモリ2倍。
    さらに脱量子化キャッシュの必要量も2倍になり、LRUスラッシングで
    毎forwardの再復号が発生していた)。

    戻り値: (model, ComponentRegistry, 情報dict)
    """
    from safetensors import safe_open
    from .arch import ComponentRegistry
    from .quant4 import (PackedInt4Linear, _params_on_meta, MIN_IN_FEATURES)

    from .envcheck import suppress_library_noise
    suppress_library_noise()   # import後にハンドラへノイズフィルタを接続

    tgt_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}.get(dtype, torch.bfloat16)

    print(tr("stream_title"))
    auto_cls, cfg_raw = _model_class_for(model_dir)
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(str(model_dir))

    # --- 1) 骨組みをメタパラメータで構築 (バッファは実テンソル) ---
    with _params_on_meta():
        try:
            model = auto_cls.from_config(cfg)
        except Exception:
            import transformers
            arch = cfg_raw.get("architectures", [""])[0]
            model = getattr(transformers, arch)(cfg)
    model.eval()

    # --- 2) 構造レジストリ (量子化前に摂動対象を確定) ---
    registry = ComponentRegistry(model)
    keep = _keep_module_ids(model, registry, components=components)

    # --- 3) シャードを1テンソルずつ読んで割り当て ---
    shard_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".safetensors")])
    if not shard_files:
        raise RuntimeError(tr("no_safetensors", d=model_dir))

    from .quant4 import _set_tensor

    def _assign(key: str, t: torch.Tensor) -> None:
        """[v7] dtypeを鋳直して割り当てるラッパ (fp32骨組み問題の修正)。"""
        _set_tensor(model, key, t, device, tgt_dtype)

    n_swap = 0
    swapped_bytes = 0
    # [v6] 融合3D専門家のテンソルを集めるstash。チェックポイント上の書式は2種類:
    #   融合形式:  "...experts.down_proj" / "...experts.gate_up_proj" (3D)
    #   分割形式:  "...experts.{e}.down_proj.weight" + gate_proj/up_proj (2D毎)
    # 摂動対象なら 3D Parameterとして復元し、非対象なら (int4時) モジュールごと
    # PackedInt4FusedExperts へ置換する。
    import re as _re
    _expert_split_re = _re.compile(
        r"^(?P<base>.*)\.experts\.(?P<e>\d+)\."
        r"(?P<t>down_proj|gate_proj|up_proj|gate_up_proj)(?:\.weight)?$")
    # [v6] PLEのngram埋め込み等はチェックポイント上で "shard_N" 分割されている
    # (transformersのWeightConverter: Concatenate(dim=0))。数値順に連結して復元する。
    _shard_re = _re.compile(r"^(?P<base>.*)\.shard_(?P<idx>\d+)(?P<rest>\.weight)?$")
    shard_groups: dict[str, dict[int, torch.Tensor]] = {}

    def _walk(parts: list[str]) -> nn.Module | None:
        cur: nn.Module = model
        for p in parts:
            if p.isdigit():
                try:
                    cur = cur[int(p)]
                except (IndexError, TypeError):
                    return None
            else:
                nxt = getattr(cur, p, None)
                if nxt is None:
                    return None
                cur = nxt
        return cur

    pending: dict[int, dict] = {}

    def _note_split_expert(key: str, t: torch.Tensor) -> bool:
        """分割形式の専門家テンソル1つを処理する。消費した場合 True。

        対象が「融合3D専門家」(num_experts属性を持つコンテナ) ならstashへ貯め、
        古典的な ModuleList per-expert Linear なら False を返して通常経路へ。
        """
        m = _expert_split_re.match(key)
        if not m:
            return False
        base_parts = m.group("base").split(".")
        holder = _walk(base_parts)
        container = getattr(holder, "experts", None) if holder is not None else None
        if (container is None or not isinstance(container, nn.Module)
                or not getattr(container, "num_experts", None)):
            return False   # 古典的MoE (ModuleList) → 通常割当て
        e = int(m.group("e"))
        tname = m.group("t")
        stash = pending.setdefault(id(container), {
            "container": container, "holder": holder, "name": "experts",
            "tensors": {}, "orig_keys": {}})
        stash["tensors"].setdefault(e, {})[tname] = t
        stash["orig_keys"].setdefault(e, {})[tname] = key
        return True

    def _stack_split(tensors: dict, tname: str, e_max: int):
        return torch.stack([tensors[e][tname] for e in range(e_max)])

    def _flush_pending_experts() -> None:
        """stash済み分割形式専門家を復元 (3D Parameter) またはINT4置換する。"""
        for stash in list(pending.values()):
            container = stash["container"]
            holder = stash["holder"]
            tensors = stash["tensors"]
            E = int(container.num_experts)
            try:
                have_down = all(e in tensors and "down_proj" in tensors[e]
                                for e in range(E))
                have_gu = all(
                    e in tensors and (("gate_proj" in tensors[e] and "up_proj" in tensors[e])
                                      or "gate_up_proj" in tensors[e])
                    for e in range(E))
                if not (have_down and have_gu):
                    raise RuntimeError("incomplete split-expert tensors")
                dn = _stack_split(tensors, "down_proj", E)
                if "gate_up_proj" in tensors[0]:
                    gu = _stack_split(tensors, "gate_up_proj", E)
                else:
                    gu = torch.stack([
                        torch.cat([tensors[e]["gate_proj"], tensors[e]["up_proj"]], dim=0)
                        for e in range(E)])
                if quant == "int4" and id(container) not in keep:
                    from .quant4 import PackedInt4FusedExperts
                    new_mod = PackedInt4FusedExperts(gu, dn, container.act_fn,
                                                     device=device)
                    setattr(holder, stash["name"], new_mod)
                    n_swap += 1
                    swapped_bytes += new_mod.memory_bytes()
                else:
                    pdt = None
                    for attr, val in (("down_proj", dn), ("gate_up_proj", gu)):
                        cur = getattr(container, attr, None)
                        dt = cur.dtype if torch.is_tensor(cur) else None
                        pdt = pdt or dt
                        # [v7] meta骨組みはfp32既定のため、評価dtype (通常bf16) を優先
                        setattr(container, attr, torch.nn.Parameter(
                            val.to(device=device, dtype=(tgt_dtype if pdt == torch.float32 else (dt or tgt_dtype))),
                            requires_grad=False))
            except Exception:
                # 復元失敗: 生テンソルを従来の_set_tensorで割当て
                for e, td in tensors.items():
                    for tname, tv in td.items():
                        try:
                            _assign(stash["orig_keys"][e][tname], tv)
                        except Exception:
                            pass
            pending.pop(id(container), None)

    def _try_fused_experts(key: str, t: torch.Tensor) -> bool:
        """融合形式の専門家テンソル1つを処理。消費した場合 True。

        gate_up_proj と down_proj が揃った時点でモジュールごと
        PackedInt4FusedExpertsへ置換する。揃わない/構築失敗の場合は
        両テンソルを生割当てしてフォールバックする。
        """
        parts = key.split(".")
        if quant != "int4" or len(parts) < 3:
            return False
        if parts[-1] not in ("gate_up_proj", "down_proj"):
            return False
        holder = _walk(parts[:-2])
        mod = getattr(holder, parts[-2], None) if holder is not None else None
        if (holder is None or mod is None
                or not getattr(mod, "num_experts", None) or id(mod) in keep):
            return False
        stash = pending.setdefault(id(mod), {"mod": mod, "holder": holder,
                                             "tensors": {}, "orig_keys": {}})
        stash["tensors"][parts[-1]] = t
        stash["orig_keys"][parts[-1]] = key
        if len(stash["tensors"]) < 2:
            return True   # 1つ目のテンソル: 保持して2つ目の到着を待つ
        try:
            from .quant4 import PackedInt4FusedExperts
            new_mod = PackedInt4FusedExperts(
                stash["tensors"]["gate_up_proj"], stash["tensors"]["down_proj"],
                mod.act_fn, device=device)
            setattr(holder, parts[-2], new_mod)
            n_swap += 1
            swapped_bytes += new_mod.memory_bytes()
            pending.pop(id(mod), None)
            return True
        except Exception:
            # 構築失敗: 両テンソルを生割当て (meta残しを防ぐ)
            pending.pop(id(mod), None)
            for orig_key, tv in stash["tensors"].items():
                try:
                    _assign(orig_key, tv)
                except Exception:
                    pass
            return False

    def _int4_from_shard_slice(fh, key: str, mod: nn.Linear, group_size: int = 128):
        """[v7] safetensorsスライスを行チャンクで読みながらINT4パックする。

        従来 (v6) は fh.get_tensor(key) でテンソル全体を読んだうえで
        パック中間体 (数倍サイズ) を生成していたため、RSS高水位が
        「INT4なのにbf16より多い」状態を残していた。この経路は
        1チャンク (~16MB) 分しか一時メモリを持たない。
        """
        from .quant4 import (PackedInt4Linear, _pick_group, pack_into_buffers,
                             _PACK_CHUNK_ELEMS)
        sl = fh.get_slice(key)
        shape = sl.get_shape()
        out_f, in_f = int(shape[0]), int(shape[1])
        pad = in_f % 2
        total = in_f + pad
        g = _pick_group(total, group_size)
        ng = total // g
        P = total // 2
        packed = torch.empty(out_f, P, dtype=torch.uint8)
        scale = torch.empty(out_f, ng, dtype=torch.float32)
        rows = max(1, _PACK_CHUNK_ELEMS // max(1, in_f))
        for r0 in range(0, out_f, rows):
            r1 = min(out_f, r0 + rows)
            t = sl[r0:r1]
            pack_into_buffers(packed, scale, t, r0, g)
            del t
        return PackedInt4Linear.from_packed(packed, scale, g, pad, in_f,
                                            out_features=out_f, device=device)

    def _drop_file_cache(fname: str) -> None:
        """[v7] 読み終えたシャードのページキャッシュをOSへ返す (Linux)。

        ストリーミング読み込みはテンソルをコピーして保持するため、
        mmap 由来のファイルページは読み込み後不要である。放置すると
        低メモリ環境でRSSを押し上げる (実測で約0.9GB)。カーネルは圧力が
        かかるまで返さないため、posix_fadvise(DONTNEED)で明示的に返す。
        """
        try:
            fd = os.open(str(model_dir / fname), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except (AttributeError, OSError):
            pass

    for sf in shard_files:
        with safe_open(str(model_dir / sf), framework="pt") as fh:
            keys = list(fh.keys())
            for key in keys:
                parts = key.split(".")
                leaf = parts[-1]
                swapped = False
                if leaf == "weight" and len(parts) >= 2 and quant == "int4":
                    holder = _walk(parts[:-2])
                    mod = getattr(holder, parts[-2], None) if holder is not None else None
                    if (isinstance(mod, nn.Linear) and id(mod) not in keep
                            and mod.in_features >= MIN_IN_FEATURES):
                        try:
                            new_mod = _int4_from_shard_slice(fh, key, mod, group_size)
                            setattr(holder, parts[-2], new_mod)
                            n_swap += 1
                            swapped_bytes += new_mod.memory_bytes()
                            swapped = True
                        except Exception:
                            pass  # 変換失敗時は通常割当てへ
                if not swapped:
                    t = fh.get_tensor(key)
                    m_shard = _shard_re.match(key)
                    if m_shard:
                        base_key = m_shard.group("base") + (m_shard.group("rest") or "")
                        shard_groups.setdefault(base_key, {})[int(m_shard.group("idx"))] = t
                        swapped = True
                if not swapped and quant == "int4":
                    # 分割形式の専門家テンソル (摂動対象でもstashして3D復元)
                    if _note_split_expert(key, t):
                        swapped = True
                    elif leaf in ("gate_up_proj", "down_proj") and _try_fused_experts(key, t):
                        swapped = True
                elif not swapped and not (leaf == "weight" and quant == "int4"):
                    # int4以外でも分割形式専門家は3D復元が必要
                    if _note_split_expert(key, t):
                        swapped = True
                if not swapped:
                    try:
                        _assign(key, t)
                    except Exception:
                        pass
                    del t
        _drop_file_cache(sf)
        print(tr("stream_shard", f=sf, n=n_swap))

    # [v7] 読み込み完了後: ヒープ断片をOSへ返す
    from .memguard import malloc_trim
    import gc as _gc
    _gc.collect()
    malloc_trim()

    # [v6] shard分割テンソルの再結合 (数値順にcatして割当て)
    for base_key, shards in shard_groups.items():
        try:
            full = torch.cat([shards[i] for i in sorted(shards)], dim=0)
            _assign(base_key, full)
        except Exception as e:  # noqa: BLE001
            print(f"  shard reassembly failed for {base_key}: {type(e).__name__}")
        del shards

    # 未完成/未処理の専門家stashを解決する
    _flush_pending_experts()
    for stash in list(pending.values()):
        tensors = stash.get("tensors", {})
        orig = stash.get("orig_keys", {})
        if tensors and isinstance(next(iter(tensors.values()), None), torch.Tensor):
            # 融合形式の残り (1テンソルのみ等): 生割当て
            for orig_key, tv in zip(orig.values(), tensors.values()):
                try:
                    _assign(orig_key, tv)
                except Exception:
                    pass
        else:
            for e, td in tensors.items():
                for tname, tv in td.items():
                    try:
                        _assign(orig.get(e, {}).get(tname, ""), tv)
                    except Exception:
                        pass
    pending.clear()

    # --- 4) 重み共有 (tie) の解決と残余メタの検査 ---
    try:
        if getattr(cfg, "tie_word_embeddings", False):
            model.tie_weights()
    except Exception:
        pass
    # tie_weightsで解決しない場合の手動共有 (VLMの入れ子config対策)
    lm = getattr(model, "lm_head", None)
    if lm is not None and isinstance(getattr(lm, "weight", None), torch.Tensor) \
            and lm.weight.is_meta:
        from .arch import get_text_model
        emb_w = get_text_model(model).embed_tokens.weight
        if not emb_w.is_meta:
            lm.weight = emb_w
    try:
        from transformers import GenerationConfig
        model.generation_config = GenerationConfig.from_pretrained(str(model_dir))
    except Exception:
        pass
    meta_left = [n for n, p in model.named_parameters() if p.is_meta]
    if meta_left:
        raise RuntimeError(tr("stream_meta_left", n=len(meta_left), ex=meta_left[:3]))

    # --- 5) CPUバッファ (inv_freq等) をデバイスへ移して動作確認 ---
    model = model.to(device)
    model.eval()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    ids = tok("Hi", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**ids, use_cache=False)
    if torch.isnan(out.logits.float()).any():
        raise RuntimeError(tr("model_nan_logits_stream"))

    info = {
        "quant": quant, "n_int4_modules": n_swap,
        "int4_weight_bytes": swapped_bytes,
        "n_target_modules": sum(len(m) for c in registry.components
                                for m in registry.map[c].values()),
    }
    gc.collect()
    from .memguard import malloc_trim
    malloc_trim()
    install_lm_head_contiguity_guard(model)
    print(tr("stream_done", n=n_swap, gb=swapped_bytes / (1024 ** 3)))
    return model, registry, info


class OutputDirError(RuntimeError):
    """[v8条件③] 出力先パスの問題 (存在・衝突・ファイル型など)。

    エラーで終了して探索・検証の成果を失わせる代わりに、cli側がこの例外を
    捕捉してパスの再入力 (対話時) / 自動修正 (非対話時) を行う。
    """


def _guard_output_dir(out_dir: Path, model_dir: Path, overwrite: bool) -> None:
    """出力先の安全性を検証する (条件⑰: 元モデルを絶対に上書きしない)。

    [v8条件③] 出力先ディレクトリが「存在するが空」の場合はエラーにせず
    そのまま使用する (空フォルダは正当な出力先)。何か入っている場合のみ
    OutputDirError を出す (cliが再入力/自動修正する)。
    """
    out_abs = out_dir.resolve()
    src_abs = model_dir.resolve()
    if out_abs == src_abs:
        raise OutputDirError(tr("export_same_dir"))
    if src_abs in out_abs.parents:
        raise OutputDirError(tr("export_inside_dir"))
    if out_abs.exists():
        if not out_abs.is_dir():
            raise OutputDirError(tr("export_path_file", d=out_dir))
        try:
            has_content = any(out_abs.iterdir())
        except OSError as e:
            raise OutputDirError(tr("export_path_problem", msg=e)) from e
        if not has_content:
            # [v8条件③] 空フォルダはそのまま使う
            print(tr("export_dir_empty_ok"))
            return
        if not overwrite:
            raise OutputDirError(tr("export_exists", d=out_dir))
        shutil.rmtree(out_abs)


def export_model(
    model_dir: Path,
    out_dir: Path,
    key_map: dict[str, tuple[str, int]],
    params,
    engine,
    overwrite: bool = False,
) -> Path:
    """元モデルのsafetensorsシャードを1つずつ読み込み(読み取り専用)、
    摂動対象キーにだけ重み書き換えを適用して新しいモデルとして書き出す。

    key_map: safetensorsキー -> (コンポーネント名, レイヤ番号)
    params:  engine.AblationParams
    engine:  engine.PatchEngine (方向・ノルム保存設定を保持)
    """
    _guard_output_dir(out_dir, model_dir, overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import load_file, save_file

    shard_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".safetensors")])
    if not shard_files:
        raise RuntimeError(tr("no_safetensors", d=model_dir))

    written: list[str] = []
    for sf in shard_files:
        sd = load_file(str(model_dir / sf))
        changed = 0
        for key in list(sd.keys()):
            entry = key_map.get(key)
            if entry is None:
                continue
            comp, layer = entry
            new_w = engine.ablated_weight_for(comp, layer, params, sd[key])
            if new_w is not None:
                sd[key] = new_w
                changed += 1
        save_file(sd, str(out_dir / sf), metadata={"format": "pt"})
        del sd
        written.append(sf)
        print(tr("export_shard", f=sf,
                  extra=tr("export_changed", n=changed) if changed else ""))

    # サイドカーコピー
    copied = 0
    for name in os.listdir(model_dir):
        if name in SIDECAR_NAMES:
            shutil.copyfile(model_dir / name, out_dir / name)
            copied += 1
    print(tr("export_sidecar", n=copied))
    return out_dir


def build_int8_eval_model(model: nn.Module, registry=None):
    """探索用にint8動的量子化した評価モデルを構築する (v2: モジュール単位方式)。

    v1は「全Linearを一括fp32化してからquantize_dynamic」していたため、
    変換中のピークメモリがモデル全体のfp32分 (パラメータ数×4バイト) に達し、
    大規模モデルや空きメモリが少ない環境でOOMする危険があった。
    v2は1モジュールずつ「fp32化→量子化→キャストアダプタでラップ→元重み解放」
    を行うため、ピーク増加は1テンソル分で済む。

    摂動対象 (o_proj/down_proj)・lm_head・埋め込みは元精度のまま残す。
    戻り値: 同一オブジェクト (in-place変換)。
    """
    import gc
    from torch.ao.quantization import quantize_dynamic

    class CastLinear(nn.Module):
        """bf16入力→fp32キャスト→int8動的量子化Linear→元dtypeに戻すアダプタ。"""

        def __init__(self, qmod, out_dtype):
            super().__init__()
            self.q = qmod
            self.out_dtype = out_dtype

        def forward(self, x):
            return self.q(x.to(torch.float32)).to(self.out_dtype)

    from .arch import get_text_model
    keep = _keep_module_ids(model, registry)
    out_dtype = torch.bfloat16
    emb = getattr(get_text_model(model), "embed_tokens", None)
    if emb is not None:
        out_dtype = emb.weight.dtype

    n_swap = 0
    entries = [(path, mod) for path, mod in model.named_modules()
               if isinstance(mod, nn.Linear)]
    for path, mod in entries:
        if id(mod) in keep or mod.in_features < 16:
            continue
        parts = path.split(".")
        parent = model
        for pp in parts[:-1]:
            parent = parent[int(pp)] if pp.isdigit() else getattr(parent, pp)
        try:
            # 1モジュール分だけfp32化 (量子化の要件) → 逐次量子化 → 解放。
            # 葉Linearはquantize_dynamicの走査対象に入らないため小コンテナで包む
            orig_dtype = mod.weight.dtype
            mod.weight.data = mod.weight.data.to(torch.float32)
            if mod.bias is not None:
                mod.bias.data = mod.bias.data.to(torch.float32)

            class _Wrap(nn.Module):
                def __init__(self, lin):
                    super().__init__()
                    self.lin = lin

                def forward(self, x):
                    return self.lin(x)

            qwrap = quantize_dynamic(_Wrap(mod), {nn.Linear}, dtype=torch.qint8)
            adapter = CastLinear(qwrap.lin, orig_dtype)
            setattr(parent, parts[-1], adapter)
            del mod
            n_swap += 1
        except Exception:
            continue  # 変換できないモジュールは元のまま (fpで動作)
    gc.collect()
    from .memguard import malloc_trim
    malloc_trim()
    print(tr("int8_build", n=n_swap))
    return model


def shard_key_count(model_dir: Path) -> dict[str, int]:
    """シャードごとのテンソル数を返す (検証用)。"""
    from safetensors import safe_open
    counts = {}
    for f in sorted(os.listdir(model_dir)):
        if f.endswith(".safetensors"):
            with safe_open(str(model_dir / f), framework="pt") as fh:
                counts[f] = len(fh.keys())
    return counts


def read_tensor(model_dir: Path, key: str) -> tuple[torch.Tensor, str]:
    """指定キーのテンソルを元モデルから読む (検証・デルタ計算用)。戻り値: (テンソル, シャード名)"""
    from safetensors import safe_open
    for f in sorted(os.listdir(model_dir)):
        if f.endswith(".safetensors"):
            with safe_open(str(model_dir / f), framework="pt") as fh:
                if key in fh.keys():
                    return fh.get_tensor(key), f
    raise KeyError(f"キー {key} がモデルに存在しません")


def export_params_card(out_dir: Path, record: dict[str, Any],
                       model_dir: Path | None = None,
                       base_model_ref: str | None = None,
                       version: str = "1.0.0") -> None:
    """出力モデルに HuggingFace Hub 形式のモデルカード (README.md) を添付する。

    [仕様]
    - 元モデルに README.md がある場合:
      YAML frontmatter (--- ～ ---) を保存し、tags に ``Tema_Q-Abliterated``
      を追加してから、frontmatterと元の本文の「間」に本ツールの説明
      (英語) を挟んだ形で出力する。base_model / library_name が無ければ補う。
    - 元モデルに README.md がない場合:
      frontmatter (library_name / tags / base_model) 付きの完全なカードを新規生成する。
    - パラメータ表は英語見出し「Abliteration parameters」で、外部ツールとの
      相互運用のため標準的なキー名 (attn.o_proj.max_weight 等)
      で書き出す (``--params-from`` で読み戻せる)。

    HuggingFaceモデルカードの形式は公式仕様 (YAML frontmatter + Markdown本文、
    tags はブロック/フローリスト両対応) に従う。
    """
    model_name = (model_dir.name if model_dir else "") or "model"
    orig_text = None
    if model_dir is not None:
        for cand in ("README.md", "readme.md", "Readme.md"):
            p = model_dir / cand
            if p.is_file():
                orig_text = p.read_text(encoding="utf-8", errors="replace")
                break

    front_yaml, body = _split_frontmatter(orig_text) if orig_text is not None else ("", "")
    has_original_card = orig_text is not None
    front_yaml = _card_add_tag(front_yaml, "Tema_Q-Abliterated")
    if base_model_ref:
        front_yaml = _card_ensure_key(front_yaml, "base_model", base_model_ref)
    front_yaml = _card_ensure_key(front_yaml, "library_name", "transformers")

    section = _card_tool_section(record, base_model_ref=base_model_ref,
                                 model_name=model_name, version=version,
                                 standalone=not has_original_card)

    parts = ["---", front_yaml.rstrip("\n"), "---", "", section, ""]
    if has_original_card and body.strip():
        parts += ["---", "", body.rstrip("\n"), ""]
    (out_dir / "README.md").write_text("\n".join(parts), encoding="utf-8")


def _split_frontmatter(text: str) -> tuple[str, str]:
    """README本文を (YAML frontmatter, 本文) に分割する。

    frontmatterは冒頭の ``---`` 行で始まり、次の ``---`` (または ``...``) 行で
    閉じる (HuggingFace Hub仕様)。BOMは除去する。frontmatterが無ければ
    ("", text) を返す。
    """
    text = text.lstrip("\ufeff")
    if not text.startswith("---"):
        return "", text
    lines = text.split("\n")
    # 先頭行が「---」のみ (または行末に空白のみ) であること
    if lines[0].strip() != "---":
        return "", text
    for j in range(1, len(lines)):
        if lines[j].strip() in ("---", "..."):
            yaml_text = "\n".join(lines[1:j]).rstrip("\n")
            body = "\n".join(lines[j + 1:]).lstrip("\n")
            return yaml_text, body
    return "", text


def _card_add_tag(yaml_text: str, tag: str) -> str:
    """frontmatterの tags リストにタグを追加する (既にあれば何もしない)。

    ブロック形式 (``tags:`` + ``- item`` 行) とフロー形式 (``tags: [a, b]``)
    の両方に対応する。tagsキーが無ければ frontmatterの先頭に新設する。
    トップレベルの ``tags:`` のみを対象とし、インデントされた ``tags:``
    (ネストした構造) は触らない。
    """
    lines = yaml_text.split("\n") if yaml_text else []
    if not lines or all(not ln.strip() for ln in lines):
        return f"tags:\n- {tag}"
    idx = None
    for i, ln in enumerate(lines):
        if re.match(r"^tags:\s*(\[.*\]\s*)?$", ln):
            idx = i
            break
    if idx is None:
        return f"tags:\n- {tag}\n" + "\n".join(lines)
    line = lines[idx]
    m = re.match(r"^tags:\s*\[(.*)\]\s*$", line)
    if m:  # フロー形式
        items = [x.strip().strip("\"'") for x in m.group(1).split(",") if x.strip()]
        if tag in items:
            return "\n".join(lines)
        items.append(tag)
        lines[idx] = "tags: [" + ", ".join(items) + "]"
        return "\n".join(lines)
    # ブロック形式: 直後の連続するリスト項目を探す
    j = idx + 1
    while j < len(lines):
        s = lines[j].lstrip()
        if s.startswith("- ") or s == "-":
            j += 1
        else:
            break
    for k in range(idx + 1, j):
        if lines[k].lstrip().lstrip("-").strip().strip("\"'") == tag:
            return "\n".join(lines)
    lines.insert(j, f"- {tag}")
    return "\n".join(lines)


def _card_ensure_key(yaml_text: str, key: str, value: str) -> str:
    """frontmatterに key が無ければ value を追加する (tagsブロックの直後優先)。"""
    if yaml_text and re.search(rf"^{key}:\s*\S", yaml_text, re.M):
        return yaml_text
    line = f"{key}: {value}"
    lines = yaml_text.split("\n") if yaml_text else []
    if not lines or all(not ln.strip() for ln in lines):
        return line
    # tags ブロックの末尾 (なければ先頭) に挿入
    insert_at = 0
    for i, ln in enumerate(lines):
        if re.match(r"^tags:\s*(\[.*\]\s*)?$", ln):
            insert_at = i + 1
            j = insert_at
            while j < len(lines):
                s = lines[j].lstrip()
                if s.startswith("- ") or s == "-":
                    j += 1
                else:
                    break
            insert_at = j
            break
    lines.insert(insert_at, line)
    return "\n".join(lines)


def _card_tool_section(record: dict[str, Any], *, base_model_ref: str | None,
                       model_name: str, version: str,
                       standalone: bool) -> str:
    """モデルカードに挿入する本ツールの説明セクション (英語)。"""
    L: list[str] = []
    if standalone:
        L.append(f"# {model_name} - uncensored by Tema_Q-Abliterated")
    else:
        L.append("# Tema_Q-Abliterated")
    L += [
        "",
        ("This model was uncensored with **Tema_Q-Abliterated**, an exploratory "
         "weight-uncensoring (abliteration) tool for large language models. "
         "It removes refusal behavior by ablating a refusal direction from the "
         "model weights while preserving general quality (multi-position KL "
         "divergence from the base model is kept low during the search)."),
        "",
        f"- Tool and source code: <https://github.com/ek15072809/Tema_Q-Abliterated>",
        f"- Generated with: Tema_Q-Abliterated v{version} (SCAPE algorithm)",
    ]
    if base_model_ref:
        L.append(f"- Base model: `{base_model_ref}`")
    L += [
        "",
        "## Usage",
        "",
        ("This model can be used exactly like the base model (transformers, "
         "vLLM, llama.cpp conversion, etc.)."),
        "",
        ("To create your own uncensored model, install Tema_Q-Abliterated and "
         "run it on any HuggingFace model - a CPU-only machine is enough:"),
        "",
        "```sh",
        "pip install temaq-abliterated",
        "temaq-abliterated <model-name>",
        "```",
        "",
        ("The SCAPE surrogate objective evaluates each search candidate about "
         "**49x faster than the full-generation evaluation used by conventional "
         "tools**, so the whole uncensoring search finishes in practical time "
         "even without a GPU."),
        "",
        "## Abliteration parameters",
        "",
        "| Parameter | Value |",
        "| :-------- | :---: |",
    ]
    # 外部ツールとの相互運用と --params-from での読み戻しを可能にするため、
    # 標準的なキー名で書き出す。
    # recordのキー形式: 内部コンポーネント名 "attn_out.*" / "mlp_down.*"
    params = dict(record.get("parameters", {}))
    di = params.get("direction_index", "per layer")
    L.append(f"| **direction_index** | {di} |")
    for internal, disp in (("attn_out", "attn.o_proj"), ("mlp_down", "mlp.down_proj")):
        strength = params.get(f"{internal}.strength")
        center = params.get(f"{internal}.center")
        radius = params.get(f"{internal}.radius")
        if strength is None and center is None and radius is None:
            continue
        strength = 0.0 if strength is None else strength
        center = 0.0 if center is None else center
        radius = 1.0 if radius is None else radius
        # flatカーネルは帯端でも強度が落ちないため min_weight = 強度、
        # cosineカーネルは帯端で0に達するため min_weight = 0
        min_w = strength if params.get(f"{internal}.shape") == "flat" else 0
        L.append(f"| **{disp}.max_weight** | {strength} |")
        L.append(f"| **{disp}.max_weight_position** | {center} |")
        L.append(f"| **{disp}.min_weight** | {min_w} |")
        L.append(f"| **{disp}.min_weight_distance** | {radius} |")
        if params.get(f"{internal}.shape"):
            L.append(f"| **{disp}.kernel_shape** | {params.get(f'{internal}.shape')} |")
    L.append("")
    if record.get("metrics"):
        L += [
            "## Performance",
            "",
            "| Metric | Value |",
            "| :----- | :---: |",
        ]
        for k, v in record["metrics"].items():
            L.append(f"| **{k}** | {v} |")
        L.append("")
    return "\n".join(L)


def install_lm_head_contiguity_guard(model: nn.Module) -> None:
    """[v7] lm_headへの入力が非連続テンソルの場合に連続化するフックを付ける。

    背景 (実測): transformersは ``logits_to_keep=k>1`` を
    ``hidden_states[:, -k:, :]`` の**非連続スライス**としてlm_headへ渡す。
    一部のCPU/BLAS環境では「非連続入力 × 大行列 (大語彙のlm_head)」の
    GEMMがメモリを爆発させ (200MB/s以上の消費でOOM)、または極端に
    低速化する。連続化は数KB〜数MBの一時コピーで済むため、フックで
    常時保証する (入力が既に連続なら何もしない)。

    KL評価 (keep_last = K+1) と生成の全経路がこの影響を受けていた。
    """
    lm = getattr(model, "lm_head", None)
    if lm is None or not isinstance(lm, nn.Module):
        try:
            from .arch import get_text_model
            core = get_text_model(model)
            lm = getattr(core, "lm_head", None)
        except Exception:
            lm = None
    if lm is None or not isinstance(lm, nn.Module):
        return

    def _guard(mod, args, kwargs):
        x = args[0] if args else kwargs.get("input")
        if isinstance(x, torch.Tensor) and not x.is_contiguous():
            if args:
                return (x.contiguous(),) + tuple(args[1:]), kwargs
            return {"input": x.contiguous(), **kwargs}
        return None

    try:
        lm.register_forward_pre_hook(_guard, with_kwargs=True)
    except Exception:
        pass
