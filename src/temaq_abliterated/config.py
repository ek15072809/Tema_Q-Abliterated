# -*- coding: utf-8 -*-
"""設定管理: CLI引数とTOML設定ファイルを統合し、実行設定オブジェクトを構築する。"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # Python 3.10 以前
    import tomli as tomllib  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    """コマンドライン引数パーサを構築する。"""
    p = argparse.ArgumentParser(
        prog="temaq-abliterated",
        usage="temaq-abliterated <model> [options]",
        description="temaq-abliterated: exploratory weight-uncensoring tool for LLMs (SCAPE algorithm)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("model", nargs="?", help="Target model: local directory path or HuggingFace repo id")
    p.add_argument("--config", default=None, help="Path to a TOML settings file (optional)")
    p.add_argument("--workdir", default=None,
                   help="Working directory for caches, checkpoints, reports and output "
                        "(default: ./tq_workspace/<model-name>, created relative to the "
                        "directory you launch the command from)")
    p.add_argument("--output", default=None, help="Output directory of the uncensored model (auto if omitted)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Execution device")
    p.add_argument("--eval-dtype", default="auto",
                   choices=["auto", "bfloat16", "float16", "float32", "int8", "int4"],
                   help="Evaluation dtype for the search phase (auto = planned from free memory; "
                        "int4/int8 quantized evaluation saves RAM/VRAM; "
                        "int4 additionally enables streaming load for models that do not fit in bulk)")
    p.add_argument("--dequant-cache-gb", type=float, default=-1.0,
                   help="INT4 weight cache budget in GB (negative = auto, safe value from free memory)")
    p.add_argument("--language", default="en", choices=["en", "jp"],
                   help="Output message language (English by default; jp = Japanese)")
    p.add_argument("--time-budget-min", type=float, default=0.0,
                   help="Soft time budget for the search phase in minutes (0=unlimited). "
                        "When reached, the search stops at the best-so-far point and "
                        "proceeds to final verification")
    p.add_argument("--trials", type=int, default=0, help="Number of search trials (0 = auto)")
    p.add_argument("--target-refusal", type=float, default=0.03, help="Target refusal rate of the surrogate metric (soft)")
    p.add_argument("--penalty", type=float, default=8.0, help="Penalty coefficient lambda for exceeding the target")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--harmful-file", default=None, help="Harmful prompt file, one prompt per line (built-in set or TOML if omitted)")
    p.add_argument("--harmless-file", default=None, help="Harmless prompt file, one prompt per line")
    p.add_argument("--verify-dataset", default=None,
                   help="HuggingFace dataset for final-verification prompts (uses the first "
                        "--verify-probes rows). Examples: 'mlabonne/harmful_behaviors' or "
                        "'mlabonne/harmful_behaviors:test' (split after a colon)")
    p.add_argument("--n-probe", type=int, default=0, help="Number of probes for exploration (0 = auto, default 64)")
    p.add_argument("--no-probe-hardening", action="store_true",
                   help="Disable harmful-probe hardening: by default, probes the baseline "
                        "model does not refuse to are dropped from direction computation "
                        "and search evaluation. Use this flag to keep all probes")
    p.add_argument("--kl-tokens", type=int, default=8,
                   help="Teacher-forced tokens for multi-position KL (0 = legacy first-token only)")
    p.add_argument("--verify", default="auto", choices=["auto", "full", "none"],
                   help="Final verification by full generation (auto = verify top candidates)")
    p.add_argument("--verify-probes", type=int, default=0,
                   help="Number of harmful probes for verification (0 = auto, 100 prompts)")
    p.add_argument("--verify-topk", type=int, default=5,
                   help="Number of top candidates to verify by full generation (more = accurate but slower)")
    p.add_argument("--verify-gate", type=float, default=0.4,
                   help="Skip full-generation verification for candidates whose first-token "
                        "surrogate refusal exceeds this value (they cannot reach the target "
                        "anyway). The mass-weighted surrogate has a grammatical floor of "
                        "~0.3-0.4 on fully uncensored models, so the gate uses the "
                        "floorless first-token ratio")
    p.add_argument("--max-new-tokens", type=int, default=64, help="Generated tokens for verification")
    p.add_argument("--norm-preserve", default="on", choices=["on", "off"], help="Enable row-norm-preserving orthogonalization")
    p.add_argument("--components", default="both", choices=["both", "attn", "mlp"], help="Components to explore")
    p.add_argument("--select", default="interactive", choices=["interactive", "auto"],
                   help="Final candidate selection (interactive = always choose from a list "
                        "before writing the model, auto = pick the best automatically; "
                        "falls back to auto in non-interactive environments)")
    p.add_argument("--params-from", default=None,
                   help="Path or URL of an external model card (README.md) to load Abliteration parameters from")
    p.add_argument("--params-mode", default="focus", choices=["apply", "focus"],
                   help="apply = use loaded parameters as-is without searching / "
                        "focus = narrow the search around the loaded parameters")
    p.add_argument("--resume", action="store_true", help="Resume an interrupted run from the checkpoint (auto-resume is also the default)")
    p.add_argument("--restart", action="store_true", help="Discard the existing checkpoint and start over")
    p.add_argument("--no-export", action="store_true", help="Do not export the model")
    p.add_argument("--overwrite-output", action="store_true", help="Overwrite the output directory if it exists")
    p.add_argument("--no-report", action="store_true", help="Do not write report PNGs")
    p.add_argument("--report-file", default=None, help="Path of the report PNG (auto if omitted)")
    p.add_argument("--cache-dir", default=None, help="Cache root directory (<workdir>/cache if omitted)")
    p.add_argument("--dry-run", action="store_true", help="Run the search only; do not write the model output")
    p.add_argument("--quiet", action="store_true", help="Suppress progress display")
    p.add_argument("--hf-repo", default=None,
                   help="HuggingFace repository to upload the exported model to. The suffix "
                        "'-Tema_Q-Abliterated' is appended to the name automatically. "
                        "With --hf-token this enables non-interactive upload")
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace API token with write access "
                        "(create one at https://huggingface.co/settings/tokens)")
    return p


@dataclass
class Settings:
    """統合された実行設定。"""

    model: str = ""
    workdir: str = ""
    output: str | None = None
    device: str = "auto"
    eval_dtype: str = "auto"
    dequant_cache_gb: float = -1.0
    language: str = "en"
    time_budget_min: float = 0.0
    trials: int = 0
    target_refusal: float = 0.03
    penalty: float = 8.0
    seed: int | None = None
    harmful_file: str | None = None
    harmless_file: str | None = None
    n_probe: int = 0
    probe_hardening: bool = True
    kl_tokens: int = 8
    verify: str = "auto"
    verify_probes: int = 0
    max_new_tokens: int = 64
    norm_preserve: bool = True
    components: str = "both"
    select: str = "interactive"
    params_from: str | None = None
    params_mode: str = "focus"
    resume: bool = False
    restart: bool = False
    no_export: bool = False
    overwrite_output: bool = False
    no_report: bool = False
    report_file: str | None = None
    cache_dir: str | None = None
    dry_run: bool = False
    quiet: bool = False
    verify_dataset: str | None = None
    verify_gate: float = 0.4
    hf_repo: str | None = None
    hf_token: str | None = None
    workdir_is_default: bool = False   # 既定ワークスペース (tq_workspace/<モデル名>) を使っているか
    # TOMLから与えられる追加設定
    system_prompt: str = "You are a helpful assistant."
    harmful_dataset: dict[str, Any] = field(default_factory=dict)
    harmless_dataset: dict[str, Any] = field(default_factory=dict)
    auto_trials_gpu: int = 200
    auto_trials_cpu: int = 120
    checkpoint_every: int = 1
    verify_topk: int = 5

    @property
    def cache_root(self) -> Path:
        return Path(self.cache_dir) if self.cache_dir else Path(self.workdir) / "cache"

    @property
    def output_dir(self) -> Path:
        if self.output:
            return Path(self.output)
        name = Path(self.model).name or "model"
        return Path(self.workdir) / "output" / f"{name}-uncensored"


def default_workdir_for(model_ref: str) -> Path:
    """既定のワークスペース: ./tq_workspace/<モデル名> (起動ディレクトリ基準)。

    モデル名はパス区切り ("/" および "\\") の末尾要素から取り、
    ファイル名に使えない文字 (Windowsの予約文字等) は "_" に置き換える。
    これにより ``tq_workspace`` フォルダの下がモデルごとに1フォルダに整理される:

        tq_workspace/<モデル名>/cache/…    キャッシュ・チェックポイント
        tq_workspace/<モデル名>/output/…   非検閲化モデルの書き出し先
        tq_workspace/<モデル名>/report/…   レポートPNG・実行記録
    """
    name = (model_ref or "").replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].strip()
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", name).rstrip(". ")
    if not name:
        name = "model"
    return Path("tq_workspace") / name


def _merge_toml(settings: Settings, toml_path: str, ns_defaults: dict) -> None:
    """TOML設定ファイルの値をsettingsへ反映する (CLI明示指定より優先度は低い)。

    [v8] v7までは「現在値が空相当 (None/""/0/False) のときのみTOML値を採用」
    だったため、既定値が非空の設定 (probe_hardening=True, select="interactive"
    など) はTOMLから変更できなかった。ここでは「現在値がパーサ既定値のまま
    (=CLIで明示指定されていない)」場合もTOML値を採用する。
    """
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    for key, value in data.items():
        if hasattr(settings, key) and value is not None:
            cur = getattr(settings, key)
            at_default = (key in ns_defaults and cur == ns_defaults[key])
            if at_default or cur in (None, "", 0, False) or key in ("system_prompt",):
                if key == "trials" and value == 0:
                    continue
                setattr(settings, key, value)


def _parser_defaults(parser: argparse.ArgumentParser) -> dict:
    """パーサの既定値一覧 (設定属性名キー)。TOMLマージの「CLI未指定」判定に使う。

    store_true の逆転フラグ (--no-probe-hardening → probe_hardening) や
    選択型文字列 (--norm-preserve on/off → bool) は設定属性名・型に合わせて
    変換する。
    """
    try:
        d = {a.dest: a.default for a in parser._actions if hasattr(a, "default")}
    except Exception:
        return {}
    if "no_probe_hardening" in d:
        d["probe_hardening"] = not d["no_probe_hardening"]
    if "norm_preserve" in d:
        d["norm_preserve"] = (d["norm_preserve"] == "on")
    return d


def load_settings(argv: list[str] | None = None) -> tuple[Settings, argparse.Namespace]:
    """CLI引数とTOMLを統合してSettingsを返す。"""
    parser = build_parser()
    ns = parser.parse_args(argv)
    s = Settings()
    s.model = ns.model or ""
    s.workdir = ns.workdir or s.workdir
    s.output = ns.output
    s.device = ns.device
    s.eval_dtype = ns.eval_dtype
    s.dequant_cache_gb = ns.dequant_cache_gb
    s.language = ns.language
    s.time_budget_min = ns.time_budget_min
    s.trials = ns.trials
    s.target_refusal = ns.target_refusal
    s.penalty = ns.penalty
    s.seed = ns.seed
    s.harmful_file = ns.harmful_file
    s.harmless_file = ns.harmless_file
    s.n_probe = ns.n_probe
    s.probe_hardening = not ns.no_probe_hardening
    s.kl_tokens = ns.kl_tokens
    s.verify = ns.verify
    s.verify_probes = ns.verify_probes
    if ns.verify_topk:
        s.verify_topk = ns.verify_topk
    s.max_new_tokens = ns.max_new_tokens
    s.norm_preserve = (ns.norm_preserve == "on")
    s.components = ns.components
    s.select = ns.select
    s.params_from = ns.params_from
    s.params_mode = ns.params_mode
    s.resume = ns.resume
    s.restart = ns.restart
    s.no_export = ns.no_export
    s.overwrite_output = ns.overwrite_output
    s.no_report = ns.no_report
    s.report_file = ns.report_file
    s.cache_dir = ns.cache_dir
    s.dry_run = ns.dry_run
    s.quiet = ns.quiet
    s.verify_dataset = ns.verify_dataset
    s.verify_gate = ns.verify_gate
    s.hf_repo = ns.hf_repo
    s.hf_token = ns.hf_token

    if ns.config:
        _merge_toml(s, ns.config, _parser_defaults(parser))

    if not s.model:
        # [v7条件⑮] モデル指定なしで実行された場合はヘルプを表示する
        # (v6はargparseのusageエラーだけを出していた)。
        parser.print_help()
        raise SystemExit(0)

    # [ワークスペース既定値] --workdir / TOMLのいずれでも指定されていない
    # 場合は ./tq_workspace/<モデル名> を使う (起動ディレクトリ基準)。
    # CLI・TOML・既定値の優先順位は CLI > TOML > 既定 (他の設定と同一)。
    if not s.workdir:
        s.workdir = str(default_workdir_for(s.model))
        s.workdir_is_default = True

    # すべてのキャッシュ・出力をプロジェクト作業フォルダへ集約 (条件⑪)
    os.makedirs(s.workdir, exist_ok=True)
    os.makedirs(s.cache_root, exist_ok=True)
    # 出力言語を確定 (以降の全メッセージは msg 経由でローカライズされる)
    from .msg import set_language
    set_language(s.language)
    return s, ns
