<div align="center">

# Tema_Q-Abliterated

**SCAPE: search-based LLM uncensoring that runs on a CPU-only PC — up to 49× faster candidate evaluation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-1.0.0-informational.svg)](pyproject.toml)
[![PyPI](https://img.shields.io/pypi/v/temaq-abliterated)](https://pypi.org/project/temaq-abliterated/)

[Overview](#overview) •
[Highlights](#highlights) •
[How SCAPE Works](#how-scape-works) •
[Key Features](#key-features) •
[Installation](#installation) •
[Quick Start](#quick-start) •
[Usage](#usage) •
[Experimental Results](#experimental-results) •
[Supported Architectures](#supported-architectures) •
[Notes](#notes) •
[Citation](#citation)

</div>

---

## Overview

Tema_Q-Abliterated is a research tool that **automatically finds good settings for directional ablation** (also called abliteration). It builds on the finding of Arditi et al. (2024) that *refusal behavior is mediated by a single direction in the residual stream*.

Choosing which layers to modify, and how strongly, is the hard part: too strong destroys the model's general ability, too weak leaves refusals in place. The search algorithm in this tool, **SCAPE (Structured Cached Abliteration with Prefix Evaluation)**, automates that choice. It runs on modest hardware, including a **CPU-only PC with 16 GB RAM**, which existing GPU-based tools do not support.

> **Research use only.** Uncensored models operate with their safety filters removed. Always follow the original model's license and terms of use, and handle the output strictly within research and educational contexts.

For the algorithm and full experiments, see the technical report: [Tema_Q-Abliterated_Technical_Report](https://github.com/ek15072809/Tema_Q-Abliterated/blob/main/docs/Tema_Q-Abliterated_Technical_Report.pdf)

---

## Highlights

| | |
|---|---|
| **Fast candidate evaluation** | A surrogate metric replaces full-text generation and is up to **~49× faster**. |
| **Works without a GPU** | A 4B-class model finishes on a Core i7 with 16 GB RAM. |
| **Better results in less time** | On the same Colab T4 GPU, Qwen3.5-4B reached **8/100** refusals in **61 min**, versus 27–74/100 in 4 h+ for heretic. |
| **Quality-first objective** | Minimizes KL divergence from the original model *subject to* a target refusal rate, so the model is not over-modified. |
| **Safe by design** | Never writes out a model without your confirmation unless you pass `--select auto`. |

---

## How SCAPE Works

1. **Seed grid (≈ 8 evaluations).** The search starts by testing a small set of systematic settings across the strength × layer-band range. This always includes the uniform all-layer orthogonalization of Arditi et al. (the *reference solution*), so the search does not get stuck in a narrow local optimum.
2. **Floor calibration.** The surrogate metric cannot reach zero, even for a fully uncensored model, because tokens that grammatically continue a refusal sentence always keep some probability mass. We call this the **grammatical floor** (≈ 0.3–0.4). SCAPE measures it from the seed grid and adjusts the target accordingly, so the search keeps making progress instead of stalling.
3. **Pattern search.** Starting from the best seed, a pattern search refines the settings. Hopeless candidates are rejected early using an exact lower bound.
4. **Verification.** Only a few top candidates are checked with real text generation. The gate uses the **first-token ratio**, which has no floor.
5. **Human-in-the-loop export.** You choose the final candidate. The output weights are always built from the **original safetensors** in fp32, so quantization used during the search never reduces output precision.

Speed comes from three techniques: a **rank-1 hook** (no weight rewriting per trial), a **multi-boundary prefix cache** (layers before the perturbation are computed only once), and **INT4 streaming with a memory guard**.

---

## Key Features

- **CPU-only search** — Each trial takes only a few seconds, so 0.5B–9B models can be processed without a GPU.
- **Seed-grid initialization** — About 8 evaluations at the start cover the whole strength × bandwidth spectrum, including the reference solution.
- **Grammatical-floor calibration** — The floor is measured from the seed grid and the effective target is redefined accordingly.
- **Quality-preserving objective** — KL divergence from the original model is minimized under a refusal-rate constraint.
- **INT4 evaluation and memory guard** — Weights use about 1/3.6 the memory of bf16, with streaming load, an elastic cache, and automatic OOM fallback.
- **Safe export** — If no candidate passes the verification gate, nothing is exported without your confirmation.
- **Interrupt and resume** — A checkpoint is saved after every trial; on re-run, resume options appear before model loading.
- **Hugging Face integration** — Optional upload after export (the repository name automatically gets the suffix `-Tema_Q-Abliterated`), plus an auto-generated model card that keeps the original model's tags.
- **Minimal dependencies** — Only torch, transformers, safetensors, huggingface-hub, matplotlib, and psutil (no Optuna, PEFT, or bitsandbytes).

---

## Installation

### Requirements

- Python 3.10 or higher
- PyTorch 2.2 or higher (install beforehand for your environment)
- transformers 5.x for the Qwen4Exp and Gemma4 series

### From PyPI (recommended)

```bash
pip install temaq-abliterated
```

This places the `temaq-abliterated` command in your PATH. To use Hugging Face datasets as verification prompts:

```bash
pip install "temaq-abliterated[datasets]"
```

### From source (for development)

```bash
git clone https://github.com/ek15072809/Tema_Q-Abliterated.git
cd Tema_Q-Abliterated
pip install -e .
```

---

## Quick Start

```bash
# A model on Hugging Face
temaq-abliterated Qwen/Qwen3.5-0.8B

# A local model directory
temaq-abliterated /path/to/gemma4-E2B

# Limited memory or a larger model: use INT4 evaluation
temaq-abliterated Qwen/Qwen3.5-4B --eval-dtype int4

# Recommended: cap the runtime
temaq-abliterated Qwen/Qwen3.5-4B --eval-dtype int4 --time-budget-min 60 --trials 200
```

Unless `--workdir` is given, this structure is created in the launch directory:

```
tq_workspace/
└── <model-name>/
    ├── cache/    … downloads and checkpoints
    ├── output/   … uncensored model (safetensors)
    └── report/   … refusal-rate curve, memory timeline, run record
```

### Rough runtime guide

Processing time scales roughly linearly with model size. These are guidelines, not guarantees.

| Model size | CPU (12–16 threads) | Colab T4 GPU |
|---|---|---|
| 0.5–1B | 42–66 min | 3–12 min |
| 4B | 180–240 min | 30–42 min |

---

## Usage

### Output files

After a run completes:

| Path | Contents |
|---|---|
| `<workspace>/output/<model-name>-uncensored/` | Uncensored model (safetensors + model card + LICENSE) |
| `<workspace>/report/refusal_curve.png` | Refusal-rate convergence graph |
| `<workspace>/report/memory_timeline.png` | Memory usage timeline |
| `<workspace>/report/run_summary.json` | Run record |

### Final candidate selection

Unless `--select auto` is given, a selection UI is always shown before export. Candidates that fail the verification gate (first-token ratio above `--verify-gate`, default 0.4) are excluded from the list. If no candidate passes, the model is not exported without your confirmation.

### Upload to Hugging Face Hub

After export you will be asked "Upload to Hugging Face?". In non-interactive environments:

```bash
temaq-abliterated Qwen/Qwen3.5-4B \
    --hf-repo my-uncensored --hf-token hf_xxxxxxxx
```

The suffix `-Tema_Q-Abliterated` is appended to the repository name automatically.

### Interrupt and resume

A checkpoint is saved after every trial. Re-running the same command shows this menu before the model is loaded:

1. Continue search
2. Add more trials
3. Export model from recorded candidates
4. Start over
5. Exit

### Options

Run `temaq-abliterated --help` for the full list. The main options are grouped below.

**Input / output**

| Option | Default | Description |
|---|---|---|
| `--workdir DIR` | `./tq_workspace/<model-name>` | Working directory (cache, checkpoints, output, reports). |
| `--output DIR` | auto | Output directory. It cannot be the same as, or inside, the original model. Empty folders are used as-is; on conflict you are asked to re-enter or an automatic number is added. |
| `--hf-repo NAME` | — | Hugging Face repository name (the suffix `-Tema_Q-Abliterated` is appended). Use with `--hf-token` for non-interactive upload. |
| `--hf-token TOKEN` | — | Hugging Face API key (write permission). |
| `--language` | `en` | Language of output logs: `en` (English) or `jp` (Japanese). |
| `--no-export` / `--dry-run` | — | Skip writing the output model. |

**Search**

| Option | Default | Description |
|---|---|---|
| `--trials N` | auto | Number of search trials (200 on GPU, 120 on CPU). |
| `--time-budget-min F` | 0 (unlimited) | Time budget for the search phase, in minutes. When reached, the best solution so far proceeds to verification. **Use this to control runtime.** |
| `--target-refusal F` | 0.03 | Target soft refusal rate of the surrogate metric (lower = stronger uncensoring). |
| `--components` | `both` | Components to search: `attn`, `mlp`, or `both`. When `attn` is specified, fused MoE experts are also INT4-quantized. |
| `--n-probe N` | 0 (auto) | Number of probes used in the search (auto: 48–24 depending on model scale). |
| `--kl-tokens N` | 8 | Teacher-forced tokens for the multi-position KL (0 = first token only). |
| `--no-probe-hardening` | off | Disable hardening of harmful probes (use only probes that the base model refuses). |
| `--params-from FILE` | — | Path or URL to an external model card (README.md) to load parameters from. |
| `--params-mode` | `focus` | `apply` = use loaded values as they are; `focus` = narrow the search around them. |
| `--norm-preserve` | on | Row-norm-preserving orthogonalization (turn off for comparison experiments). |
| `--seed N` | random | Random seed. |
| `--resume` / `--restart` | — | Resume from the checkpoint / discard it and start over. |

**Memory / precision**

| Option | Default | Description |
|---|---|---|
| `--eval-dtype` | `auto` | `auto` (planned from free memory) / `bfloat16` / `float16` / `float32` / `int8` / **`int4`**. |
| `--dequant-cache-gb F` | auto | Budget for the INT4 weight cache in GB. A negative value means "decide from free RAM after loading". |

**Verification and selection**

| Option | Default | Description |
|---|---|---|
| `--verify` | `auto` | `auto` = verify top candidates with full generation; `full` = best candidate only; `none` = no verification. |
| `--verify-probes N` | 100 | Number of prompts used for verification. |
| `--verify-dataset DS` | — | Hugging Face dataset for verification prompts (`repo` or `repo:split`; the first N samples are used, and the test split is preferred if no split is given). |
| `--verify-topk N` | 5 | Number of top candidates verified with full generation (on CPU: original + 2 candidates = 3 in total). |
| `--verify-gate F` | 0.4 | Threshold for skipping full-generation verification. It is judged on the floor-free first-token ratio, because the mass-weighted ratio has a grammatical floor of about 0.3–0.4. |
| `--max-new-tokens N` | 64 | Tokens generated during verification. |
| `--select` | `interactive` | `interactive` = always show the selection UI before export; `auto` = choose automatically. Switches automatically in non-interactive environments. |
| `--harmful-file` / `--harmless-file` | built-in | Use your own prompt files (one prompt per line). |

---

## Experimental Results

All numbers come from the run records (JSON and PNG reports) produced by this tool. Refusal rates are measured by full-text generation on 100 harmful prompts (heretic's prompts were used so the two tools can be compared). **KL** is the KL divergence from the original model; lower means the original behavior is better preserved.

### Results on each model

| | Qwen3.5-0.8B | Qwen3.5-4B | gemma4-E2B | LFM2.5-2.6B |
|---|---|---|---|---|
| Trials | 12 | 59 | 200 | 85 |
| Refusal rate of best candidate | **0/100** | **8/100** | 20/100 | **2/100** |
| KL of best candidate | 0.0186 | 0.0266 | 0.0199 | 0.0143 |

### Comparison with heretic (Qwen3.5-4B)

heretic was run in a reduced configuration (85 and 95 trials) so that results would not be lost to the Colab session limit of 5 h 30 min.

| | heretic run 1<br>(Colab T4) | heretic run 2<br>(Colab T4) | **SCAPE**<br>(Colab T4) | **SCAPE**<br>(Windows CPU, 16 GB) |
|---|---|---|---|---|
| Total run time | 4 h 2 min | 4 h 25 min | **1 h 1 min** | 6 h 35 min |
| Search time | 3 h 35 min | 3 h 25 min | **40 min** | 3 h 0 min |
| Trials (random init.) | 85 (20) | 95 (5) | 59 (0) | 21 (0) |
| Refusal rate of best candidate | 74/100 | 27/100 | **8/100** | **7/100** |
| KL of best candidate | 0.0164 | 0.0391 | 0.0266 | 0.0335 |

- On the same GPU, SCAPE finished in about **one quarter of the time** and reached a lower refusal rate.
- On a **GPU-less Windows PC (Core i7, 16 GB RAM)**, an environment where heretic cannot run, SCAPE still reached 7/100.
- The KL computation is not guaranteed to be identical across the two tools, so treat the KL columns as a rough guide, not a strict comparison.

### Evaluation speed (Qwen3.5-0.8B, INT4, 2 CPU cores)

| Method | Content | Time per candidate |
|---|---|---|
| **SCAPE surrogate evaluation** | 16 probes × 1 forward pass | **4–7.5 s** |
| Full-text generation | 24 prompts × 32 tokens (measured) | about 50 s |
| Full-text generation | 100 prompts × 32 tokens (scaled from measurement) | about 208 s (≈ 28–49× slower) |

### Effect of the seed grid and floor calibration (Qwen3.5-0.8B, INT4, 2 CPU cores)

| | Without seed grid and floor calibration | **SCAPE** |
|---|---|---|
| Refusal rate (40 prompts) | 0.94 → 0.10 (36 trials, 395 s) | **0.900 → 0.000 (12 trials, 148 s)** |
| KL of best candidate | 0.0237 | **0.0186** |
| Reference solution reached | not within 31 trials | **at the 2nd seed evaluation** |

---

## Supported Architectures

Model structure is detected automatically by attribute probing:

- **Attention output projection**: `self_attn.o_proj`, `out_proj`, `linear_attn.out_proj`, etc.
- **MLP down projection**: `mlp.down_proj`, fused 3D expert tensors, `layer.experts`, etc.
- Dense, MoE, hybrid linear attention, and multimodal models (language-model part)
- Hyper-connection width mismatch in the Qwen4Exp series
- Shared-KV safe bypass depth in the Gemma4 series
- Automatic resolution of the language backbone in VLMs

All 8 automated tests with synthetic tiny models pass, and the numerical equivalence of the partial forward pass was confirmed to match exactly on real hardware.

---

## Project Structure

```
Tema_Q-Abliterated/
├── README.md                  … This file
├── LICENSE                    … MIT
├── pyproject.toml
├── src/temaq_abliterated/     … Core
│   ├── cli.py                 … Pipeline orchestration
│   ├── search.py              … SCAPE search, seed grid, floor calibration
│   ├── surrogate.py           … Fast evaluator
│   ├── quant4.py              … INT4 quantization
│   ├── modelio.py             … Model I/O and model-card generation
│   ├── arch.py                … Architecture detection
│   ├── engine.py              … Perturbation engine
│   └── ...
└── docs/                      … Research materials
```

---

## Notes

- This tool is designed for research into the internal refusal mechanisms of models.
- Uncensored models operate with safety filters removed. Handle them with care and follow the original model's license and terms of use.
- Use for harmful purposes is not intended. Please use the tool responsibly, within research and educational contexts.
- The quality metrics of an output model (KL, refusal rate, collapse check) are reference values; actual behavior varies with prompts, model size, and seed.

---

## References

1. Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. "Refusal in Language Models Is Mediated by a Single Direction." [arXiv:2406.11717](https://arxiv.org/abs/2406.11717), 2024 (NeurIPS 2024). — The finding on which directional ablation is based.
2. Philipp Emanuel Weidmann. "heretic: Fully automatic censorship removal for language models." <https://github.com/p-e-w/heretic> (AGPL-3.0). — A pioneering tool for automated parameter search, used as the comparison baseline in this project.

Tema_Q-Abliterated is an independent implementation and is not affiliated with the projects above.

---

## License

MIT License.

The directional ablation technique itself is publicly available knowledge originating in the academic literature (Arditi et al., 2024).

---

## Citation

If Tema_Q-Abliterated is useful in your work, please consider citing it:

```bibtex
@software{temaq_abliterated2026,
  author = {ek15072809},
  title  = {Tema_Q-Abliterated},
  year   = {2026},
  url    = {https://github.com/ek15072809/Tema_Q-Abliterated}
}
```

---

## Author

**ek15072809**
