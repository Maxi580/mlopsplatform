# Which finetuning library should the `finetune` Stage use?

Research ticket: [#4](https://github.com/Maxi580/mlopsplatform/issues/4) (parent map: [#1](https://github.com/Maxi580/mlopsplatform/issues/1)).
Researched on **2026-10-01**. Every fact below applies to the version and date named next to it. Check them again before pinning versions.

## Question

Which library should the `finetune` Stage be built on? The candidates are TRL + PEFT, Axolotl, LLaMA-Factory, Unsloth and torchtune. The Stage must:

- cover full finetuning, LoRA, QLoRA and other adapter methods;
- train any Hugging Face Base Model without per-model code;
- scale to several GPUs later (FSDP or DeepSpeed);
- be driven by config generated from a Pipeline Request;
- log to MLflow;
- run on Blackwell (RTX 5060 Ti, CUDA 12.8+);
- train on response-level Distillation Datasets as supervised finetuning (SFT) data;
- be free for commercial use and actively maintained.

## Recommendation

**Build the `finetune` Stage on TRL (`SFTTrainer`) + PEFT, on top of Transformers and Accelerate.** Package it as a small entrypoint in a platform-owned container image. The entrypoint turns the Pipeline Request's `finetune` section into an `SFTConfig`, a `LoraConfig` and an Accelerate config.

Why:

1. **Every method from one API.** Full finetuning is the default. LoRA, DoRA, rsLoRA and about 40 other PEFT methods (IA3, OFT/BOFT, LoKr, VeRA, prefix/prompt tuning and more) come in through `peft_config`. QLoRA is `quantization_config` (bitsandbytes 4-bit) plus `peft_config` ([SFTTrainer docs, TRL v1.14.1](https://huggingface.co/docs/trl/sft_trainer); [PEFT tuners in source](https://github.com/huggingface/peft/tree/main/src/peft/tuners)).
2. **Any Hugging Face causal LM, with no per-model code.** `SFTTrainer` takes a Hub model id and loads it with `<ModelArchitecture>.from_pretrained`. `trust_remote_code` is an explicit opt-in ([SFTTrainer API](https://huggingface.co/docs/trl/sft_trainer)). PEFT's `target_modules="all-linear"` picks adapter targets on any architecture ([LoraConfig source](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py)).
3. **Distillation Datasets fit natively.** A prompt → response pair is TRL's *prompt-completion* dataset type. `SFTTrainer` computes the loss on the completion only by default (`completion_only_loss=None` resolves to completion-only for prompt-completion data) ([SFTConfig](https://huggingface.co/docs/trl/sft_trainer); [dataset formats](https://huggingface.co/docs/trl/dataset_formats)).
4. **Multi-GPU needs no code change.** TRL ships Accelerate presets `multi_gpu`, `fsdp1`, `fsdp2`, `zero1`, `zero2` and `zero3`, selected with `accelerate_config` ([TRL CLI docs](https://huggingface.co/docs/trl/clis)).
5. **MLflow is built in.** Set `report_to: mlflow`, and the Transformers `MLflowCallback` reads `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, `MLFLOW_RUN_ID`, `MLFLOW_TAGS` and `HF_MLFLOW_LOG_ARTIFACTS` ([Transformers v5.17.0 callbacks](https://huggingface.co/docs/transformers/main_classes/callback)). With `MLFLOW_RUN_ID`, the API can pre-create the Run and the training Job attaches to it.
6. **Fewest layers, first-party maintenance, no telemetry, Apache-2.0.** Hugging Face maintains the library, and TRL released v1.13.0, v1.14.0 and v1.14.1 in September 2026 alone ([TRL releases](https://github.com/huggingface/trl/releases)). Axolotl, LLaMA-Factory and Unsloth are all built on these same libraries.
7. **Blackwell support comes from the image.** TRL has no CUDA code of its own. PyTorch ≥ 2.7 ships CUDA 12.8 wheels with Blackwell support ([PyTorch 2.7 blog](https://pytorch.org/blog/pytorch-2-7/)). bitsandbytes (MIT) added sm_100/sm_120 builds in 0.45.3 ([release](https://github.com/bitsandbytes-foundation/bitsandbytes/releases/tag/0.45.3)).

**Runner-up: Axolotl** (Apache-2.0). It is a strong choice if the team would rather adopt a ready-made, pydantic-validated YAML schema with more performance features (multipacking, Liger, Cut Cross Entropy, sequence/tensor/context parallelism) than own a thin entrypoint. Its costs:

- a second layer of abstraction over TRL/Transformers, with its own compatibility rules ([support matrix](https://github.com/axolotl-ai-cloud/axolotl/blob/main/docs/support-matrix.qmd));
- **telemetry to PostHog that is on by default**. It must be disabled with `AXOLOTL_DO_NOT_TRACK=1`, and it otherwise blocks startup for 10 s ([telemetry docs](https://github.com/axolotl-ai-cloud/axolotl/blob/main/docs/telemetry.qmd));
- releases that require Python ≥ 3.12 and PyTorch ≥ 2.13 ([README](https://github.com/axolotl-ai-cloud/axolotl/blob/main/README.md)).

The Pipeline Request should not expose library-specific keys (see open questions). Then swapping TRL for Axolotl later only touches the entrypoint.

**Not recommended:**

- **Unsloth.** Its multi-GPU support is DDP only, with no FSDP or ZeRO sharding for models bigger than one GPU. Its docs still say official multi-GPU support is coming "soon" ([multi-GPU docs](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth), [DDP docs](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth/ddp)). The core is Apache-2.0, but the required dependency `unsloth_zoo` is **LGPL-3.0** and Unsloth Studio is **AGPL-3.0** ([README license section](https://github.com/unslothai/unsloth/blob/main/README.md#license), [unsloth-zoo repo](https://github.com/unslothai/unsloth-zoo), [pyproject dependency](https://github.com/unslothai/unsloth/blob/main/pyproject.toml)). It is configured in code, not by config file.
- **LLaMA-Factory.** Chat models need a per-model `template` from its registry, and custom datasets must be registered in `dataset_info.json` ([README](https://github.com/hiyouga/LLaMA-Factory/blob/main/README.md), [data/README](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md)). Its official Docker image is built on CUDA 12.4, which is not Blackwell-ready, and its last release is v0.9.5 from 2026-05-30.
- **torchtune.** Active development stopped on 2025-07-15, with no new features and fixes only through 2025 ([issue #2883](https://github.com/meta-pytorch/torchtune/issues/2883)). The last release is v0.6.1 from 2025-04-07, and every model needs a native PyTorch builder. This also rules out Kubeflow Trainer's only `BuiltinTrainer`, which is the TorchTune LLM trainer ([kubeflow/sdk types.py](https://github.com/kubeflow/sdk/blob/main/kubeflow/trainer/types/types.py)).

## Comparison table

Status as of 2026-10-01.

| | **TRL + PEFT** | **Axolotl** | **LLaMA-Factory** | **Unsloth** | **torchtune** |
|---|---|---|---|---|---|
| **Version / date** | TRL v1.14.1 (2026-09-29), PEFT v0.21.1 (2026-09-29) | v0.20.0 (2026-09-30) | v0.9.5 (2026-05-30) | v0.1.900-beta (2026-09-28), PyPI date-versioned | v0.6.1 (2025-04-07) |
| **Full finetuning** | Yes (default) | Yes (omit `adapter`) | Yes (`full`) | Yes | Yes |
| **LoRA / QLoRA** | Yes / Yes (`peft_config` + bnb 4-bit) | Yes / Yes (`adapter: lora\|qlora`) | Yes / Yes (2–8 bit via bnb, GPTQ, AWQ, HQQ and others) | Yes / Yes | Yes / Yes |
| **Other adapters** | All PEFT methods: DoRA, rsLoRA, IA3, OFT/BOFT, LoKr, VeRA, prefix/prompt tuning and more | DoRA, rsLoRA, LoRA+, LoftQ, ReLoRA, LISA, Spectrum, MoRA | Freeze-tuning, DoRA, OFT, LoRA+, PiSSA, GaLore, BAdam, APOLLO | LoRA-family, FP8 | LoRA, DoRA, QLoRA |
| **Model coverage** | Any HF causal LM via `from_pretrained`, no per-model code | "Any HuggingFace causal/seq2seq LM trains out of the box" | Model list plus a per-model chat `template` registry | "Any model supported by transformers"; some need tweaks | Only models with a native torchtune builder |
| **Multi-GPU** | Accelerate: DDP, FSDP1/FSDP2, DeepSpeed ZeRO 1/2/3, multi-node | FSDP2, DeepSpeed ZeRO 1–3, TP, CP, EP, multi-node (torchrun, Ray) | DeepSpeed, FSDP (+QLoRA), Megatron-core backend | DDP; `device_map="balanced"` model split; no FSDP/ZeRO | FSDP2, multi-node |
| **Config style** | Python dataclasses (`SFTConfig`); `trl sft --config x.yaml` CLI | One YAML per run, pydantic-validated; `axolotl train x.yml` | YAML; `llamafactory-cli train x.yaml`; plus web UI | Python code / notebooks; Studio web UI | YAML recipes + `tune run` |
| **Distillation Dataset (prompt → response)** | Native prompt-completion type; completion-only loss by default | `type: input_output`, `chat_template` with `roles_to_train`, or alpaca | Must register in `dataset_info.json` (alpaca/sharegpt columns) | Via TRL `SFTTrainer` (Unsloth wraps TRL) | Instruct dataset builders |
| **MLflow** | `report_to: mlflow` + `MLFLOW_*` env vars (Transformers `MLflowCallback`) | `use_mlflow`, `mlflow_tracking_uri`, `mlflow_experiment_name`, `mlflow_run_name`, `hf_mlflow_log_artifacts` | `report_to: mlflow` (listed monitor) | Via TRL/Transformers `report_to` | MLflow metric logger |
| **Blackwell / CUDA 12.8+** | Via PyTorch ≥ 2.7 cu128 wheels and bitsandbytes ≥ 0.45.3; we build the image | Official images are CUDA 13.0 (`main-py3.12-cu130-2.13.0`) | README names RTX 5060 Ti bnb builds on Windows; official image CUDA 12.4 | Blackwell guide in README | Not updated for new stacks after 2025 |
| **License** | Apache-2.0 (TRL, PEFT, Transformers, Accelerate); bitsandbytes MIT | Apache-2.0; **telemetry on by default** (opt-out) | Apache-2.0 | Core Apache-2.0; **`unsloth_zoo` LGPL-3.0 (required)**; **Studio AGPL-3.0** | BSD-3-Clause |
| **Maintenance** | Very active: HF-maintained, several releases a month | Very active: releases every 2–8 weeks | Active commits, slower releases | Very active | **Unmaintained** since 2025-07 |

Sources for each cell are listed under [Sources](#sources).

## How a Pipeline Request would map onto the chosen library's config

The Pipeline Request's shape is decided in [#8](https://github.com/Maxi580/mlopsplatform/issues/8), so the keys below are only illustrative. The idea: the Pipeline Request uses **platform-level, library-neutral names**, and the `finetune` Stage entrypoint translates them. Illustrative `finetune` section:

```yaml
finetune:
  base_model: Qwen/Qwen3-0.6B          # any HF model id
  method: qlora                        # full | lora | qlora | <other peft method>
  dataset: <Distillation Dataset ref>  # or an uploaded/HF dataset
  adapter: { r: 16, alpha: 32, dropout: 0.05, target_modules: all-linear }
  training: { epochs: 3, learning_rate: 2e-4, batch_size: 4, grad_accum: 4, max_length: 2048 }
  gpus: 1                              # >1 -> distributed preset
  distributed: auto                    # auto | ddp | fsdp2 | zero3
```

The entrypoint maps it onto TRL/PEFT like this:

| Pipeline Request (illustrative) | TRL / PEFT / Accelerate target |
|---|---|
| `base_model` | `SFTTrainer(model=...)` / `model_name_or_path`; loaded from the Model Cache path; `HF_TOKEN` env for gated models |
| `method: full` | no `peft_config`; `bf16: true` |
| `method: lora` | `peft_config=LoraConfig(r, lora_alpha, lora_dropout, target_modules)`; CLI equivalents `use_peft`, `lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_modules` ([ModelConfig source](https://github.com/huggingface/trl/blob/v1.14.1/trl/trainer/model_config.py)) |
| `method: qlora` | the above plus `quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", ...)`; CLI `load_in_4bit`, `bnb_4bit_quant_type`, `use_bnb_nested_quant` |
| `method: dora` / `rslora` | `LoraConfig(use_dora=True)` / `use_rslora=True` |
| other PEFT methods | the matching `*Config` class from `peft` (IA3Config, OFTConfig, …), chosen from an allow-list |
| `dataset` (Distillation Dataset) | a `datasets.Dataset` with conversational prompt-completion rows `{"prompt":[{"role":"user",...}], "completion":[{"role":"assistant",...}]}`; `completion_only_loss` default |
| `training.*` | `SFTConfig(num_train_epochs, learning_rate, per_device_train_batch_size, gradient_accumulation_steps, max_length, packing, ...)` |
| `gpus`, `distributed` | `accelerate launch --num_processes N --config_file <preset>` using TRL presets `multi_gpu`, `fsdp2` or `zero3` ([TRL CLI](https://huggingface.co/docs/trl/clis)) |
| tracking (platform-set, not user-set) | `report_to: mlflow`; env `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, `MLFLOW_RUN_ID` (API pre-creates the Run), `MLFLOW_TAGS` (Owner, Pipeline id) |
| output | `output_dir` written to the object store; adapter weights (PEFT) or full weights, optionally merged with `merge_and_unload()` |

Two ways to run it:

1. **(Preferred)** A small Python entrypoint that builds `SFTConfig` and `LoraConfig` objects directly. This gives full control over the dataset loading from the object store, merging and MLflow model logging.
2. Render a YAML file and call `trl sft --config finetune.yaml`. This has fewer lines of our own code, but the stock script only loads datasets by `dataset_name` / `datasets` mixtures ([sft.py](https://github.com/huggingface/trl/blob/v1.14.1/trl/scripts/sft.py)).

Errors such as CUDA out-of-memory propagate from the trainer as exceptions. The entrypoint can catch them and write them to the Run and the Job's logs, which fits the map's rule to "show the error to the user".

## Open questions for the grilling tickets

- **[#8](https://github.com/Maxi580/mlopsplatform/issues/8) Pipeline Request shape.**
  - Does the `finetune` section use library-neutral names (recommended) or pass TRL `SFTConfig` keys through?
  - Is there a raw "extra args" escape hatch?
  - Which PEFT methods are on the allow-list, and which hyperparameters get platform defaults?
  - Is `trust_remote_code` allowed per request?
- **[#8](https://github.com/Maxi580/mlopsplatform/issues/8) / [#10](https://github.com/Maxi580/mlopsplatform/issues/10) Chat template.** Which chat template applies when the Base Model is a *base* (non-instruct) model without one? TRL supports `chat_template_path` and `eos_token` for this. And which on-disk format does a Distillation Dataset use? Conversational prompt-completion in JSONL or Parquet is the natural fit.
- **[#10](https://github.com/Maxi580/mlopsplatform/issues/10) / [#11](https://github.com/Maxi580/mlopsplatform/issues/11) Stage output.** Does `finetune` publish the adapter only, merged full weights, or both? This affects `evaluate` (#5) and `serve` (#6), for example whether the serving engine loads LoRA adapters at runtime.
- **[#2](https://github.com/Maxi580/mlopsplatform/issues/2) How to launch.** Kubeflow Trainer's only built-in LLM trainer is TorchTune, which is unmaintained. Does the `finetune` Stage run as a plain Kubernetes Job from the KFP step, or as a Kubeflow Trainer `TrainJob` with a custom container (`torch_distributed` / `deepspeed_distributed` runtime)? For multi-GPU on one VM, a single pod with `accelerate launch` is enough.
- **[#3](https://github.com/Maxi580/mlopsplatform/issues/3) MLflow Run layout.** Is there one Run per Stage, or one parent Run per Pipeline with nested Runs (`MLFLOW_NESTED_RUN`)? And should checkpoints go to MLflow artifacts (`HF_MLFLOW_LOG_ARTIFACTS`) or only to the object store?
- **[#12](https://github.com/Maxi580/mlopsplatform/issues/12) Smoke Test.** The Smoke Test can run TRL SFT with LoRA on a tiny model on CPU in CI, which checks the Pipeline Request → config mapping without a GPU.

## Sources

All accessed 2026-10-01.

**TRL / PEFT / Transformers / PyTorch / bitsandbytes**
- TRL repo, license and releases (Apache-2.0; v1.14.1 2026-09-29): https://github.com/huggingface/trl , https://github.com/huggingface/trl/releases
- TRL SFT Trainer and SFTConfig (v1.14.1): https://huggingface.co/docs/trl/sft_trainer
- TRL dataset formats (v1.14.1): https://huggingface.co/docs/trl/dataset_formats
- TRL CLI, YAML config and Accelerate presets: https://huggingface.co/docs/trl/clis
- TRL ModelConfig (LoRA/QLoRA CLI flags), v1.14.1: https://github.com/huggingface/trl/blob/v1.14.1/trl/trainer/model_config.py
- TRL SFT script dataset loading, v1.14.1: https://github.com/huggingface/trl/blob/v1.14.1/trl/scripts/sft.py
- TRL pyproject (license, dependencies), v1.14.1: https://github.com/huggingface/trl/blob/v1.14.1/pyproject.toml
- PEFT repo and releases (Apache-2.0; v0.21.1 2026-09-29): https://github.com/huggingface/peft , https://github.com/huggingface/peft/releases
- PEFT tuners (list of methods): https://github.com/huggingface/peft/tree/main/src/peft/tuners
- PEFT LoraConfig `all-linear`: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py
- Transformers MLflowCallback (v5.17.0): https://huggingface.co/docs/transformers/main_classes/callback
- PyTorch 2.7 Blackwell / CUDA 12.8 wheels: https://pytorch.org/blog/pytorch-2-7/
- bitsandbytes 0.45.3 (sm_100/sm_120 builds): https://github.com/bitsandbytes-foundation/bitsandbytes/releases/tag/0.45.3 ; license MIT: https://github.com/bitsandbytes-foundation/bitsandbytes

**Axolotl**
- Repo, README, license (Apache-2.0; v0.20.0 2026-09-30): https://github.com/axolotl-ai-cloud/axolotl , https://github.com/axolotl-ai-cloud/axolotl/blob/main/README.md , https://github.com/axolotl-ai-cloud/axolotl/releases
- Support matrix (methods, model coverage, parallelism, compatibility): https://github.com/axolotl-ai-cloud/axolotl/blob/main/docs/support-matrix.qmd
- Telemetry (default on, opt-out): https://github.com/axolotl-ai-cloud/axolotl/blob/main/docs/telemetry.qmd
- Docker image tags (cu130): https://github.com/axolotl-ai-cloud/axolotl/blob/main/docs/docker.qmd
- MLflow config schema: https://github.com/axolotl-ai-cloud/axolotl/blob/main/src/axolotl/utils/schemas/integrations.py
- Dataset formats: https://github.com/axolotl-ai-cloud/axolotl/blob/main/docs/dataset-formats/index.qmd

**LLaMA-Factory**
- Repo, README, license (Apache-2.0; v0.9.5 2026-05-30): https://github.com/hiyouga/LLaMA-Factory , https://github.com/hiyouga/LLaMA-Factory/blob/main/README.md , https://github.com/hiyouga/LLaMA-Factory/releases
- Dataset registration: https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md

**Unsloth**
- Repo, README and license section (dual Apache-2.0 / AGPL-3.0): https://github.com/unslothai/unsloth , https://github.com/unslothai/unsloth/blob/main/README.md
- AGPL text (COPYING): https://github.com/unslothai/unsloth/blob/main/COPYING
- pyproject (requires `unsloth_zoo`): https://github.com/unslothai/unsloth/blob/main/pyproject.toml
- unsloth-zoo (LGPL-3.0): https://github.com/unslothai/unsloth-zoo
- Multi-GPU docs: https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth , https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth/ddp
- Model coverage FAQ: https://unsloth.ai/docs/basics/troubleshooting-and-faqs

**torchtune / Kubeflow**
- torchtune repo and README notice (BSD-3-Clause; v0.6.1 2025-04-07): https://github.com/pytorch/torchtune
- torchtune MLFlowLogger: https://github.com/pytorch/torchtune/blob/main/torchtune/training/metric_logging.py
- "The future of torchtune" (2025-07-15): https://github.com/meta-pytorch/torchtune/issues/2883
- Kubeflow SDK BuiltinTrainer = TorchTuneConfig only: https://github.com/kubeflow/sdk/blob/main/kubeflow/trainer/types/types.py
- Kubeflow Trainer runtimes (v2.3.0 2026-08-07): https://github.com/kubeflow/trainer/tree/master/manifests/base/runtimes
