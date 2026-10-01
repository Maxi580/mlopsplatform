# Which training backends should the `finetune` Stage support, and how are they plugged in?

Research ticket: [Which training backends (e.g. Unsloth) should finetune support, and how are they plugged in?](https://github.com/Maxi580/mlopsplatform/issues/23) (parent map: [#1](https://github.com/Maxi580/mlopsplatform/issues/1)).
Researched on **2026-10-01** against Unsloth `main` (release v0.1.900-beta, 2026-09-28), `unsloth_zoo` ≥ 2026.9.8, TRL v1.14.1, Liger Kernel v0.8.4 (2026-09-30), Axolotl v0.20.0 (2026-09-30) and LLaMA-Factory v0.9.5 (2026-05-30). Re-check before pinning versions.

Builds on earlier decisions:

- [Which finetuning library should the finetune Stage use?](https://github.com/Maxi580/mlopsplatform/issues/4): TRL + PEFT on Transformers/Accelerate behind a thin platform-owned entrypoint; Axolotl runner-up ([findings](https://github.com/Maxi580/mlopsplatform/blob/research/finetuning-library/docs/research/finetuning-library.md)). This doc reuses its Unsloth, Axolotl and LLaMA-Factory findings and only adds what changes the backend question.
- [Which training methods should the finetune Stage offer in v1?](https://github.com/Maxi580/mlopsplatform/issues/22): algorithms SFT, DPO, KTO, GRPO, RLOO (standard TRL trainers); methods `full`, `lora` (default), `qlora` ([findings](https://github.com/Maxi580/mlopsplatform/blob/research/training-methods/docs/research/training-methods.md)).
- [Which serving engine and inference optimizations for the serve Stage?](https://github.com/Maxi580/mlopsplatform/issues/6): vLLM serves **plain LoRA only**. It rejects `use_dora`, Adapter bias and `modules_to_save` (other than classifier heads).
- [Where do datasets and models live, and how are they registered and versioned?](https://github.com/Maxi580/mlopsplatform/issues/10): the `finetune` Stage publishes an Adapter for PEFT runs and full weights otherwise, with no merge.

## Recommendation

1. **Default backend `hf`** (TRL + PEFT, as already decided). It covers every algorithm and method, any Hugging Face causal LM, multi-GPU, and is all permissive licences.
2. **Second backend `unsloth`, opt-in, single-GPU, if the licence review below is signed off.** It reuses the same TRL trainers (Unsloth patches them), so it covers SFT, DPO, KTO, GRPO and RLOO with `full`/`lora`/`qlora`. It is the only candidate that adds real speed and memory gains on a single 16 GB GPU. Without sign-off, v1 ships `hf` only and the extension point stays in place.
3. **Liger Kernel is not a backend.** It is a kernel add-on that TRL switches on with one flag (`use_liger_kernel=True`). If it is offered at all, it is a boolean setting of the `hf` backend, default off. TRL already marks the flag as deprecated in its GRPO and RLOO trainers, in favour of Hugging Face Hub kernels.
4. **Axolotl and LLaMA-Factory are not backends.** They are complete training front-ends with their own config schemas, so wrapping them means translating the Pipeline Request into a second schema. Neither covers the full v1 algorithm set: Axolotl has no RLOO, and LLaMA-Factory has neither GRPO nor RLOO. Axolotl also requires an LGPL package and has telemetry on by default.

**Extension point (keep it simple):**

- **One setting** in the `finetune` Stage: `backend: hf | unsloth`, default `hf`.
- **One container image per backend**, each with the same platform entrypoint package. This is required anyway: Unsloth caps TRL at ≤ 1.13.0 and Transformers at ≤ 5.17.0, while the `hf` image tracks TRL 1.14.x (see [Dependency pins](#unsloth-dependency-pins)). Two images avoid sharing one resolver.
- **A backend is one small module** in the entrypoint that does three things: load the Base Model, attach the LoRA/QLoRA Adapter (or set up full finetuning), and return the TRL trainer class for the algorithm. Everything else is shared: Dataset loading, settings mapping, MLflow logging (`report_to=mlflow`) and the output check.
- **A capability table per backend** (algorithms × methods × multi-GPU) is checked by validation at submission. An unsupported combination is a validation error, not a silent switch.
- **No automatic fallback.** If `unsloth` fails on a model, the Pipeline fails with the error, and the user resubmits with `backend: hf`. A silent fallback would make speed, memory use and the produced Model Version depend on something the user did not ask for. (Unsloth already falls back internally to a generic, slower path for architectures it has no hand-written kernels for; see [Model coverage](#model-coverage-and-unsupported-models).)
- **One shared output check after training**, for every backend: read `adapter_config.json` and fail the Stage before registering if `modules_to_save` is set, `use_dora` is true, `bias` is not `none`, or `r` > 512. This is where a backend that silently trains embeddings gets caught (Unsloth can, see [Outputs](#outputs-plain-lora-adapters-for-vllm)).

Adding a backend later means: one image, one module, one capability-table row. Nothing else in the platform changes.

## Comparison

| | **`hf` (TRL + PEFT)** | **Unsloth** | **Liger Kernel** | **Axolotl** | **LLaMA-Factory** |
|---|---|---|---|---|---|
| What it is | Trainers + adapters | Patches Transformers/TRL with custom kernels | Triton kernels plugged into HF models | YAML front-end over TRL | YAML/web front-end over Transformers/TRL |
| Licence of what it pulls in | Apache-2.0; bitsandbytes MIT | Core Apache-2.0; **`unsloth_zoo` LGPL-3.0-or-later (required)**; **Studio AGPL-3.0 shipped inside the same wheel**; `cut_cross_entropy` Apple licence (permissive, no patent grant); **paid Pro/Enterprise tiers** | BSD-2-Clause; needs only torch + triton | Apache-2.0; **`axolotl-contribs-lgpl` LGPL-3.0 (required)**; telemetry on by default | Apache-2.0 |
| SFT / DPO / KTO / GRPO / RLOO | All five | All five (patched TRL trainers) | Through TRL: SFT, DPO, KTO, GRPO (RLOO: layer kernels only; flag deprecated in GRPO/RLOO) | SFT, DPO, KTO, GRPO; **no RLOO** | SFT, DPO, KTO (and PPO); **no GRPO, no RLOO** |
| `full` / `lora` / `qlora` | All | All | Orthogonal (kernels only) | All | All |
| Plain LoRA output vLLM accepts | Yes (platform fixes `bias=none`, no `modules_to_save`) | Yes, **unless** it auto-adds `embed_tokens`/`lm_head` to `modules_to_save` | Not affected | Yes (PEFT) | Yes (PEFT) |
| Open-source multi-GPU | DDP, FSDP2, ZeRO-3 via Accelerate | Manual DDP/FSDP/DeepSpeed setup; "official multi-GPU support … soon"; enhanced multi-GPU in paid Pro | Works with FSDP, DeepSpeed, DDP | DDP, FSDP, DeepSpeed, sequence parallelism | DDP, DeepSpeed, FSDP |
| Blackwell (sm_120, CUDA 12.8+) | PyTorch cu128+, bitsandbytes ≥ 0.45.3 | Supported; `triton>=3.3.1`, cu128 | Inherits Triton's hardware support | cu128/cu130 images | Official image CUDA 12.4 (not Blackwell-ready) |
| MLflow | `report_to=mlflow` | Same (TRL trainers) | n/a | `mlflow_*` config keys | `report_to` |
| Fit as a backend | **Default** | **Opt-in, single-GPU** | Flag on `hf` | Not a backend | Not a backend |

## Unsloth

### Licensing

- **Core package `unsloth`:** `license = "Apache-2.0"` ([pyproject.toml](https://github.com/unslothai/unsloth/blob/main/pyproject.toml)). The README describes "a dual-licensing model of Apache 2.0 and AGPL-3.0. The core Unsloth package remains licensed under Apache 2.0, while certain optional components, such as the Unsloth Studio UI are licensed under … AGPL-3.0" ([README, License](https://github.com/unslothai/unsloth/blob/main/README.md#license), [COPYING](https://github.com/unslothai/unsloth/blob/main/COPYING)).
- **Surprise: the AGPL Studio code ships in the same `unsloth` wheel.** `[tool.setuptools.packages.find]` includes `"studio"` and `"studio.backend*"`, and package data includes the Studio frontend build ([pyproject.toml](https://github.com/unslothai/unsloth/blob/main/pyproject.toml)). `studio/` carries its own `LICENSE.AGPL-3.0` ([studio/](https://github.com/unslothai/unsloth/tree/main/studio)). So `pip install unsloth` puts AGPL code into the image even though the platform never runs it. AGPL's network clause applies only if that code is modified and offered to users over a network. Unused, it is a compliance note, not a blocker, but it needs sign-off. The image could also delete `site-packages/studio` after install (an assumption: not tested that `unsloth` imports without it).
- **`unsloth_zoo` is required and LGPL-3.0-or-later.** `unsloth` depends on `"unsloth_zoo>=2026.9.8"` ([pyproject.toml](https://github.com/unslothai/unsloth/blob/main/pyproject.toml)), and `unsloth_zoo` declares `license = "LGPL-3.0-or-later"` ([unsloth-zoo pyproject.toml](https://github.com/unslothai/unsloth-zoo/blob/main/pyproject.toml); GitHub reports LGPL-3.0). It holds the RL replacements, the compiler and the saving utilities, so it is not optional. LGPL allows commercial use of an unmodified library. Its obligations (keep the licence notice, let users swap in a modified version of the library) apply when the image is distributed. An internal platform pulling it from PyPI unmodified is the low-risk case; vendoring or patching it is not.
- **`cut_cross_entropy`** (pulled in by `unsloth_zoo`) uses Apple's sample-code licence: it permits use, modification and redistribution, but "no other rights or licenses, express or implied, are granted … including but not limited to any patent rights" ([apple/ml-cross-entropy LICENSE](https://github.com/apple/ml-cross-entropy/blob/main/LICENSE)). It is permissive but has no patent grant.
- **Paid tiers.** Unsloth's pricing page lists Free, **Pro** ("2.5x faster training … enhanced MultiGPU support up to 8 GPUs") and **Enterprise** (multi-node, "up to 30% accuracy improvement") ([pricing](https://www.unsloth.ai/pricing), search-engine snapshot because the page returned 403 to direct fetch). The open-source version is the "Free" tier. Nothing in this recommendation needs a paid tier.
- **Usage statistics on by default.** `get_statistics()` downloads a README from Hugging Face to record "which environment is in use". It is skipped when `UNSLOTH_DISABLE_STATISTICS` is set or `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` is on ([`unsloth/models/_utils.py`](https://github.com/unslothai/unsloth/blob/main/unsloth/models/_utils.py)). The Model Cache decision already runs Stages offline. The image should set `UNSLOTH_DISABLE_STATISTICS=1` as well.

### Algorithms and methods

- Unsloth does not ship its own algorithms. It **rewrites the source of TRL's trainers at import time** (`_patch_trl_rl_trainers` / `patch_trl_rl_trainers`, regex replacements over TRL code) and generates `_Unsloth*Trainer` classes ([`unsloth/models/rl.py`](https://github.com/unslothai/unsloth/blob/main/unsloth/models/rl.py), [`rl_replacements.py`](https://github.com/unslothai/unsloth/blob/main/unsloth/models/rl_replacements.py)). The code has explicit handling for KTO (`kto_trainer_get_batch_logps`), GRPO, and RLOO (reference model via `disable_adapter()`). SFT and DPO go through the same mechanism (there is also a dedicated `dpo.py`). So **all five v1 algorithms are covered**, with the same TRL configs and Dataset row formats.
- This patching is why Unsloth pins TRL tightly: a TRL release that moves code breaks the regexes. Expect Unsloth to lag TRL.
- README: "Supports … LoRA, QLoRA, full fine tuning, pretraining, RL, GRPO, DPO, and FP8" ([README](https://github.com/unslothai/unsloth/blob/main/README.md)). Full finetuning goes through `FastModel` ("Full finetuning delegates to FastModel", [`loader.py`](https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py)); QLoRA is `load_in_4bit=True`.
- GRPO can run vLLM in-process (`fast_inference=True`) ([RL guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)). How RL generation runs is decided by [How does reinforcement learning fit the platform: Stage, methods, rewards, environments?](https://github.com/Maxi580/mlopsplatform/issues/19).

### Outputs: plain LoRA Adapters for vLLM

`model.save_pretrained(dir)` on an Unsloth model is PEFT's save, so the result is a normal `adapter_config.json` + `adapter_model.safetensors`. Three traps:

1. **Embeddings moved into `modules_to_save`.** In `get_peft_model`, `_redirect_embedding_targets` moves `embed_tokens`/`lm_head` from `target_modules` into `modules_to_save` ("so they are trained as full weight matrices"). If new tokens were added to the tokenizer, Unsloth turns that on itself: "You added new tokens but did not specify if you wanted to train the lm_head and embed_tokens. We must turn it on for you." ([`unsloth/models/llama.py`](https://github.com/unslothai/unsloth/blob/main/unsloth/models/llama.py)). vLLM rejects such an Adapter. **Mitigation:** the `unsloth` backend never adds tokens or targets embeddings, and the shared output check rejects any `modules_to_save`.
2. **Repo remapping.** `FastLanguageModel.from_pretrained` may swap the requested repo for a pre-quantized mirror (for example `unsloth/…-bnb-4bit`) unless `use_exact_model_name=True` ([`loader.py`](https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py)). With the offline Model Cache that would fail, or train against a different repo than the pinned Base Model. **Mitigation:** always pass `use_exact_model_name=True` and the cached path.
3. **Defaults differ from the `hf` backend.** Unsloth's `get_peft_model` defaults are `r=16`, `lora_alpha=16`, `lora_dropout=0.0`, `bias="none"`, `modules_to_save=None`. Any `lora_dropout != 0` turns off Unsloth's fused LoRA kernels ("The fused LoRA kernels were skipped because lora_dropout = …"). The training-methods default of `dropout: 0.05` would therefore lose much of Unsloth's speed-up; the `unsloth` backend should default `dropout` to 0. Its optimized `target_modules` are the seven projections `q/k/v/o/gate/up/down_proj`; other names work but print "Unsloth hasn't optimized for this … might be noticeably slower" ([`llama.py`](https://github.com/unslothai/unsloth/blob/main/unsloth/models/llama.py)). The backend should translate `all-linear` into that list rather than pass `all-linear` through (not verified whether Unsloth resolves `all-linear` itself).

Full finetuning writes normal full weights, as with `hf`.

### Hardware

- **Multi-GPU (open source):** "Unsloth currently supports multi-GPU setups through libraries like Accelerate and DeepSpeed … We know that the process can be complex and requires manual setup … we'll be announcing official multi-GPU support for Unsloth soon" ([multi-GPU docs](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth), [DDP guide](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth/ddp)). The `hf` backend already scales with Accelerate presets. **v1: `unsloth` is single-GPU only**, enforced by validation.
- **Blackwell:** supported. "`triton>=3.3.1` is required for Blackwell support"; cu128 PyTorch; the `unsloth/unsloth` Docker image covers 50-series ([Blackwell guide](https://unsloth.ai/docs/blog/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth)). Build our own image from pinned wheels rather than using `unsloth/unsloth`, which is built around Studio and notebooks.
- **Speed and memory:** Unsloth's own notebook table claims about "2x faster" and "70–80% less" VRAM for GRPO notebooks ([README](https://github.com/unslothai/unsloth/blob/main/README.md)). These are vendor numbers, not measured here. The Smoke Test or a one-off benchmark on the RTX 5060 Ti should confirm them before `unsloth` is recommended to users.

### Model coverage and unsupported models

- "Unsloth works with any model supported by `transformers`"; newer models may need `trust_remote_code=True` ([FAQ](https://unsloth.ai/docs/basics/troubleshooting-and-faqs)). Hand-written fast paths exist for Llama, Mistral, Qwen2/Qwen3 (+MoE), Gemma/Gemma2, Cohere, Granite, Falcon-H1, GLM-4-MoE, Llama 4 and others ([`unsloth/models/`](https://github.com/unslothai/unsloth/tree/main/unsloth/models)). Other architectures go through the generic `FastModel` path: they train, but with less speed-up.
- The real limit is the **Transformers ceiling**: a model that needs a newer Transformers than Unsloth allows fails with "`<model>` is not supported yet in `transformers==<version>`" ([`loader.py`](https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py)). The user then resubmits with `backend: hf`.

### Unsloth dependency pins

- `unsloth_zoo`: `trl>=0.18.2,!=0.19.0,<=1.13.0` and `transformers … <=5.17.0` on Linux ([unsloth-zoo pyproject.toml](https://github.com/unslothai/unsloth-zoo/blob/main/pyproject.toml)).
- `unsloth`'s `huggingface` extra is stricter still: `trl … <=0.24.0`, `transformers … <=5.5.0`, `datasets … <4.4.0` ([pyproject.toml](https://github.com/unslothai/unsloth/blob/main/pyproject.toml)). Install without that extra and let `unsloth_zoo` set the bounds.
- The `hf` backend is on TRL v1.14.1. So the two backends cannot share one image without holding `hf` back: hence **one image per backend**.

### MLflow

Unsloth's trainers are TRL trainers, so `report_to=mlflow` and the `MLFLOW_*` env vars work as in `hf`. The backend name should be logged as a Run param, so speed and quality can be compared across backends.

## Liger Kernel

- **Licence:** BSD-2-Clause; dependencies are only `torch` and `triton` ([repo](https://github.com/linkedin/Liger-Kernel), [README](https://github.com/linkedin/Liger-Kernel/blob/main/README.md)). Clean.
- **What it is:** fused Triton kernels (RMSNorm, RoPE, SwiGLU, fused linear cross-entropy). "Increase multi-GPU training throughput by 20% and reduces memory usage by 60%"; post-training losses "up to 80% memory savings" ([README](https://github.com/linkedin/Liger-Kernel/blob/main/README.md)). Works with FSDP, DeepSpeed and DDP. Hardware support follows Triton's.
- **Integration:** TRL supports it in SFT, DPO, GRPO and KTO with `use_liger_kernel=True`. In DPO, GRPO and KTO, the chunked log-prob path "does not support … PEFT adapters on `lm_head`" ([TRL Liger integration](https://huggingface.co/docs/trl/liger_kernel_integration)). In TRL v1.14.1, `RLOOTrainer` and `GRPOTrainer` warn: "`use_liger_kernel=True` is deprecated and will be removed in v2.0.0. Use the Hub kernels instead, with `model_init_kwargs={"use_kernels": True}`." `RLOOTrainer` also disables Liger's fused cross-entropy ([`rloo_trainer.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/rloo_trainer.py), [`grpo_trainer.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)).
- **Model coverage:** 52 model types (Llama, Mistral, Mixtral, Qwen2/3/3.5 + MoE + VL, Gemma 1–4, Phi-3, GLM-4, OLMo 2/3, gpt-oss, DeepSeek-V3/V4, Granite, SmolLM3 and more). For other models it logs "There are currently no Liger kernels supported for model type: …" and **trains normally without it** ([`monkey_patch.py`](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/monkey_patch.py)). So it fails safe.
- **Verdict:** not a backend. Optionally a `liger` boolean on the `hf` backend (default off), or leave it out of v1 and revisit once TRL's Hub-kernels path settles.

## Axolotl and LLaMA-Factory as wrappers

- **Axolotl** (Apache-2.0, v0.20.0): `RLType` is `dpo, gdpo, grpo, ipo, orpo, kto, simpo, ebft`, so **no RLOO** ([`schemas/enums.py`](https://github.com/axolotl-ai-cloud/axolotl/blob/main/src/axolotl/utils/schemas/enums.py), [RLHF docs](https://github.com/axolotl-ai-cloud/axolotl/blob/main/docs/rlhf.qmd)). It requires **`axolotl-contribs-lgpl==0.0.7` (LGPL-3.0)** ([pyproject.toml](https://github.com/axolotl-ai-cloud/axolotl/blob/main/pyproject.toml), [PyPI](https://pypi.org/project/axolotl-contribs-lgpl/)), pins `trl==1.13.0` and `liger-kernel==0.8.1`, and has PostHog telemetry on by default (see the finetuning-library findings). As a backend it would need the Pipeline Request translated into its YAML schema, which is a second config surface to keep in sync. Not worth it in v1; its kernel benefits are reachable through Liger on `hf`.
- **LLaMA-Factory** (Apache-2.0, v0.9.5): `stage: Literal["pt", "sft", "rm", "ppo", "dpo", "kto"]`, so **no GRPO and no RLOO** ([`finetuning_args.py`](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/hparams/finetuning_args.py)). It has its own `use_unsloth` and `enable_liger_kernel` switches, which shows the same kernels can be had without it. It also needs per-model templates and dataset registration, and its last release is from May 2026 (finetuning-library findings). Not a backend.

## Licence review needed before shipping `unsloth`

The map's rule is "free for commercial use; flag AGPL, SSPL, BSL". For `unsloth` that means flagging:

1. **LGPL-3.0-or-later `unsloth_zoo`** (required). Fine to use unmodified; keep notices; don't vendor or patch it.
2. **AGPL-3.0 Studio code inside the `unsloth` wheel** (never run). Either accept it as unused, or strip `site-packages/studio` in the image build (needs a test that `import unsloth` still works).
3. **Apple licence `cut_cross_entropy`** (no patent grant).
4. **Paid Pro/Enterprise tiers exist.** v1 uses only the free tier, and the missing official multi-GPU support is the visible gap.

If any of these is unacceptable, v1 ships `hf` only, and `unsloth` stays a documented future backend behind the same `backend:` setting.

## Open questions for other tickets

- **[What is the shape of the Pipeline Request, and what does validation check?](https://github.com/Maxi580/mlopsplatform/issues/8):** the `backend` key and its default (`hf`); per-backend defaults (`dropout` 0 for `unsloth`); validation that rejects `unsloth` with more than one GPU; whether a `liger` flag exists.
- **Container images** (map "Not yet specified"): two `finetune` images, `hf` and `unsloth`, each pinned and on cu128+.
- **[How does reinforcement learning fit the platform: Stage, methods, rewards, environments?](https://github.com/Maxi580/mlopsplatform/issues/19):** whether GRPO/RLOO under `unsloth` use its in-process vLLM (`fast_inference`) or the same generation setup as `hf`.
- **Smoke Test:** one `lora` SFT run per backend on a tiny model, plus the shared output check.

## Sources

All accessed 2026-10-01.

**Unsloth**
- Repo, README (licence section, feature claims, Blackwell pointer): https://github.com/unslothai/unsloth , https://github.com/unslothai/unsloth/blob/main/README.md
- `pyproject.toml` (Apache-2.0 core, `unsloth_zoo` dependency, Studio packages in the wheel, extras pins): https://github.com/unslothai/unsloth/blob/main/pyproject.toml
- AGPL text and Studio licence: https://github.com/unslothai/unsloth/blob/main/COPYING , https://github.com/unslothai/unsloth/tree/main/studio
- TRL patching: https://github.com/unslothai/unsloth/blob/main/unsloth/models/rl.py , https://github.com/unslothai/unsloth/blob/main/unsloth/models/rl_replacements.py
- `get_peft_model` defaults, embedding redirect, new-token handling: https://github.com/unslothai/unsloth/blob/main/unsloth/models/llama.py
- Loader (`use_exact_model_name`, FastModel, unsupported Transformers message): https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py
- Usage statistics: https://github.com/unslothai/unsloth/blob/main/unsloth/models/_utils.py
- unsloth-zoo (LGPL-3.0-or-later, TRL/Transformers caps, `cut_cross_entropy`): https://github.com/unslothai/unsloth-zoo , https://github.com/unslothai/unsloth-zoo/blob/main/pyproject.toml
- Docs: multi-GPU https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth ; DDP https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth/ddp ; Blackwell https://unsloth.ai/docs/blog/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth ; RL guide https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide ; FAQ https://unsloth.ai/docs/basics/troubleshooting-and-faqs ; pricing https://www.unsloth.ai/pricing

**Apple Cut Cross Entropy**
- Licence: https://github.com/apple/ml-cross-entropy/blob/main/LICENSE

**Liger Kernel**
- Repo, README (BSD-2-Clause, claims, requirements): https://github.com/linkedin/Liger-Kernel , https://github.com/linkedin/Liger-Kernel/blob/main/README.md
- Model coverage and unsupported-model behaviour: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/monkey_patch.py

**TRL**
- Liger integration: https://huggingface.co/docs/trl/liger_kernel_integration
- Deprecation in RLOO/GRPO (v1.14.1): https://github.com/huggingface/trl/blob/main/trl/trainer/rloo_trainer.py , https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py

**Axolotl**
- RL types: https://github.com/axolotl-ai-cloud/axolotl/blob/main/src/axolotl/utils/schemas/enums.py ; RLHF docs: https://github.com/axolotl-ai-cloud/axolotl/blob/main/docs/rlhf.qmd
- Dependencies: https://github.com/axolotl-ai-cloud/axolotl/blob/main/pyproject.toml ; `axolotl-contribs-lgpl`: https://pypi.org/project/axolotl-contribs-lgpl/

**LLaMA-Factory**
- Stages and `use_unsloth`/`enable_liger_kernel`: https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/hparams/finetuning_args.py , https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/hparams/model_args.py

**Earlier platform research**
- Finetuning library: https://github.com/Maxi580/mlopsplatform/blob/research/finetuning-library/docs/research/finetuning-library.md
- Training methods: https://github.com/Maxi580/mlopsplatform/blob/research/training-methods/docs/research/training-methods.md
