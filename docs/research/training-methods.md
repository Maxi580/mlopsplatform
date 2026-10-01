# Which training methods should the `finetune` Stage offer in v1?

Research ticket: [#22](https://github.com/Maxi580/mlopsplatform/issues/22) (parent map: [#1](https://github.com/Maxi580/mlopsplatform/issues/1)).
Researched on **2026-10-01** against TRL v1.14.1, PEFT v0.21.1, vLLM `main` (v0.30.x), lm-evaluation-harness `main` (v0.4.13) and bitsandbytes 0.50.2. Re-check before pinning versions.

Builds on earlier decisions:

- [Which finetuning library should the finetune Stage use?](https://github.com/Maxi580/mlopsplatform/issues/4): TRL `SFTTrainer` + PEFT ([findings](https://github.com/Maxi580/mlopsplatform/blob/research/finetuning-library/docs/research/finetuning-library.md)).
- [Which evaluation harness should the evaluate Stage use?](https://github.com/Maxi580/mlopsplatform/issues/5): lm-evaluation-harness.
- [Which serving engine and inference optimizations for the serve Stage?](https://github.com/Maxi580/mlopsplatform/issues/6): vLLM, serving Adapters unmerged.
- [Where do datasets and models live, and how are they registered and versioned?](https://github.com/Maxi580/mlopsplatform/issues/10): `finetune` publishes an **Adapter** for PEFT runs and **full weights** for full finetuning, with **no merge option**. Datasets are JSONL in TRL's standard row formats.

## Recommendation

**v1 offers one training objective (SFT) and three methods: `full`, `lora`, `qlora`.**

| Method | Output (Model Version) | vLLM serves it | lm-eval evaluates it | Fits 16 GB RTX 5060 Ti |
|---|---|---|---|---|
| `full` | full weights | Yes, as a normal model | Yes (`hf` or `vllm` backend, or the Endpoint) | Small models only (about ≤ 1B params, estimate) |
| `lora` (default) | Adapter | Yes, unmerged (`--enable-lora`) | Yes (`peft=` on `hf`, `lora_local_path=` on `vllm`) | Yes, for small-to-mid models (about ≤ 3–4B, estimate) |
| `qlora` | Adapter (same file format as `lora`) | Yes, on the bf16 Base Model | Yes, same as `lora` | Yes, up to about 7–8B (estimate) |

Everything else is left out of v1, mainly because the registry decision forbids merging and **vLLM serves plain LoRA only**. Any PEFT method that vLLM cannot load as an Adapter would produce a Model Version the `serve` Stage cannot serve.

Why this list:

1. **It is the common path.** Full finetuning, LoRA and QLoRA are what TRL's own SFT docs show. The `peft_config` and `quantization_config` arguments of `SFTTrainer` exist exactly for LoRA and QLoRA ([SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)).
2. **Every output works in every later Stage without merging.** vLLM's adapter loader rejects DoRA and anything but plain LoRA weights (see [Serving constraints](#serving-constraints-vllm)). lm-eval's `hf` backend loads any PEFT Adapter, but the `serve` Stage is the stricter gate.
3. **All three run on the dev GPU.** bitsandbytes ships sm_120 (Blackwell, RTX 50 series) wheels for CUDA 12.8 and 13.x ([bitsandbytes install](https://huggingface.co/docs/bitsandbytes/main/en/installation)).
4. **Small configuration surface.** QLoRA adds no user settings over LoRA; the quantization settings are fixed by the platform.

**rsLoRA** is the one optional extra worth a boolean. vLLM reads `use_rslora` from `adapter_config.json` and applies the `alpha/sqrt(r)` scaling ([vLLM peft_helper.py](https://github.com/vllm-project/vllm/blob/main/vllm/lora/peft_helper.py)). It costs one flag, defaults to off, and matters only at high ranks. Dropping it is also fine.

## Training objective: SFT, with the loss following the Dataset format

TRL's `SFTTrainer` decides which tokens count toward the loss from the row format ([SFTConfig `completion_only_loss`](https://huggingface.co/docs/trl/sft_trainer)):

| Dataset row format | Loss by default (TRL v1.14.1) | Use |
|---|---|---|
| prompt-completion (`{"prompt":…, "completion":…}`, standard or conversational) | **Completion only** (`completion_only_loss=None` resolves to completion-only) | Distillation Datasets, instruction data. The recommended format. |
| language modeling, text (`{"text": …}`) | Full sequence | **Continued pretraining** on raw text. No extra method needed. |
| language modeling, conversational (`{"messages": […]}`) | **Full sequence, including user turns** | Multi-turn chats. Assistant-only loss needs `assistant_only_loss=True` **and** a chat template with `{% generation %}` tags. TRL patches the template only for known families such as Qwen3 ([SFTTrainer: train on assistant messages only](https://huggingface.co/docs/trl/sft_trainer)). |

So "completion-only vs full-sequence loss" and "continued pretraining" are **not separate methods** in v1. They fall out of the Dataset's row format, which the registry already validates on upload. No loss toggle is needed in v1.

Other SFT knobs left at TRL defaults and not exposed: `loss_type` (default `chunked_nll`, the same math as plain NLL with less memory; `dft` is an opt-in research variant), `packing` (default off), NEFTune (`neftune_noise_alpha`, off).

## Settings per method and defaults

Library-neutral names; the final keys belong to [What is the shape of the Pipeline Request, and what does validation check?](https://github.com/Maxi580/mlopsplatform/issues/8).

**Common to all methods**

| Setting | Default | Notes |
|---|---|---|
| `method` | `lora` | `full` \| `lora` \| `qlora` |
| `epochs` | 3 | TRL default `num_train_epochs=3.0` |
| `learning_rate` | `2e-5` for `full`, `1e-4` for `lora`/`qlora` | TRL default is `2e-5`; TRL's docs recommend "≈1e-4" for adapters ([SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)) |
| `batch_size` | 8 | per device; TRL default `per_device_train_batch_size=8`. Lower it if the job runs out of memory. |
| `gradient_accumulation` | 1 | TRL default |
| `max_length` | 1024 | TRL default; longer sequences are truncated |

**Fixed by the platform (not user settings):** `bf16=True`, `gradient_checkpointing=True` (both TRL defaults), `report_to=mlflow`, and **`model_init_kwargs={"dtype": torch.bfloat16}`**. The last one matters: when `SFTTrainer` gets a model id, "if `dtype` is not specified … it defaults to `float32`" ([SFTTrainer `model` argument](https://huggingface.co/docs/trl/sft_trainer)). Without it, full finetuning would load fp32 weights and double the memory.

**`lora` and `qlora`**

| Setting | Default | PEFT default | Notes |
|---|---|---|---|
| `rank` (`r`) | 16 | 8 | Must be ≤ the `serve` Stage's `--max-lora-rank`, which accepts only 1, 8, 16, 32, 64, 128, 256, 320 or 512 (default 16) ([vLLM LoRAConfig](https://github.com/vllm-project/vllm/blob/main/vllm/config/lora.py)). Validation can cap `rank` at 512. |
| `alpha` | 32 (2 × rank) | 8 | |
| `dropout` | 0.05 | 0.0 | |
| `target_modules` | `all-linear` | `None` (per-architecture defaults) | "To apply LoRA to all the linear layers, like in QLoRA, set `target_modules="all-linear"`" ([PEFT LoRA guide](https://huggingface.co/docs/peft/main/en/developer_guides/lora)) |
| `rslora` (optional) | `false` | `false` | Scales by `alpha/sqrt(r)` instead of `alpha/r` ([PEFT LoRA guide](https://huggingface.co/docs/peft/main/en/developer_guides/lora)) |

Platform-fixed for LoRA: `bias="none"` and no `modules_to_save`, because vLLM rejects both (see below).

Platform-fixed for `qlora`: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=bfloat16)`, passed as `quantization_config` to `SFTTrainer`, which TRL documents as "Combine with `peft_config` for QLoRA training" ([SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)). The **8-bit** variant is left out: 4-bit NF4 is the QLoRA standard, and 8-bit saves less memory with no other benefit here.

## Serving constraints (vLLM)

From vLLM's adapter loader ([`vllm/lora/peft_helper.py`](https://github.com/vllm-project/vllm/blob/main/vllm/lora/peft_helper.py)), which reads `adapter_config.json`:

- **DoRA is rejected:** `"vLLM does not yet support DoRA."` when `use_dora` is true.
- **`modules_to_save` is rejected** unless it is `None` or `["classifier", "score"]` (classification heads). So an Adapter that also trains `embed_tokens` or `lm_head` (for example after adding special tokens for a new chat template) cannot be served.
- **Adapter bias is rejected:** `"Adapter bias is not supported."`
- **Rank** must be ≤ `max_lora_rank`.
- **rsLoRA is supported:** `use_rslora` switches the scaling to `lora_alpha / math.sqrt(r)`.

vLLM's LoRA docs cover only LoRA adapters ([vLLM LoRA docs](https://docs.vllm.ai/en/latest/features/lora/)). Prompt adapters (prompt tuning) were removed with the V0 engine; the `--enable-prompt-adapter` flag has no effect ([vLLM v0.10.0 CLI reference](https://docs.vllm.ai/en/v0.10.0/cli/index.html), [V0 deprecation RFC #18571](https://github.com/vllm-project/vllm/issues/18571)). IA3, OFT/BOFT, LoKr, VeRA, prefix tuning and the other PEFT tuners have no vLLM loader at all.

**QLoRA Adapters are served on the bf16 Base Model.** The Adapter file is a plain LoRA; vLLM loads it on the unquantized Base Model from the Model Cache. That is a small train/serve mismatch (trained against NF4 weights, served against bf16), which is the standard way QLoRA Adapters are used.

**Consequence for the base-model chat template.** If a *base* (non-instruct) Base Model gets a chat template whose special tokens are new, the embeddings must be trained (`modules_to_save` or `trainable_token_indices`), which vLLM will not serve. In v1, either use `full` for that case or pick a template that reuses existing tokens. TRL's own docs warn that a Jinja template's special tokens must be added and the embedding resized ([SFTConfig `chat_template_path`](https://huggingface.co/docs/trl/sft_trainer)).

## Evaluation constraints (lm-evaluation-harness)

- **`hf` backend:** `peft=<path>` loads the Adapter with `PeftModel.from_pretrained(self._model, peft, revision=revision)`, so **any PEFT method** loads. It also resizes embeddings if the tokenizer grew ([`lm_eval/models/huggingface.py`](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py)).
- **`vllm` backend:** `lora_local_path=` builds a vLLM `LoRARequest`, so the same plain-LoRA limits as serving apply ([`lm_eval/models/vllm_causallms.py`](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/vllm_causallms.py)).
- **Endpoint:** `local-completions` against a running Endpoint evaluates whatever the `serve` Stage serves.

So lm-eval is never the limiting factor; vLLM is.

## Every method considered

"TRL+PEFT" = trainable via `SFTTrainer` with `peft_config` / `quantization_config`. "vLLM" = servable as an unmerged Adapter or as full weights.

| Method | TRL+PEFT | vLLM | lm-eval | Registry fit (no merge) | v1? | Why |
|---|---|---|---|---|---|---|
| Full finetuning | Yes (no `peft_config`) | Yes | Yes | Yes (full weights) | **Yes** | Baseline; best quality for small models |
| LoRA | Yes | Yes | Yes | Yes (Adapter) | **Yes (default)** | Most common, cheap, servable unmerged |
| QLoRA 4-bit (NF4) | Yes (bitsandbytes) | Yes (on bf16 base) | Yes | Yes (Adapter) | **Yes** | Bigger models on 16 GB |
| QLoRA 8-bit | Yes | Yes | Yes | Yes | No | Less savings than 4-bit; no extra benefit |
| rsLoRA | Yes (`use_rslora`) | Yes (scaling handled) | Yes | Yes | **Optional flag** | One boolean; helps only at high rank |
| DoRA | Yes (`use_dora`) | **No** ("does not yet support DoRA") | Yes (`hf`) | Would need merging | No | Not servable without merge; PEFT itself recommends merging for inference due to overhead ([PEFT LoRA guide](https://huggingface.co/docs/peft/main/en/developer_guides/lora)) |
| LoRA+ | PEFT `create_loraplus_optimizer`; needs a custom optimizer passed to the trainer | Yes (output is plain LoRA) | Yes | Yes | No | Not an `SFTConfig` option; needs our own optimizer wiring for a claimed "up to 2x" speed / "1–2%" quality gain |
| PiSSA / OLoRA / CorDA init | Yes (`init_lora_weights`) | Only after converting to plain LoRA with `path_initial_model_for_weight_conversion` (these inits modify the base weights) | Same | Only with the conversion step | No | Extra save-time conversion step; incompatible with rsLoRA + rank patterns ([PEFT LoraConfig](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py)) |
| LoftQ / EVA / other inits | Yes | Mostly plain LoRA output | Yes | Yes | No | Niche; more surface than value in v1 |
| IA3 | Yes | **No** | Yes (`hf`) | Would need merging | No | Not servable |
| Prompt / prefix / P-tuning | Yes | **No** (removed in vLLM V1) | Yes (`hf`) | Cannot be merged at all | No | Not servable |
| OFT/BOFT, LoKr, LoHa, VeRA, others | Yes | **No** | Yes (`hf`) | Would need merging | No | Not servable |
| Continued pretraining | Yes (text rows, full-sequence loss) | Yes | Yes | Yes | **Yes, via Dataset format** | Not a method; any of the three methods on `{"text":…}` rows |

## Hardware: the 16 GB RTX 5060 Ti and bigger GPUs

- **Blackwell support.** bitsandbytes' Linux x86-64 wheels for CUDA 12.8 and 13.0–13.2 include `sm120` ([bitsandbytes install](https://huggingface.co/docs/bitsandbytes/main/en/installation)). Latest release 0.50.2 (2026-08-27). PyTorch ≥ 2.7 cu128 covers the rest (see the [finetuning-library findings](https://github.com/Maxi580/mlopsplatform/blob/research/finetuning-library/docs/research/finetuning-library.md)). So QLoRA works on the dev GPU as long as the image uses a cu128+ build.
- **Rough memory fit (estimates, not measured):** full finetuning with AdamW needs about 16 bytes per parameter before activations, so about ≤ 1B params on 16 GB. LoRA keeps bf16 weights (2 bytes/param) plus small adapter state, so about ≤ 3–4B. QLoRA keeps about 0.5 bytes/param, so about 7–8B. `gradient_checkpointing` (on by default) and a smaller `batch_size`/`max_length` stretch these.
- **Bigger GPUs.** `full` on bigger models needs sharding: Accelerate presets `fsdp2` or `zero3` (from the library decision). `lora` scales with DDP or FSDP2. Keep `qlora` **single-GPU in v1**; QLoRA with FSDP needs extra settings (quantized storage dtype) that are not worth supporting until someone needs it. Not verified here.

## Preference and RL methods (listed only)

The choice belongs to [How does reinforcement learning fit the platform: Stage, methods, rewards, environments?](https://github.com/Maxi580/mlopsplatform/issues/19). How they relate, per TRL v1.14.1's taxonomy ([TRL index](https://huggingface.co/docs/trl/index)):

- **Offline preference (stable):** `DPOTrainer` (chosen/rejected pairs), `KTOTrainer` (unpaired good/bad labels). They run after SFT on preference data, and take the same `peft_config`, so the same `full`/`lora`/`qlora` choice and Adapter output would apply.
- **Offline preference (experimental in TRL):** `ORPOTrainer`, `CPOTrainer`. ORPO folds SFT and preference into one step.
- **Online RL (stable):** `GRPOTrainer`, `RLOOTrainer`, plus `RewardTrainer`. They need a reward function or model and generation during training (TRL can co-locate vLLM).
- **PPO** no longer appears in TRL's trainer list.
- **`DistillationTrainer`** (stable) does logit-level distillation; the platform's Distillation Dataset is response-level, which is plain SFT.

## Open questions for other tickets

- **[What is the shape of the Pipeline Request, and what does validation check?](https://github.com/Maxi580/mlopsplatform/issues/8):** final key names; whether to keep the `rslora` flag; whether validation warns on `{"messages": …}` rows that the loss also covers user turns.
- **[How does reinforcement learning fit the platform: Stage, methods, rewards, environments?](https://github.com/Maxi580/mlopsplatform/issues/19):** whether DPO/KTO reuse the same three methods and Adapter output.
- **Serve Stage:** set `--max-lora-rank` to the smallest allowed value ≥ the Adapter's `r` (read from `adapter_config.json`).
- **Smoke Test:** one `lora` run on a tiny model covers the default path; `qlora` needs a GPU (bitsandbytes CUDA).

## Sources

All accessed 2026-10-01.

- TRL SFT Trainer and SFTConfig (v1.14.1): https://huggingface.co/docs/trl/sft_trainer
- TRL trainer taxonomy (v1.14.1): https://huggingface.co/docs/trl/index
- PEFT LoRA developer guide (rsLoRA, DoRA, LoRA+, PiSSA/OLoRA conversion, `all-linear`): https://huggingface.co/docs/peft/main/en/developer_guides/lora
- PEFT `LoraConfig` source (defaults, init options): https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py
- vLLM LoRA docs: https://docs.vllm.ai/en/latest/features/lora/
- vLLM adapter config validation (`peft_helper.py`): https://github.com/vllm-project/vllm/blob/main/vllm/lora/peft_helper.py
- vLLM `LoRAConfig` (`max_lora_rank` allowed values): https://github.com/vllm-project/vllm/blob/main/vllm/config/lora.py
- vLLM prompt adapter removal: https://docs.vllm.ai/en/v0.10.0/cli/index.html , https://github.com/vllm-project/vllm/issues/18571
- lm-evaluation-harness `hf` backend (`peft=`): https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py
- lm-evaluation-harness `vllm` backend (`lora_local_path`): https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/vllm_causallms.py
- bitsandbytes installation (CUDA/sm targets): https://huggingface.co/docs/bitsandbytes/main/en/installation
- bitsandbytes releases (0.50.2, 2026-08-27; MIT): https://github.com/bitsandbytes-foundation/bitsandbytes/releases

Licenses: TRL, PEFT, Transformers, Accelerate and vLLM are Apache-2.0; bitsandbytes and lm-evaluation-harness are MIT. All are free for commercial use.
