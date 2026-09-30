# Research: serving engine and inference optimizations for the `serve` Stage

Resolves [#6](https://github.com/Maxi580/mlopsplatform/issues/6). Researched 2026-10-01. The facts apply to **vLLM v0.30.0** (2026-09-22), **SGLang v0.5.20** (2026-09-18), **TGI v3.3.7** (2025-12-19, the last release), **KServe v0.21.0** (2026-09-25) and **llm-compressor 0.14.0** (2026-09-22). Versions and dates come from each project's GitHub releases.

## Question

Which serving engine should the `serve` Stage use, and which inference optimizations can it offer? The same engine may also run an in-cluster **Teacher** for the `distill` Stage behind an OpenAI-compatible API. The candidates are vLLM, SGLang and TGI, each with or without KServe in front. They are compared on serving (OpenAI API, unmerged LoRA), optimizations, Blackwell support, Kubernetes deployment, S3 weight loading, licensing and maintenance.

## Recommendation

**Engine: vLLM** (Apache-2.0), using the official `vllm/vllm-openai` image.

- **TGI is out.** Its repository is archived and its README says it "is now in maintenance mode", and it recommends vLLM and SGLang instead [TGI-README]. It has had no release since v3.3.7 in December 2025.
- **vLLM beats SGLang for this platform**, although the two are close on features. Both are Apache-2.0 and very actively maintained. Both have OpenAI-compatible servers, multi-LoRA serving without merging, AWQ/GPTQ/FP8/NVFP4, EAGLE-3 and n-gram speculative decoding, prefix caching, and `runai_streamer` S3 loading. vLLM is ahead on three points that matter here:
  1. **Offline quantization comes from the same project.** llm-compressor is a vLLM-project tool and absorbed AutoAWQ. Its `compressed-tensors` output loads directly in vLLM [LLMC] [AUTOAWQ].
  2. **The KServe ecosystem runs on vLLM.** KServe's Hugging Face runtime and `LLMInferenceService` both use vLLM [KS-GEN] [KS-LLMISVC]. Picking vLLM keeps KServe open as a later option without changing engines.
  3. **Adapters address cleanly in the OpenAI API.** A LoRA adapter is selected just by setting `model` to the adapter's name [VLLM-LORA]. SGLang needs `base:adapter` syntax [SGL-LORA]. That is fine for our own clients, but plain OpenAI clients such as the `distill` Stage's Teacher client work more naturally with vLLM.

  SGLang stays a credible fallback. It's worth re-checking if throughput for the Teacher ever becomes the bottleneck.

**KServe or a plain Deployment: a plain Deployment + Service for v1.** KServe doesn't earn its place yet:

- The `serve` Stage runs one model, or one base model with adapters, per Deployment. The API already knows the model's S3 location and the engine flags. vLLM's own Kubernetes guide uses exactly a Deployment + Service with `/health` probes and a shared-memory volume [VLLM-K8S].
- KServe Standard mode needs cert-manager and a Gateway API or Ingress controller [KS-K8S]. **Scale from zero isn't supported in Standard mode** for HTTP [KS-K8S]. Scale-to-zero needs the Knative (Serverless) mode [KS-ADMIN], which brings Knative into the install. So the one feature that would clearly beat a Deployment, scale-to-zero, costs the heaviest dependency.
- `LLMInferenceService` (llm-d based) targets prefix-aware routing and disaggregated serving across many replicas [KS-ADMIN] [KS-LLMISVC]. That is overkill on one VM with few GPUs.
- KServe's storage initializer copies the whole model from S3 to local disk before start-up [KS-S3]. vLLM can stream from S3 directly (see [Loading weights from S3](#loading-weights-from-s3)).
- Stopping a served model is simply scaling or deleting the Deployment. That decision belongs to grilling ticket #11.

Revisit KServe if the VM grows to many GPUs and many concurrent served models, or if scale-to-zero becomes a hard requirement.

**Optimizations to offer in the `serve` Stage for v1:**

- **Prefix caching:** always on. It's the vLLM default.
- **Quantization:**
  - Serve pre-quantized checkpoints as they are (AWQ, GPTQ, FP8, NVFP4, `compressed-tensors`).
  - Offer on-the-fly FP8 at load time (`--quantization fp8`) as the zero-effort option.
  - Offline quantization with llm-compressor could be an optional step that writes a new model to S3. Where that step lives, and whether it's in v1, is an open question.
- **FP8 KV cache** (`--kv-cache-dtype fp8`).
- **Speculative decoding:** n-gram needs no extra model. Draft-model or EAGLE-3 needs a user-supplied draft or speculator.

## Comparison table

| | **vLLM** | **SGLang** | **TGI** |
|---|---|---|---|
| Latest release | v0.30.0, 2026-09-22 | v0.5.20, 2026-09-18 | v3.3.7, 2025-12-19; repo **archived** |
| License | Apache-2.0 [GH-VLLM] | Apache-2.0 [GH-SGL] | Apache-2.0 [GH-TGI] |
| Maintenance, 2026 | Very active; releases every ~2 weeks | Very active | **Maintenance mode**, archived [TGI-README] |
| OpenAI-compatible API | Yes (`vllm serve`) | Yes | Yes (historically) |
| Unmerged multi-LoRA | Yes: `--enable-lora`, `--lora-modules`, `--max-loras`, `--max-lora-rank`; runtime `/v1/load_lora_adapter` behind `VLLM_ALLOW_RUNTIME_LORA_UPDATING`; adapter = `model` name [VLLM-LORA] | Yes: `--enable-lora`, `--lora-paths`, `--max-loras-per-batch` (default 8); `/load_lora_adapter`; OpenAI name `base:adapter` [SGL-LORA] | Yes, but frozen |
| LoRA from S3 | Via a LoRAResolver plugin. Built-ins are filesystem and HF Hub only; S3 needs a custom resolver (example given) [VLLM-LORA] | Local paths (not verified for S3) | n/a |
| AWQ / GPTQ | Yes. GPTQ `g_idx` (act-order) **removed** in v0.30.0 [VLLM-REL] | Yes [SGL-ARGS] | Frozen |
| FP8 | Pre-quantized and on-the-fly (`fp8`, `fp8_per_tensor`, `fp8_per_block`, …) [VLLM-QSRC] | Pre-quantized and `w8a8_fp8`, `modelopt_fp8` [SGL-ARGS] | Frozen |
| NVFP4 (Blackwell) | Yes; "W4A4 NVFP4 preferred on SM120/121" in v0.30.0 [VLLM-REL] | Yes (`modelopt_fp4`, `nvfp4_online`) [SGL-ARGS] | No |
| bitsandbytes | **Out-of-tree** now: `vllm-bnb-plugin` [VLLM-BNB] | In-tree (`--quantization bitsandbytes`) [SGL-ARGS] | Frozen |
| GGUF | Out-of-tree `vllm-gguf-plugin`, experimental [VLLM-GGUF] | In-tree [SGL-ARGS] | No |
| Speculative decoding | EAGLE, MTP, draft model, PARD, MLP, n-gram, suffix decoding via `--speculative-config` [VLLM-SD] | EAGLE/EAGLE-3, MTP, UNO, DFlash, STANDALONE draft, NGRAM; EAGLE-3 recommended [SGL-SD] | Medusa / n-gram (frozen) |
| Prefix caching | On by default (`enable_prefix_caching: bool = True`) [VLLM-CACHE] | RadixAttention on by default (`--disable-radix-cache`) [SGL-ARGS] | Yes (frozen) |
| Blackwell / CUDA | Wheels and image on **CUDA 13.0** in v0.30.0; NVFP4 tuned for SM120 [VLLM-REL]; CUDA 12.8 wheels still documented [VLLM-INST] | **CUDA 13 only**; last CUDA 12 image is `v0.5.19-cu129` [SGL-INST] | Last images predate the CUDA 13 move |
| S3 weights | `--load-format runai_streamer`, `s3://` model path, custom `AWS_ENDPOINT_URL`; safetensors only [VLLM-RUNAI] | `--load-format runai_streamer` / `remote` [SGL-ARGS] | Download only |
| K8s guidance | Plain Deployment + Service; also KServe, KubeRay, KAITO, production-stack [VLLM-K8S] | Sample Deployment / StatefulSet YAML; OME operator [SGL-INST] | — |
| Behind KServe | HF runtime and LLMInferenceService both use vLLM [KS-GEN] [KS-LLMISVC] | Custom ServingRuntime only | Formerly a KServe runtime |

**KServe (v0.21.0, Apache-2.0)** adds the following on top of any engine:

- **Modes:** Standard (Deployment + HPA, optional KEDA; no HTTP scale-from-zero), Knative (scale-to-zero) and `LLMInferenceService` [KS-ADMIN] [KS-K8S].
- **Requirements:** Kubernetes 1.32+, cert-manager 1.15+, and Gateway API or Ingress [KS-K8S].
- **Model loading:** S3 download through the storage initializer [KS-S3], plus LocalModelCache [KS-GEN].

## Inference optimizations catalogue

"Offline" means a separate step that writes a new model artifact before serving. "Load time" means the engine does it when the Deployment starts, with no extra artifact.

| Optimization | What it is | vLLM | SGLang | Offline or load time |
|---|---|---|---|---|
| **Prefix caching** | Reuses the KV cache of shared prompt prefixes such as system prompts and few-shot examples | Default on [VLLM-CACHE] | Default on, RadixAttention [SGL-ARGS] | Runtime; nothing to do |
| **AWQ (W4A16)** | Activation-aware 4-bit weight-only quantization; needs calibration data | Serves AWQ checkpoints [VLLM-QSRC] | Yes [SGL-ARGS] | **Offline.** llm-compressor; AutoAWQ is deprecated and was adopted into it [AUTOAWQ] [LLMC] |
| **GPTQ (W4A16 / W8A16)** | Second-order 4/8-bit weight-only quantization; needs calibration data | Yes, but no act-order `g_idx` since v0.30.0 [VLLM-REL] | Yes | **Offline.** llm-compressor or GPTQModel [SGL-Q] |
| **FP8 (W8A8)** | 8-bit float weights and activations; about half the memory of BF16 with small quality loss | Pre-quantized **or** on the fly (`--quantization fp8`) [VLLM-QSRC] | Pre-quantized or on the fly | **Either.** Offline with llm-compressor (better scales), or at load time with dynamic scales |
| **NVFP4 / MXFP4** | 4-bit float micro-scaled formats with native Blackwell tensor-core support | Yes, tuned for SM120 [VLLM-REL] | Yes (`modelopt_fp4`, `nvfp4_online`) | Mostly **offline** (llm-compressor, NVIDIA ModelOpt); SGLang also has online NVFP4 [SGL-Q] |
| **bitsandbytes (NF4 / INT8)** | In-flight 4/8-bit quantization at load, no calibration; the QLoRA base format | Out-of-tree plugin `vllm-bnb-plugin` [VLLM-BNB] | In-tree | **Load time**; it can also load pre-quantized bnb checkpoints |
| **FP8 KV cache** | Stores the KV cache in FP8, which doubles context and batch capacity per GB | `--kv-cache-dtype fp8` (also int8/int4/nvfp4 variants) [VLLM-ARGS] | Supported | **Load time** |
| **Speculative decoding: n-gram / suffix** | Drafts tokens by matching n-grams from the prompt; no extra model | Yes [VLLM-SD] | NGRAM [SGL-SD] | **Load time**; config only |
| **Speculative decoding: draft model** | A small model of the same tokenizer family proposes tokens and the target verifies them | Yes [VLLM-SD] | STANDALONE [SGL-SD] | **Load time**, but needs a second model in S3 |
| **Speculative decoding: EAGLE / EAGLE-3 / MTP** | Small trained head on the target's hidden states; MTP uses heads built into the model | Yes [VLLM-SD] | Yes; EAGLE-3 recommended [SGL-SD] | Needs a **trained speculator for that exact target** (offline training). MTP only for models that ship MTP heads |
| **Unmerged multi-LoRA** | Serve many adapters on one base without merging | Yes [VLLM-LORA] | Yes [SGL-LORA] | **Load time** (or runtime load) |

Notes:

- **SGLang advises against online quantization.** It says "offline quantization is recommended over online quantization" [SGL-Q]. Our default can still be on-the-fly FP8, because it needs no extra step, while offline is the quality/speed upgrade.
- **A finetuned model breaks its EAGLE head.** A published EAGLE head is trained against the original base model. After full finetuning it will accept fewer tokens, so the speedup drops. LoRA-served models are affected less. N-gram has no such issue. This is an inference from how EAGLE works, not a documented claim.
- **llm-compressor runs as a GPU Job,** since calibration needs a forward pass. It writes a `compressed-tensors` safetensors checkpoint, which vLLM loads directly [LLMC].

## Loading weights from S3

vLLM has two options, and a third comes with KServe.

1. **Stream directly from S3** with `--load-format runai_streamer` [VLLM-RUNAI].
   - Install the `vllm[runai]` extra. Check whether the official image already includes it before building a custom image.
   - Serve with `vllm serve s3://bucket/path --load-format runai_streamer`.
   - For an S3-compatible store, set `AWS_ENDPOINT_URL`, `RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0` (path-style) and `AWS_EC2_METADATA_DISABLED=true`. Credentials come from the usual `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in a Kubernetes Secret.
   - **Safetensors only.** Tensors stream straight to GPU memory with configurable concurrency, and nothing is copied to local disk.
   - This works with any S3-compatible store chosen in #3 or #7, as long as it supports path-style addressing.
2. **Download first, then serve.** An init container (or the Model Cache from #7) copies the model to a PVC or `emptyDir`, and vLLM serves the local path. This is simpler to debug and works for non-safetensors checkpoints. It costs a full copy per start, unless the Model Cache is a shared volume.
3. **KServe storage initializer.** It is the same as option 2, but configured with `serving.kserve.io/s3-endpoint` annotations on a Secret or ServiceAccount [KS-S3]. It is only relevant if KServe is adopted.

**LoRA adapters from S3:** vLLM's built-in LoRA resolvers cover only the local filesystem and HF Hub [VLLM-LORA]. The pragmatic way is to sync adapters to a local directory (init container or sidecar) and pass `--lora-modules` or use `lora_filesystem_resolver`. The alternative is a small custom S3 LoRAResolver plugin; the docs give an example [VLLM-LORA].

## Open questions for the grilling tickets

- **#8 (Pipeline Request shape):**
  - Which `serve` fields are exposed? Candidates: `quantization` (none / `fp8` on the fly / use checkpoint's own), `kv_cache_dtype`, `speculative` (none / `ngram` / `draft_model` + model ref / `eagle3` + speculator ref), `max_model_len`, `gpu_memory_utilization`, `max_loras`.
  - Is it a curated subset, or a raw `engine_args` passthrough?
- **#8 / #10:** Is **offline quantization** its own Stage, a `serve` sub-step, or out of v1? Its output (a quantized model in S3) needs registering and versioning like any other model.
- **#11 (after a Pipeline finishes):** A plain Deployment doesn't scale to zero. Is a served model kept running, scaled to 0 replicas manually, or given a TTL? Is GPU contention with `finetune` Jobs handled by the GPU-queueing decision?
- **#7 / #10:** Is the S3 store's layout **safetensors-only**? That lets `runai_streamer` load directly. Are adapters stored separately from Base Models, and synced to a local directory?
- **#12 (Smoke Test):** The Smoke Test `serve` should use a tiny model with an on-the-fly FP8 (or no) quantization and n-gram speculation. That exercises the flags without an extra artifact.
- **New: GPU driver baseline.** vLLM v0.30.0 and SGLang both ship CUDA 13 images now [VLLM-REL] [SGL-INST]. CUDA 13.x needs an NVIDIA driver ≥ 580 [CUDA-RN]. The map says "CUDA 12.8+", so the Windows host driver (for WSL k3s) and the production VM driver must be ≥ 580, or the platform must pin a CUDA 12.x build. The CUDA 12.8 wheels are still documented [VLLM-INST], but there's no image. This also affects the `finetune` image (#4).
- **#9 (secrets):** `runai_streamer` needs S3 credentials in the serve pod. The HF token is needed only if the pod pulls from the Hub, which should be avoided by always loading from S3 or the Model Cache.

## Sources

All accessed 2026-10-01.

- [GH-VLLM] vLLM repository, license Apache-2.0 — https://github.com/vllm-project/vllm
- [VLLM-REL] vLLM v0.30.0 release notes (CUDA 13.0 default, NVFP4 on SM120, GPTQ `g_idx` removed) — https://github.com/vllm-project/vllm/releases/tag/v0.30.0
- [VLLM-LORA] vLLM LoRA adapters docs — https://docs.vllm.ai/en/latest/features/lora/
- [VLLM-SD] vLLM speculative decoding docs — https://docs.vllm.ai/en/latest/features/speculative_decoding/
- [VLLM-Q] vLLM quantization overview — https://docs.vllm.ai/en/latest/features/quantization/
- [VLLM-QSRC] vLLM `QuantizationMethods` at v0.30.0 — https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/quantization/__init__.py
- [VLLM-BNB] vLLM BitsAndBytes (`vllm-bnb-plugin`) — https://docs.vllm.ai/en/latest/features/quantization/bnb/
- [VLLM-GGUF] vLLM GGUF (`vllm-gguf-plugin`) — https://docs.vllm.ai/en/latest/features/quantization/gguf/
- [VLLM-CACHE] vLLM `CacheConfig.enable_prefix_caching = True` at v0.30.0 — https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/cache.py
- [VLLM-ARGS] vLLM engine arguments (`--load-format`, `--kv-cache-dtype`) — https://docs.vllm.ai/en/latest/configuration/engine_args/
- [VLLM-RUNAI] vLLM Run:ai Model Streamer — https://docs.vllm.ai/en/latest/models/extensions/runai_model_streamer/
- [VLLM-INST] vLLM GPU installation (Blackwell needs CUDA ≥ 12.8) — https://docs.vllm.ai/en/latest/getting_started/installation/gpu/
- [VLLM-K8S] vLLM on Kubernetes — https://docs.vllm.ai/en/latest/deployment/k8s/
- [GH-SGL] SGLang repository, license Apache-2.0 — https://github.com/sgl-project/sglang
- [SGL-LORA] SGLang LoRA serving — https://docs.sglang.io/advanced_features/lora.html
- [SGL-Q] SGLang quantization — https://docs.sglang.io/advanced_features/quantization.html
- [SGL-SD] SGLang speculative decoding — https://docs.sglang.io/advanced_features/speculative_decoding.html
- [SGL-ARGS] SGLang server arguments — https://docs.sglang.io/advanced_features/server_arguments.html
- [SGL-INST] SGLang install (CUDA 13 required; last CUDA 12 image v0.5.19-cu129; Kubernetes YAML) — https://docs.sglang.io/docs/get-started/install
- [GH-TGI] TGI repository (archived, Apache-2.0) — https://github.com/huggingface/text-generation-inference
- [TGI-README] TGI README maintenance-mode notice — https://github.com/huggingface/text-generation-inference/blob/main/README.md
- [KS-GEN] KServe generative inference overview — https://kserve.github.io/website/docs/model-serving/generative-inference/overview
- [KS-LLMISVC] KServe LLMInferenceService overview — https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/llmisvc-overview
- [KS-ADMIN] KServe admin guide, deployment modes — https://kserve.github.io/website/docs/admin-guide/overview
- [KS-K8S] KServe Standard-mode prerequisites and autoscaling — https://kserve.github.io/website/docs/admin-guide/kubernetes-deployment
- [KS-S3] KServe S3 storage provider — https://kserve.github.io/website/docs/model-serving/storage/providers/s3
- [KS-REL] KServe v0.21.0 release — https://github.com/kserve/kserve/releases/tag/v0.21.0
- [LLMC] llm-compressor (Apache-2.0) — https://github.com/vllm-project/llm-compressor
- [AUTOAWQ] AutoAWQ deprecation notice (archived, adopted by vLLM) — https://github.com/casper-hansen/AutoAWQ
- [CUDA-RN] CUDA Toolkit release notes (CUDA 13.x needs driver ≥ 580 under minor-version compatibility) — https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
