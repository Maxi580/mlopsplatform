# mlopsplatform

A Kubernetes-native platform to distill, finetune, quantize, evaluate and serve open-source LLMs, with Speculators trained for faster decoding. See `CONTEXT.md` for the vocabulary.

## Install

On a WSL or Ubuntu host with an NVIDIA driver (`nvidia-smi` must work) and systemd:

```sh
git clone https://github.com/Maxi580/mlopsplatform && cd mlopsplatform
./install.sh dev
```

It installs only missing host prerequisites (k3s, nerdctl + BuildKit, NVIDIA container toolkit, gVisor, helm, yq), builds our images, deploys the platform and prints its URLs and the path to `ca.crt`. Import `ca.crt` into your OS trust store once. The NVIDIA driver must be 580 or newer (`versions.minNvidiaDriver`).

On any other Ubuntu host (a cloud VM, a bare-metal server), set `domain` in `deploy/values-prod.yaml` to a DNS name that points at the host, open port 443 to it, and run `./install.sh prod`. One name is enough, since everything is routed by path; a cloud VM's free DNS name works. `values-prod.yaml` holds only the domain, the GPU count and disk sizes; everything else comes from `deploy/values.yaml`.

Running `install.sh` again is the upgrade path. If Pipelines or Endpoints are running, it lists them and asks once `Continue? [y/N]` before changing anything (`MLP_YES=1` skips the prompt). It then cancels the Pipelines (`cancelled`), stops the Endpoints (`stopped`) and replaces our pods and Deployments; all data is kept (Postgres, SeaweedFS, MLflow, Model Cache). At the end it removes outdated images (ours from older builds, and upstream ones a newer tag replaced) and prunes the BuildKit cache to `buildCacheGb`.

Each run also registers the default calibration Dataset `llm-compression-calibration` (the rows of `neuralmagic/LLM_compression_calibration` at a pinned commit, which `quantize` calibrates on) unless it exists; delete it with `mlp datasets delete` and the next install registers it again.

The first run asks for the shared account's password (or reads `MLP_PASSWORD`). To reset it later:

```sh
kubectl -n mlp exec -it deploy/api -- set-password
```

There are no database migrations: the API only creates missing tables. After a change to existing tables, empty the `platform` database (Pipelines, Datasets and the password are lost; MLflow, SeaweedFS and the Model Cache are kept), then reinstall:

```sh
kubectl -n mlp exec statefulset/postgres -- psql -U mlp -d platform -c 'DROP SCHEMA public CASCADE; CREATE SCHEMA public'
./install.sh dev
```

`./uninstall.sh` deletes everything the platform created, including all data, but keeps k3s, nerdctl/BuildKit and the driver.

## CLI login

Write a CLI Profile to `~/.mlp/profile.yaml`, then run `mlp login`:

```yaml
url: https://<domain>
ca_cert: ~/ca.crt
```

The token is stored in `~/.mlp/token` and is valid for 12 hours.

## Datasets

```sh
mlp datasets upload chat.jsonl --name chat   # each upload adds the next version: chat@1, chat@2, ...
mlp datasets                                 # every version with its size and row format
mlp datasets download chat@2 [-o chat.jsonl]
mlp datasets delete chat@1                   # its number is never handed out again
```

A Dataset is one JSONL file whose rows all share one of TRL's row formats: `messages`, `text`, prompt-completion, prompt-only, preference (`chosen`/`rejected`), unpaired preference (`label`) or stepwise supervision. An upload with a bad row is rejected with that row's line number. Downloads come straight from the object store through a presigned URL.

## Models

```sh
mlp models                                                   # every Model Version with size and lineage
mlp models upload ./my-model --name my-model                 # full weights, as the next version
mlp models upload ./my-lora --name my-lora --base hf:org/name # an Adapter, on a Base Model or model:name@v
mlp models download my-model@1 -o ./my-model
mlp models delete my-model@1
```

Every `finetune` Phase registers a Model Version; an upload registers one too, tagged `source: uploaded` and `owner`, and is referenced as `model:name@version` like any other. The files go straight to the object store in parts through presigned URLs (`POST /models/uploads`, a PUT per part, `POST /models/uploads/{id}/complete`), so nothing heavy passes through the API. File names are checked before anything travels, and sizes and contents once the parts arrived; a rejected upload's files are deleted (parts of an upload never completed stay in SeaweedFS). Hidden files such as `.git` are skipped.

- **Full weights** need `config.json`, `tokenizer_config.json` with `tokenizer.json`, `tokenizer.model` or `vocab.json`, and only `*.safetensors` weights: one `model.safetensors`, or shards with a `model.safetensors.index.json` that lists only files that are there. `.bin`, `.pt` and other formats that run code on load are refused, as is an `auto_map` (remote code). Their `tool_parser` tag comes from the `model_type`, or from `--tool-parser` for one the platform doesn't know; otherwise it is `none` and the model serves without tool calling.
- **Adapters** need `adapter_config.json` and `--base`: a Base Model (pinned to a commit) or a full-weight Model Version. No `auto_map`, and vLLM's Adapter rules apply: no DoRA, `modules_to_save` or bias, rank at most 512.

`GET /models/{name}/versions/{v}/files` answers a presigned URL per file, which `mlp models download` saves under the directory.

## Model Cache

```sh
mlp cache                                      # every cached Base Model and benchmark with size and last use
mlp cache free hf:Qwen/Qwen2.5-0.5B-Instruct@<commit>
mlp cache free lm_eval:gsm8k
```

`fetch` downloads each Base Model and each benchmark's datasets once into the Model Cache, a host directory (`modelCacheHostPath`) that the steps and the API mount: Base Models into the Hugging Face cache (`HF_HOME`), benchmarks into `benchmarks/<harness>/<task>/`. Past `model_cache_high_water_mark` of `model_cache_size`, the API evicts the least recently used of them, never one an unfinished Pipeline or an Endpoint uses; a submit or an Endpoint start also evicts to make room for its downloads. `fetch` fails before downloading if the whole download still doesn't fit. Freeing one in use is refused with 409.

## Endpoints

```sh
mlp endpoints                                              # every Endpoint with model, status and URL
mlp endpoints start hf:Qwen/Qwen3-0.6B --name chat         # a Base Model
mlp endpoints start model:qwen-sft@1 --name chat-sft -o max_model_len=8192 -o kv_cache_dtype=fp8
mlp endpoints start model:qwen-sft@1 --name fast -o 'speculative={method: eagle3, model: "model:qwen-sft-speculator@1"}'
mlp endpoints stop chat
```

An Endpoint is a vLLM Deployment (the upstream `images.vllm`, pinned) serving one model on `gpus_per_endpoint` GPUs from the platform settings: a Base Model from the Model Cache, a full-weight Model Version, or an Adapter on its base. Its pod first downloads what vLLM loads, with `fetch` for a Base Model (public ones only: Endpoints take no Hugging Face token, so gated models are rejected) and `mlp-stage download` for Model Version files, then runs vLLM offline. It shows `pending` while GPUs are busy, however long that takes, then `running` once vLLM is ready, or `failed` once vLLM crashed (its pod logs say why). It runs until `mlp endpoints stop` deletes it (`stopped`), or its Deployment disappears, e.g. in a platform upgrade. A name is free again once its Endpoint stopped. The Base Models and Model Versions an Endpoint serves from can't be deleted or evicted while it runs.

The OpenAI-compatible URL is `https://<domain>/endpoints/<name>/v1`, behind the same login as everything else; clients call the model by the Endpoint's name and send the token `mlp login` stored:

```python
from pathlib import Path
from openai import OpenAI  # run with SSL_CERT_FILE=ca.crt

client = OpenAI(
    base_url="https://<domain>/endpoints/chat/v1", api_key=(Path.home() / ".mlp/token").read_text()
)
client.chat.completions.create(model="chat", messages=[{"role": "user", "content": "Hi"}])
```

Serving options, all optional (`POST /endpoints` takes `name`, `model` and these; anything else, `tensor_parallel_size` included, is rejected). Where vLLM's default doesn't depend on the model, it is the option's default, so forms start from it; `max_model_len` and `tool_parser` are left to vLLM, which reads them from the model:

| Option | vLLM flag |
|---|---|
| `max_model_len` | `--max-model-len` |
| `prefix_caching` (default `true`) | `--[no-]enable-prefix-caching` |
| `dtype` (default `auto`): `auto`, `half`, `float16`, `bfloat16`, `float32` | `--dtype` |
| `gpu_memory_utilization` (default 0.9), `max_num_seqs` (default 256), `max_num_batched_tokens` | the same names |
| `async_scheduling` | `--[no-]async-scheduling` |
| `kv_cache_dtype` (default `auto`): `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | `--kv-cache-dtype` |
| `quantization`: `fp8` or `bitsandbytes`, on the fly | `--quantization` |
| `tool_parser`: one of vLLM's built-in parsers | `--tool-call-parser` |
| `speculative`: `method`, `model`, `num_speculative_tokens` (default 3), `prompt_lookup_min`/`max` | `--speculative-config` |

`speculative` turns on speculative decoding: a cheap guess at the next tokens, which the model checks in one pass, so the answer stays the same. `method: ngram` needs no `model`; it looks the last `prompt_lookup_min` to `prompt_lookup_max` tokens up earlier in the context and proposes what followed them there, which helps where answers repeat the prompt. Every other method drafts with `model`: `draft` with any small full-weight model sharing the vocabulary (a Base Model or Model Version), and `eagle3`, `dflash`, `dspark` and `peagle` with a Speculator of that type (P-EAGLE is served as `eagle3` drafting all its tokens at once). A Speculator Model Version is checked against the served model: it must have been trained for exactly that pinned model (its `verifier` tag), and served with its own type as `method`; otherwise the start is rejected. A published Speculator on Hugging Face (`hf:…`) is the hub's to answer for. The pod downloads the Speculator next to the model, and a Speculator can't be served on its own.

Tool calling is on (`--enable-auto-tool-choice`) whenever a parser is known: `tool_parser`, else the model's `tool_parser` tag, else the one its Base Model's `model_type` maps to. `mlp_core.endpoint_spec.vllm_args` turns a spec into vLLM's arguments, so `evaluate` and in-cluster Teachers use the same flags.

Live stats come from each running Endpoint's vLLM `/metrics`, which the API reads through its Service (all Endpoints at once, waiting `ENDPOINT_METRICS_TIMEOUT` for each), so they count every request the Endpoint serves; nothing is stored and no Prometheus is installed. `GET /endpoints` adds a `stats` field per Endpoint: running and waiting requests, generated tokens with the time they were read (two readings give tokens/s), and the time-to-first-token p50; it is `null` while the Endpoint is pending or loading, or when it doesn't answer. `GET /endpoints/<name>/stats` answers the curated sections (Requests, Tokens, KV cache, Latency, and Speculative decoding when it is on), each value with a label and a one-line explanation; histograms as p50, p95 and mean since start (quantiles estimated from the buckets, as Prometheus' `histogram_quantile` does) with their raw sums and counts; and every metric with vLLM's own `# HELP` text. The raw Prometheus text is at `https://<domain>/endpoints/<name>/metrics`, behind the login like the OpenAI URL, as the route strips `/endpoints/<name>` before vLLM; point a Prometheus or `curl` with the token at it to keep history the platform doesn't.

## Pipeline Requests

The CLI Profile also holds reusable Stage settings, named Phase variants and Secrets. A command builds a Pipeline Request from only the Stages and Phases it names; Secrets travel beside the request, never inside it. The API publishes the request's JSON Schema at `/schema`.

```yaml
name: qwen-sft
secrets:
  hf_token: hf_...
finetune:
  base_model: hf:Qwen/Qwen2.5-0.5B-Instruct
  backend: hf  # the default; or unsloth
  phases:
    sft:  # the algorithm defaults to the variant's name
      dataset: "dataset:chat"
      method: lora
      settings: {learning_rate: 1.0e-4, num_train_epochs: 3, per_device_train_batch_size: 8,
                 gradient_accumulation_steps: 1, max_length: 1024}
      lora: {r: 16, lora_alpha: 32, lora_dropout: 0.05, target_modules: all-linear}
    dpo:  # no `lora`: as a later Phase it continues the Adapter
      # `settings` and `lora` may be left out, in whole or in part: the algorithm's defaults fill them
      dataset: "dataset:preferences"
      method: lora
      settings: {learning_rate: 5.0e-6, num_train_epochs: 1, per_device_train_batch_size: 4,
                 gradient_accumulation_steps: 2, max_length: 1024}
```

`mlp run --finetune sft,dpo` builds a request whose `phases` are the named variants, in that order; see [Phase chaining](#phase-chaining). To train on a full-weight Model Version, e.g. an uploaded one, put `from: model:my-model` (or `model:my-model@2`) in place of `base_model`; Adapters and quantized Model Versions can't be started from.

`mlp validate --finetune sft` prints the resolved request (Base Model pinned to a commit, `dataset:chat` to its latest version, e.g. `dataset:chat@2`) or every error with its path. Validate and submit both answer `downloads`, each `{kind, ref, bytes, cached}`, `download_bytes`, the total not in the Model Cache yet, and `cached_bytes`. Validation first checks that the request holds every basic value above (`max_length` is `max_completion_length` for `distillation`, `grpo` and `rloo`), then that each `settings` key exists in the algorithm's TRL config (`SFTConfig`, `DPOConfig`, `KTOConfig`, `DistillationConfig`, `GRPOConfig`, `RLOOConfig`; each `lora` key in PEFT's `LoraConfig`) with the right type, and that each Phase's Dataset has rows its algorithm trains on. That second check uses schemas in `packages/core/src/mlp_core/pipeline_request/trainer_configs/`, regenerated by `packages/core/src/mlp_core/pipeline_request/generate_trainer_configs.py` when TRL or PEFT is bumped, with each field's TRL, PEFT or `TrainingArguments` help as its `description`; without a usable schema it is skipped.

## Pipelines

```sh
mlp run --finetune sft   # submits the request and prints the Pipeline ID right away
mlp run --finetune sft,dpo        # two Phases: dpo continues the Adapter sft registered
mlp run --finetune sft --dry-run  # lists what it would download, without submitting
mlp run --finetune sft --evaluate # then runs the Profile's benchmarks on the new Model Version
mlp run --finetune sft --quantize # then quantizes the new Model Version as the Profile's `quantize:` says
mlp run --finetune sft --speculate # then trains a Speculator for it as the Profile's `speculate:` says
mlp run --evaluate                # only the benchmarks, on the Profile's `evaluate.model`
mlp run --distill --finetune sft  # distills the Profile's prompts, then trains on the replies
mlp run --sweep --finetune sft    # searches the Profile's `sweep:`, then trains sft with the best parameters
mlp ls                   # every Pipeline with its Owner, status, Stages, links and best swept parameters
mlp cancel 7             # stops the run and deletes its Secrets
mlp rerun 7              # submits Pipeline 7's resolved request again as a new Pipeline
mlp resume 7             # continues failed or cancelled Pipeline 7 as a new Pipeline
```

Each Pipeline first runs `fetch`, which downloads the Base Models at their pinned commits and the benchmarks into the Model Cache (a Pipeline with nothing to download has no `fetch`), and ends with a `cleanup` step that runs even after a failure. The Hugging Face token comes from the Profile's `secrets`, else `HF_TOKEN` or `hf auth login`, else a hidden prompt (Enter skips it; public models download anonymously). It lives in a per-Pipeline Kubernetes Secret that only `fetch` sees, is redacted from step logs, and is deleted once the Pipeline finishes or is cancelled (any left over are swept after 48 hours). A step whose GPUs are busy waits and shows as `waiting for GPU`. Pipelines are never retried automatically; `mlp rerun` reads the Secrets afresh, as `mlp run` does, since a Pipeline's own are never kept.

Every `checkpoint_minutes` (a platform setting, default 30) a Phase uploads a Checkpoint to `platform/checkpoints/<pipeline-id>/<phase-index>/`, keeping only the newest, and deletes it once the Phase succeeds; TRL's own `save_*` and `resume_from_checkpoint` settings are refused. `mlp resume 7` starts a new Pipeline from Pipeline 7's resolved request, with Secrets read afresh, that records `resumed_from: 7`: a finished `distill`, a `sweep` that reported its best parameters (a failed one runs again from its first Trial) and every Phase that registered a Model Version are skipped, the next Phase starts from that Model Version and continues from its Checkpoint if it saved one, and the Stages after `finetune` run as usual. A failed or cancelled Pipeline's Checkpoints stay until deleted on the Storage page, which refuses while a running resume needs one.

With a `distill` block (`mlp run --distill` takes the Profile's `distill:`), a `distill` step first turns a Dataset of `prompt` rows (`prompt_only`, a string or messages) into a Distillation Dataset:

```yaml
distill:
  dataset: dataset:prompts
  teacher: hf:Qwen/Qwen3-8B   # or model:qwen-sft@2, endpoint:chat, or with api_url the API's model
  # api_url: https://api.openai.com/v1   # an OpenAI-compatible API; its key is the `teacher_api_key` Secret
  tools:                       # optional, offered with every prompt
    - type: function
      function: {name: get_weather, parameters: {type: object, properties: {city: {type: string}}, required: [city]}}
  parallel_tool_calls: false   # the default: one call per reply, replies with several are dropped
  max_tokens: 1024             # default 4096; temperature defaults to 0.7
  serving: {max_model_len: 8192}  # optional: options for the Teacher's vLLM, as for Endpoints
```

The Teacher is a Base Model or Model Version, which the step serves on vLLM with `gpus_per_stage` GPUs and the Endpoint defaults; a running Endpoint, reached through its Service without a GPU; or a model behind an OpenAI-compatible `api_url`. An API Teacher's key is the `teacher_api_key` Secret (in the Profile's `secrets` or the Web UI's Secrets), which only the `distill` step receives, and which is redacted from its log. Each prompt gets one reply, asked 16 at a time, which is a text or calls to the offered `tools`; no tool runs. With `tools`, validation pins the Teacher's tool parser into `serving.tool_parser` and rejects a Base Model or Model Version Teacher without one (name it in `serving.tool_parser`); tool parameters must be a JSON Schema. A reply is dropped as malformed when it is an error or empty, was cut off at `max_tokens`, calls a tool that isn't offered, has arguments that aren't JSON or don't match the tool's schema, or makes several calls while `parallel_tool_calls` is off. The kept and dropped counts are the Run's `distill/kept` and `distill/dropped` metrics, and the first 5 dropped replies with their reasons are its `dropped_replies.json`. If more than 20 % are dropped, the step fails and registers nothing. Otherwise every kept row (`prompt` messages, `completion` with the one assistant message, its calls' arguments as objects, and `tools` when given) becomes the next version of the Dataset named after the Pipeline, a `prompt_completion` Dataset that the step registers through the API with the step token. A Phase with `dataset: "@distill"` trains on it; it must train on `prompt_completion` rows, and `distill` must be enabled.

With a `sweep` block (`mlp run --sweep` takes the Profile's `sweep:`), a `sweep` step searches the hyperparameters of one Phase configuration with [Optuna](https://optuna.org) (MIT):

```yaml
sweep:
  model: hf:Qwen/Qwen2.5-0.5B-Instruct  # or a full-weight model:…; default finetune's starting model
  backend: hf
  algorithm: sft                # then every field of a Phase except `output`
  dataset: dataset:chat
  method: lora
  settings: {learning_rate: 1e-4, num_train_epochs: 1, per_device_train_batch_size: 8, gradient_accumulation_steps: 1, max_length: 1024}
  lora: {r: 16, lora_alpha: 32, lora_dropout: 0.05, target_modules: all-linear}
  parameters:                   # what the Trials vary, over `settings` and `lora`
    settings:
      learning_rate: {min: 1e-5, max: 1e-3, scale: log}   # two integers sample integers
    lora:
      r: {values: [8, 16, 32]}
  objective: {metric: eval_loss, goal: minimize}  # the default: the algorithm's first metric
  trials: 10                    # the default
  sampler: tpe                  # the default; or random, or grid (every parameter then has `values`)
  eval_split: 0.1               # the default share of rows held out; or name an `eval_dataset`
```

Parameter names are checked against the algorithm's TRL config and PEFT's `LoraConfig`, with the deny-list applied as for a Phase's settings; a `full` Sweep varies no `lora`. The objective's metric comes from the algorithm's list in `mlp_core.config.ALGORITHMS`: `sft` `eval_loss` or `eval_mean_token_accuracy`; `dpo` `eval_loss`, `eval_rewards/accuracies` or `eval_rewards/margins`; `kto` and `distillation` `eval_loss`; `grpo` and `rloo` `reward` (the last training value) or `eval_reward`. An `eval_` metric is measured once a Trial trained, on the held-out rows. The Trials run one after another in one step on the backend's trainer image with that backend's GPUs, offline from the Model Cache, each a Run nested under the step's MLflow Run with its values as `settings.<name>` and `lora.<name>` params. Trials save no Checkpoints and their weights are deleted with them; nothing is registered. A Trial that fails, e.g. out of GPU memory, is logged as a failed Run and the Sweep goes on; the step fails only if every Trial failed. The best Trial's values (`best/<block>.<name>` params) and objective (`best/<metric>`) go into the step's Run, and the step reports them to the API with the step token, which stores them on the Pipeline: `GET /pipelines` lists them as `sweep`, the Web UI's Pipelines page and `mlp ls` show them. A Phase with `params_from: "@sweep"` trains with them over its own `settings` and `lora`, and its Model Version's `pipeline_request.json` names the values it trained with; validation requires its `algorithm`, `method` and `backend` to match the `sweep` block's, and refuses `lora` parameters for a Phase that continues an Adapter.

Next comes `finetune`, which trains each Phase in its own step (`finetune-<algorithm>` in the KFP UI) on the backend's trainer image with `gpus_per_stage` GPUs from the platform settings. `backend: hf` (the default) trains with TRL + PEFT. `backend: unsloth` trains the same TRL trainers patched by Unsloth, on one GPU only, with Unsloth's pins (TRL 0.24, Transformers 5.5, vLLM 0.11, while settings are checked against `hf`'s TRL); it trains `sft`, `dpo` and `kto` with every method, `grpo` and `rloo` with `lora` and `qlora` (their rollouts run on Unsloth's own vLLM, which shares the Adapter's weights), and no `distillation`. Validation rejects any other algorithm or method on it, as `mlp_core.config.BACKENDS` lists; nothing falls back to `hf`, so a model Unsloth can't load fails the step, and you resubmit with `backend: hf`. On `unsloth`, `target_modules: all-linear` means Unsloth's default projections, and an Adapter kept as one fails before registering if Unsloth moved `embed_tokens` or `lm_head` into `modules_to_save`, which vLLM can't serve. It runs offline from the Model Cache (or downloads the Model Version it starts from) and logs its params and metrics into its own MLflow Run (KFP's MLflow plugin creates one per step, under the Pipeline's Run). Each Phase's output is registered in the MLflow Model Registry as the next Model Version of the Registered Model named after the Pipeline (`qwen-sft` version 1, 2, …), and the next Phase starts from it; `@finetune` is the last one. Each version holds the Adapter or full weights, the starting model's tokenizer and chat template, and `pipeline_request.json` with the resolved request. Its tags are `weights` (`adapter` or `full`), `base_model` (an Adapter's base: the Base Model, or the full-weight Model Version it was trained on), `merged: true` for an Adapter merged into its base, `method`, `parent` (the previous Phase's Model Version, or for the first Phase its starting model), `pipeline`, `phase` (1, 2, …), `algorithm`, `teacher` (a `distillation` Phase's Teacher), `backend`, `tool_parser` (the vLLM tool parser for the Base Model's `model_type`, or `none`) and `tools_rendered` (`false` when the chat template drops the tools a client sends, so tool calling won't work). There are no size limits: a Phase that runs out of GPU memory fails with a plain message naming the settings to lower.

With a `quantize` block (`mlp run --quantize` takes the Profile's `quantize:`), a `quantize` step runs after `finetune` and quantizes a model with llm-compressor:

```yaml
quantize:
  model: hf:Qwen/Qwen2.5-0.5B-Instruct  # or model:qwen-sft@2; default @finetune
  scheme: w4a16-gptq        # fp8-dynamic (the default), w4a16-gptq, w4a16-awq or w8a8-int8
  ignore: [lm_head]         # the default: layers kept unquantized
  calibration: {}           # every scheme but fp8-dynamic; {} means these defaults:
  # calibration: {dataset: "dataset:llm-compression-calibration", samples: 512, max_length: 2048}
```

`fp8-dynamic` takes its scales from the weights alone, so it has no `calibration`; the others measure activations on `samples` random rows of a `messages` or `text` Dataset, each cut at `max_length` tokens (conversations in the model's chat template), and validation rejects them without the block. `w4a16-gptq` and `w4a16-awq` store 4-bit weights (GPTQ or AWQ) and compute in 16 bits; `w8a8-int8` runs SmoothQuant and then GPTQ to 8-bit weights and activations, for GPUs older than Ada without FP8. The recipes are `mlp_core.config.QUANTIZATION_SCHEMES`. `model` is a Base Model, any Model Version or `@finetune`, the last Phase's Model Version and the default when `finetune` runs; an Adapter is merged into its base first, and only the quantized result is registered. A model that is already quantized (its `config.json` has a `quantization_config`) is rejected. The step runs on the stages image with `gpus_per_stage` GPUs, offline from the Model Cache, and registers the result in compressed-tensors format, which vLLM serves as is, as the next full-weight Model Version of the Registered Model named after the Pipeline, tagged `weights: full`, `quantization: <scheme>`, `parent` (the model it quantized), `pipeline`, `tool_parser` and `tools_rendered`. `@quantize` is that Model Version.

With a `speculate` block (`mlp run --speculate` takes the Profile's `speculate:`), a `speculate` step runs after `quantize` and trains a Speculator, a small draft model for exactly one verifier, with [`speculators`](https://github.com/vllm-project/speculators) (Apache-2.0):

```yaml
speculate:
  speculator: eagle3        # the default; or dflash, dspark, peagle
  model: hf:Qwen/Qwen3-8B   # the verifier, or model:qwen-sft@2; default @quantize or @finetune
  dataset: dataset:chats    # messages rows, or @distill
  settings: {samples: 1000, seq_length: 8192, epochs: 5, learning_rate: 0.0001, draft_vocab_size: 32000}  # the defaults
```

The step serves the verifier on its own vLLM (`gpus_per_stage` GPUs, offline from the Model Cache) with `extract_hidden_states`, so it writes each prompt's hidden states of an early, a middle and a late layer and the last one to disk instead of answering; `speculators prepare-data` renders `samples` conversations through the verifier's chat template, cut at `seq_length` tokens, and `generate-offline-data` collects their hidden states. vLLM then stops and the drafter trains on the freed GPU for `epochs`, proposing from the `draft_vocab_size` most frequent tokens, with `speculate_dataloader_workers` (a platform setting, default 4) processes reading the hidden states. Conversations like the traffic the Endpoint will see raise acceptance most, so the Pipeline's `@distill` replies (a prompt and its reply as one conversation) are a good choice. Each conversation's hidden states take about 10 MB of disk for a 0.5B verifier and 50 MB for an 8B one. The best epoch, without its optimizer state, registers as the next Model Version of `<pipeline>-speculator`, its `config.json` naming the verifier's Reference, tagged `weights: speculator`, `speculator: <type>`, `verifier: <pinned model>` and `pipeline`. The verifier is a Base Model, a full-weight Model Version, `@quantize` (the default when `quantize` runs) or `@finetune` (the default otherwise); a Speculator reads the hidden states of full weights, so an Adapter is rejected (set `output: merged` on the last Phase, or quantize it), as is a Speculator. The stages image patches one word of `speculators`, which hands an unset `head_dim` to transformers for Qwen2 verifiers; its build fails once a release fixes it, so the patch gets removed then.

With an `evaluate` block (`mlp run --evaluate` takes the Profile's `evaluate:`), an `evaluate` step runs after `speculate` and logs every metric into its own MLflow Run:

```yaml
evaluate:
  model: hf:Qwen/Qwen2.5-0.5B-Instruct  # or model:qwen-sft@2, endpoint:chat; default @quantize or @finetune
  benchmarks: [lm_eval:gsm8k, lm_eval:mmlu]
  limit: 50  # optional: samples per task, for a quick look
  serving: {max_model_len: 8192}  # optional: options for the step's own vLLM, as for Endpoints
  performance: {prompt_tokens: 256, output_tokens: 128, concurrency: 1, requests: 100}  # optional
```

`model` is a Base Model, any Model Version (an Adapter too), a running Endpoint (not `pending`), `@finetune` or `@quantize`; the Pipeline's last Model Version is the default: `@quantize` when `quantize` runs, else `@finetune` when `finetune` does. Benchmarks come from the catalog in `mlp_core.config.BENCHMARKS` (`GET /benchmarks`, the Web UI picker): each is `harness:task` with a category, a one-line description, its dataset licence and download size. Only datasets without a non-commercial or share-alike licence are listed. The harness is lm-evaluation-harness (`lm_eval:`) or, for benchmarks it lacks, EvalScope (`evalscope:`). Tool calling is checked with BFCL (`bfcl:<category>`, e.g. `bfcl:simple_python` or `bfcl:multi_turn_base`), whose datasets ship inside its package, so `fetch` downloads nothing for it; it runs in its own virtualenv in the stages image (`versions.bfcl` in `deploy/values.yaml`), as it pins libraries the other harnesses can't share. BFCL calls the model through its OpenAI handler with real `tools` over chat completions, never its per-family prompt formats, so the score covers the chat template and tool parser too, as an Endpoint would serve them. Validation pins that parser into `serving.tool_parser`: the override, else the one the model (for `@finetune`, its starting model; for `@quantize`, the model it quantized) maps to, exactly as an Endpoint picks it; a model without one is rejected for BFCL, and an Endpoint uses its own. `limit` runs a category's first entries. The step starts vLLM on `gpus_per_stage` GPUs with the Endpoint defaults (an Endpoint is reached through its Service instead, without a GPU) and runs each benchmark in its own harness process, offline from the Model Cache: lm-eval against vLLM's completions API, EvalScope against its chat completions API, with its `EVALSCOPE_CACHE` and `MODELSCOPE_CACHE` in the benchmark's Model Cache directory. Coding benchmarks (`lm_eval:humaneval`, `lm_eval:mbpp`, `evalscope:mbpp_plus`) score generated code only in the Sandbox: each harness process replaces the one place its tasks run code (HF evaluate's `code_eval` for lm-eval, `CodeExecutionSandboxMixin` for EvalScope) with a call that sends the program and its tests to the Sandbox, so no generated code runs in the step; lm-eval runs code-executing tasks only for catalog entries in the `coding` category. `fetch` and `evaluate` both reach the Sandbox, as `fetch` scores one sample to download all a benchmark needs. Metrics are logged as `<harness>/<task>/<metric>` (BFCL's as `bfcl/<category>/accuracy`, with the parser as the Run's `tool_parser` tag); a benchmark that fails is tagged `NA` in the Run and the others still run. A second evaluation of the same benchmark downloads nothing.

`performance` (any of its values may be left out; those shown are the defaults) adds a GuideLLM run after the benchmarks: `requests` synthetic chat requests of `prompt_tokens` tokens, each asking for `output_tokens`, `concurrency` at a time, against the step's own vLLM started with the `serving` options, its prompts tokenized with the model's own tokenizer from the Model Cache. It logs time to first token, inter-token latency and output tokens/s over the successful requests as `guidellm/<metric>/<mean|median|p99>`, e.g. `guidellm/time_to_first_token_ms/p99`; if GuideLLM fails or no request succeeds (e.g. `prompt_tokens` past `max_model_len`), the Run is tagged `guidellm: NA`. Without `performance` nothing is measured, so evaluations stay fast. An Endpoint is shared, so it can't be measured; `benchmarks` may be empty when `performance` is set.

A Pipeline starts no Endpoint: once it registered its Model Versions, they are listed under "Our models" on the Serving page, which starts Endpoints (as does `mlp endpoints start`).

## Smoke Test

```sh
mlp smoke-test                              # every case; prints each result, exits 1 if one failed
mlp smoke-test --phases sft --backends hf   # a custom one: only the finetune cases named
mlp smoke-test --sandbox                    # a custom one: only the sandbox case
mlp smoke-test --uploaded-model             # a custom one: only the uploaded-model case
mlp smoke-test --serving                    # a custom one: only serving the Base Model and tiny model
mlp smoke-test --evaluate                   # a custom one: only evaluating the Base Model
mlp smoke-test --distill                    # a custom one: only distilling and training on it
mlp smoke-test --sweep                      # a custom one: only sweeping two Trials and training with the best
mlp smoke-test --chain                      # a custom one: only the sft → dpo chain
mlp smoke-test --weights                    # a custom one: only rsLoRA, merged outputs, full after an Adapter
mlp smoke-test --resume                     # a custom one: only stopping a Phase at a Checkpoint and resuming it
mlp smoke-test --quantize                   # a custom one: only quantizing the Base Model with each scheme
mlp smoke-test --speculate --serving        # a custom one: each Speculator type, then serving with each and n-gram
```

The API routes are `POST /smoke-tests/complete` and `POST /smoke-tests/custom` (body e.g. `{"finetune": {"phases": ["sft"]}}`; an unnamed list means all of it). Both answer 202 with the Kubeflow run link, or 409 while a Smoke Test runs. It is one Kubeflow run, `smoketest-YYMMDD-HHMMSS`, on `Qwen/Qwen2.5-0.5B-Instruct`, with one node per case: `fetch`, then `sandbox` (custom body `{"sandbox": true}`), a step that sends the Sandbox a batch of snippets and passes if one runs, one finds itself under gVisor, one finds no network, one stops at the memory limit and one is killed at the timeout, then one per finetune Phase × method × backend the backend supports (`sft-lora-hf`, `sft-lora-unsloth`, `sft-qlora-hf`, …, `grpo-qlora-unsloth`, …, `distillation-lora-hf` with the Base Model as its Teacher, `grpo-lora-hf` and `rloo-lora-hf` with two weighted rewards, `correct` and `short`, on bundled maths prompts and `num_generations: 2`), each training 3 steps on its algorithm's bundled Dataset from `packages/core/src/mlp_core/smoke_test_datasets/`, with `sft` and `lora` chosen also `sft-assistant-only-<backend>` per backend, an `sft` Phase with `assistant_only_loss` on the Base Model's own template, and `uploaded-model` (custom body `{"uploaded_model": true}`): the API downloads `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`, uploads it as `smoketest-…-uploaded` through the same checks as a user's upload, and the case trains an `sft` Phase `from` it. A case passes if it runs without errors; what it learns is ignored. Each node runs once the one before it ended, even if that failed, so one failure shows red and the rest still run. The evaluate cases (custom body `{"evaluate": true}`) each run 5 samples of a benchmark that `fetch` downloads too: `lm_eval:truthfulqa_mc2` on the Base Model (`evaluate-base-model`) and, with a finetune case, on its Adapter (`evaluate-adapter`), plus the coding benchmarks `lm_eval:humaneval` (`evaluate-coding`) and the EvalScope task `evalscope:mbpp_plus` (`evaluate-evalscope`) on the Base Model, whose generated code runs in the Sandbox, `bfcl:simple_python` on the Base Model with its `hermes` tool parser (`evaluate-tool-calling`), and one short GuideLLM performance run of 10 requests on the Base Model (`evaluate-performance`). The distill case (custom body `{"distill": true}`) is two nodes: a `distill` step (`distill-tools-distill`) with the Base Model as in-cluster Teacher, offered one `get_weather` tool, on the bundled `distill.jsonl` prompts, then an `sft` Phase on `@distill` (`distill-tools`), which runs only once `distill` passed. The sweep case (custom body `{"sweep": true}`) is two nodes: a `sweep` step (`sft-sweep-sweep`) that tries two learning rates for an `sft` `lora` Phase on the bundled `sft.jsonl`, holding out 2 of its 8 rows, then that Phase with `params_from: "@sweep"` (`sft-sweep`), which runs only once `sweep` passed. The chain case (custom body `{"chain": true}`) is two nodes too: an `sft` Phase (`sft-dpo-chain-finetune-sft`), then a `dpo` Phase that continues its Adapter (`sft-dpo-chain`), which runs only once `sft` passed. The weight cases (custom body `{"weights": true}`) train an rsLoRA Adapter (`sft-rslora-hf`), a QLoRA Adapter merged into full weights (`sft-qlora-merged-hf`), a DoRA Adapter merged into full weights (`sft-dora-merged-hf`), an `sft` Adapter followed by a `full` `dpo` Phase that merges it first (`sft-lora-dpo-full-hf-finetune-sft`, then `sft-lora-dpo-full-hf`), and an Adapter Unsloth merges into full weights (`sft-lora-merged-unsloth`). The resume case (custom body `{"resume": true}`) is two nodes: an `sft` Phase that saves a Checkpoint after its first step and stops there, as a cancel would (`resume-interrupted`), then the same Phase continued from that Checkpoint (`resume`); skipping finished Phases is covered by the API's tests. The quantize cases (custom body `{"quantize": true}`) quantize the Base Model with each scheme (`quantize-fp8-dynamic`, `quantize-w4a16-gptq`, `quantize-w4a16-awq`, `quantize-w8a8-int8`), the calibrated ones on 16 rows of the bundled `calibration.jsonl` cut at 256 tokens, and, with a finetune case, quantize its Adapter to FP8 after merging it (`quantize-adapter`). The speculate cases (custom body `{"speculate": true}`) each train one Speculator type for the Base Model (`speculate-eagle3`, `speculate-dflash`, `speculate-dspark`, `speculate-peagle`) on 16 of the bundled `sft.jsonl` conversations cut at 256 tokens, for one epoch. The serving cases (custom body `{"serving": true}`) run outside the Kubeflow run: the API starts an Endpoint each for the Base Model (`serve-base-model`), the uploaded tiny model (`serve-full-weights`) and, once each is registered, the Adapter of the first finetune case that keeps one (`serve-adapter`), with the weight cases, the merged QLoRA model (`serve-merged`) and, with the quantize cases, each scheme's quantized Base Model (`serve-quantize-<scheme>`), the Base Model drafting with n-gram (`serve-ngram`) and, with the speculate cases, the Base Model drafting with each Speculator (`serve-speculate-<type>`); a case passes once vLLM is ready, answers one real chat request and its stats then count a finished request and generated tokens, and its Endpoint is then stopped. The Smoke Test finishes once the run and these cases did. It also lists in `mlp ls`. Once it finished, the API stops its Endpoints and deletes every Dataset and Registered Model named `smoketest-YYMMDD-HHMMSS-…` and its Checkpoints; only the Kubeflow run, its logs and its MLflow Runs remain, and the Base Model stays in the Model Cache.

## Sandbox

The only place untrusted Python runs: RL rewards (see [Writing rewards](#writing-rewards)) and code generated in benchmarks. It is a small stateless server (`packages/sandbox`) that Pipeline steps call inside the cluster with a batch of snippets, `POST /snippets` with `[{"code": "...", "input": "..."}]`. Each snippet runs as a script in a fresh Python process with `input` on its stdin, in an empty directory and without the server's environment. The answer holds each one's `status` (`ok`, `error`, `timeout` or `memory_limit`), `stdout` and `stderr`, in order. A snippet is killed with every process it started after `sandbox_timeout_seconds`, stops at `sandbox_memory_mb`, and is stopped once it writes more than 1 MiB to any file, its output included. Each of the `sandbox_replicas` replicas runs at most `sandbox_cpus_per_replica` snippets at once, across all batches. All four are platform settings. The pods run under gVisor (the `gvisor` RuntimeClass), with no Kubernetes credentials and no outbound network. A NetworkPolicy lets only Kubeflow run pods reach them, and they are never routed through Traefik. Its tests run the server in-process without gVisor, on Linux only.

## Phase algorithms

Each algorithm is one row of `ALGORITHMS` in `packages/core/src/mlp_core/config.py`: its TRL trainer and config, the Dataset row formats it trains on, the setting bounding a row's tokens, which always has a value, its blocked settings, what the platform sets, the defaults of its settings and Adapter, and whether it learns from a Teacher or from rewards. A Dataset whose rows the algorithm can't train on is rejected at validation. Each Phase picks a weight method, which every algorithm supports:

- `lora` (the default): trains an Adapter on the base at its saved precision. A Phase training a new Adapter takes `lora` over its algorithm's defaults, which goes to PEFT's `LoraConfig` with `task_type: CAUSAL_LM` (blocked); `use_rslora: true` there gives rsLoRA.
- `qlora`: the same, on the base loaded in 4 bits (NF4, computing in bfloat16, via bitsandbytes), for less GPU memory.
- `full`: trains every weight and takes no `lora`.

Each Phase also picks its `output`: `adapter` (the default) registers the Adapter, `merged` registers only the Adapter merged into its base, as full weights tagged `merged: true`; the base is reloaded in bfloat16 before merging, so `qlora` merges into full precision weights, not 4 bits. `full` ignores `output`. Merge when the result must stand alone: to export it, to start another Pipeline `from` it, or to use options vLLM can't serve in an Adapter. An Adapter kept as an Adapter must be servable by vLLM, so with `output: adapter` validation rejects `use_dora`, `modules_to_save`, `bias` other than `none` and `r` above 512; with `output: merged` they are allowed, as the result is plain weights.

A Phase's `settings` and `lora` go over its algorithm's defaults (`default_settings` and `default_lora` in `ALGORITHMS`, published by `GET /schema`), tuned for LoRA training of a 1-8B model; a Phase of only `algorithm` and `dataset` trains with them:

| algorithm | `learning_rate` | `num_train_epochs` | batch × gradient accumulation | length setting | `warmup_steps` |
|---|---|---|---|---|---|
| `sft` | 2e-4 | 3 | 4 × 4 | `max_length: 2048` | 0.03 |
| `dpo`, `kto` | 5e-6 | 1 | 2 × 8 | `max_length: 2048` | 0.1 |
| `distillation` | 2e-5 | 1 | 4 × 4 | `max_completion_length: 1024` | 0.1 |
| `grpo`, `rloo` | 2e-6 | 1 | 4 × 4 | `max_completion_length: 1024` | 0.1 |

All of them also set `lr_scheduler_type: cosine` and `bf16: true`; `dpo` and `kto` `beta: 0.1`; `grpo` and `rloo` `num_generations: 8` and `vllm_gpu_memory_utilization: 0.3`, `rloo` also `beta: 0.05`. A new Adapter defaults to `r: 16`, `lora_alpha: 32`, `lora_dropout: 0.05` (0 for `grpo` and `rloo`) and `target_modules: all-linear`. Anything else is TRL's default for its config, plus `report_to: mlflow`, `save_strategy: no` and `disable_tqdm: true`. Every algorithm blocks `output_dir`, `report_to`, `logging_dir` (the platform stores and logs the output), `save_strategy`, `save_steps`, `save_total_limit`, `resume_from_checkpoint` (the platform owns Checkpoints), `push_to_hub` and `hub_*` (outputs go to the Model Registry), and `model_init_kwargs` and `trust_remote_code` (could ask for remote code).

### Phase chaining

`finetune.phases` is an ordered list. Each Phase runs in its own step and starts from the previous Phase's output, and every Phase's output is registered as a Model Version, so `sft` then `dpo` registers `qwen-sft@1` and then `qwen-sft@2`, trained from the first. What a Phase does with the previous Phase's output depends on what that output is:

- **An Adapter** (`lora`/`qlora` with `output: adapter`): a `lora` or `qlora` Phase continues training that same Adapter, keeping its rank and targets, so it leaves out `lora`. A `full` Phase merges the Adapter into its base first, then trains every weight. To train a new Adapter instead, set `output: merged` on the previous Phase.
- **Full weights** (a `full` Phase, `output: merged`, or the first Phase's Base Model or `from` Model Version): a `lora` or `qlora` Phase trains a new Adapter on them, so it sets `lora`; that Adapter's base is the full-weight Model Version.

Validation rejects a Phase that sets `lora` where it would continue an Adapter, one that leaves it out where it trains a new one, and a `full` Phase with `lora`. `evaluate` and Endpoints load every Model Version of a chain, an Adapter on a Base Model or on a full-weight Model Version alike. Another Pipeline can continue the work through `from` only from full weights, so end with `output: merged` or a `full` Phase to hand it on.

### `sft`: supervised finetuning

Trains the model to produce the Dataset's text, with TRL's `SFTTrainer`. Use it to teach a format, a style, a domain or a task from examples of the answers you want; it is the usual first Phase.

- **Rows**: `messages` (conversations, rendered with the Base Model's own chat template; a Base Model without one is refused when the Phase starts), prompt-completion (loss on the completion only), or `text` (plain language modelling):
  ```jsonl
  {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
  {"prompt": "Translate to French: cat", "completion": "chat"}
  {"text": "Any text the model should learn to continue."}
  ```
- **Defaults**: those of every algorithm, over TRL's [`SFTConfig`](https://huggingface.co/docs/trl/sft_trainer#trl.SFTConfig).
- **Blocked settings**: those of every algorithm, and `chat_template_path` (the Base Model's chat template is always kept).
- **`assistant_only_loss`** (default `false`, `messages` rows only; prompt-completion rows already learn only from the completion, so validation rejects it there):

  > **Assistant-only loss.** Off: the model learns from every token of a conversation, including what the user wrote, so it also learns to imitate users and tool results. On: only the assistant's replies count towards the loss; everything else is context it reads but isn't graded on. Turn it on for chat and instruction tuning where the user turns aren't text you want the model to produce. Leave it off to teach the model the whole conversation style, or when your data is mostly assistant text anyway. Requires a chat template that marks assistant turns (checked at submit on the `hf` backend).

  On `hf`, TRL finds the assistant's turns through `{% generation %}` markers in the chat template. Validation reads the starting model's template (from Hugging Face, or from the Model Version's files) and accepts it if it has the markers, or if it is one TRL swaps for a marked version of its own while training, as it does for Qwen2.5's, Llama 3's and Gemma's, among others; otherwise it rejects the request. That list is generated with the trainer configs (`TrainingChatTemplates.json`). On `unsloth`, Unsloth's `train_on_responses_only` finds the assistant's turns by rendering the chat template itself, so no markers are needed and nothing is checked at submit.

### `dpo`: direct preference optimization

Trains the model to prefer one answer over another, with TRL's `DPOTrainer`: it raises the likelihood of the chosen answer relative to the rejected one, measured against a frozen reference. Use it after `sft` when you can say which of two answers to a prompt is better, e.g. to make answers more helpful, safer or better formatted, without a reward model. As a later Phase, the reference is the Adapter as the previous Phase left it; as the first Phase, the base without an Adapter.

- **Rows**: `preference`, a prompt with a chosen and a rejected answer, as strings or messages (rendered with the base's own chat template); `prompt` may be left out when it begins both conversations:
  ```jsonl
  {"prompt": [{"role": "user", "content": "What is 2 + 2?"}], "chosen": [{"role": "assistant", "content": "4."}], "rejected": [{"role": "assistant", "content": "5."}]}
  {"prompt": "The sky is", "chosen": " blue.", "rejected": " green."}
  ```
- **Defaults**: those of every algorithm, over TRL's [`DPOConfig`](https://huggingface.co/docs/trl/dpo_trainer#trl.DPOConfig), e.g. `beta: 0.1` (how far it may move from the reference) and `loss_type: sigmoid`.
- **Blocked settings**: those of every algorithm.

### `kto`: Kahneman-Tversky optimization

Trains the model from single answers labelled good or bad, with TRL's `KTOTrainer`, against a frozen reference as `dpo` does. Use it after `sft` when you have thumbs-up/thumbs-down feedback rather than two answers to the same prompt. The Dataset needs both labels, and `per_device_train_batch_size` must be at least 2, as KTO compares the answers within a batch.

- **Rows**: `unpaired_preference`, a prompt, one completion and whether it is good (`true`) or bad (`false`), as strings or messages:
  ```jsonl
  {"prompt": [{"role": "user", "content": "What is 2 + 2?"}], "completion": [{"role": "assistant", "content": "4."}], "label": true}
  {"prompt": "The sky is", "completion": " green.", "label": false}
  ```
- **Defaults**: those of every algorithm, over TRL's [`KTOConfig`](https://huggingface.co/docs/trl/kto_trainer#trl.KTOConfig), e.g. `beta: 0.1`, and `desirable_weight` and `undesirable_weight` of 1 (raise one when the labels are unbalanced).
- **Blocked settings**: those of every algorithm.

### `distillation`: logit-level distillation

Trains the model, the Student, to match a Teacher's token probabilities, with TRL's `DistillationTrainer`: for each prompt the Student writes its own completion of up to `max_completion_length` tokens, and learns the Teacher's whole next-token distribution at every token of it. The Teacher is loaded in the same Job as the Student, on the step's GPUs, so both must fit there.

```yaml
    distillation:
      dataset: "dataset:prompts"
      teacher: hf:Qwen/Qwen2.5-7B-Instruct  # or a Model Version, e.g. model:qwen-sft@2
      method: lora
      settings: {learning_rate: 1.0e-4, num_train_epochs: 1, per_device_train_batch_size: 4,
                 gradient_accumulation_steps: 2, max_completion_length: 512}
      lora: {r: 16, lora_alpha: 32, lora_dropout: 0.05, target_modules: all-linear}
```

- **Teacher**: a Base Model, which `fetch` pulls into the Model Cache, or a Model Version; an Adapter is merged into its base when loaded. It must have the Student's vocabulary, in practice a model of the same family, which TRL checks when the Phase starts. An API model or an Endpoint is rejected at validation, as neither hands over its token probabilities; distill from those with the `distill` Stage.
- **`distillation` or the `distill` Stage**: the `distill` Stage asks any Teacher (in-cluster, an Endpoint, an API model, any tokenizer) once for a reply to each prompt, on vLLM, and keeps the replies as a Distillation Dataset that `sft` then trains on, reusable and with tool calls. A `distillation` Phase gives the Student far more signal per token, the Teacher's full distribution rather than one sampled reply, on completions the Student wrote itself, so it learns to recover from its own mistakes; but it needs a Teacher of the same vocabulary loaded beside it, and generates during training without vLLM, so it is slower. Use the `distill` Stage to reach a hosted or differently tokenized Teacher, or to build a Dataset; use `distillation` to squeeze a larger model of the same family into a smaller one, e.g. after an `sft` Phase on a Distillation Dataset.
- **Rows**: `prompt_only`, a prompt as a string or messages; the completions are the Student's own:
  ```jsonl
  {"prompt": [{"role": "user", "content": "Why is the sky blue?"}]}
  {"prompt": [{"role": "user", "content": "Write a haiku about autumn."}]}
  ```
- **Defaults**: those of every algorithm, over TRL's [`DistillationConfig`](https://huggingface.co/docs/trl/distillation_trainer), e.g. `beta: 1.0` (reverse KL; 0 is forward KL, 0.5 the Jensen-Shannon divergence) and `temperature: 1.0`. It requires `max_completion_length` in place of `max_length`.
- **Blocked settings**: those of every algorithm, `teacher_model_name_or_path`, `teacher_model_revision` and `teacher_model_init_kwargs` (the Phase's `teacher` names the Teacher), and `use_vllm` (the trainer image has no vLLM).

### `grpo`: group relative policy optimization

Trains the model on its own completions, scored by your [rewards](#writing-rewards), with TRL's `GRPOTrainer`: for each prompt it writes `num_generations` completions (default 8), every reward scores each one in the Sandbox, and the model is pushed towards the completions that scored above their group's mean and away from those below. Use it when you can check an answer in code but have no example answers to imitate, e.g. maths with a known result, a format, or code that must pass tests; usually after `sft`, so the model already writes answers worth scoring. The completions are generated with vLLM in the Phase's own pod, on the same GPUs as training (`vllm_mode: colocate`), so both must fit there; TRL hands the current weights (an Adapter merged in) to vLLM before each generation.

```yaml
    grpo:
      dataset: "dataset:maths"
      method: lora
      settings: {learning_rate: 1.0e-6, num_train_epochs: 1, per_device_train_batch_size: 8,
                 gradient_accumulation_steps: 1, max_completion_length: 512, num_generations: 8}
      lora: {r: 16, lora_alpha: 32, lora_dropout: 0.05, target_modules: all-linear}
      rewards:
        correct:
          weight: 0.8
          source: |
            def reward(sample, item):
                return float(item["answer"] in sample["output_text"])
        short:
          weight: 0.2
          source: |
            def reward(sample, item):
                return 1.0 if len(sample["output_text"]) < 400 else 0.0
```

- **Rows**: `prompt_only`, a prompt as a string or messages, plus any columns your rewards read (here `answer`); the completions are the model's own:
  ```jsonl
  {"prompt": [{"role": "user", "content": "What is 2 + 2?"}], "answer": "4"}
  {"prompt": [{"role": "user", "content": "What is 7 - 2?"}], "answer": "5"}
  ```
- **Batches**: `per_device_train_batch_size` × `gradient_accumulation_steps` counts completions, not prompts, and must be a multiple of `num_generations`, which TRL checks when the Phase starts. More completions per prompt give a better baseline and cost more memory and time.
- **Defaults**: those of every algorithm, `use_vllm: true` and `vllm_mode: colocate`, over TRL's [`GRPOConfig`](https://huggingface.co/docs/trl/grpo_trainer#trl.GRPOConfig), e.g. `beta: 0.0` (no KL penalty towards the starting model), `temperature: 1.0` and `vllm_gpu_memory_utilization: 0.3`, the share of GPU memory vLLM takes. It requires `max_completion_length` in place of `max_length`. `use_vllm: false` generates with transformers instead, which needs no memory for vLLM but is much slower.
- **Blocked settings**: those of every algorithm, `vllm_mode` and the vLLM server settings (`vllm_server_*`, `vllm_group_port`), as there is no separate vLLM server, and `reward_weights` (each reward's `weight` sets it).
- **Metrics** in the Run: TRL's `reward` and `reward_std` (of the weighted sum), `rewards/<name>/mean` and `/std` per reward, `rewards/<name>/errors` (the completions a reward failed on, per step), `kl`, `entropy` and the completion lengths.

### `rloo`: REINFORCE leave-one-out

Like `grpo`, with TRL's `RLOOTrainer`: the same rows, rewards, rollouts on colocated vLLM, platform defaults, blocked settings and metrics, over TRL's [`RLOOConfig`](https://huggingface.co/docs/trl/rloo_trainer#trl.RLOOConfig). Only the baseline differs: each completion is compared with the mean reward of the *other* completions to its prompt, so its own score stays out of its baseline. Its defaults differ too: `num_generations: 2`, and `beta: 0.05`, a KL penalty towards the starting model. Use `grpo` by default; try `rloo` when training is unstable, or with few `num_generations` (2 to 4), where leaving one out matters most.

### Writing rewards

A `grpo` or `rloo` Phase needs `rewards`, a map from a name to a `weight` and Python `source`; validation rejects one without, and any other Phase with them. In the Web UI each Phase has a **Rewards** box with this text in its ⓘ, where "Add reward" gives a name, a weight and a compact code box for the source that grows while you edit it and starts from a commented template of `def reward(sample, item)`. In the CLI Profile, write the source as a YAML block (`source: |`), as above. Each reward is one function:

```python
def reward(sample, item):
    # sample["output_text"]: the reply the model wrote, as text
    # sample["output_tools"]: the tool calls in that reply (OpenAI format), usually []
    # item: the Dataset row, e.g. item["prompt"], item["answer"]
    return 1.0  # a float, higher is better, or None where the reward doesn't apply
```

> **Rewards.** grpo and rloo only. Each reward has a name, a weight and Python `source` that defines `def reward(sample, item)`, returning a float (higher is better) or None where it doesn't apply. `sample["output_text"]` is the reply the model wrote and `sample["output_tools"]` the tool calls in it; `item` is the Dataset row, so it can read extra columns such as `answer`. For every prompt the model writes several completions, and every reward scores each one in the Sandbox: the Python standard library only, no network, within the Sandbox's time and memory limits. A reward that raises or times out scores 0.0 and counts towards rewards/<name>/errors. The model learns from the sum of each reward times its weight. Example: def reward(sample, item): return float(item["answer"] in sample["output_text"])

- **How it runs**: your code never runs in the training Job. Each step, each reward sends one batch to the [Sandbox](#sandbox), one snippet per completion: a fresh Python 3.12 process that defines your source, calls `reward(sample, item)` with that completion and its Dataset row, and prints the result. So a reward keeps no state between calls, can import only the standard library (`re`, `json`, `math`, `ast`, …), reaches no network and no files, and must finish within `sandbox_timeout_seconds` (default 10) and `sandbox_memory_mb` (default 1024), both platform settings. What it prints is ignored.
- **Score**: the model learns from the sum of each reward times its `weight` (TRL's `reward_weights`). `None` leaves that reward out of the sum for that completion, e.g. a format check that doesn't apply to a tool call. Names are 1 to 64 letters, digits, `_` or `-`, as they name the `rewards/<name>/…` metrics.
- **Failures**: a reward that raises, times out, runs out of memory, exits, or returns anything but a number or `None` scores 0.0 for that completion; the step's log says e.g. ``Reward `correct` failed on 3 of 16 completions; the first: KeyError: 'answer'``, and `rewards/<name>/errors` counts them. Training goes on, so watch that metric: a reward that always fails trains on zeros. Validation rejects a source that isn't Python or has no top-level `def reward(sample, item)`, without running it, and a source over 64 KiB.
- **Tool calls**: `output_tools` holds the calls TRL parsed out of the reply with the tokenizer's response schema; with a tokenizer that has none, calls stay text in `output_text`. No tool is offered in the prompt and no call is run, as rollouts are single-turn.
- **Tips**: give partial credit rather than all-or-nothing where you can, so a group's completions score differently and there is something to learn; weight the rewards that matter most highest; and try a new reward on a short run (`max_steps: 10`), reading its `rewards/<name>/mean` and `/errors` in MLflow, before a long one.

```python
import re


def reward(sample, item):
    # Partial credit: the right number scores 1, a wrong one 0.2, none 0.
    numbers = re.findall(r"-?\d+(?:\.\d+)?", sample["output_text"])
    if not numbers:
        return 0.0
    return 1.0 if numbers[-1] == str(item["answer"]) else 0.2
```

## Web UI

The Web UI (`packages/interface/webui`, React) is a client of the API like the CLI (`packages/interface/cli`). Open `https://<domain>/` and log in with the shared password; the cookie lasts 12 hours and also opens the KFP UI (`/pipeline/`) and the MLflow UI (`/mlflow/`), which send a logged-out browser to the same login page. Login returns to the page that asked for it; an unknown path goes to Pipelines. "Log out" under the user's name in the sidebar clears the cookie (`POST /auth/logout`).

- **Pipelines**: every Pipeline with its Owner, status, Stages and links to its Kubeflow run and MLflow Run, updated live, with Cancel; the sidebar shows the platform's GPU count.
- **Storage**: object store usage, every Dataset and Model Version with Download and Delete (a Model Version lists a link per file), kept Checkpoints with their size and Delete, and a form that uploads a model directory as `mlp models upload` does.
- **Serving**: every Endpoint with its model, status, load (running · waiting requests), generated tokens/s, time to first token (p50) and URL, updated live, with Stop, and a form that starts an Endpoint for any Model Version or Base Model with the serving options (`ServingOptions` from `GET /schema`) and a Speculator picker: off, n-gram, or a Speculator Model Version, where only those trained for the model typed in can be picked and the others show which model they draft for. An Endpoint's name opens its stats page (`/serving/<name>`), reloaded every 5 s: a KV cache bar (how full the GPU memory for in-flight requests is, not the hit rate), charts of requests, tokens/s, mean latency and prefix-cache hit rate over the last 15 minutes the page was open, every curated value with what it means, and all of vLLM's metrics.
- **New Pipeline**: a form built from `GET /schema` that makes the same Pipeline Request as `mlp run`. Every field that needs explaining has an ⓘ showing its `description` from the schema on hover and keyboard focus. It shows only the fields that apply to the current choices, by the `applies_if` of each field in the schema and what `GET /schema` publishes per algorithm: a Teacher only for `distillation`, rewards only for `grpo` and `rloo`, `assistant_only_loss` only for `sft`, LoRA settings only for a new Adapter (not for `full`, nor where a Phase continues the previous one's Adapter), `output` only for `lora` and `qlora`, and `distill.api_url` and the Teacher API key only for a Teacher that is a model at an external API; a hidden field is left out of the request. A comma in a list field (e.g. `target_modules`) makes a list. A Phase shows its algorithm's common settings directly (`SHOWN_SETTINGS` and the algorithm's `shown_settings` in `config.py`, e.g. `beta` and `loss_type` for `dpo`) and LoRA's `r` (a choice of the ranks vLLM serves), `lora_alpha`, `lora_dropout` and `target_modules`; "All settings" opens every other allowed field of its TRL config and `LoraConfig`, typed (choices, booleans, numbers), filterable by name, with TRL's default as placeholder and its help as ⓘ, and takes custom keys at its end. Blocked settings never show, nor `use_dora`, `modules_to_save` and `bias`, which make an Adapter vLLM can't serve (the API still takes them with `output: merged`). and a Phase's "Add reward" adds a named reward with a weight and a code box for its source, explained by an infobox (see [Writing rewards](#writing-rewards)). While the request changes, it shows how much is still to download. Validate shows the resolved request, Submit starts the Pipeline, and every error appears next to its field. The Hugging Face token is sent as a Secret beside the request. An optional Stage such as `quantize` joins the request once its switch is on.

## Development

```sh
uv sync --all-packages   # add --extra hf or --extra unsloth for a backend's trainer libraries (vLLM on Linux only); their tests skip without them
uv run pytest
uv run ruff check . && uv run ruff format --check .

cd packages/interface/webui
npm ci
npm run lint && npm run build && npm test   # npm run format fixes formatting
MLP_API_URL=http://localhost:8000 npm run dev   # http://localhost:5173/ui/, API calls go to MLP_API_URL
```
