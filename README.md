# mlopsplatform

A Kubernetes-native platform to distill, finetune, quantize, evaluate and serve open-source LLMs. See `CONTEXT.md` for the vocabulary.

## Install

On a WSL or Ubuntu host with an NVIDIA driver (`nvidia-smi` must work) and systemd:

```sh
git clone https://github.com/Maxi580/mlopsplatform && cd mlopsplatform
./install.sh dev
```

It installs only missing host prerequisites (k3s, nerdctl + BuildKit, NVIDIA container toolkit, gVisor, helm, yq), builds our images, deploys the platform and prints its URLs and the path to `ca.crt`. Import `ca.crt` into your OS trust store once. The NVIDIA driver must be 580 or newer (`versions.minNvidiaDriver`).

On any other Ubuntu host (a cloud VM, a bare-metal server), set `domain` in `deploy/values-prod.yaml` to a DNS name that points at the host, open port 443 to it, and run `./install.sh prod`. One name is enough, since everything is routed by path; a cloud VM's free DNS name works. `values-prod.yaml` holds only the domain, the GPU count and disk sizes; everything else comes from `deploy/values.yaml`.

Running `install.sh` again is the upgrade path. If Pipelines or Endpoints are running, it lists them and asks once `Continue? [y/N]` before changing anything (`MLP_YES=1` skips the prompt). It then cancels the Pipelines (`cancelled`), stops the Endpoints (`stopped`) and replaces our pods and Deployments; all data is kept (Postgres, SeaweedFS, MLflow, Model Cache). At the end it removes outdated images (ours from older builds, and upstream ones a newer tag replaced) and prunes the BuildKit cache to `buildCacheGb`.

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

Serving options, all optional (`POST /endpoints` takes `name`, `model` and these; anything else, `tensor_parallel_size` included, is rejected):

| Option | vLLM flag |
|---|---|
| `max_model_len` | `--max-model-len` |
| `prefix_caching` (default `true`) | `--[no-]enable-prefix-caching` |
| `dtype`: `auto`, `half`, `float16`, `bfloat16`, `float32` | `--dtype` |
| `gpu_memory_utilization`, `max_num_seqs`, `max_num_batched_tokens` | the same names |
| `async_scheduling` | `--[no-]async-scheduling` |
| `kv_cache_dtype`: `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | `--kv-cache-dtype` |
| `quantization`: `fp8` or `bitsandbytes`, on the fly | `--quantization` |
| `tool_parser`: one of vLLM's built-in parsers | `--tool-call-parser` |

Tool calling is on (`--enable-auto-tool-choice`) whenever a parser is known: `tool_parser`, else the model's `tool_parser` tag, else the one its Base Model's `model_type` maps to. `mlp_core.endpoint_spec.vllm_args` turns a spec into vLLM's arguments, so `evaluate` and in-cluster Teachers use the same flags.

## Pipeline Requests

The CLI Profile also holds reusable Stage settings, named Phase variants and Secrets. A command builds a Pipeline Request from only the Stages and Phases it names; Secrets travel beside the request, never inside it. The API publishes the request's JSON Schema at `/schema`.

```yaml
name: qwen-sft
secrets:
  hf_token: hf_...
finetune:
  base_model: hf:Qwen/Qwen2.5-0.5B-Instruct
  backend: hf
  phases:
    sft:  # the algorithm defaults to the variant's name
      dataset: "dataset:chat"
      method: lora
      settings: {learning_rate: 1.0e-4, num_train_epochs: 3, per_device_train_batch_size: 8,
                 gradient_accumulation_steps: 1, max_length: 1024}
      lora: {r: 16, lora_alpha: 32, lora_dropout: 0.05, target_modules: all-linear}
    dpo:  # no `lora`: as a later Phase it continues the Adapter
      dataset: "dataset:preferences"
      method: lora
      settings: {learning_rate: 5.0e-6, num_train_epochs: 1, per_device_train_batch_size: 4,
                 gradient_accumulation_steps: 2, max_length: 1024}
```

`mlp run --finetune sft,dpo` builds a request whose `phases` are the named variants, in that order; see [Phase chaining](#phase-chaining). To train on a full-weight Model Version, e.g. an uploaded one, put `from: model:my-model` (or `model:my-model@2`) in place of `base_model`; Adapters can't be started from yet.

`mlp validate --finetune sft` prints the resolved request (Base Model pinned to a commit, `dataset:chat` to its latest version, e.g. `dataset:chat@2`) or every error with its path. Validate and submit both answer `downloads`, each `{kind, ref, bytes, cached}`, `download_bytes`, the total not in the Model Cache yet, and `cached_bytes`. Validation first checks that the request holds every basic value above (`max_length` is `max_completion_length` for `distillation`), then that each `settings` key exists in the algorithm's TRL config (`SFTConfig`, `DPOConfig`, `KTOConfig`, `DistillationConfig`; each `lora` key in PEFT's `LoraConfig`) with the right type, and that each Phase's Dataset has rows its algorithm trains on. That second check uses schemas in `packages/core/src/mlp_core/pipeline_request/trainer_configs/`, regenerated by `packages/core/src/mlp_core/pipeline_request/generate_trainer_configs.py` when TRL or PEFT is bumped; without a usable schema it is skipped.

## Pipelines

```sh
mlp run --finetune sft   # submits the request and prints the Pipeline ID right away
mlp run --finetune sft,dpo        # two Phases: dpo continues the Adapter sft registered
mlp run --finetune sft --dry-run  # lists what it would download, without submitting
mlp run --finetune sft --serve    # ends with an Endpoint for the last Model Version
mlp run --finetune sft --evaluate # then runs the Profile's benchmarks on the new Model Version
mlp run --evaluate                # only the benchmarks, on the Profile's `evaluate.model`
mlp run --distill --finetune sft  # distills the Profile's prompts, then trains on the replies
mlp ls                   # every Pipeline with its Owner, status, Stages and Kubeflow/MLflow links
mlp cancel 7             # stops the run and deletes its Secrets
mlp rerun 7              # submits Pipeline 7's resolved request again as a new Pipeline
mlp resume 7             # continues failed or cancelled Pipeline 7 as a new Pipeline
```

Each Pipeline first runs `fetch`, which downloads the Base Models at their pinned commits and the benchmarks into the Model Cache (a Pipeline with nothing to download has no `fetch`), and ends with a `cleanup` step that runs even after a failure. The Hugging Face token comes from the Profile's `secrets`, else `HF_TOKEN` or `hf auth login`, else a hidden prompt (Enter skips it; public models download anonymously). It lives in a per-Pipeline Kubernetes Secret that only `fetch` sees, is redacted from step logs, and is deleted once the Pipeline finishes or is cancelled (any left over are swept after 48 hours). A step whose GPUs are busy waits and shows as `waiting for GPU`. Pipelines are never retried automatically; `mlp rerun` reads the Secrets afresh, as `mlp run` does, since a Pipeline's own are never kept.

Every `checkpoint_minutes` (a platform setting, default 30) a Phase uploads a Checkpoint to `platform/checkpoints/<pipeline-id>/<phase-index>/`, keeping only the newest, and deletes it once the Phase succeeds; TRL's own `save_*` and `resume_from_checkpoint` settings are refused. `mlp resume 7` starts a new Pipeline from Pipeline 7's resolved request, with Secrets read afresh, that records `resumed_from: 7`: a finished `distill` and every Phase that registered a Model Version are skipped, the next Phase starts from that Model Version and continues from its Checkpoint if it saved one, and the Stages after `finetune` run as usual. A failed or cancelled Pipeline's Checkpoints stay until deleted on the Storage page, which refuses while a running resume needs one.

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
  max_tokens: 1024             # optional, as is temperature
  serving: {max_model_len: 8192}  # optional: options for the Teacher's vLLM, as for Endpoints
```

The Teacher is a Base Model or Model Version, which the step serves on vLLM with `gpus_per_stage` GPUs and the Endpoint defaults; a running Endpoint, reached through its Service without a GPU; or a model behind an OpenAI-compatible `api_url`. An API Teacher's key is the `teacher_api_key` Secret (in the Profile's `secrets` or the Web UI's Secrets), which only the `distill` step receives, and which is redacted from its log. Each prompt gets one reply, asked 16 at a time, which is a text or calls to the offered `tools`; no tool runs. With `tools`, validation pins the Teacher's tool parser into `serving.tool_parser` and rejects a Base Model or Model Version Teacher without one (name it in `serving.tool_parser`); tool parameters must be a JSON Schema. A reply is dropped as malformed when it is an error or empty, was cut off at `max_tokens`, calls a tool that isn't offered, has arguments that aren't JSON or don't match the tool's schema, or makes several calls while `parallel_tool_calls` is off. The kept and dropped counts are the Run's `distill/kept` and `distill/dropped` metrics, and the first 5 dropped replies with their reasons are its `dropped_replies.json`. If more than 20 % are dropped, the step fails and registers nothing. Otherwise every kept row (`prompt` messages, `completion` with the one assistant message, its calls' arguments as objects, and `tools` when given) becomes the next version of the Dataset named after the Pipeline, a `prompt_completion` Dataset that the step registers through the API with the step token. A Phase with `dataset: "@distill"` trains on it; it must train on `prompt_completion` rows, and `distill` must be enabled.

Next comes `finetune`, which trains each Phase in its own step (`finetune-<algorithm>` in the KFP UI) on the backend's trainer image (`hf`: TRL + PEFT) with `gpus_per_stage` GPUs from the platform settings. It runs offline from the Model Cache (or downloads the Model Version it starts from) and logs its params and metrics into its own MLflow Run (KFP's MLflow plugin creates one per step, under the Pipeline's Run). Each Phase's output is registered in the MLflow Model Registry as the next Model Version of the Registered Model named after the Pipeline (`qwen-sft` version 1, 2, …), and the next Phase starts from it; `@finetune` is the last one. Each version holds the Adapter or full weights, the starting model's tokenizer and chat template, and `pipeline_request.json` with the resolved request. Its tags are `weights` (`adapter` or `full`), `base_model` (an Adapter's base: the Base Model, or the full-weight Model Version it was trained on), `merged: true` for an Adapter merged into its base, `method`, `parent` (the previous Phase's Model Version, or for the first Phase its starting model), `pipeline`, `phase` (1, 2, …), `algorithm`, `teacher` (a `distillation` Phase's Teacher), `backend`, `tool_parser` (the vLLM tool parser for the Base Model's `model_type`, or `none`) and `tools_rendered` (`false` when the chat template drops the tools a client sends, so tool calling won't work). There are no size limits: a Phase that runs out of GPU memory fails with a plain message naming the settings to lower.

With an `evaluate` block (`mlp run --evaluate` takes the Profile's `evaluate:`), an `evaluate` step runs after `finetune` and logs every metric into its own MLflow Run:

```yaml
evaluate:
  model: hf:Qwen/Qwen2.5-0.5B-Instruct  # or model:qwen-sft@2, endpoint:chat; default @finetune
  benchmarks: [lm_eval:gsm8k, lm_eval:mmlu]
  limit: 50  # optional: samples per task, for a quick look
  serving: {max_model_len: 8192}  # optional: options for the step's own vLLM, as for Endpoints
  performance: {prompt_tokens: 256, output_tokens: 128, concurrency: 1, requests: 100}  # optional
```

`model` is a Base Model, any Model Version (an Adapter too), a running Endpoint (not `pending`), or `@finetune`, the Pipeline's last Model Version and the default when `finetune` runs. Benchmarks come from the catalog in `mlp_core.config.BENCHMARKS` (`GET /benchmarks`, the Web UI picker): each is `harness:task` with a category, a one-line description, its dataset licence and download size. Only datasets without a non-commercial or share-alike licence are listed. The harness is lm-evaluation-harness (`lm_eval:`) or, for benchmarks it lacks, EvalScope (`evalscope:`). Tool calling is checked with BFCL (`bfcl:<category>`, e.g. `bfcl:simple_python` or `bfcl:multi_turn_base`), whose datasets ship inside its package, so `fetch` downloads nothing for it; it runs in its own virtualenv in the stages image (`versions.bfcl` in `deploy/values.yaml`), as it pins libraries the other harnesses can't share. BFCL calls the model through its OpenAI handler with real `tools` over chat completions, never its per-family prompt formats, so the score covers the chat template and tool parser too, as an Endpoint would serve them. Validation pins that parser into `serving.tool_parser`: the override, else the one the model (for `@finetune`, its starting model) maps to, exactly as `serve` picks it; a model without one is rejected for BFCL, and an Endpoint uses its own. `limit` runs a category's first entries. The step starts vLLM on `gpus_per_stage` GPUs with the Endpoint defaults (an Endpoint is reached through its Service instead, without a GPU) and runs each benchmark in its own harness process, offline from the Model Cache: lm-eval against vLLM's completions API, EvalScope against its chat completions API, with its `EVALSCOPE_CACHE` and `MODELSCOPE_CACHE` in the benchmark's Model Cache directory. Coding benchmarks (`lm_eval:humaneval`, `lm_eval:mbpp`, `evalscope:mbpp_plus`) score generated code only in the Sandbox: each harness process replaces the one place its tasks run code (HF evaluate's `code_eval` for lm-eval, `CodeExecutionSandboxMixin` for EvalScope) with a call that sends the program and its tests to the Sandbox, so no generated code runs in the step; lm-eval runs code-executing tasks only for catalog entries in the `coding` category. `fetch` and `evaluate` both reach the Sandbox, as `fetch` scores one sample to download all a benchmark needs. Metrics are logged as `<harness>/<task>/<metric>` (BFCL's as `bfcl/<category>/accuracy`, with the parser as the Run's `tool_parser` tag); a benchmark that fails is tagged `NA` in the Run and the others still run. A second evaluation of the same benchmark downloads nothing.

`performance` (any of its values may be left out; those shown are the defaults) adds a GuideLLM run after the benchmarks: `requests` synthetic chat requests of `prompt_tokens` tokens, each asking for `output_tokens`, `concurrency` at a time, against the step's own vLLM started with the `serving` options, its prompts tokenized with the model's own tokenizer from the Model Cache. It logs time to first token, inter-token latency and output tokens/s over the successful requests as `guidellm/<metric>/<mean|median|p99>`, e.g. `guidellm/time_to_first_token_ms/p99`; if GuideLLM fails or no request succeeds (e.g. `prompt_tokens` past `max_model_len`), the Run is tagged `guidellm: NA`. Without `performance` nothing is measured, so evaluations stay fast. An Endpoint is shared, so it can't be measured; `benchmarks` may be empty when `performance` is set.

With a `serve` block (`mlp run --serve` takes the Profile's `serve:` settings, or none), the Pipeline ends with a `serve` step that asks the API to start an Endpoint for the last Model Version it registered, with the serving options listed under Endpoints. The Endpoint is named after the Pipeline unless `serve.name` names it; validation rejects a name that can't name an Endpoint (e.g. one with a dot) or that a running Endpoint holds. The step calls the API inside the cluster with a step token from the Pipeline's Secret, which reaches only that Pipeline's step routes (`serve` and `distill`) and is no login; like the Secret, it lasts 48 hours, so a Pipeline that runs longer can't serve and its Model Version is served from the Serving page instead. The Endpoint outlives the Pipeline and runs until stopped.

## Smoke Test

```sh
mlp smoke-test                              # every case; prints each result, exits 1 if one failed
mlp smoke-test --phases sft --backends hf   # a custom one: only the finetune cases named
mlp smoke-test --sandbox                    # a custom one: only the sandbox case
mlp smoke-test --uploaded-model             # a custom one: only the uploaded-model case
mlp smoke-test --serving                    # a custom one: only serving the Base Model and tiny model
mlp smoke-test --evaluate                   # a custom one: only evaluating the Base Model
mlp smoke-test --distill                    # a custom one: only distilling and training on it
mlp smoke-test --chain                      # a custom one: only the sft → dpo chain
mlp smoke-test --weights                    # a custom one: only rsLoRA, merged outputs, full after an Adapter
mlp smoke-test --resume                     # a custom one: only stopping a Phase at a Checkpoint and resuming it
```

The API routes are `POST /smoke-tests/complete` and `POST /smoke-tests/custom` (body e.g. `{"finetune": {"phases": ["sft"]}}`; an unnamed list means all of it). Both answer 202 with the Kubeflow run link, or 409 while a Smoke Test runs. It is one Kubeflow run, `smoketest-YYMMDD-HHMMSS`, on `Qwen/Qwen2.5-0.5B-Instruct`, with one node per case: `fetch`, then `sandbox` (custom body `{"sandbox": true}`), a step that sends the Sandbox a batch of snippets and passes if one runs, one finds itself under gVisor, one finds no network, one stops at the memory limit and one is killed at the timeout, then one per finetune Phase × method × backend the backend supports (`sft-lora-hf`, `sft-qlora-hf`, `sft-full-hf`, `dpo-lora-hf`, …, `distillation-lora-hf` with the Base Model as its Teacher), each training 3 steps on its algorithm's bundled Dataset from `packages/core/src/mlp_core/smoke_test_datasets/`, with `sft` and `lora` chosen also `sft-assistant-only-hf`, an `sft` Phase with `assistant_only_loss` on the Base Model's own template, and `uploaded-model` (custom body `{"uploaded_model": true}`): the API downloads `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`, uploads it as `smoketest-…-uploaded` through the same checks as a user's upload, and the case trains an `sft` Phase `from` it. A case passes if it runs without errors; what it learns is ignored. Each node runs once the one before it ended, even if that failed, so one failure shows red and the rest still run. The evaluate cases (custom body `{"evaluate": true}`) each run 5 samples of a benchmark that `fetch` downloads too: `lm_eval:truthfulqa_mc2` on the Base Model (`evaluate-base-model`) and, with a finetune case, on its Adapter (`evaluate-adapter`), plus the coding benchmarks `lm_eval:humaneval` (`evaluate-coding`) and the EvalScope task `evalscope:mbpp_plus` (`evaluate-evalscope`) on the Base Model, whose generated code runs in the Sandbox, `bfcl:simple_python` on the Base Model with its `hermes` tool parser (`evaluate-tool-calling`), and one short GuideLLM performance run of 10 requests on the Base Model (`evaluate-performance`). The distill case (custom body `{"distill": true}`) is two nodes: a `distill` step (`distill-tools-distill`) with the Base Model as in-cluster Teacher, offered one `get_weather` tool, on the bundled `distill.jsonl` prompts, then an `sft` Phase on `@distill` (`distill-tools`), which runs only once `distill` passed. The chain case (custom body `{"chain": true}`) is two nodes too: an `sft` Phase (`sft-dpo-chain-finetune-sft`), then a `dpo` Phase that continues its Adapter (`sft-dpo-chain`), which runs only once `sft` passed. The weight cases (custom body `{"weights": true}`) train an rsLoRA Adapter (`sft-rslora-hf`), a QLoRA Adapter merged into full weights (`sft-qlora-merged-hf`), a DoRA Adapter merged into full weights (`sft-dora-merged-hf`), and an `sft` Adapter followed by a `full` `dpo` Phase that merges it first (`sft-lora-dpo-full-hf-finetune-sft`, then `sft-lora-dpo-full-hf`). The resume case (custom body `{"resume": true}`) is two nodes: an `sft` Phase that saves a Checkpoint after its first step and stops there, as a cancel would (`resume-interrupted`), then the same Phase continued from that Checkpoint (`resume`); skipping finished Phases is covered by the API's tests. The serving cases (custom body `{"serving": true}`) run outside the Kubeflow run: the API starts an Endpoint each for the Base Model (`serve-base-model`), the uploaded tiny model (`serve-full-weights`) and, once each is registered, the Adapter of the first finetune case that keeps one (`serve-adapter`) and, with the weight cases, the merged QLoRA model (`serve-merged`); a case passes once vLLM is ready, and its Endpoint is then stopped. With a finetune case, `{"serving": true}` also runs `finetune-serve` as the last two nodes: an `sft` Phase like the first finetune case's (`finetune-serve-finetune-sft`), then its `serve` step (`finetune-serve`), which passes once the API started the Endpoint; that Endpoint is stopped right away, so it holds no GPU the other cases wait for. The Smoke Test finishes once the run and these cases did. It also lists in `mlp ls`. Once it finished, the API stops its Endpoints and deletes every Dataset and Registered Model named `smoketest-YYMMDD-HHMMSS-…` and its Checkpoints; only the Kubeflow run, its logs and its MLflow Runs remain, and the Base Model stays in the Model Cache.

## Sandbox

The only place untrusted Python runs: RL rewards and code generated in benchmarks. It is a small stateless server (`packages/sandbox`) that Pipeline steps call inside the cluster with a batch of snippets, `POST /snippets` with `[{"code": "...", "input": "..."}]`. Each snippet runs as a script in a fresh Python process with `input` on its stdin, in an empty directory and without the server's environment. The answer holds each one's `status` (`ok`, `error`, `timeout` or `memory_limit`), `stdout` and `stderr`, in order. A snippet is killed with every process it started after `sandbox_timeout_seconds`, stops at `sandbox_memory_mb`, and is stopped once it writes more than 1 MiB to any file, its output included. Each of the `sandbox_replicas` replicas runs at most `sandbox_cpus_per_replica` snippets at once, across all batches. All four are platform settings. The pods run under gVisor (the `gvisor` RuntimeClass), with no Kubernetes credentials and no outbound network. A NetworkPolicy lets only Kubeflow run pods reach them, and they are never routed through Traefik. Its tests run the server in-process without gVisor, on Linux only.

## Phase algorithms

Each algorithm is one row of `ALGORITHMS` in `packages/core/src/mlp_core/config.py`: its TRL trainer and config, the Dataset row formats it trains on, the setting bounding a row's tokens that every Phase must set, its blocked settings, the platform's defaults, which the Phase's `settings` override, and whether it learns from a Teacher. A Dataset whose rows the algorithm can't train on is rejected at validation. Each Phase picks a weight method, which every algorithm supports:

- `lora` (the default): trains an Adapter on the base at its saved precision. A Phase training a new Adapter sets `lora`, which goes to PEFT's `LoraConfig` with `task_type: CAUSAL_LM` (blocked); `use_rslora: true` there gives rsLoRA.
- `qlora`: the same, on the base loaded in 4 bits (NF4, computing in bfloat16, via bitsandbytes), for less GPU memory.
- `full`: trains every weight and takes no `lora`.

Each Phase also picks its `output`: `adapter` (the default) registers the Adapter, `merged` registers only the Adapter merged into its base, as full weights tagged `merged: true`; the base is reloaded in bfloat16 before merging, so `qlora` merges into full precision weights, not 4 bits. `full` ignores `output`. Merge when the result must stand alone: to export it, to start another Pipeline `from` it, or to use options vLLM can't serve in an Adapter. An Adapter kept as an Adapter must be servable by vLLM, so with `output: adapter` validation rejects `use_dora`, `modules_to_save`, `bias` other than `none` and `r` above 512; with `output: merged` they are allowed, as the result is plain weights.

Every algorithm takes TRL's defaults for its config, plus `report_to: mlflow`, `save_strategy: no` and `disable_tqdm: true`. Every algorithm blocks `output_dir`, `report_to`, `logging_dir` (the platform stores and logs the output), `save_strategy`, `save_steps`, `save_total_limit`, `resume_from_checkpoint` (the platform owns Checkpoints), `push_to_hub` and `hub_*` (outputs go to the Model Registry), and `model_init_kwargs` and `trust_remote_code` (could ask for remote code).

### Phase chaining

`finetune.phases` is an ordered list. Each Phase runs in its own step and starts from the previous Phase's output, and every Phase's output is registered as a Model Version, so `sft` then `dpo` registers `qwen-sft@1` and then `qwen-sft@2`, trained from the first. What a Phase does with the previous Phase's output depends on what that output is:

- **An Adapter** (`lora`/`qlora` with `output: adapter`): a `lora` or `qlora` Phase continues training that same Adapter, keeping its rank and targets, so it leaves out `lora`. A `full` Phase merges the Adapter into its base first, then trains every weight. To train a new Adapter instead, set `output: merged` on the previous Phase.
- **Full weights** (a `full` Phase, `output: merged`, or the first Phase's Base Model or `from` Model Version): a `lora` or `qlora` Phase trains a new Adapter on them, so it sets `lora`; that Adapter's base is the full-weight Model Version.

Validation rejects a Phase that sets `lora` where it would continue an Adapter, one that leaves it out where it trains a new one, and a `full` Phase with `lora`. `evaluate`, `serve` and Endpoints load every Model Version of a chain, an Adapter on a Base Model or on a full-weight Model Version alike. Another Pipeline can continue the work through `from` only from full weights, so end with `output: merged` or a `full` Phase to hand it on.

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

  On `hf`, TRL finds the assistant's turns through `{% generation %}` markers in the chat template. Validation reads the starting model's template (from Hugging Face, or from the Model Version's files) and accepts it if it has the markers, or if it is one TRL swaps for a marked version of its own while training, as it does for Qwen2.5's, Llama 3's and Gemma's, among others; otherwise it rejects the request. That list is generated with the trainer configs (`TrainingChatTemplates.json`).

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

## Web UI

The Web UI (`packages/interface/webui`, React) is a client of the API like the CLI (`packages/interface/cli`). Open `https://<domain>/` and log in with the shared password; the cookie lasts 12 hours and also opens the KFP UI (`/pipeline/`) and the MLflow UI (`/mlflow/`), which send a logged-out browser to the same login page.

- **Pipelines**: every Pipeline with its Owner, status, Stages and links to its Kubeflow run and MLflow Run, updated live, with Cancel; the sidebar shows the platform's GPU count.
- **Storage**: object store usage, every Dataset and Model Version with Download and Delete (a Model Version lists a link per file), kept Checkpoints with their size and Delete, and a form that uploads a model directory as `mlp models upload` does.
- **Serving**: every Endpoint with its model, status and URL, updated live, with Stop, and a form that starts an Endpoint for any Model Version or Base Model with the serving options (the same as `serve` takes, from `GET /schema`).
- **New Pipeline**: a form built from `GET /schema` that makes the same Pipeline Request as `mlp run`. A comma in a list field (e.g. `target_modules`) makes a list, and "More settings" takes further `SFTConfig`/`LoraConfig` keys. While the request changes, it shows how much is still to download. Validate shows the resolved request, Submit starts the Pipeline, and every error appears next to its field. The Hugging Face token is sent as a Secret beside the request. An optional Stage such as `serve` joins the request once its switch is on.

## Development

```sh
uv sync --all-packages   # add --extra hf for the trainer libraries; their tests skip without them
uv run pytest
uv run ruff check . && uv run ruff format --check .

cd packages/interface/webui
npm ci
npm run lint && npm run build && npm test   # npm run format fixes formatting
MLP_API_URL=http://localhost:8000 npm run dev   # http://localhost:5173/ui/, API calls go to MLP_API_URL
```
