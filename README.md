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
```

To train on a full-weight Model Version, e.g. an uploaded one, put `from: model:my-model` (or `model:my-model@2`) in place of `base_model`; Adapters can't be started from yet.

`mlp validate --finetune sft` prints the resolved request (Base Model pinned to a commit, `dataset:chat` to its latest version, e.g. `dataset:chat@2`) or every error with its path. Validate and submit both answer `downloads`, each `{kind, ref, bytes, cached}`, `download_bytes`, the total not in the Model Cache yet, and `cached_bytes`. Validation first checks that the request holds every basic value above, then that each `settings` key exists in TRL's `SFTConfig` (and each `lora` key in PEFT's `LoraConfig`) with the right type. That second check uses schemas in `packages/core/src/mlp_core/pipeline_request/trainer_configs/`, regenerated by `packages/core/src/mlp_core/pipeline_request/generate_trainer_configs.py` when TRL or PEFT is bumped; without a usable schema it is skipped.

## Pipelines

```sh
mlp run --finetune sft   # submits the request and prints the Pipeline ID right away
mlp run --finetune sft --dry-run  # lists what it would download, without submitting
mlp run --finetune sft --serve    # ends with an Endpoint for the last Model Version
mlp run --finetune sft --evaluate # then runs the Profile's benchmarks on the new Model Version
mlp run --evaluate                # only the benchmarks, on the Profile's `evaluate.model`
mlp run --distill --finetune sft  # distills the Profile's prompts, then trains on the replies
mlp ls                   # every Pipeline with its Owner, status, Stages and Kubeflow/MLflow links
mlp cancel 7             # stops the run and deletes its Secrets
mlp rerun 7              # submits Pipeline 7's resolved request again as a new Pipeline
```

Each Pipeline first runs `fetch`, which downloads the Base Models at their pinned commits and the benchmarks into the Model Cache (a Pipeline with nothing to download has no `fetch`), and ends with a `cleanup` step that runs even after a failure. The Hugging Face token comes from the Profile's `secrets`, else `HF_TOKEN` or `hf auth login`, else a hidden prompt (Enter skips it; public models download anonymously). It lives in a per-Pipeline Kubernetes Secret that only `fetch` sees, is redacted from step logs, and is deleted once the Pipeline finishes or is cancelled (any left over are swept after 48 hours). A step whose GPUs are busy waits and shows as `waiting for GPU`. Pipelines are never retried automatically; `mlp rerun` reads the Secrets afresh, as `mlp run` does, since a Pipeline's own are never kept.

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

Next comes `finetune`, which trains the Phase on the backend's trainer image (`hf`: TRL + PEFT) with `gpus_per_stage` GPUs from the platform settings. It runs offline from the Model Cache (or downloads the Model Version it starts from) and logs its params and metrics into its own MLflow Run (KFP's MLflow plugin creates one per step, under the Pipeline's Run). Its output is registered in the MLflow Model Registry as the next Model Version of the Registered Model named after the Pipeline (`qwen-sft` version 1, 2, …). Each version holds the Adapter, the Base Model's tokenizer and chat template, and `pipeline_request.json` with the resolved request. Its tags are `weights: adapter`, `base_model`, `pipeline`, `phase`, `algorithm`, `backend`, `tool_parser` (the vLLM tool parser for the Base Model's `model_type`, or `none`) and `tools_rendered` (`false` when the chat template drops the tools a client sends, so tool calling won't work). There are no size limits: a Phase that runs out of GPU memory fails with a plain message naming the settings to lower.

With an `evaluate` block (`mlp run --evaluate` takes the Profile's `evaluate:`), an `evaluate` step runs after `finetune` and logs every metric into its own MLflow Run:

```yaml
evaluate:
  model: hf:Qwen/Qwen2.5-0.5B-Instruct  # or model:qwen-sft@2, endpoint:chat; default @finetune
  benchmarks: [lm_eval:gsm8k, lm_eval:mmlu]
  limit: 50  # optional: samples per task, for a quick look
  serving: {max_model_len: 8192}  # optional: options for the step's own vLLM, as for Endpoints
  performance: {prompt_tokens: 256, output_tokens: 128, concurrency: 1, requests: 100}  # optional
```

`model` is a Base Model, any Model Version (an Adapter too), a running Endpoint (not `pending`), or `@finetune`, the Pipeline's last Model Version and the default when `finetune` runs. Benchmarks come from the catalog in `mlp_core.config.BENCHMARKS` (`GET /benchmarks`, the Web UI picker): each is `harness:task` with a category, a one-line description, its dataset licence and download size. Only datasets without a non-commercial or share-alike licence are listed. The harness is lm-evaluation-harness (`lm_eval:`) or, for benchmarks it lacks, EvalScope (`evalscope:`). The step starts vLLM on `gpus_per_stage` GPUs with the Endpoint defaults (an Endpoint is reached through its Service instead, without a GPU) and runs each benchmark in its own harness process, offline from the Model Cache: lm-eval against vLLM's completions API, EvalScope against its chat completions API, with its `EVALSCOPE_CACHE` and `MODELSCOPE_CACHE` in the benchmark's Model Cache directory. Coding benchmarks (`lm_eval:humaneval`, `lm_eval:mbpp`, `evalscope:mbpp_plus`) score generated code only in the Sandbox: each harness process replaces the one place its tasks run code (HF evaluate's `code_eval` for lm-eval, `CodeExecutionSandboxMixin` for EvalScope) with a call that sends the program and its tests to the Sandbox, so no generated code runs in the step; lm-eval runs code-executing tasks only for catalog entries in the `coding` category. `fetch` and `evaluate` both reach the Sandbox, as `fetch` scores one sample to download all a benchmark needs. Metrics are logged as `<harness>/<task>/<metric>`; a benchmark that fails is tagged `NA` in the Run and the others still run. A second evaluation of the same benchmark downloads nothing.

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
```

The API routes are `POST /smoke-tests/complete` and `POST /smoke-tests/custom` (body e.g. `{"finetune": {"phases": ["sft"]}}`; an unnamed list means all of it). Both answer 202 with the Kubeflow run link, or 409 while a Smoke Test runs. It is one Kubeflow run, `smoketest-YYMMDD-HHMMSS`, on `Qwen/Qwen2.5-0.5B-Instruct`, with one node per case: `fetch`, then `sandbox` (custom body `{"sandbox": true}`), a step that sends the Sandbox a batch of snippets and passes if one runs, one finds itself under gVisor, one finds no network, one stops at the memory limit and one is killed at the timeout, then one per finetune Phase × method × backend the backend supports (e.g. `sft-lora-hf`), each training 3 steps on a bundled Dataset from `packages/core/src/mlp_core/smoke_test_datasets/`, and `uploaded-model` (custom body `{"uploaded_model": true}`): the API downloads `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`, uploads it as `smoketest-…-uploaded` through the same checks as a user's upload, and the case trains an `sft` Phase `from` it. A case passes if it runs without errors; what it learns is ignored. Each node runs once the one before it ended, even if that failed, so one failure shows red and the rest still run. The evaluate cases (custom body `{"evaluate": true}`) each run 5 samples of a benchmark that `fetch` downloads too: `lm_eval:truthfulqa_mc2` on the Base Model (`evaluate-base-model`) and, with a finetune case, on its Adapter (`evaluate-adapter`), plus the coding benchmarks `lm_eval:humaneval` (`evaluate-coding`) and the EvalScope task `evalscope:mbpp_plus` (`evaluate-evalscope`) on the Base Model, whose generated code runs in the Sandbox, and one short GuideLLM performance run of 10 requests on the Base Model (`evaluate-performance`). The distill case (custom body `{"distill": true}`) is two nodes: a `distill` step (`distill-tools-distill`) with the Base Model as in-cluster Teacher, offered one `get_weather` tool, on the bundled `distill.jsonl` prompts, then an `sft` Phase on `@distill` (`distill-tools`), which runs only once `distill` passed. The serving cases (custom body `{"serving": true}`) run outside the Kubeflow run: the API starts an Endpoint each for the Base Model (`serve-base-model`), the uploaded tiny model (`serve-full-weights`) and, once it is registered, the first finetune case's Adapter (`serve-adapter`); a case passes once vLLM is ready, and its Endpoint is then stopped. With a finetune case, `{"serving": true}` also runs `finetune-serve` as the last two nodes: an `sft` Phase like the first finetune case's (`finetune-serve-finetune`), then its `serve` step (`finetune-serve`), which passes once the API started the Endpoint; that Endpoint is stopped right away, so it holds no GPU the other cases wait for. The Smoke Test finishes once the run and these cases did. It also lists in `mlp ls`. Once it finished, the API stops its Endpoints and deletes every Dataset and Registered Model named `smoketest-YYMMDD-HHMMSS-…`; only the Kubeflow run, its logs and its MLflow Runs remain, and the Base Model stays in the Model Cache.

## Sandbox

The only place untrusted Python runs: RL rewards and code generated in benchmarks. It is a small stateless server (`packages/sandbox`) that Pipeline steps call inside the cluster with a batch of snippets, `POST /snippets` with `[{"code": "...", "input": "..."}]`. Each snippet runs as a script in a fresh Python process with `input` on its stdin, in an empty directory and without the server's environment. The answer holds each one's `status` (`ok`, `error`, `timeout` or `memory_limit`), `stdout` and `stderr`, in order. A snippet is killed with every process it started after `sandbox_timeout_seconds`, stops at `sandbox_memory_mb`, and is stopped once it writes more than 1 MiB to any file, its output included. Each of the `sandbox_replicas` replicas runs at most `sandbox_cpus_per_replica` snippets at once, across all batches. All four are platform settings. The pods run under gVisor (the `gvisor` RuntimeClass), with no Kubernetes credentials and no outbound network. A NetworkPolicy lets only Kubeflow run pods reach them, and they are never routed through Traefik. Its tests run the server in-process without gVisor, on Linux only.

## Phase algorithms

Each algorithm is one row of `ALGORITHMS` in `packages/core/src/mlp_core/config.py`: its TRL trainer and config, the Dataset row formats it trains on, its blocked settings and the platform's defaults, which the Phase's `settings` override. Every algorithm uses LoRA (`method: lora`): `lora` goes to PEFT's `LoraConfig` with `task_type: CAUSAL_LM`. Adapters must be servable by vLLM, so `r` is at most 512 and `use_dora`, `modules_to_save`, `bias` and `task_type` are blocked.

### `sft`: supervised finetuning

Trains the model to produce the Dataset's text, with TRL's `SFTTrainer`. Use it to teach a format, a style, a domain or a task from examples of the answers you want; it is the usual first Phase.

- **Rows**: `messages` (conversations, rendered with the Base Model's own chat template; a Base Model without one is refused when the Phase starts), prompt-completion (loss on the completion only), or `text` (plain language modelling):
  ```jsonl
  {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
  {"prompt": "Translate to French: cat", "completion": "chat"}
  {"text": "Any text the model should learn to continue."}
  ```
- **Defaults**: TRL's `SFTConfig` defaults, plus `report_to: mlflow`, `save_strategy: no` and `disable_tqdm: true`.
- **Blocked settings**: `output_dir`, `report_to`, `logging_dir` (the platform stores and logs the output), `save_strategy`, `save_steps`, `save_total_limit`, `resume_from_checkpoint` (the platform owns Checkpoints), `push_to_hub` and `hub_*` (outputs go to the Model Registry), `model_init_kwargs` (could ask for remote code) and `chat_template_path` (the Base Model's chat template is always kept).

## Web UI

The Web UI (`packages/interface/webui`, React) is a client of the API like the CLI (`packages/interface/cli`). Open `https://<domain>/` and log in with the shared password; the cookie lasts 12 hours and also opens the KFP UI (`/pipeline/`) and the MLflow UI (`/mlflow/`), which send a logged-out browser to the same login page.

- **Pipelines**: every Pipeline with its Owner, status, Stages and links to its Kubeflow run and MLflow Run, updated live, with Cancel; the sidebar shows the platform's GPU count.
- **Storage**: object store usage, every Dataset and Model Version with Download and Delete (a Model Version lists a link per file), and a form that uploads a model directory as `mlp models upload` does.
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
