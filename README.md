# mlopsplatform

A Kubernetes-native platform to distill, finetune, quantize, evaluate and serve open-source LLMs. See `CONTEXT.md` for the vocabulary.

## Install

On a WSL or Ubuntu host with an NVIDIA driver (`nvidia-smi` must work) and systemd:

```sh
git clone https://github.com/Maxi580/mlopsplatform && cd mlopsplatform
./install.sh dev
```

It installs only missing host prerequisites (k3s, nerdctl + BuildKit, NVIDIA container toolkit, gVisor, helm, yq), builds our images, deploys the platform and prints its URLs and the path to `ca.crt`. Import `ca.crt` into your OS trust store once. Running it again is safe and keeps all data.

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
mlp cache                                      # every cached Base Model with size and last use
mlp cache free hf:Qwen/Qwen2.5-0.5B-Instruct@<commit>
```

`fetch` downloads each Base Model once into the Model Cache, a host directory (`modelCacheHostPath`) that the steps and the API mount. Past `model_cache_high_water_mark` of `model_cache_size`, the API evicts the least recently used Base Models, never one an unfinished Pipeline or an Endpoint uses; a submit or an Endpoint start also evicts to make room for its Base Model. `fetch` fails before downloading if the whole download still doesn't fit. Freeing a Base Model in use is refused with 409.

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

Tool calling is on (`--enable-auto-tool-choice`) whenever a parser is known: `tool_parser`, else the model's `tool_parser` tag, else the one its Base Model's `model_type` maps to. `mlp_core.endpoint_spec.vllm_args` turns a spec into vLLM's arguments, so `evaluate` and in-cluster Teachers will use the same flags.

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
mlp ls                   # every Pipeline with its Owner, status, Stages and Kubeflow/MLflow links
mlp cancel 7             # stops the run and deletes its Secrets
mlp rerun 7              # submits Pipeline 7's resolved request again as a new Pipeline
```

Each Pipeline first runs `fetch`, which downloads the Base Model at its pinned commit into the Model Cache (a Pipeline starting `from` a Model Version has no `fetch`), and ends with a `cleanup` step that runs even after a failure. The Hugging Face token comes from the Profile's `secrets`, else `HF_TOKEN` or `hf auth login`, else a hidden prompt (Enter skips it; public models download anonymously). It lives in a per-Pipeline Kubernetes Secret that only `fetch` sees, is redacted from step logs, and is deleted once the Pipeline finishes or is cancelled (any left over are swept after 48 hours). A step whose GPUs are busy waits and shows as `waiting for GPU`. Pipelines are never retried automatically; `mlp rerun` reads the Secrets afresh, as `mlp run` does, since a Pipeline's own are never kept.

Next comes `finetune`, which trains the Phase on the backend's trainer image (`hf`: TRL + PEFT) with `gpus_per_stage` GPUs from the platform settings. It runs offline from the Model Cache (or downloads the Model Version it starts from) and logs its params and metrics into its own MLflow Run (KFP's MLflow plugin creates one per step, under the Pipeline's Run). Its output is registered in the MLflow Model Registry as the next Model Version of the Registered Model named after the Pipeline (`qwen-sft` version 1, 2, …). Each version holds the Adapter, the Base Model's tokenizer and chat template, and `pipeline_request.json` with the resolved request. Its tags are `weights: adapter`, `base_model`, `pipeline`, `phase`, `algorithm`, `backend`, `tool_parser` (the vLLM tool parser for the Base Model's `model_type`, or `none`) and `tools_rendered` (`false` when the chat template drops the tools a client sends, so tool calling won't work). There are no size limits: a Phase that runs out of GPU memory fails with a plain message naming the settings to lower.

## Smoke Test

```sh
mlp smoke-test                              # every case; prints each result, exits 1 if one failed
mlp smoke-test --phases sft --backends hf   # a custom one: only the finetune cases named
mlp smoke-test --uploaded-model             # a custom one: only the uploaded-model case
mlp smoke-test --serving                    # a custom one: only serving the Base Model and tiny model
```

The API routes are `POST /smoke-tests/complete` and `POST /smoke-tests/custom` (body e.g. `{"finetune": {"phases": ["sft"]}}`; an unnamed list means all of it). Both answer 202 with the Kubeflow run link, or 409 while a Smoke Test runs. It is one Kubeflow run, `smoketest-YYMMDD-HHMMSS`, on `Qwen/Qwen2.5-0.5B-Instruct`, with one node per case: `fetch`, then one per finetune Phase × method × backend the backend supports (e.g. `sft-lora-hf`), each training 3 steps on a bundled Dataset from `packages/core/src/mlp_core/smoke_test_datasets/`, and `uploaded-model` (custom body `{"uploaded_model": true}`): the API downloads `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`, uploads it as `smoketest-…-uploaded` through the same checks as a user's upload, and the case trains an `sft` Phase `from` it. A case passes if it runs without errors; what it learns is ignored. Each node runs once the one before it ended, even if that failed, so one failure shows red and the rest still run. The serving cases (custom body `{"serving": true}`) run outside the Kubeflow run: the API starts an Endpoint each for the Base Model (`serve-base-model`), the uploaded tiny model (`serve-full-weights`) and, once it is registered, the first finetune case's Adapter (`serve-adapter`); a case passes once vLLM is ready, and its Endpoint is then stopped. The Smoke Test finishes once the run and these cases did. It also lists in `mlp ls`. Once it finished, the API stops its Endpoints and deletes every Dataset and Registered Model named `smoketest-YYMMDD-HHMMSS-…`; only the Kubeflow run, its logs and its MLflow Runs remain, and the Base Model stays in the Model Cache.

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
- **New Pipeline**: a form built from `GET /schema` that makes the same Pipeline Request as `mlp run`. A comma in a list field (e.g. `target_modules`) makes a list, and "More settings" takes further `SFTConfig`/`LoraConfig` keys. While the request changes, it shows how much is still to download. Validate shows the resolved request, Submit starts the Pipeline, and every error appears next to its field. The Hugging Face token is sent as a Secret beside the request.

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
