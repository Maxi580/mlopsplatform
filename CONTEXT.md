# MLOps Platform

An internal, Kubernetes-native platform where a team finetunes, evaluates, distills and serves open-source language models, submitting work through a CLI or Web UI.

## Language

### Execution

**Pipeline Request**:
The document describing what to do: which Stages are enabled and all their settings. Built by the CLI from a CLI Profile or produced by the Web UI form; the one schema the API accepts and validates.
_Avoid_: Workflow, job spec, config (when meaning the whole request)

**CLI Profile**:
A local file holding a user's reusable settings for every Stage, including named Phase variants. The CLI builds a Pipeline Request from it, taking only the Stages and Phases named on the command line. It is never sent to the API as is.
_Avoid_: Config (when meaning this file), Pipeline Request

**Pipeline**:
One submitted Pipeline Request as it executes; the thing users list, watch, cancel and rerun.
_Avoid_: Workflow, experiment, job

**Stage**:
One enabled step inside a Pipeline: `distill`, `sweep`, `finetune`, `quantize`, `speculate` or `evaluate`, always run in that order. Serving is no Stage: Endpoints start only from the Serving tab or `mlp endpoints start`.
_Avoid_: Workflow type, step, task

**Phase**:
One training algorithm run inside the `finetune` Stage (e.g. SFT, then DPO, then GRPO). Phases run in order, each starting from the previous Phase's output, and every Phase output becomes a Model Version.
_Avoid_: Step, sub-stage

**Checkpoint**:
A snapshot taken during a Phase that the Phase can be resumed from.

**Sweep**:
The `sweep` Stage: a hyperparameter search over one Phase configuration. Its output is the best parameters found, not a model; a later Phase can train with them.

**Trial**:
One training run within a Sweep. Its weights are thrown away.

**Endpoint**:
A running, OpenAI-compatible server for one model (a Model Version or a Base Model), started by a user from the Serving tab or the CLI. It outlives its Pipeline and runs until its Owner stops it or a platform upgrade resets it.
_Avoid_: Deployment, served model, inference service

**Smoke Test**:
A built-in Pipeline, started by a user, that runs every Stage, every Phase algorithm, every weight method and every training backend on the smallest Qwen model, to prove the platform runs without errors. Output quality is ignored. A failed Case doesn't stop the rest. Afterwards it deletes everything it created except its Kubeflow run.
_Avoid_: Health check, e2e test

**Case**:
One independent check inside a Smoke Test (e.g. `fetch` or `sft-lora-hf`), run as one Kubeflow node, or for a serving case (e.g. `serve-adapter`) as an Endpoint, that passes if it runs without errors.
_Avoid_: Check, test

**Owner**:
The user a Pipeline belongs to. Recorded on every Pipeline even while there is only one user.

**Secret**:
A credential such as the Hugging Face token or a Teacher's API key, supplied with each submission and kept only while its Pipeline runs. A Pipeline Request may name a Secret but never contains its value.
_Avoid_: Credential, key, token (when meaning the platform concept)

**Job**:
A Kubernetes object that executes part of a user's submitted work. Internal: users never see or manage Jobs directly.
_Avoid_: Task, pod (when meaning the unit of work)

**Sandbox**:
The isolated place where the platform runs user-written and model-generated Python as Snippets, each in its own fresh process that keeps no state. Nothing else executes untrusted code.
_Avoid_: Code runner, executor

**Snippet**:
One piece of Python sent to the Sandbox, run as a script with its input on stdin; its result is a status (`ok`, `error`, `timeout`, `memory_limit`) and its output. Callers send Snippets in batches.
_Avoid_: Job, task, script (when meaning what the Sandbox runs)

**Run**:
An MLflow run: the tracked record of params, metrics and artifacts. Used for nothing else.
_Avoid_: Execution, experiment (when meaning a single run)

### Models and data

**Base Model**:
An open-source model pulled from a model hub, used as the starting point for finetuning or as a Teacher.
_Avoid_: Pretrained model, foundation model

**Model Cache**:
Shared storage holding downloaded Base Models and datasets, so each is fetched from the hub once rather than per Job.
_Avoid_: Model store, HF cache

**Teacher**:
The model a Student learns from: through its responses to a set of prompts (the `distill` Stage), or through its token probabilities during a `finetune` Phase. Either a model run in-cluster, or, for responses only, an external model reached through an API.

**Student**:
The model being finetuned to learn from a Teacher, on a Distillation Dataset or on the Teacher's token probabilities.

**Distillation Dataset**:
A Dataset of prompt → response pairs produced by a Teacher, used to finetune a Student. Response-level only: it contains no logits. A prompt may offer tools, in which case the response may be a tool call.
_Avoid_: Synthetic dataset, teacher data

**Dataset**:
A named set of examples that the platform stores and versions, either uploaded by a user or produced by a Pipeline. Pipeline Requests refer to Datasets by name, never by storage path.
_Avoid_: Data, corpus, file (when meaning the platform concept)

**Dataset Version**:
One immutable snapshot of a Dataset, numbered 1, 2, 3… by the platform. Every upload or Pipeline output creates a new one.

**Reference**:
How a Pipeline Request names a Base Model (`hf:org/name@revision`) or a Dataset (`dataset:name@version`). The API pins every Reference to an exact commit or version when it resolves the request.
_Avoid_: Path, URI

**Registered Model**:
A named model that a Pipeline produced. Every `finetune`, `quantize` and `speculate` output becomes a new Model Version of one. Base Models are not Registered Models.
_Avoid_: Finetuned model, checkpoint (when meaning the platform concept)

**Model Version**:
One immutable output of a Registered Model, numbered 1, 2, 3… by the platform. It holds either full weights or an Adapter.

**Adapter**:
A Model Version that holds only parameter-efficient weights (e.g. LoRA). It is usable only together with the model it was trained on: a Base Model at the exact revision, or a full-weight Model Version.
_Avoid_: LoRA (when meaning any adapter), delta

**Uploaded Model**:
A Model Version that came from a user's upload rather than from a Pipeline.

**Benchmark**:
One evaluation from the platform's catalog, named `harness:task` (e.g. `lm_eval:gsm8k`), that the `evaluate` Stage runs against a model and scores. Its datasets are kept in the Model Cache.
_Avoid_: Eval, test (when meaning the platform concept)

**Speculator**:
A small draft model trained for exactly one verifier (a Model Version or Base Model), used for speculative decoding. The `speculate` Stage registers each as a Model Version of `<pipeline>-speculator`, tagged with its verifier; an Endpoint drafts only with one trained for the model it serves.
_Avoid_: Draft model (when meaning one the platform trained)
