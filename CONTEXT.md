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
One submitted Pipeline Request as it executes; the thing users list, watch and cancel.
_Avoid_: Workflow, experiment, job

**Stage**:
One enabled step inside a Pipeline: `distill`, `finetune`, `evaluate` or `serve`.
_Avoid_: Workflow type, step, task

**Phase**:
One training algorithm run inside the `finetune` Stage (e.g. SFT, then DPO, then GRPO). Phases run in order, each starting from the previous Phase's output, and every Phase output becomes a Model Version.
_Avoid_: Step, sub-stage

**Endpoint**:
A running, OpenAI-compatible server for one model (a Model Version or a Base Model), created by the `serve` Stage. It outlives its Pipeline and runs until its Owner stops it.
_Avoid_: Deployment, served model, inference service

**Smoke Test**:
A built-in Pipeline, started by a user, that runs every Stage, every Phase algorithm, every weight method and every training backend on the smallest Qwen model, to prove the platform runs without errors. Output quality is ignored. A failed check doesn't stop the rest. Afterwards it deletes everything it created except its Kubeflow run.
_Avoid_: Health check, e2e test

**Owner**:
The user a Pipeline belongs to. Recorded on every Pipeline even while there is only one user.

**Secret**:
A credential such as the Hugging Face token or a Teacher's API key, supplied with each submission and kept only while its Pipeline runs. A Pipeline Request may name a Secret but never contains its value.
_Avoid_: Credential, key, token (when meaning the platform concept)

**Job**:
A Kubernetes object that executes part of a user's submitted work. Internal: users never see or manage Jobs directly.
_Avoid_: Task, pod (when meaning the unit of work)

**Sandbox**:
The isolated place where the platform runs user-written and model-generated Python, one stateless execution at a time. Nothing else executes untrusted code.
_Avoid_: Code runner, executor

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

**Registered Model**:
A named model that a Pipeline produced. Every `finetune` output becomes a new Model Version of one. Base Models are not Registered Models.
_Avoid_: Finetuned model, checkpoint (when meaning the platform concept)

**Model Version**:
One immutable output of a Registered Model, numbered 1, 2, 3… by the platform. It holds either full weights or an Adapter.

**Adapter**:
A Model Version that holds only parameter-efficient weights (e.g. LoRA). It is usable only together with the model it was trained on: a Base Model at the exact revision, or a full-weight Model Version.
_Avoid_: LoRA (when meaning any adapter), delta
