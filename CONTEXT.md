# MLOps Platform

An internal, Kubernetes-native platform where a team finetunes, evaluates, distills and serves open-source language models, submitting work through a CLI or Web UI.

## Language

### Execution

**Pipeline Request**:
The config document describing what to do: which Stages are enabled and all their settings. Written as a file for the CLI or produced by the Web UI form; the one schema the API accepts and validates.
_Avoid_: Workflow, job spec, config (when meaning the whole request)

**Pipeline**:
One submitted Pipeline Request as it executes; the thing users list, watch and cancel.
_Avoid_: Workflow, experiment, job

**Stage**:
One enabled step inside a Pipeline: `distill`, `finetune`, `evaluate` or `serve`.
_Avoid_: Workflow type, step, task

**Smoke Test**:
A built-in Pipeline Request that runs every Stage end to end on a tiny model, to prove the platform works.
_Avoid_: Health check, e2e test

**Owner**:
The user a Pipeline belongs to. Recorded on every Pipeline even while there is only one user.

**Secret**:
A credential such as the Hugging Face token or a Teacher's API key, supplied with each submission and kept only while its Pipeline runs. A Pipeline Request may name a Secret but never contains its value.
_Avoid_: Credential, key, token (when meaning the platform concept)

**Job**:
A Kubernetes object that executes part of a user's submitted work. Internal: users never see or manage Jobs directly.
_Avoid_: Task, pod (when meaning the unit of work)

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
The model whose responses to a set of prompts become training data for a Student. Either a Base Model run in-cluster or an external model reached through an API.

**Student**:
The model being finetuned on a Distillation Dataset.

**Distillation Dataset**:
A set of prompt → response pairs produced by a Teacher, used to finetune a Student. Response-level only; it contains no logits.
_Avoid_: Synthetic dataset, teacher data
