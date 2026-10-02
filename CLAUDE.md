## What we're building

A Kubernetes-native MLOps platform to distill, finetune, quantize, evaluate and serve open-source LLMs. Components:

- **CLI / Web UI**: thin clients; users build a Pipeline Request and talk only to the API / Can acces Kubeflow and MLFLow UIs.
- **API** (FastAPI): validates the Pipeline Request, compiles it into a Kubeflow pipeline, tracks Pipelines, Datasets and Endpoints. All decisions live here.
- **Stage entrypoints**: one container per Stage, run by Kubeflow; do the heavy lifting with TRL/PEFT, lm-eval and vLLM, log to MLflow, register outputs via the API.
- **Sandbox**: the only place untrusted Python runs (RL rewards, generated benchmark code).
- **Endpoints**: vLLM servers for a model, alive until stopped.
- **Installed, not ours**: Kubeflow Pipelines, MLflow, Postgres, SeaweedFS (one object store), Traefik (login + HTTPS).

Details: spec #24, architecture #25 (★ = undecided), decisions #2–#23 (`gh issue view <n>`). Vocabulary: `CONTEXT.md`.

## Coding Guidelines

### Scope and Complexity

This is a small internal platform for one team. Keep the architecture and requirements as simple as possible; prefer the boring solution and small solution.
Humans should be able to understand it, so try to minimize lines of code without giving up quality/ features.

### Coding Best Practices

Judgement calls only. Anything a linter can check lives in the linter config.

#### Structure
- Each module/directory has one job; one should be able to only from the directory and filename understand what happens inside.
- Group code by domain concept (terms from CONTEXT.md), not by technical layer.

#### Naming
- File, function and variable names use the domain terms from CONTEXT.md.
- A function name says what it returns or does (`load_model_config`, not `process`/`handle`/`utils`).

#### Comments
- One line, explaining why the code is the way it is if necessary. Don´t do comments if they are unecessary because its obvious what happens.
- Don't put huge Docstrings at the beginning of files.
- Delete commented-out code; git keeps history.
- A function that runs several steps gets a one-line docstring saying what it returns, and a numbered one-line comment above each step, so it reads top to bottom:
  ```python
  def validate_pipeline_request(data, secrets, hugging_face):
      """The request with its Base Model pinned to a commit, or None and every error with its path."""
      # 1. The schema: required values, types, no unknown fields.
      ...
      # 2. The settings TRL/PEFT would receive, and no Secret value anywhere.
      ...
  ```
- Put the main function first in a file and its helpers below, in the order it calls them.

#### Size
- Reuse existing helpers before writing new ones.
- Prefer deleting code to adding configuration options.
- Keep everything as small as possible.

#### Constants
- Declare Constants in env files and not at the top of the file

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues on Maxi580/mlopsplatform, using the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the five default labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
