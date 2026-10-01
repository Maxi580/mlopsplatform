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

#### Size
- Reuse existing helpers before writing new ones.
- Prefer deleting code to adding configuration options.
- Keep everything as small as possible.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues on Maxi580/mlopsplatform, using the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the five default labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
