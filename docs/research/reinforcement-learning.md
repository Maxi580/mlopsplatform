# Research: Reinforcement learning for LLMs on the platform

_Researched 2026-10-01. Primary sources only (official docs, source repos, specs, papers). Every factual claim is linked to its source. Anything labelled **Recommendation** or **Sketch** is our own synthesis and has no source behind it._

**Question.** How can the platform let users configure reinforcement learning (RL) for LLMs, meaning when and how the model is rewarded, and optionally an environment, inside a **Pipeline Request**?

**Vocabulary** follows `CONTEXT.md`: Pipeline Request, Pipeline, Stage (`distill` / `finetune` / `evaluate` / `serve`), Base Model, Teacher / Student, Job, Run (= MLflow run only). "Policy" means the model being trained by RL. In this document it is a Base Model or the output of an earlier `finetune` Stage.

---

## 1. TL;DR

- "RL for LLMs" covers four families, and each asks something different of the user:
  1. **Offline preference optimisation** (DPO, ORPO, KTO). The user supplies a **dataset** of preferences or binary labels. There is no reward function and no generation during training. Operationally this is a `finetune` variant.
  2. **RLHF with a learned reward model** (PPO). The user supplies a **reward model**, which usually has to be trained first on preference pairs.
  3. **RL with verifiable rewards** (GRPO / RLVR). The user supplies **prompts plus a reward function or grader**. This is where "defining when and how the model is rewarded" matters most.
  4. **Online multi-turn / agentic RL**. The user supplies an **environment**: tools, state, termination and a reward.
- Frameworks are converging on the same core shape. A reward is a function `(prompt, completion, dataset columns…) -> float`, and several of them are combined with weights. An environment is an object with `reset()` plus tools/`step()`, and the reward is read from its state at the end. Examples: TRL `reward_funcs` + `reward_weights` + `environment_factory`; veRL `custom_reward_function` + agent loop; OpenRLHF `reward_func` + `AgentInstanceBase`; verifiers `@vf.reward` on a Task.
- OpenAI's RFT **grader spec** is the most complete *declarative* reward design: `string_check`, `text_similarity`, `score_model`, sandboxed `python`, and `multi` with an arithmetic formula. It is a strong template for a Pipeline Request `reward:` block.
- **User-supplied code** is the main risk. It has to run sandboxed: gVisor/Kata through `RuntimeClass`, no network, resource limits. A remote sandbox service such as Sandbox Fusion, or a Kubernetes sandbox orchestrator such as `agent-sandbox`, are the known patterns.
- Online RL is **two workloads in one Job**: generation (vLLM) and training. Rewards sometimes add a third (judge or reward-model server, code sandbox). This decides GPU topology: colocated on the same GPUs, or split across GPUs/pods.

---

## 2. Landscape of LLM post-training RL methods

| Family | Representative methods | What the user must supply | Online generation during training? | Source |
|---|---|---|---|---|
| RLHF (reward model + PPO) | InstructGPT-style PPO | SFT model, **reward model** (trained on human rankings), prompts | Yes | [InstructGPT](https://arxiv.org/abs/2203.02155), [TRL PPO](https://huggingface.co/docs/trl/main/en/ppo_trainer) |
| Offline preference optimisation | DPO (+ IPO, SimPO, … as loss variants), ORPO | **Preference pairs** `prompt / chosen / rejected` | No | [DPO paper](https://arxiv.org/abs/2305.18290), [TRL DPO](https://huggingface.co/docs/trl/main/en/dpo_trainer), [ORPO](https://arxiv.org/abs/2403.07691) |
| Binary-feedback alignment | KTO | **Binary desirable / undesirable** label per output | No | [KTO](https://arxiv.org/abs/2402.01306) |
| RL with verifiable rewards | GRPO, RLVR, DAPO / Dr.GRPO / GSPO-style variants | **Prompts + reward function(s) / grader(s)**, often a ground-truth column | Yes (N samples per prompt) | [DeepSeekMath](https://arxiv.org/abs/2402.03300), [Tulu 3](https://arxiv.org/abs/2411.15124), [DeepSeek-R1](https://arxiv.org/html/2501.12948v1), [TRL GRPO](https://huggingface.co/docs/trl/main/en/grpo_trainer) |
| Multi-turn / agentic RL | GRPO with environments or tools, Search-R1 | **Environment** (tools, state, termination) + reward | Yes, multi-turn | [TRL OpenEnv](https://huggingface.co/docs/trl/main/en/openenv), [Search-R1](https://arxiv.org/abs/2503.09516) |

### 2.1 RLHF / PPO (needs a reward model)

- InstructGPT is the canonical recipe: SFT on demonstrations, then a reward model trained on human rankings of outputs, then PPO against that reward model ([arXiv 2203.02155](https://arxiv.org/abs/2203.02155)).
- In TRL, `PPOTrainer` now lives under `trl.experimental.ppo`. It takes a policy `model`, an optional `ref_model` for the KL penalty, a `reward_model` **and** a `value_model`, i.e. up to four models in memory. Config includes `kl_coef`, `cliprange`, `vf_coef`, `gamma`, `lam` and `missing_eos_penalty` ([TRL PPO docs](https://huggingface.co/docs/trl/main/en/ppo_trainer)).
- The reward model is usually trained separately. TRL's `RewardTrainer` loads an `AutoModelForSequenceClassification` with `num_labels` fixed to 1 and trains it with a Bradley-Terry loss on `chosen` / `rejected` pairs ([TRL Reward docs](https://huggingface.co/docs/trl/main/en/reward_trainer)).
- **Implication.** RLHF is really two Stages: reward-model training, then PPO. It also has the heaviest GPU footprint.

### 2.2 DPO / ORPO / KTO (preference data, no online reward)

- DPO solves the RLHF objective with "a simple classification loss", and is "eliminating the need for sampling from the LM during fine-tuning" ([DPO paper via TRL docs](https://huggingface.co/docs/trl/main/en/dpo_trainer)).
- TRL `DPOTrainer` expects `prompt` / `chosen` / `rejected` in standard or conversational format. It uses a reference model, which defaults to the initial policy, and offers `beta` plus a list of `loss_type`s (`sigmoid`, `ipo`, `hinge`, `sigmoid_norm` (SimPO), `apo_zero`, …) that can be combined with `loss_weights` ([TRL DPO](https://huggingface.co/docs/trl/main/en/dpo_trainer)).
- ORPO folds SFT and preference alignment into one phase "without reference model" ([arXiv 2403.07691](https://arxiv.org/abs/2403.07691)).
- KTO needs only "a binary signal of whether an output is desirable", not pairs ([arXiv 2402.01306](https://arxiv.org/abs/2402.01306)).
- Axolotl exposes all of these behind one `rl:` key (`dpo`, `ipo`, `orpo`, `kto`, `simpo`, `grpo`, …), with preference dataset types such as `chatml.intel` and `chat_template.default` ([Axolotl RLHF docs](https://docs.axolotl.ai/docs/rlhf.html)).
- **Implication.** These methods have no reward function and no environment. They are a `finetune` Stage with a different loss and a different dataset schema, and they fit the existing Stage model cleanly.

### 2.3 GRPO / RLVR (verifiable reward functions)

- GRPO "obviates the need for additional value function approximation as in PPO, and instead uses the average reward of multiple sampled outputs, produced in response to the same question, as the baseline" ([DeepSeekMath §4.1.1](https://arxiv.org/html/2402.03300v3)). The paper distinguishes outcome supervision (one reward at the end) from process supervision (a reward per reasoning step) (same source).
- Tulu 3 introduced the name **RLVR** (RL with Verifiable Rewards) as one of its three post-training algorithms, alongside SFT and DPO ([arXiv 2411.15124](https://arxiv.org/abs/2411.15124)).
- DeepSeek-R1-Zero used only rule-based **accuracy** and **format** rewards. It explicitly avoided neural reward models because they "may suffer from reward hacking in the large-scale reinforcement learning process" and complicate the pipeline ([DeepSeek-R1 §2.2.2](https://arxiv.org/html/2501.12948v1)).
- OpenAI's RFT works the same way: it "samples several responses per prompt, scores them with the grader, and applies policy-gradient updates". It notes that the model must already have *some* success on the task, because RFT "cannot bootstrap" from 0% ([OpenAI RFT guide](https://developers.openai.com/api/docs/guides/reinforcement-fine-tuning)).

### 2.4 Online multi-turn / agentic RL (environments)

- TRL distinguishes **tools**, where each call is stateless, from **environments**, which "maintain state across turns, enabling genuine multi-turn interaction where the agent's actions shape future observations" ([TRL OpenEnv](https://huggingface.co/docs/trl/main/en/openenv)).
- Search-R1 trains interleaved search-tool calls with RL. It masks retrieved tokens out of the loss and uses an outcome-based reward ([arXiv 2503.09516](https://arxiv.org/abs/2503.09516)).

---

## 3. How open-source frameworks let users specify rewards and environments

### 3.1 Hugging Face TRL

**Rewards (`GRPOTrainer`)** — [TRL GRPO docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- `reward_funcs` takes a single item or a list. Each item is a model-id string (a reward model), a `PreTrainedModel`, or a Python callable.
- Callables receive keyword arguments: `prompts`, `completions`, `completion_ids`, `trainer_state`, **every dataset column** (so a `ground_truth` column arrives as a keyword argument), and `environments` when environments are used.
- A callable returns `list[float]`, with one value per completion. `None` means "not applicable to this sample".
- `reward_weights` sets weights per function. `multi_objective_aggregation` chooses `sum_then_normalize` or `normalize_then_sum`.
- Async reward functions run concurrently via `asyncio.gather`.
- Key knobs: `num_generations` (group size, default 8), `beta` (KL, default 0.0), `scale_rewards`, and `loss_type` (`dapo` default, `dr_grpo`, `sapo`, `cispo`, …).

**Environments and tools** — [TRL GRPO](https://huggingface.co/docs/trl/main/en/grpo_trainer), [TRL OpenEnv](https://huggingface.co/docs/trl/main/en/openenv)
- `tools=[fn, …]` takes plain typed functions with docstrings, and the schema is derived from them.
- `environment_factory=Cls`, or a `dict` of them routed by an `environment` dataset column. The trainer creates one instance per generation and calls `reset(**dataset_columns)`, which may return the first observation. **Every public method becomes a tool.** An optional `get_reward()` lets the environment own the reward.
- TRL runs the multi-turn tool-call loop itself, and tool exceptions are fed back to the model as tool responses.
- `max_completion_length` caps the *whole* multi-turn episode.
- `rollout_func` is the escape hatch for a fully custom generation loop.
- OpenEnv environments run as separate servers (HF Space, Docker container or local uvicorn) reached over WebSocket. By default a server allows only one concurrent session, so it must be configured with `max_concurrent_envs` ≥ `generation_batch_size`.
- An experimental **loop-owning** mode (`AsyncGRPOTrainer` + `HarnessRolloutWorker`) trains a black-box agent harness such as opencode. A proxy records token ids and logprobs, a `verify()` method scores the final workspace, and the user supplies `rollout_reward_fn`, `train_turn_fn` and `agent_turn_fn`.

**vLLM** — [TRL GRPO](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- `use_vllm` with `vllm_mode="colocate"` (default) puts generation on the training GPUs. `vllm_mode="server"` uses a separate vLLM server and syncs weights over NCCL.
- `vllm_importance_sampling_correction` (default on) corrects the mismatch between training and inference logprobs.

### 3.2 OpenRLHF — [docs](https://openrlhf.readthedocs.io/en/latest/), [agent training](https://openrlhf.readthedocs.io/en/latest/agent_training.html)
- Rewards come from a pretrained reward model (`--reward_pretrain`) or a remote URL / Python file (`--remote_rm_url`, written `--reward.remote_url` in newer docs).
- The Python form is `reward_func(queries, prompts, labels)`. It returns a dict with `rewards` (used for the advantage), `scores` (used for dynamic filtering) and `extra_logs`.
- Multi-turn agents subclass `AgentInstanceBase` with async `reset()` and `step()`. `step()` returns `rewards`, `scores`, `environment_feedback` and `done`. The agent is enabled via `--train.agent_func_path`.
- Ray + vLLM architecture; algorithms PPO, REINFORCE++, GRPO and RLOO all work with any reward source.

### 3.3 veRL
- **Custom reward:** `custom_reward_function.path` + `.name`, with signature `compute_score(data_source, solution_str, ground_truth, extra_info=None)` ([veRL reward function](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)).
- **Reward Loop** ([docs](https://verl.readthedocs.io/en/latest/advance/reward_loop.html)) distributes reward computation over `RewardWorker`s (`config.reward.num_workers`). It supports rule-based rewards, discriminative reward models, generative reward models (LLM judges) and hybrids. Built-in reward managers are `naive`, `dapo` (overlong penalty), `limit` (concurrency cap for rate-limited APIs) and `remote` (separate process for CPU-heavy verification). Rewards can be streamed while rollout is still running.
- **Multi-turn / tools** ([multiturn docs](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)): `multi_turn: True`. Stateful tools subclass `BaseTool` with `create` / `execute` / `calc_reward` / `release`, configured via a tools YAML. Stateless tools use the `@function_tool` decorator. The docs advise keeping `BaseTool` "for per-trajectory state that must be torn down between rollouts (sandbox VMs, scratch directories…)" ([source](https://github.com/volcengine/verl/blob/main/docs/sglang_multiturn/multiturn.rst)).
- **Agent loop** ([docs](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)): the user implements `AgentLoopBase.run()`, which returns `prompt_ids`, `response_ids` and a `response_mask` that separates model tokens from tool tokens. Rollout servers are vLLM or SGLang.
- **Sandbox Fusion** ([docs](https://verl.readthedocs.io/en/latest/examples/sandbox_fusion_example.html)) is a remote code-execution service used to score generated code. It is configured with `sandbox_fusion.url`, `max_concurrent` and `memory_limit_mb`.

### 3.4 Axolotl / Unsloth
- **Axolotl** wraps TRL. In YAML, `rl: grpo` plus `trl.reward_funcs: ["rewards.my_fn"]` references functions as `'{file_name}.{fn_name}'`, alongside `reward_weights`, `num_generations` and `use_vllm` with a separate vLLM server ([Axolotl RLHF](https://docs.axolotl.ai/docs/rlhf.html)). This is the closest existing analogue to a YAML Pipeline Request, and it simply points at user Python.
- **Unsloth** exposes GRPO via `GRPOConfig` `loss_type` options (`grpo`, `dr_grpo`, `gspo`). It separates a *verifier* ("does not assign a numerical score") from a *reward function* (which "converts verification results… into a numerical score"), and shares GPU memory with vLLM ([Unsloth RL guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide.md)).

### 3.5 OpenAI RFT graders (reference design for declarative rewards)
Source: [OpenAI Graders guide](https://developers.openai.com/api/docs/guides/graders), [RFT guide](https://developers.openai.com/api/docs/guides/reinforcement-fine-tuning).

| Grader `type` | Fields | Notes |
|---|---|---|
| `string_check` | `input`, `reference`, `operation` | `eq`, `neq`, `like` (contains), `ilike` (contains, case-insensitive) |
| `text_similarity` | `input`, `reference`, `evaluation_metric`, `pass_threshold` | `fuzzy_match`, `bleu`, `gleu`, `meteor`, `cosine`, `rouge_1`…`rouge_l` |
| `score_model` | `model`, `input` (message array), `range`, `pass_threshold`, `sampling_params` | LLM-as-judge that returns a number |
| `python` | `source`, `image_tag` | Sandbox: 2 min, 2 GB RAM, 1 GB disk, 2 CPU, **no network**, 256 kB code, fixed package allowlist (numpy, sympy, jsonschema, rapidfuzz, …) |
| `multi` | `graders` (map), `calculate_output` | Formula over sub-grader results using `+ - * / ^`, `min`, `max`, `abs`, `exp`, `log`, … |

- Templating uses `{{ item.<field> }}` for dataset columns and `{{ sample.output_text }}` / `output_json` / `output_tools` for model output.
- RFT jobs set `method.type: "reinforcement"`. The dataset is JSONL with `messages` plus arbitrary extra fields that graders reference. A `response_format` JSON schema is optional.
- Design guidance in the docs: avoid rewards a lucky guess can earn, require tasks where experts agree, and check that baseline scores are not already at the minimum or maximum.

### 3.6 verifiers / Prime Intellect environments
Source: [verifiers repo](https://github.com/PrimeIntellect-ai/verifiers), [v1 overview](https://github.com/PrimeIntellect-ai/verifiers/blob/main/docs/v1/overview.md), [tasksets](https://github.com/PrimeIntellect-ai/verifiers/blob/main/docs/v1/tasksets.md).
- The v1 API (the legacy v0 `vf.SingleTurnEnv`-style stack "has been removed") splits an environment into these parts:
  - **Taskset**: loader plus `TasksetConfig`.
  - **Task**: holds `TaskData` (prompt, references, resource needs) and behaviour (tools, stop conditions, rewards). Rewards are `@vf.reward async def …(self, trace) -> float`.
  - **Toolset**: tools exposed to harnesses as **MCP servers**.
  - **Harness**: the agent program, e.g. Claude Code, Codex or mini-swe-agent.
  - **Trace**: the message graph plus rewards plus per-call token/logprob records.
- `vf.Judge` provides LLM-judge rewards for semantic tasks.
- Environments are pip-installable packages scaffolded with `vf-init` and shared through the **Environments Hub**. Users configure them via TOML or CLI overrides (`--env.taskset.num-tasks`).
- prime-rl trains on them with separate **trainer / orchestrator / vLLM inference** components, runs fully asynchronous off-policy, and documents Slurm and Kubernetes multi-node support ([prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)).

### 3.7 Gymnasium-style interfaces applied to LLMs
- Gymnasium: `reset(seed, options) -> (observation, info)`, `step(action) -> (observation, reward, terminated, truncated, info)`. `terminated` is the MDP's natural end; `truncated` is an external cut-off such as a time limit ([Gymnasium Env API](https://gymnasium.farama.org/api/env/)).
- LLM frameworks keep `reset` but **replace the generic `step(action)` with named tools**. TRL warns against a generic `step(action)` "since the model needs meaningful tool names and argument descriptions to learn tool calling" ([TRL OpenEnv](https://huggingface.co/docs/trl/main/en/openenv)). OpenRLHF's `step()` keeps the Gym shape (`environment_feedback`, `rewards`, `done`) ([OpenRLHF](https://openrlhf.readthedocs.io/en/latest/agent_training.html)).
- The LLM analogue of `truncated` is the token or turn budget, e.g. TRL `max_completion_length` across the episode.

---

## 4. Reward specification options and tradeoffs

| Option | Examples | Pros | Cons / risks |
|---|---|---|---|
| **Declarative graders** (exact match, contains, regex, JSON-schema validity, similarity metric) | OpenAI `string_check` / `text_similarity`; DeepSeek-R1 accuracy + format rewards | Serialisable in YAML; validated by the API; no user code; cheap CPU work; reproducible | Only covers checkable tasks; similarity metrics can be gamed; regex/format rewards invite format-only hacking |
| **Composition** of graders | OpenAI `multi.calculate_output`; TRL `reward_weights`; verifiers multiple `@vf.reward` | Shaping (correctness + format + length penalty) stays declarative | Weight tuning matters; TRL notes binary rewards often train more cleanly than partial credit ([TRL OpenEnv](https://huggingface.co/docs/trl/main/en/openenv)) |
| **LLM-as-judge** (generative reward) | OpenAI `score_model`; verifiers `vf.Judge`; veRL GenRM | Handles open-ended quality; rubric written in natural language | Known position, verbosity and self-enhancement biases ([Zheng et al.](https://arxiv.org/abs/2306.05685)); costs a GPU or API per sample; rate limits (veRL ships a `limit` manager for this); the judge can be gamed |
| **Trained reward model** (discriminative) | TRL `RewardTrainer` → PPO / GRPO `reward_funcs="<model id>"`; OpenRLHF `--reward_pretrain`; veRL DisRM | Learns human preference at scale; one forward pass per sample | Needs a preference dataset and an extra training Stage; reward hacking at scale ([DeepSeek-R1](https://arxiv.org/html/2501.12948v1)); extra GPU memory |
| **User-supplied code** | TRL callables; veRL `custom_reward_function.path`; Axolotl `file.fn`; OpenRLHF `reward_func`; OpenAI `python` grader | Unlimited expressiveness (unit tests, sympy equivalence, simulators) | **Arbitrary code execution in the cluster.** Needs a sandbox, dependency management, timeouts, and handling of crashes or `None` scores |
| **Environment-owned reward** | TRL `get_reward()` / reading `env.reward`; verifiers Task rewards; OpenEnv `verify()` | Reward lives next to the state it judges; reusable for `evaluate` | Same code-execution concerns as above, plus long-lived stateful services |

### Sandboxing user code (security)
- **Reference limits.** OpenAI runs Python graders with no network, 2 min, 2 GB RAM, 1 GB disk, 2 CPU and a package allowlist ([Graders](https://developers.openai.com/api/docs/guides/graders)). That is a reasonable baseline for a platform contract.
- **Kubernetes mechanism.** `RuntimeClass` selects a hardened runtime per pod. The docs cite hardware-virtualised runtimes for "workloads requiring high security assurance", and `overhead` / `scheduling` fields support placement ([K8s RuntimeClass](https://kubernetes.io/docs/concepts/containers/runtime-class/)). gVisor plugs in as `runtimeClassName: gvisor` with the `runsc` handler ([gVisor](https://gvisor.dev/docs/user_guide/containerd/quick_start/)).
- **Orchestration.** `kubernetes-sigs/agent-sandbox` provides `Sandbox`, `SandboxTemplate`, `SandboxClaim` and `SandboxWarmPool` CRDs explicitly aimed at "reinforcement learning environments". It delegates isolation to gVisor or Kata via RuntimeClass ([agent-sandbox](https://github.com/kubernetes-sigs/agent-sandbox)).
- **Remote sandbox service.** veRL's Sandbox Fusion keeps code execution out of the training pods and adds concurrency and memory limits ([veRL Sandbox Fusion](https://verl.readthedocs.io/en/latest/examples/sandbox_fusion_example.html)).
- **Recommendation.** Two different things need sandboxing:
  - the *reward code* itself (if allowed);
  - code the *model* generates and that the reward executes (e.g. unit tests for code tasks).

  The second is needed **even if users write no code**, as soon as code-generation rewards are offered.
- In-process callables (the TRL default) run inside the trainer pod with GPU and credential access. Without isolation that is the least safe option.

---

## 5. Environment specification

### 5.1 What "environment" means for an LLM
1. **Single-turn: prompt + verifier.** The "environment" is just a dataset row (prompt, ground truth) plus a reward. This is the GRPO/RLVR case and needs no extra infrastructure (TRL `reward_funcs` with dataset columns).
2. **Multi-turn with stateless tools.** The model may call functions such as a calculator or search, and rewards come at the end (TRL `tools=`, veRL `@function_tool`).
3. **Multi-turn with stateful environment.** Each rollout has its own state: a game, a browser, a repo or sandbox VM, or a simulated user. Requirements: `reset`, tools/`step`, termination, and per-episode teardown (TRL `environment_factory`, veRL `BaseTool` `create` / `release`, OpenRLHF `AgentInstanceBase`, verifiers Task + Toolset).
4. **Black-box harness.** An existing agent program owns the loop. The trainer only proxies its LLM calls and scores the final state (TRL `AsyncGRPOTrainer` loop-owning mode, verifiers Harness).

### 5.2 Common interface elements across frameworks
- **Episode start:** `reset(**task_fields)` → initial observation (TRL, OpenRLHF, Gymnasium).
- **Actions:** named, typed tools with docstrings, exported as JSON schema or MCP (TRL, veRL, verifiers).
- **Termination:** model stops calling tools, a `done` flag, or a token/turn budget (TRL `max_completion_length`, OpenRLHF `done`).
- **Reward:** read from environment state at episode end, or computed by separate reward functions over the trajectory (TRL `environments` kwarg, verifiers `trace`, TRL `rollout_reward_fn` with tool-call counts and `timed_out`).
- **Loss masking:** tool or environment tokens are excluded from the policy loss (veRL `response_mask`, Search-R1 retrieved-token masking).

### 5.3 Infrastructure implications for Kubernetes Jobs
- **Generation engine.** Every online method needs fast sampling. TRL, veRL, OpenRLHF, Unsloth and prime-rl all use vLLM, and veRL also supports SGLang (sources above).
- **GPU topology choices:**
  - *Colocated.* vLLM shares the training GPUs (TRL `vllm_mode="colocate"`, Unsloth). One pod, simplest scheduling, but memory is contended (TRL suggests lowering `vllm_gpu_memory_utilization`).
  - *Split.* A dedicated vLLM server on separate GPUs, with weight sync over NCCL (TRL `vllm_mode="server"`, Axolotl). That means two pods or containers with GPU-to-GPU networking, and gang scheduling matters.
  - *Disaggregated async.* Trainer, orchestrator and inference run as separate services with off-policy updates (prime-rl, veRL fully-async, TRL `AsyncGRPOTrainer`). Highest throughput, most moving parts.
- **Reward servers.** A judge LLM or reward model can sit in the trainer, in a separate resource pool, or behind an API (veRL `reward_model.enable_resource_pool`, OpenRLHF `--remote_rm_url`). A judge pointing at an external API resembles the platform's external **Teacher** concept from `CONTEXT.md`.
- **Environment servers.** Stateful environments run as separate services with concurrency limits sized to the generation batch (OpenEnv `max_concurrent_envs`). One sandbox per rollout matches `agent-sandbox` warm pools.
- **PPO memory.** PPO holds policy, reference, reward and value models ([TRL PPO](https://huggingface.co/docs/trl/main/en/ppo_trainer)). GRPO drops the value model ([DeepSeekMath](https://arxiv.org/html/2402.03300v3)), and TRL's GRPO default `beta=0.0` means no reference model is loaded either ([TRL GRPO](https://huggingface.co/docs/trl/main/en/grpo_trainer)).
- **Tracking.** Per-reward-function metrics (`train/reward_func_0`, …) and sample completions should go into the Run. TRL recommends monitoring per-function rewards in multi-environment training ([TRL OpenEnv](https://huggingface.co/docs/trl/main/en/openenv)).

---

## 6. Relation to tool calling (brief)

- Tool calling in RL comes in two forms, with tools either stateless (`tools=`) or methods on a stateful environment ([TRL OpenEnv](https://huggingface.co/docs/trl/main/en/openenv)). The model must emit tool calls in its chat-template format, so tokenisation consistency matters (veRL warns about "inconsistent training and inference tokenization" ([source](https://github.com/volcengine/verl/blob/main/docs/sglang_multiturn/multiturn.rst))). The vLLM server needs tool-call parsing enabled for harness mode, e.g. `--enable-auto-tool-choice --tool-call-parser hermes` ([TRL OpenEnv](https://huggingface.co/docs/trl/main/en/openenv)).
- Offline methods also support tool calling: `DPOTrainer` and `RewardTrainer` accept `tool_calls` and `tool` messages plus a `tools` column ([TRL DPO](https://huggingface.co/docs/trl/main/en/dpo_trainer), [TRL Reward](https://huggingface.co/docs/trl/main/en/reward_trainer)).
- verifiers ships tools as MCP servers (Toolsets) ([tasksets](https://github.com/PrimeIntellect-ai/verifiers/blob/main/docs/v1/tasksets.md)). A platform tool spec based on MCP or JSON schema could therefore be reused by both `serve` and RL.
- Graders can score tool use directly: OpenAI exposes `sample.output_tools` ([Graders](https://developers.openai.com/api/docs/guides/graders)), and TRL's `rollout_reward_fn` sees `tool_calls_by_name` ([TRL OpenEnv](https://huggingface.co/docs/trl/main/en/openenv)).

---

## 7. Sketch: how rewards could appear in a Pipeline Request

_Sketch only. This is not a decision, and the field names are illustrative. It borrows OpenAI's grader taxonomy and TRL's weighting._

```yaml
stages:
  rl:                                # or finetune: { method: grpo, ... } — see open decisions
    method: grpo                     # grpo | dpo | orpo | kto | ppo
    policy: { base_model: Qwen/Qwen3-1.7B }
    dataset: { source: hf://org/math-prompts, prompt_field: question }
    rollout: { num_generations: 8, max_completion_tokens: 1024, engine: vllm, placement: colocate }
    reward:
      combine: "0.8 * correct + 0.2 * format"     # like OpenAI multi.calculate_output
      graders:
        correct: { type: string_check, operation: eq,
                   input: "{{ sample.final_answer }}", reference: "{{ item.answer }}" }
        format:  { type: regex, input: "{{ sample.output_text }}", pattern: "<think>.*</think>" }
        # judge:   { type: model_judge, model: <Base Model or external Teacher-like endpoint>, rubric: "..." }
        # custom:  { type: python, source_ref: <artifact>, sandbox: default }   # only if user code allowed
    environment: null                # or { ref: <registered environment>, max_turns: 8 }
```

---

## 8. Open decisions for the platform

1. **RL as its own Stage, or a `finetune` mode?**
   - *`finetune.method: dpo|orpo|kto`* fits offline methods naturally: same Job shape, different loss and dataset schema.
   - *A new `rl` Stage (or `align`)* fits online methods (GRPO / PPO / agentic). These need rollout engines, reward and environment services, and a different GPU topology.
   - Middle ground: offline preference methods become `finetune` modes and online RL becomes a new Stage. **Tradeoff:** one more Stage name in `CONTEXT.md` versus overloading `finetune` with options that only make sense when generation is involved.
2. **Which method families to support first?** DPO-family is cheapest (no generation, no reward). GRPO with declarative graders gives the most "define the reward" value per unit of infrastructure. PPO needs a reward-model training step and roughly 4 models in memory. Agentic RL needs environment servers and sandboxes. Phasing in that order is a natural option.
3. **How rewards appear in a Pipeline Request.**
   - (a) A closed set of declarative graders: validated by the API, safe, shows well in the Web UI.
   - (b) (a) plus LLM-judge, which reuses the Base Model / external-Teacher plumbing.
   - (c) (b) plus user code.

   Also undecided: whether to adopt OpenAI's grader schema and templating (`{{ item.* }}` / `{{ sample.* }}`) more or less verbatim so the format is familiar, or a TRL-style list with weights.
4. **Is user code allowed, and where does it run?**
   - Options:
     - None: graders only.
     - Python in a sandbox pod (gVisor/Kata RuntimeClass, no network, OpenAI-like limits).
     - A remote sandbox service (Sandbox Fusion-like).
     - Trusted in-process callables, which is fastest but unsafe.
   - Also undecided: how code is shipped. Inline `source` (like OpenAI's 256 kB limit), a reference to an artifact, or a container image (`image_tag`)? Each has different reproducibility and supply-chain implications.
5. **Environments: built-in catalogue, registered images, or arbitrary?**
   - A curated set (math, code-with-tests, JSON-schema tasks) is simplest.
   - Accepting user environments as container images with a fixed HTTP or MCP contract (OpenEnv-style `reset` + tools, or verifiers packages) gives extensibility, but adds a long-lived per-rollout service, concurrency sizing, and teardown.
   - Separately: whether "Environment" becomes a first-class domain term in `CONTEXT.md`.
6. **Rollout topology per Job.** Colocated vLLM (one pod, simpler, memory pressure) versus a split vLLM server (multi-pod, needs gang scheduling and an NCCL path) versus fully async. This affects Job templates, GPU quotas and failure handling.
7. **Reward-model training.** Should a "train reward model" capability exist (a `finetune` variant producing a sequence-classifier), so PPO/GRPO can reference a platform-produced reward model?
8. **Reuse between `evaluate` and RL.** The same graders and environments could score the `evaluate` Stage. OpenAI uses graders for both evals and RFT, and verifiers environments serve both eval and training. A shared grader spec avoids two reward languages.
9. **Observability contract.** Which per-grader metrics, sample completions and trajectories get logged to the Run, and how large trajectory artifacts are stored.
10. **Guardrails against unproductive Pipelines.** Should the platform run a quick baseline `evaluate` of the reward on the policy before an RL Job starts? OpenAI's guidance is that RFT cannot bootstrap from 0% success, and saturated rewards give no signal.

---

## Sources
- InstructGPT — https://arxiv.org/abs/2203.02155
- DPO — https://arxiv.org/abs/2305.18290
- ORPO — https://arxiv.org/abs/2403.07691
- KTO — https://arxiv.org/abs/2402.01306
- DeepSeekMath / GRPO — https://arxiv.org/abs/2402.03300, https://arxiv.org/html/2402.03300v3
- Tulu 3 / RLVR — https://arxiv.org/abs/2411.15124
- DeepSeek-R1 — https://arxiv.org/html/2501.12948v1
- Search-R1 — https://arxiv.org/abs/2503.09516
- LLM-as-a-Judge biases — https://arxiv.org/abs/2306.05685
- TRL GRPO — https://huggingface.co/docs/trl/main/en/grpo_trainer
- TRL OpenEnv — https://huggingface.co/docs/trl/main/en/openenv
- TRL DPO — https://huggingface.co/docs/trl/main/en/dpo_trainer
- TRL PPO — https://huggingface.co/docs/trl/main/en/ppo_trainer
- TRL Reward — https://huggingface.co/docs/trl/main/en/reward_trainer
- OpenRLHF — https://openrlhf.readthedocs.io/en/latest/, https://openrlhf.readthedocs.io/en/latest/agent_training.html
- veRL — https://verl.readthedocs.io/en/latest/preparation/reward_function.html, https://verl.readthedocs.io/en/latest/advance/reward_loop.html, https://verl.readthedocs.io/en/latest/advance/agent_loop.html, https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html, https://verl.readthedocs.io/en/latest/examples/sandbox_fusion_example.html
- Axolotl — https://docs.axolotl.ai/docs/rlhf.html
- Unsloth — https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide.md
- OpenAI Graders / RFT — https://developers.openai.com/api/docs/guides/graders, https://developers.openai.com/api/docs/guides/reinforcement-fine-tuning
- verifiers — https://github.com/PrimeIntellect-ai/verifiers, https://github.com/PrimeIntellect-ai/verifiers/blob/main/docs/v1/tasksets.md
- prime-rl — https://github.com/PrimeIntellect-ai/prime-rl
- Gymnasium — https://gymnasium.farama.org/api/env/
- Kubernetes RuntimeClass — https://kubernetes.io/docs/concepts/containers/runtime-class/
- gVisor — https://gvisor.dev/docs/user_guide/containerd/quick_start/
- agent-sandbox — https://github.com/kubernetes-sigs/agent-sandbox
