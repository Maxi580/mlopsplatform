# Tool / function calling across the platform lifecycle

Research notes, 2026-10-01. Question: how can the platform support tool (function) calling for open-source LLMs in every Stage (`distill`, `finetune`, `evaluate`, `serve`), and let users define their own tools?

Vocabulary follows `CONTEXT.md`: Pipeline Request, Stage, Base Model, Teacher, Student, Distillation Dataset. All claims are cited to primary sources (official docs, source repos, specs, model cards, papers).

---

## TL;DR

- **The model never runs a tool.** It emits a structured request, and something else runs it and feeds the result back. Hugging Face: "A model **cannot actually call the tool itself**. It requests a tool call, and it's your job to handle the call" ([HF chat_extras](https://huggingface.co/docs/transformers/main/en/chat_extras)). Meta's Llama 3.1 format says the same: "the model itself does not execute the calls; it provides structured output to facilitate calling by an executor" ([llama-models prompt_format](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/prompt_format.md)). OpenAI's flow puts execution in the application too (step 3, "Application executes the function") ([OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)).
- **Tool calling has three layers, and the platform has to keep them consistent from training to serving:**
  1. **Definition.** A JSON Schema per tool in the OpenAI shape `{type:"function", function:{name, description, parameters}}`. This shape is model-agnostic.
  2. **Rendering.** The model's **chat template** (Jinja) turns `tools`, `tool_calls` and `role:"tool"` messages into the model's own token format ([HF chat_extras](https://huggingface.co/docs/transformers/main/en/chat_extras), [HF "Tool Use, Unified"](https://huggingface.co/blog/unified-tool-use)). This part is model-specific.
  3. **Parsing.** The inference server's **tool parser** turns generated text back into OpenAI `tool_calls` (vLLM `--tool-call-parser`, SGLang `--tool-call-parser`, llama.cpp native handlers, transformers `response_template`). This part is also model-specific.
- **Serving.** vLLM and SGLang both expose OpenAI-compatible `tools` / `tool_choice` on `/v1/chat/completions`. Each needs a per-model-family parser flag and sometimes a template override. `required` and named `tool_choice` use grammar-constrained decoding. `auto` falls back to text parsing unless strict mode is on ([vLLM tool calling](https://docs.vllm.ai/en/latest/features/tool_calling.html), [SGLang tool parser](https://docs.sglang.io/advanced_features/tool_parser.html)).
- **Finetune.** TRL, Axolotl and Unsloth all accept conversational data with a `tools` column, assistant `tool_calls` and `tool` role messages. They render it through the chat template ([TRL dataset formats](https://huggingface.co/docs/trl/dataset_formats), [Axolotl conversation formats](https://docs.axolotl.ai/docs/dataset-formats/conversation.html)). A finetuned model keeps working with the serving parser only if training used the **same template** the parser expects.
- **Distill.** A Teacher can produce tool-calling trajectories, and APIGen, APIGen-MT and ToolACE are published pipelines that do this. But the current Distillation Dataset (prompt → response) cannot represent tool calls, multi-turn trajectories or tool results. It would need to become a message list plus `tools`.
- **Evaluate.** BFCL (single/multi-turn call accuracy, AST plus executable checks, with vLLM/SGLang backends) and τ-bench / τ²-bench (multi-turn agent tasks with a simulated user, scored on database end state with pass^k) are the reference harnesses ([BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html), [τ-bench paper](https://arxiv.org/abs/2406.12045)).
- **User-defined tools.** Declaring a tool means writing a JSON Schema, which the platform can own and validate. Executing a tool is a different job. By default it is client-side, and the platform only takes it on if it chooses to run sandboxed or MCP-based tools (needed for distillation with real tool results, RL, and agentic evaluation).

---

## 1. The core distinction: emitting vs executing

| Concern | Who does it | Evidence |
|---|---|---|
| Declare tools (name, description, JSON Schema params) | Request author (user / client app) | OpenAI tool fields `type`, `name`, `description`, `parameters`, `strict` ([OpenAI](https://developers.openai.com/api/docs/guides/function-calling)) |
| Render tools into the prompt | Chat template | `apply_chat_template(messages, tools=...)` ([HF](https://huggingface.co/docs/transformers/main/en/chat_extras)) |
| Decide to call, emit name + args | Model | ([HF](https://huggingface.co/docs/transformers/main/en/chat_extras)) |
| Extract structured `tool_calls` from text | Inference server parser | vLLM `--tool-call-parser` ([vLLM](https://docs.vllm.ai/en/latest/features/tool_calling.html)) |
| Execute the tool | Application / executor (client side), or a server-side tool runtime | Llama 3.1: "Both methods require the executor—not the model" ([llama-models](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/prompt_format.md)) |
| Return result as `role:"tool"` message, loop | Application | OpenAI 5-step flow ([OpenAI](https://developers.openai.com/api/docs/guides/function-calling)); HF `messages.append({"role": "tool", ...})` ([HF](https://huggingface.co/docs/transformers/main/en/chat_extras)) |

Server-side execution exists in open-source servers, but only in narrow cases. vLLM can act as an MCP client for gpt-oss "built-in tools" on the **Responses API** only: `vllm serve ... --tool-server ip-1:port-1,ip-2:port-2`. It invokes no tool through chat completions ([vLLM GPT-OSS recipe](https://docs.vllm.ai/projects/recipes/en/stable/OpenAI/GPT-OSS.html)). Its demo Python tool needs Docker or `PYTHON_EXECUTION_BACKEND=dangerously_use_uv` "to dangerously allow execution of model generated code snippets" (same source). Support for non-Harmony models (Qwen3, Kimi K2, MiniMax M2) was added later through chat templates. The tracking issue was closed with generic MCP tools and streaming still listed as remaining work ([vLLM #30115](https://github.com/vllm-project/vllm/issues/30115)).

**Wire-format gotcha:** OpenAI's API sends `tool_calls[].function.arguments` as a **JSON string**. Transformers chat templates expect a **dict**. HF warns that the string form "may cause errors or strange model behavior if used in Transformers" ([HF chat_extras](https://huggingface.co/docs/transformers/main/en/chat_extras)). Any platform code that moves data between API responses (Teacher outputs, eval logs) and training data has to normalize this.

---

## 2. Serving Stage

### 2.1 vLLM

Flags ([vLLM tool calling](https://docs.vllm.ai/en/latest/features/tool_calling.html)):
- `--enable-auto-tool-choice`: required for `tool_choice="auto"`.
- `--tool-call-parser <name>`: model-family parser.
- `--chat-template <path>`: optional override "which handles `tool`-role messages". Some families need one, for example `examples/tool_chat_template_llama3.1_json.jinja`, `..._llama3.2_json.jinja` and `..._mistral_parallel.jinja`. Hermes-style models and Qwen2.5 use their default template.
- `--tool-parser-plugin <file>`: registers a user-written parser. You subclass `ToolParser`, implement `adjust_request`, `extract_tool_calls` and `extract_tool_calls_streaming`, then register with `ToolParserManager`.
- `--tool-strict-level {auto,function,parameter}`, the env var `VLLM_ENFORCE_STRICT_TOOL_CALLING` (default true), and `--exclude-tools-when-tool-choice-none`.

Parsers listed in the docs: `hermes`, `mistral`, `llama3_json`, `llama4_pythonic`, `pythonic`, `granite`, `granite4`, `granite-20b-fc`, `internlm`, `jamba`, `xlam`, `deepseek_v3`, `deepseek_v31`, `openai` (gpt-oss), `kimi_k2`, `hunyuan_a13b`, `cohere_command3/4`, `longcat`, `glm45`, `glm47`, `functiongemma`, `qwen3_xml`, `mimo`, `olmo3`, `gigachat3`, `apertus` ([vLLM](https://docs.vllm.ai/en/latest/features/tool_calling.html)). Qwen recommends `--enable-auto-tool-choice --tool-call-parser hermes` for Qwen3 because it uses Hermes-style tool use ([Qwen function calling](https://qwen.readthedocs.io/en/latest/framework/function_call.html)).

How `tool_choice` is enforced in vLLM:

| `tool_choice` | Mechanism |
|---|---|
| `"none"` | No tool calls. Tools can be dropped from the prompt with `--exclude-tools-when-tool-choice-none` |
| `"auto"` | Text parsing: "arguments may occasionally be malformed or violate the function's parameter schema". Structural tags are added if any tool sets `strict: true` or the strict level is raised |
| `"required"` / named function | Structured outputs (grammar). The first use compiles an FSM, which costs "several seconds of latency (or more)" before it is cached |

Source: [vLLM tool calling](https://docs.vllm.ai/en/latest/features/tool_calling.html). `--tool-strict-level function` constrains the call envelope (markup plus function name) for every request. `parameter` additionally pins argument schemas "as if every tool had `strict: true`" (same source). Grammar backends (xgrammar, guidance, …) are chosen with `--structured-outputs-config.backend`, default `auto` ([vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html)).

Known model caveats in vLLM: Llama 3 does not support parallel tool calls (Llama 4 does), Mistral 7B "struggles to generate parallel tool calls correctly", and with the pythonic parser the model must not mix text and tool calls in one generation ([vLLM](https://docs.vllm.ai/en/latest/features/tool_calling.html)).

For offline/batch use (relevant to an in-cluster Teacher), `LLM.chat(..., tools=...)` takes the same tool schemas. The `chat_with_tools.py` example parses the output, executes the tool, appends the result and calls `chat` again ([vLLM chat_with_tools example](https://docs.vllm.ai/en/latest/examples/offline_inference/chat_with_tools.html)).

### 2.2 SGLang

`--tool-call-parser <name>`, optionally with `--chat-template`. Parsers: `llama3`, `llama4`, `pythonic` (recommended with `tool_chat_template_llama4_pythonic.jinja`), `mistral`, `qwen`, `qwen3_coder`, `deepseekv3` / `deepseekv31` (each with a recommended template), `deepseekv32`, `glm`, `gpt-oss`, `kimi_k2`, `step3`, `apertus2509` ([SGLang tool parser](https://docs.sglang.io/advanced_features/tool_parser.html)). `tool_choice="required"` and named functions are enforced with an EBNF grammar and need the Xgrammar backend. A new format is added by writing a detector class that inherits `BaseFormatDetector` (same source).

### 2.3 Others (for completeness)

- **HF TGI**: supports `tools` and `tool_choice` (`auto`, `none`, `required`, named) on `/v1/chat/completions`, enforced through outlines grammars ([TGI guidance](https://huggingface.co/docs/text-generation-inference/basic_tutorials/using_guidance)). However, "text-generation-inference is now in maintenance mode", and HF recommends vLLM, SGLang, llama.cpp or MLX instead ([TGI README](https://github.com/huggingface/text-generation-inference)).
- **llama.cpp server**: `--jinja` enables OpenAI-style tool calling. It has native handlers for Llama 3.x, Functionary, Hermes, Qwen 2.5, Mistral Nemo and others, plus a "Generic" fallback that "may consume more tokens". `--chat-template-file` overrides the template, `parallel_tool_calls: true` is supported, and "extreme KV quantizations (e.g. `-ctk q4_0`)... can substantially degrade the model's tool calling performance" ([llama.cpp function-calling.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)).
- **Ollama**: `tools` on `/api/chat`, and the model returns `tool_calls`. It supports parallel calls, multi-turn loops and streaming ([Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling)).

### 2.4 What the serve Stage would need in a Pipeline Request

Derived from the sources above:
- `tool_parser`: a vLLM/SGLang parser name. It is chosen by model family, so the platform should infer it from the Base Model where possible and allow an override.
- `chat_template`: optional override (path or inline Jinja). It must match the parser.
- `enable_auto_tool_choice`: bool.
- `tool_strict_level` (vLLM) or the equivalent grammar-backend setting.
- Optionally `tool_parser_plugin` (custom Python parser). This is user code inside the serving container, so it is a security and supply-chain decision.
- Optionally a `reasoning_parser`. vLLM documents reasoning parsers alongside tool parsers for some models ([vLLM](https://docs.vllm.ai/en/latest/features/tool_calling.html)).
- Tools themselves are **per request** in the OpenAI API. They are not part of server config, so the serve Stage does not need to know the user's tools unless the platform also executes them (section 6).

---

## 3. Finetune Stage

### 3.1 Data format

The de facto standard is a conversational record plus a `tools` column:

```python
{"messages": [
   {"role": "user", "content": "Turn on the living room lights."},
   {"role": "assistant", "tool_calls": [{"type": "function", "function": {
       "name": "control_light", "arguments": {"room": "living room", "state": "on"}}}]},
   {"role": "tool", "name": "control_light", "content": "The lights in the living room are now on."},
   {"role": "assistant", "content": "Done!"}],
 "tools": [ <JSON schema>, ... ]}
```

([TRL dataset formats, "Tool Calling"](https://huggingface.co/docs/trl/dataset_formats)). TRL: "it is important that your dataset includes an additional column named `tools`... usually used by the chat template to construct the system prompt". Because arguments are arbitrary JSON, use the `datasets` `Json()` feature type. On `datasets < 4.7.0`, store `tools` as a `json.dumps` string instead (same source).

### 3.2 Framework support

- **TRL SFTTrainer**: "fully supports fine-tuning models with _tool calling_ capabilities". Each example holds `tool_calls`, `tool` role messages and the `tools` column ([TRL SFT](https://huggingface.co/docs/trl/sft_trainer)). `assistant_only_loss=True` requires the chat template to contain `{% generation %}` markers. TRL patches the template for known families such as Qwen3 (same source). `chat_template_path` sets a template, and with a raw Jinja file "you must ensure that any special tokens referenced in the template are added to the tokenizer and that the model's embedding layer is resized" (same source).
- **TRL GRPOTrainer** (RL with real tool execution): the `tools=[callables]` and `environment_factory` arguments run tools during rollouts. "Ensure that the model's chat template supports tool use and that it has been fine-tuned for tool calling". The loop requires a *prefix-preserving* template, which TRL swaps in for known families ([TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer)). This is the one place in finetuning where the **training Job itself executes tools**.
- **Axolotl**: the `chat_template` dataset type with `field_tools` (default `tools`). Messages may carry `tool_calls` and `role: tool` with `tool_call_id`. Axolotl advises storing `arguments` as JSON strings when types conflict across rows, to avoid dataset casting errors, and says to check that your `chat_template` supports `tools` and which role it expects for tool answers (Llama 4 accepts `tool` or `ipython`) ([Axolotl conversation formats](https://docs.axolotl.ai/docs/dataset-formats/conversation.html)).
- **Unsloth**: integrates with TRL ([TRL SFT, "Train with Unsloth"](https://huggingface.co/docs/trl/sft_trainer)). Its tool-calling guide covers inference only (llama-server `--jinja`, client-side execution) ([Unsloth tool-calling guide](https://unsloth.ai/docs/basics/tool-calling-guide-for-local-llms)). Unsloth names RL as the route to "excel at a specific behavior (e.g., tool-calling)" ([Unsloth fine-tuning guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)).

### 3.3 Public function-calling datasets

| Dataset | Size / format | Generation | License |
|---|---|---|---|
| [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | 60k; `query`, `tools`, `answers` as JSON strings | APIGen; 3,673 executable APIs, three checks (format, execution, semantic); >95% correct on a 600-sample human check | CC-BY-4.0, gated |
| [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | ~113k; system prompt with function JSON, `<functioncall>` tags, simulated function responses | Synthetic | Apache-2.0 |
| [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | 11.3k dialogs, 26,507 APIs | Multi-agent synthesis with rule- and model-based verification ([paper](https://arxiv.org/abs/2409.00920)) | Apache-2.0 |
| APIGen-MT ([paper](https://arxiv.org/abs/2504.03601)) | 5k multi-turn trajectories | Verified blueprints → simulated human-agent interplay; trains the xLAM-2-fc-r models | see HF release |

Each dataset uses its **own** format (xLAM: query/tools/answers, Glaive: `<functioncall>` text). They have to be converted to the `messages + tools` shape before the target model's chat template is applied.

### 3.4 Does finetuning break native tool calling?

There are two separate risks.

1. **Format mismatch, which is deterministic and avoidable.** The serving parser looks for exact markup (Hermes `<tool_call>{...}</tool_call>`, Llama `<|python_tag|>` plus `<|eom_id|>`, Qwen3 `<function=...><parameter=...>`, …) ([HF response parsing](https://huggingface.co/docs/transformers/main/en/chat_response_parsing), [llama-models](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/prompt_format.md)). This leads to three cases:
   - If the finetune applies a different chat template from the one the Base Model was tool-trained with (for example `chat_template_path` set to another family, or a template with no `tools` support), the tuned model learns a different markup, and the stock parser for that family no longer matches.
   - If the training data contains tool calls rendered as plain text in `content` instead of `tool_calls`, the model learns the wrong format.
   - If tool data is absent altogether, the format is not reinforced.

   Rule: train with the Base Model's own tool-capable template, and serve with the parser that matches it.
2. **Capability forgetting, which is empirical.** Finetuning in general causes catastrophic forgetting of prior abilities ([Luo et al., empirical study](https://arxiv.org/abs/2308.08747)). In the opposite direction, Alopex reports that tuning on function-call data degrades general benchmarks and uses "a data mixing strategy... to mitigate catastrophic forgetting" ([Alopex](https://arxiv.org/abs/2411.05209)). I found no primary source that measures how much non-tool SFT degrades BFCL scores for a given family. Treat it as a risk to **measure** with an evaluate Stage (section 5), not as a known quantity. Qwen's docs say outright that "It is not guaranteed that the model generation will always follow the protocol even with proper prompting or templates" ([Qwen](https://qwen.readthedocs.io/en/latest/framework/function_call.html)).

Transformers can store a `response_template` (the inverse of the chat template) in `tokenizer_config.json`. `parse_response()` then extracts `tool_calls` per model and types arguments against the tool JSON Schema ([HF response parsing](https://huggingface.co/docs/transformers/main/en/chat_response_parsing)). This gives the platform one artifact, saved with the tokenizer, that describes the emitted format, which is useful for eval-time parsing outside vLLM/SGLang.

---

## 4. Distill Stage

### 4.1 Can a Teacher produce tool-calling data?

Yes. Several published pipelines do exactly this:
- **APIGen**: LLM-generated calls verified by format checking, **actual function execution** and semantic checks. Models trained on its 60k samples beat several GPT-4 versions on BFCL at 7B ([APIGen](https://arxiv.org/abs/2406.18518)).
- **APIGen-MT**: multi-turn trajectories from "simulated human-agent interplay" over verified blueprints ([APIGen-MT](https://arxiv.org/abs/2504.03601)).
- **ToolACE**: multi-agent (user/assistant/tool) dialog generation with rule- and model-based verification ([ToolACE](https://arxiv.org/abs/2409.00920)).
- **Agent Distillation** transfers "full task-solving behavior from LLM-based agents into sLMs with retrieval and code tools". It improves teacher trajectories with a "first-thought prefix", and 0.5B–3B students are competitive with larger CoT-distilled models ([Agent Distillation](https://arxiv.org/abs/2505.17612)).

A Teacher reached through an external API returns OpenAI-shaped `tool_calls` (arguments as a JSON string). An in-cluster Teacher on vLLM returns the same shape when served with the right parser, or through `LLM.chat(tools=...)` offline ([vLLM](https://docs.vllm.ai/en/latest/features/tool_calling.html)).

### 4.2 What this means for the Distillation Dataset

The current definition, "prompt → response pairs… Response-level only" (`CONTEXT.md`), has three gaps for tool use:
- **Single-call (one step)**: a response is a `tool_calls` list, not text. The record also needs the `tools` that were offered, because the right call depends on them. This fits if "response" becomes an assistant *message* (content or tool_calls) and the prompt carries `tools`. TRL's conversational prompt-completion format already allows that ([TRL dataset formats](https://huggingface.co/docs/trl/dataset_formats)).
- **Multi-step trajectories**: after a call, *someone* must produce the tool result before the Teacher continues. There are three options:
  - (a) **real execution**, as APIGen does, which needs tool runtimes in the distill Job;
  - (b) **simulated results** from an LLM, as Glaive and ToolACE's tool agent do;
  - (c) **recorded results** supplied by the user.

  The record then becomes a full `messages` list with `tool` role messages, which is the SFT format in section 3.1.
- **Verification**: all three published pipelines filter Teacher output (schema validity, execution success, semantic check). Unfiltered Teacher calls would teach malformed calls.

Loss masking also matters. With `assistant_only_loss`, tool *results* are not trained on, only the model's calls and answers ([TRL SFT](https://huggingface.co/docs/trl/sft_trainer)).

---

## 5. Evaluate Stage

| Harness | What it measures | How it runs against our models |
|---|---|---|
| **BFCL** (V4) ([leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html), [repo](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)) | Simple / multiple / parallel / multi-turn calls, live data, relevance and irrelevance detection, agentic (web search, memory), format sensitivity. Scoring by AST match plus executable checks. Separates native "FC" mode from "prompt" mode | `bfcl generate --model M --test-category C --backend {vllm,sglang}`, with `--local-model-path` and `--enable-lora` for tuned models, or `--skip-server-setup` with `LOCAL_SERVER_ENDPOINT`/`PORT` to point at an existing server. Then `bfcl evaluate`. A new model needs a handler class in `bfcl_eval/model_handler/local_inference/` and an entry in `model_config.py` |
| **τ-bench / τ²-bench** ([paper](https://arxiv.org/abs/2406.12045), [τ² repo](https://github.com/sierra-research/tau2-bench)) | Multi-turn agent with domain policy and tools, LLM-simulated user. Scored by comparing final DB state to the target. `pass^k` measures reliability over k trials | `tau2 run --domain airline --agent-llm ... --user-llm ...` through LiteLLM, so any OpenAI-compatible endpoint (our serve Stage) works. Needs a second LLM as the user simulator |

Implications:
- BFCL "FC" mode exercises the **serving parser and template**, not just the weights. So evaluate should run against the same serve configuration the user will deploy. This catches the format mismatch from section 3.4.
- τ-bench executes tools against its own mock environment. BFCL multi-turn also executes its own backend APIs. Both harnesses own their tools, so the platform does not need user-tool execution to run them.
- Evaluating on the **user's own tools** needs either a user-provided labeled set (expected call per prompt, AST-style match with no execution) or tool execution (section 6).

---

## 6. User-defined tools

### 6.1 Where tools are declared

- **OpenAI function schema** (what vLLM, SGLang, TGI, llama.cpp and Ollama accept): `type`, `name`, `description`, `parameters` (JSON Schema), and optionally `strict`. `strict: true` requires `additionalProperties: false` and every property in `required`, with optional fields expressed as `["type","null"]` ([OpenAI](https://developers.openai.com/api/docs/guides/function-calling)). vLLM honours per-tool `strict` ([vLLM](https://docs.vllm.ai/en/latest/features/tool_calling.html)).
- **Python callables**: transformers `get_json_schema` derives the schema from type hints and Google-style docstrings ([HF chat_extras](https://huggingface.co/docs/transformers/main/en/chat_extras)). TRL GRPO `tools=` takes callables directly ([TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer)).
- **MCP**: a tool definition has `name`, `title`, `description`, `inputSchema` (JSON Schema, 2020-12 by default), optional `outputSchema` and `annotations`. Tools are discovered with `tools/list` and invoked with `tools/call` ([MCP spec 2026-07-28, Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)). MCP's `inputSchema` maps directly onto OpenAI `parameters`, so an MCP server can be the **source** of tool definitions for every Stage. The current revision is stateless: servers keep state across calls by returning explicit handles (same source).

### 6.2 Execution models

| Model | Who runs the tool | Platform burden |
|---|---|---|
| **Client-side** (default OpenAI pattern) | User's application, after receiving `tool_calls` from the serve endpoint | None beyond correct parsing. The serve Stage stays a stateless model endpoint |
| **Server-side via MCP** | Platform component (vLLM `--tool-server`, today Responses API and limited model support; or a platform-owned agent loop) acting as MCP client to user-provided servers | Network policy, credentials, timeouts, audit. MCP says servers **MUST** validate inputs, apply access control, rate limit and sanitize outputs. Clients **SHOULD** confirm sensitive operations, show inputs, validate results, set timeouts and log usage. Annotations are untrusted unless the server is trusted ([MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)) |
| **Sandboxed code** (user uploads tool code) | Platform runs it in an isolated container | Highest. "Tools represent arbitrary code execution and must be treated with appropriate caution" ([MCP spec overview](https://modelcontextprotocol.io/specification/latest)). The vLLM gpt-oss python tool needs Docker or a flag literally named `dangerously_use_uv` ([vLLM GPT-OSS](https://docs.vllm.ai/projects/recipes/en/stable/OpenAI/GPT-OSS.html)) |

Execution is only *required* by the platform in these cases:
- distill with real tool results (APIGen-style verification by execution);
- RL finetuning (TRL GRPO `tools` / `environment_factory`);
- evaluation on the user's own tools beyond AST matching;
- a hosted "agent" product.

Plain serving and SFT on existing trajectories need declaration only.

### 6.3 Must-own vs leave-to-user (from the evidence)

Platform must own:
- the **template + parser pairing** per Base Model, and its persistence through finetune → serve;
- **schema validation** of tool definitions in the Pipeline Request;
- **data normalization**: OpenAI string `arguments` vs dict, dataset `Json()` typing;
- **serve flags**;
- **tool-calling eval wiring**.

Can be left to the user (at least initially):
- tool **implementation and execution** for inference;
- the agent loop;
- confirmation UX.

---

## 7. Open decisions for the platform

1. **Tool-call support as a Base Model property: inferred or declared?**
   - Option A: keep a curated registry mapping model family → (chat template, vLLM/SGLang parser, reasoning parser). This is safe, but each new family needs a registry update.
   - Option B: let users set `tool_parser` / `chat_template` in the Pipeline Request. This is flexible, but mismatches show up only at serve or eval time.
   - Option C: use HF `response_template` where present ([HF](https://huggingface.co/docs/transformers/main/en/chat_response_parsing)). This keeps the model self-describing, but few models ship one yet.
2. **Enforce template continuity from finetune to serve?** Locking the finetuned model to the Base Model's template guarantees the parser still matches. Allowing template overrides (`chat_template_path`) makes it possible to tune base (non-instruct) models, but then the serve parser has to be re-chosen.
3. **Distillation Dataset schema change.** Extend it from prompt → response to `messages` (+ `tools`), or add a separate trajectory type. Extending it keeps one concept but changes a core term in `CONTEXT.md`. A separate type keeps today's simple case simple but adds vocabulary.
4. **Where tool results come from during distillation.** Real execution (accurate, needs runtimes and sandboxing), LLM-simulated results (cheap, but risks hallucinated tool behaviour being learned), or user-supplied recorded results (no execution, limited coverage).
5. **Does the platform execute tools at all?** Never (client-side only; simplest, keeps serve a pure model endpoint); MCP only (a standard interface, but the platform must implement the MCP client role with security duties); or sandboxed user code (most capable, highest security cost).
6. **How tools are declared in a Pipeline Request.** Inline OpenAI JSON Schema, a reference to an MCP server whose `tools/list` is the source of truth, or both. Inline is static and versionable with the Pipeline. MCP is live, but the tool set can change between Stages, so it needs snapshotting.
7. **Strictness default at serve.** vLLM `--tool-strict-level auto|function|parameter`. Stricter means always-valid arguments, but adds grammar-compile latency on first use and only works for parsers/backends that support structural tags ([vLLM](https://docs.vllm.ai/en/latest/features/tool_calling.html)).
8. **Serving engine choice.** vLLM has the widest parser list and a parser plugin API. SGLang has comparable coverage, and BFCL notes it is faster for multi-turn but requires SM 80+ GPUs ([BFCL repo](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)). Supporting both doubles the parser-name mapping.
9. **Custom parser plugins.** Allow user-supplied `--tool-parser-plugin` Python? This unlocks custom formats from user finetunes, but runs arbitrary user code inside the serving pod.
10. **Which eval harness(es) to ship.** BFCL (broad, single-model, runs from the platform's own serve endpoint) and/or τ²-bench (agentic and reliability-focused, needs a second LLM as user simulator and an API budget). Plus whether to offer an AST-match eval over a user-provided "expected calls" set for the user's own tools.
11. **SFT vs RL for tool use.** SFT on trajectories needs no execution. GRPO with `tools` / `environment_factory` needs tool execution inside training Jobs, and TRL marks environments experimental ([TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer)).
12. **Forgetting guardrail.** Whether to run a tool-calling regression eval (for example a BFCL subset) automatically after every finetune of a tool-capable Base Model, given that forgetting is documented in general but not quantified per family. This costs GPU time on every Pipeline.

---

## Sources

- vLLM tool calling: https://docs.vllm.ai/en/latest/features/tool_calling.html
- vLLM structured outputs: https://docs.vllm.ai/en/latest/features/structured_outputs.html
- vLLM offline chat with tools: https://docs.vllm.ai/en/latest/examples/offline_inference/chat_with_tools.html
- vLLM GPT-OSS recipe (built-in tools, `--tool-server`): https://docs.vllm.ai/projects/recipes/en/stable/OpenAI/GPT-OSS.html
- vLLM MCP for non-Harmony models issue: https://github.com/vllm-project/vllm/issues/30115
- SGLang tool parser: https://docs.sglang.io/advanced_features/tool_parser.html
- TGI guidance: https://huggingface.co/docs/text-generation-inference/basic_tutorials/using_guidance ; maintenance notice: https://github.com/huggingface/text-generation-inference
- llama.cpp function calling: https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md
- Ollama tool calling: https://docs.ollama.com/capabilities/tool-calling
- OpenAI function calling: https://developers.openai.com/api/docs/guides/function-calling
- Transformers tool use: https://huggingface.co/docs/transformers/main/en/chat_extras
- Transformers response parsing: https://huggingface.co/docs/transformers/main/en/chat_response_parsing
- HF "Tool Use, Unified": https://huggingface.co/blog/unified-tool-use
- Llama 3.1 prompt format: https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/prompt_format.md
- Qwen function calling: https://qwen.readthedocs.io/en/latest/framework/function_call.html
- TRL dataset formats: https://huggingface.co/docs/trl/dataset_formats
- TRL SFT: https://huggingface.co/docs/trl/sft_trainer
- TRL GRPO: https://huggingface.co/docs/trl/grpo_trainer
- Axolotl conversation datasets: https://docs.axolotl.ai/docs/dataset-formats/conversation.html
- Unsloth: https://unsloth.ai/docs/basics/tool-calling-guide-for-local-llms , https://unsloth.ai/docs/get-started/fine-tuning-llms-guide
- xLAM 60k: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k ; APIGen: https://arxiv.org/abs/2406.18518 ; APIGen-MT: https://arxiv.org/abs/2504.03601
- Glaive v2: https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2
- ToolACE: https://huggingface.co/datasets/Team-ACE/ToolACE , https://arxiv.org/abs/2409.00920
- Agent Distillation: https://arxiv.org/abs/2505.17612
- Catastrophic forgetting: https://arxiv.org/abs/2308.08747 ; Alopex: https://arxiv.org/abs/2411.05209
- BFCL: https://gorilla.cs.berkeley.edu/leaderboard.html , https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
- τ-bench: https://arxiv.org/abs/2406.12045 ; τ²-bench: https://github.com/sierra-research/tau2-bench
- MCP spec (latest, 2026-07-28): https://modelcontextprotocol.io/specification/latest ; Tools: https://modelcontextprotocol.io/specification/2026-07-28/server/tools
