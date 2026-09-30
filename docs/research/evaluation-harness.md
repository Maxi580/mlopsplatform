# Research: evaluation harness for the `evaluate` Stage

Resolves [#5](https://github.com/Maxi580/mlopsplatform/issues/5). Facts checked on **2026-10-01** against the versions below.

| Tool | Version checked | Released | License |
|---|---|---|---|
| lm-evaluation-harness (`lm-eval`) | v0.4.13 | 2026-08-31 | MIT |
| lighteval | v0.13.0 (latest PyPI release; `main` still active) | 2025-11-24 | MIT |
| OpenCompass | 0.5.4 | 2026-08-26 | Apache-2.0 |
| GuideLLM | v0.8.0 | 2026-09-30 | Apache-2.0 |
| vLLM (`vllm bench serve`) | v0.30.0 | 2026-09-22 | Apache-2.0 |
| MLflow | v3.16.1 | 2026-09-17 | Apache-2.0 |

## Question

Which evaluation harness should the `evaluate` Stage be built on? Compare lm-evaluation-harness, lighteval and OpenCompass on benchmark coverage, model sources (local Hugging Face model, LoRA adapter, OpenAI-compatible endpoint such as vLLM), custom tasks, MLflow tracking, runtime/cost and licensing. Also: how is serving performance (latency, throughput) measured, and is it the same tool?

## Recommendation

**Benchmark quality: lm-evaluation-harness (`lm-eval`), MIT.**

- It covers every benchmark named in the ticket as a built-in task (`mmlu`, `gsm8k`, `hellaswag`, `arc`, `ifeval`, plus `mmlu_pro`, `gpqa`, `bbh`, `truthfulqa`, `winogrande`, the Open LLM Leaderboard `leaderboard` group, and more) [1][2].
- It reaches all three model sources with one CLI:
  - local HF model or path: `--model hf --model_args pretrained=<path>` [3];
  - LoRA/QLoRA adapter: add `peft=<adapter path>` to the `hf` model args [3], or `lora_local_path=<path>` on the `vllm` backend [4];
  - OpenAI-compatible endpoint (a vLLM server from the `serve` Stage): `--model local-completions --model_args base_url=http://…/v1/completions,…` or `local-chat-completions` [5].
- Custom tasks are YAML files that can point at local JSON/CSV data (`dataset_path: json` + `dataset_kwargs.data_files`), loaded with `--include_path` [6]. Since 2026-09 custom model backends, filters and metrics can also be added as plugins without forking [1].
- It has a Python API (`simple_evaluate`) that returns a results dict, which the Stage can log to MLflow directly [7].
- It is the most widely used of the three and has the steadiest release cadence (v0.4.11 Feb 2026, v0.4.12 May 2026, v0.4.13 Aug 2026) [8].

**Serving performance: GuideLLM, Apache-2.0, as a separate tool.** None of the three harnesses measures latency or throughput under load. They measure *quality*. Serving performance needs a load generator pointed at the `serve` Stage's OpenAI-compatible endpoint. GuideLLM (a `vllm-project` repo) reports full TTFT, inter-token latency (ITL) and end-to-end latency distributions plus throughput. It supports synchronous, concurrent, throughput, constant, Poisson and **sweep** profiles, with synthetic or dataset-driven prompts. It writes `benchmarks.json`/`.csv` (and optionally HTML), and ships a multi-arch container image on `ghcr.io/vllm-project/guidellm` [9]. `vllm bench serve` measures the same core metrics (TTFT, TPOT, ITL, E2EL, throughput, with `--save-result` to JSON) [10]. It's a fine fallback, but it needs the whole `vllm` package in the benchmark container, and its profile set is narrower (GuideLLM's own comparison table [9]).

Why not the others:

- **lighteval** (MIT) is capable. It covers MMLU, GSM8K, HellaSwag, IFEval and more, and supports PEFT adapters, vLLM and LiteLLM endpoints [11][12][13]. But it has had **no PyPI release since v0.13.0 on 2025-11-24**, while `main` keeps moving (e.g. vLLM 0.30 / transformers 5 support merged 2026-09-30) [14][15]. So we'd have to pin a git commit. Its adapter path merges the adapter into the base model and writes the merged copy to disk before evaluating [12]. That costs time and Model Cache space on every run. Its LiteLLM endpoint backend raises `NotImplementedError` for loglikelihood [13], so MMLU/ARC/HellaSwag-style multiple-choice tasks can't run against an endpoint. Its README also points new users at an `inspect-ai` backend (`lighteval eval`), which adds a second execution model [11].
- **OpenCompass** (Apache-2.0) has broad coverage, `peft_path` on its HF models [16], an `OpenAISDK` model with `openai_api_base` [17], vLLM/LMDeploy acceleration [18] and quick custom datasets [19]. But it is configured through Python config files, and by default it downloads datasets from its own storage server, a release zip, or ModelScope rather than the Hugging Face Hub [18]. That bypasses our HF token and Model Cache design. Its task set and leaderboard focus lean toward Chinese benchmarks (C-Eval, CMMLU, GAOKAO) [18]. It's heavier than we need.

## Comparison table

| | **lm-evaluation-harness** | **lighteval** | **OpenCompass** |
|---|---|---|---|
| License | MIT [8] | MIT [14] | Apache-2.0 [18] |
| Latest release | v0.4.13, 2026-08-31 [8] | v0.13.0, 2025-11-24 (none since) [15] | 0.5.4, 2026-08-26 [20] |
| MMLU / GSM8K / HellaSwag / ARC / IFEval | all built in [2] | all built in [11] | all built in [18] |
| Local HF model | `--model hf` [3] | `lighteval accelerate` [11] | `--hf-path` / HF model configs [18] |
| LoRA/QLoRA adapter | `peft=` on `hf`; `lora_local_path=` on `vllm` [3][4] | `AdapterModel`: merges and saves to disk first [12] | `peft_path=` on HF models [16] |
| OpenAI-compatible endpoint | `local-completions` (all task types if the server returns logprobs), `local-chat-completions` (generative only) [5] | `lighteval endpoint litellm` with `base_url`; generative only [13] | `OpenAISDK` with `openai_api_base` [17] |
| In-process fast backend | vLLM, SGLang [1] | vLLM, SGLang [11] | vLLM, LMDeploy [18] |
| Custom tasks | YAML + local JSON/CSV, `--include_path`; plugins [6][1] | Python custom task modules [11] | ChatML/CustomDataset JSON(L)/CSV, or full config [19] |
| Dataset source | HF Hub (uses `HF_TOKEN`, HF cache) [2][6] | HF Hub [11] | OpenCompass server / zip / ModelScope [18] |
| Built-in tracking | JSON output, W&B, Zeno, HF Hub; no MLflow [1] | JSON/details, HF Hub [11] | local summary files [18] |
| Smoke-sized subsets | `--limit N`, `tinyBenchmarks` tasks [7][21] | `max_samples` [11] | `demo_gsm8k_chat_gen` demo configs [18] |
| Serving latency/throughput | no | no | no |

Serving-performance tools:

| | **GuideLLM** | **`vllm bench serve`** |
|---|---|---|
| License | Apache-2.0 [9] | Apache-2.0 [10] |
| Metrics | TTFT, ITL, end-to-end latency distributions, throughput [9] | TTFT, TPOT, ITL, E2EL, throughput, chosen percentiles [10] |
| Load profiles | synchronous, concurrent, throughput, constant, Poisson, sweep [9] | request rate (incl. `inf`), `--max-concurrency` [10] |
| Data | synthetic (`prompt_tokens`, `output_tokens`), HF datasets, JSON/CSV files [9] | `random` and named datasets [10] |
| Output | JSON, YAML, CSV, HTML, plots [9] | JSON with `--save-result` [10] |
| Packaging | pip or `ghcr.io/vllm-project/guidellm` image [9] | part of the full `vllm` package |

### Performance and cost

- lm-eval's `hf` backend is the reference implementation but the slowest. The project recommends the `vllm` backend with `--batch_size auto` for speed, and notes vLLM output can differ slightly from HF. It ships a comparator script for this [1].
- Evaluating the endpoint the `serve` Stage already runs (`local-completions`) costs no extra GPU memory in the `evaluate` Job, but it needs a running server.
- Multiple-choice tasks (MMLU, ARC, HellaSwag) score by loglikelihood. That needs prompt logprobs, which `/v1/completions` can return but `/v1/chat/completions` cannot [5]. Evaluating an endpoint for these tasks therefore means using `local-completions`.
- Full MMLU is ~14k questions. tinyBenchmarks shows 100 curated examples estimate MMLU accuracy reliably [21]. Full runs belong in real Pipelines, not the Smoke Test.
- `--use_cache` and `--cache_requests` avoid recomputation on retries [1].

## How results get into MLflow

None of the three harnesses logs to MLflow natively. lm-eval integrates with W&B, Zeno and the HF Hub only [1], and code search finds no `mlflow` reference in any of the three repos. The `evaluate` Stage's Job should therefore wrap the harness in a small Python entrypoint:

1. Run `lm_eval.simple_evaluate(...)` (or the CLI with `--output_path`) [7].
2. Walk `results[<task>]`. lm-eval names metric keys `"<metric>,<filter>"`, e.g. `acc,none`, `exact_match,strict-match` [22].
3. **Rename the keys.** MLflow metric names may contain only alphanumerics, `_ - . / space` (and `:` off Windows) [23], and the comma is illegal. Map to e.g. `eval/gsm8k/exact_match/strict-match`. Also log `<metric>_stderr` values.
4. `mlflow.log_metrics(...)` into the Pipeline's Run (max 1000 metrics per batch [23]). Log task versions, `n-shot`, `--limit`, seed and model args as params. Upload the raw `results.json` and, if `--log_samples` was used, the per-sample JSONL as artifacts. They land in the shared object store.
5. For serving performance, parse GuideLLM's `benchmarks.json` and log per-profile metrics such as `serve_perf/ttft_ms_p50`, `…_p99`, `itl_ms_p50`, `output_tokens_per_s`, `requests_per_s`. Attach the JSON/HTML report as an artifact [9].

## Which tiny benchmark subset suits a Smoke Test

The Smoke Test proves the platform works end to end. It doesn't produce a meaningful score. Recommendation:

- **Default:** `lm-eval --tasks gsm8k,hellaswag --limit 10` (or `arc_easy` in place of `hellaswag`). This takes minutes on a tiny model. It exercises one generative task (`generate_until` + answer extraction) and one loglikelihood/multiple-choice task, which are the two request types that behave differently on endpoints [5]. `--limit` is documented as "for testing only" [7], which is exactly this use.
- **Also cover the endpoint path:** run the same two tasks once through `local-completions` against the Smoke Test's `serve` endpoint, so the vLLM → harness → MLflow path is tested too.
- **Not tinyBenchmarks for the Smoke Test.** `tinyMMLU`, `tinyGSM8k`, `tinyHellaswag` etc. are 100 examples each. They need the extra `tinyBenchmarks` package (MIT, installed from git) for IRT scoring [21][24], and they cost more than a smoke run needs. They do suit a cheap "quick quality check" preset for real Pipelines.
- **Serving smoke:** `guidellm benchmark` with `--profile kind=synchronous`, a small synthetic dataset (e.g. `prompt_tokens=64,output_tokens=32`) and a small request cap. This checks the metrics path, not the numbers.
- **GitHub Actions (no GPU):** don't run the harness. Unit-test the results→MLflow key mapping against a checked-in sample `results.json` and GuideLLM `benchmarks.json`.

## Open questions for the grilling tickets

- **#8 Pipeline Request shape:** How does a user pick benchmarks? Options are free task names (validated with `lm-eval ls tasks` / `validate` [7]), named presets (e.g. `quick` = tinyBenchmarks, `leaderboard`), or both. Also: `num_fewshot`, `limit`, `apply_chat_template`, and whether serving-performance runs are part of `evaluate` or of `serve`.
- **#8 / #6:** Does `evaluate` load the model in-process (lm-eval `vllm`/`hf` backend) or call the `serve` Stage's endpoint? The first works without `serve` enabled. The second evaluates exactly what is deployed, including quantization and speculative decoding. Serving-performance runs always need an endpoint.
- **#8:** How are custom evaluation datasets supplied: a task YAML in the Pipeline Request, or a registered dataset (ties to #10)?
- **#10:** Evaluation datasets come from the HF Hub through lm-eval's `datasets` usage. Should they go through the Model Cache / `HF_HOME` like models do (#7)?
- **#12:** Confirm the Smoke Test task set and limits above, and that CI only tests result parsing.
- **Licensing of benchmark data (new):** the harness is MIT, but each benchmark dataset carries its own license. Some are non-commercial or share-alike. The platform only runs them internally, but someone should check the preset list before it ships.
- **Code-executing tasks (new):** tasks like `humaneval`/`mbpp` need `--confirm_run_unsafe_code` [7]. Should the platform allow them, and in what sandbox?
- **Container image (new):** lm-eval has no official image. The platform must build and pin its own evaluate image (lm-eval + vLLM + CUDA 12.8 for the Blackwell dev GPU). GuideLLM ships one.

## Sources

1. lm-evaluation-harness README (news, vLLM/SGLang, caching, W&B/Zeno/HF Hub logging): https://github.com/EleutherAI/lm-evaluation-harness/blob/main/README.md
2. lm-evaluation-harness task directory: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks
3. lm-evaluation-harness README, "Advanced Usage Tips" (PEFT): https://github.com/EleutherAI/lm-evaluation-harness#advanced-usage-tips
4. lm-evaluation-harness vLLM backend source (`lora_local_path`, `LoRARequest`): https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/vllm_causallms.py
5. lm-evaluation-harness README, "Model APIs and Inference Servers" table: https://github.com/EleutherAI/lm-evaluation-harness#model-apis-and-inference-servers
6. lm-evaluation-harness new task guide ("Using Local Datasets", `--include_path`): https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md
7. lm-evaluation-harness CLI/interface reference (`--limit`, `--output_path`, `--log_samples`, `--confirm_run_unsafe_code`, Python API): https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
8. lm-evaluation-harness releases and LICENSE: https://github.com/EleutherAI/lm-evaluation-harness/releases , https://github.com/EleutherAI/lm-evaluation-harness/blob/main/LICENSE.md
9. GuideLLM README (metrics, profiles, outputs, container image) and releases: https://github.com/vllm-project/guidellm , https://github.com/vllm-project/guidellm/releases/tag/v0.8.0
10. vLLM `vllm bench serve` CLI docs: https://docs.vllm.ai/en/latest/cli/bench/serve.html ; vLLM releases: https://github.com/vllm-project/vllm/releases
11. lighteval README: https://github.com/huggingface/lighteval/blob/main/README.md
12. lighteval adapter model source: https://github.com/huggingface/lighteval/blob/main/src/lighteval/models/transformers/adapter_model.py
13. lighteval LiteLLM model source (`base_url`, `loglikelihood` raises `NotImplementedError`): https://github.com/huggingface/lighteval/blob/main/src/lighteval/models/endpoints/litellm_model.py
14. lighteval repository, LICENSE and commit history: https://github.com/huggingface/lighteval , https://github.com/huggingface/lighteval/commits/main
15. lighteval on PyPI (latest 0.13.0, 2025-11-24): https://pypi.org/project/lighteval/#history
16. OpenCompass HF model source (`peft_path`): https://github.com/open-compass/opencompass/blob/main/opencompass/models/huggingface_above_v4_33.py
17. OpenCompass OpenAI model source (`OpenAISDK`, `openai_api_base`): https://github.com/open-compass/opencompass/blob/main/opencompass/models/openai_api.py
18. OpenCompass README (install, datasets, accelerators, LICENSE): https://github.com/open-compass/opencompass/blob/main/README.md
19. OpenCompass custom dataset guide: https://github.com/open-compass/opencompass/blob/main/docs/en/advanced_guides/custom_dataset.md
20. OpenCompass releases: https://github.com/open-compass/opencompass/releases
21. lm-evaluation-harness tinyBenchmarks task README: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/tinyBenchmarks/README.md
22. lm-evaluation-harness metric key format (`f"{metric},{filter_key}"`): https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/evaluator_utils.py
23. MLflow metric-name validation and batch limits: https://github.com/mlflow/mlflow/blob/master/mlflow/utils/validation.py ; `mlflow.log_metrics` API: https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html#mlflow.log_metrics
24. tinyBenchmarks package (MIT): https://github.com/felipemaiapolo/tinyBenchmarks
