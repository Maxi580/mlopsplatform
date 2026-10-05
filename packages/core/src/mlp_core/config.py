import re
from datetime import timedelta
from pathlib import Path

from mlp_core import api_paths

# Pipeline Request
# JSON Schemas of the TRL/PEFT config classes, written by generate_trainer_configs.py.
TRAINER_CONFIGS_DIRECTORY = Path(__file__).parent / "pipeline_request" / "trainer_configs"
# Training backend -> the weight methods it supports.
BACKENDS = {"hf": ("lora", "qlora", "full")}
# Every algorithm's blocked settings: where outputs go, where they are logged, how they are
# checkpointed, and what could ask for remote code.
BLOCKED_TRAINER_SETTINGS = (
    "output_dir",
    "report_to",
    "logging_dir",
    "save_strategy",
    "save_steps",
    "save_total_limit",
    "resume_from_checkpoint",
    "push_to_hub",
    "hub_model_id",
    "hub_strategy",
    "hub_token",
    "hub_private_repo",
    "model_init_kwargs",
    "trust_remote_code",
)
TRAINER_DEFAULTS = {"report_to": ["mlflow"], "save_strategy": "no", "disable_tqdm": True}
# Phase algorithm -> its TRL trainer, Dataset row formats, blocked settings and defaults.
ALGORITHMS = {
    "sft": {
        "trainer": "SFTTrainer",
        "config": "SFTConfig",
        "row_formats": ("messages", "prompt_completion", "text"),
        # Could swap the Base Model's chat template (#16).
        "blocked_settings": (*BLOCKED_TRAINER_SETTINGS, "chat_template_path"),
        "defaults": TRAINER_DEFAULTS,
    },
    "dpo": {
        "trainer": "DPOTrainer",
        "config": "DPOConfig",
        "row_formats": ("preference",),
        "blocked_settings": BLOCKED_TRAINER_SETTINGS,
        "defaults": TRAINER_DEFAULTS,
    },
    "kto": {
        "trainer": "KTOTrainer",
        "config": "KTOConfig",
        "row_formats": ("unpaired_preference",),
        "blocked_settings": BLOCKED_TRAINER_SETTINGS,
        "defaults": TRAINER_DEFAULTS,
    },
}
# LoraConfig settings the platform sets.
LORA_CONFIG = "LoraConfig"
BLOCKED_LORA_SETTINGS = ("task_type",)
LORA_DEFAULTS = {"task_type": "CAUSAL_LM"}
# The largest Adapter rank vLLM serves; an Adapter merged into full weights may be larger.
MAX_LORA_RANK = 512
# How `qlora` loads the base it trains an Adapter on: 4-bit NF4, computing in bfloat16.
QLORA_QUANTIZATION = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_use_double_quant": True,
}
# Base Model `model_type` -> vLLM's tool-call parser for it (#16); other models serve without tools.
TOOL_PARSERS = {
    "qwen2": "hermes",
    "qwen2_moe": "hermes",
    "qwen3": "hermes",
    "qwen3_moe": "hermes",
    "llama": "llama3_json",
    "mistral": "mistral",
}
# The tool-call parsers built into the pinned vLLM; no parser plugins are allowed (#16).
VLLM_TOOL_PARSERS = (
    "deepseek_v3",
    "deepseek_v31",
    "glm45",
    "granite",
    "granite-20b-fc",
    "hermes",
    "hunyuan_a13b",
    "internlm",
    "jamba",
    "kimi_k2",
    "llama3_json",
    "llama4_json",
    "llama4_pythonic",
    "minimax",
    "mistral",
    "openai",
    "phi4_mini_json",
    "pythonic",
    "qwen3_coder",
    "qwen3_xml",
    "seed_oss",
    "step3",
    "xlam",
)
# The `--max-lora-rank` values vLLM accepts; an Adapter is served with the smallest that fits.
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)
# Always run in this order.
STAGES = ("distill", "sweep", "finetune", "quantize", "speculate", "evaluate", "serve")
# Names the Pipeline's last Model Version, before `finetune` registered it.
FINETUNE_OUTPUT = "@finetune"
# Names the Distillation Dataset Version the Pipeline's `distill` step registers.
DISTILL_OUTPUT = "@distill"
# Secret slot -> the environment variable of the steps that receive it. The API adds
# `step_token` itself, for the `distill` and `serve` steps to call the API with.
SECRET_ENV_VARS = {
    "hf_token": "HF_TOKEN",
    "teacher_api_key": "MLP_TEACHER_API_KEY",
    "step_token": "MLP_STEP_TOKEN",
}
# How long a step waits for the API, e.g. to start the Endpoint.
STEP_REQUEST_TIMEOUT = timedelta(minutes=5)
# Where steps mount the Model Cache; the Hugging Face cache lives inside it.
MODEL_CACHE_PATH = "/model-cache"
# The Kubernetes resource the NVIDIA device plugin offers GPUs as.
GPU_RESOURCE = "nvidia.com/gpu"
# KFP's own object store keys in its namespace, which install.sh fills with the platform's keys.
OBJECT_STORE_SECRET = "mlpipeline-minio-artifact"
OBJECT_STORE_SECRET_ENV_VARS = {
    "accesskey": "AWS_ACCESS_KEY_ID",
    "secretkey": "AWS_SECRET_ACCESS_KEY",
}

# API
MIN_PASSWORD_LENGTH = 12
SESSION_COOKIE = "mlp_session"
TOKEN_LIFETIME = timedelta(hours=12)
JWT_ALGORITHM = "HS256"
# Marks a step token: it only reaches its Pipeline's step routes, and lives as long as Secrets do.
STEP_TOKEN_CLAIM = "pipeline_step"
MAX_FAILED_LOGINS = 5
FAILED_LOGIN_WINDOW = timedelta(minutes=15)
PUBLIC_PATHS = {api_paths.HEALTH, api_paths.LOGIN}
DEFAULT_HF_REVISION = "main"
# Shorter Secret values would match ordinary request strings.
MIN_SECRET_LENGTH = 8
HF_TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{30,}")
DOWNLOAD_URL_LIFETIME = timedelta(hours=1)
# Long enough to upload a large model over a slow line.
UPLOAD_URL_LIFETIME = timedelta(hours=24)
# Where the API sends a browser without login; the Web UI is served under /ui/.
WEB_UI_LOGIN_URL = "/ui/login"

# Pipelines
# Every Pipeline's Owner while the shared account is the only user.
OWNER = "shared"
PIPELINE_SECRET_NAME = "pipeline-{id}"
# Marks the Secrets the platform created, so the backstop sweep never touches others.
PIPELINE_SECRET_LABEL = "mlp-pipeline-secret"
# Kubeflow run state -> Pipeline status; other states count as pending.
PIPELINE_STATUSES = {
    "PENDING": "pending",
    "RUNNING": "running",
    "PAUSED": "running",
    "CANCELING": "running",
    "SUCCEEDED": "succeeded",
    "FAILED": "failed",
    "SKIPPED": "failed",
    "CANCELED": "cancelled",
}
FINISHED_STATUSES = ("succeeded", "failed", "cancelled")
WAITING_FOR_GPU = "waiting for GPU"
RECONCILE_INTERVAL = timedelta(seconds=10)
# Backstop for Secrets the reconciler missed, e.g. while the API was down.
SECRET_MAX_AGE = timedelta(hours=48)
# The KFP UI's run page, behind the platform's /pipeline/ route.
KUBEFLOW_RUN_URL = "/pipeline/#/runs/details/{run_id}"

# Endpoints
# Names an Endpoint's Deployment, Service and route.
ENDPOINT_OBJECT_NAME = "endpoint-{name}"
# Labels an Endpoint's pods with its name.
ENDPOINT_LABEL = "mlp-endpoint"
# Where OpenAI clients reach an Endpoint, behind the login; Traefik strips the prefix before vLLM.
ENDPOINT_PATH_PREFIX = "/endpoints/{name}/"
ENDPOINT_URL = "/endpoints/{name}/v1"
# The Traefik middlewares of the platform namespace on every Endpoint route.
ENDPOINT_ROUTE_MIDDLEWARES = ("login", "endpoint-strip-prefix")
# Traefik's IngressRoute resource: its API group, version and plural.
TRAEFIK_ROUTES = ("traefik.io", "v1alpha1", "ingressroutes")
VLLM_PORT = 8000
# Where an Endpoint's init container puts the Model Version files vLLM loads.
ENDPOINT_MODEL_DIRECTORY = "/models"
ENDPOINT_WEIGHTS_DIRECTORY = f"{ENDPOINT_MODEL_DIRECTORY}/weights"
ENDPOINT_ADAPTER_DIRECTORY = f"{ENDPOINT_MODEL_DIRECTORY}/adapter"
# Prefixed with `endpoint-`, an Endpoint's name names Kubernetes objects, which allow 63 characters.
ENDPOINT_NAME_MAX_LENGTH = 54
# The object store keys in the platform namespace, which Endpoints download Model Versions with.
PLATFORM_CREDENTIALS_SECRET = "mlp-credentials"
PLATFORM_OBJECT_STORE_SECRET_ENV_VARS = {
    "s3AccessKey": "AWS_ACCESS_KEY_ID",
    "s3SecretKey": "AWS_SECRET_ACCESS_KEY",
}

# Datasets
# TRL's standard Dataset row formats -> required field -> its value; the first match wins.
ROW_FORMATS = {
    "preference": {"chosen": "a string or messages", "rejected": "a string or messages"},
    "unpaired_preference": {
        "prompt": "a string or messages",
        "completion": "a string or messages",
        "label": "true or false",
    },
    "stepwise_supervision": {
        "prompt": "a string",
        "completions": "a list of strings",
        "labels": "a list of true or false",
    },
    "prompt_completion": {"prompt": "a string or messages", "completion": "a string or messages"},
    "prompt_only": {"prompt": "a string or messages"},
    "messages": {"messages": "messages"},
    "text": {"text": "a string"},
}
# `distill` asks the Teacher about each prompt, and stores the prompt with the Teacher's reply.
PROMPT_ROW_FORMAT = "prompt_only"
DISTILLATION_ROW_FORMAT = "prompt_completion"

# Storage
# Suffixes of the Kubernetes sizes the settings use, e.g. `object_store_size: 100Gi`.
QUANTITY_SUFFIXES = {"": 1, "Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40, "Pi": 2**50}

# Model Cache
# Where the Model Cache holds Base Models: the Hugging Face cache, HF_HOME/hub.
HUB_DIRECTORY = "hub"
# What a Model Cache entry holds, by its kind.
CACHE_ENTRY_KINDS = {"base_model": "Base Model", "benchmark": "Benchmark"}
# How often the janitor evicts Base Models past the high-water mark; submits also make room.
EVICTION_INTERVAL = timedelta(minutes=10)

# Registered Models
# Model Versions per page of an MLflow registry search.
MLFLOW_PAGE_SIZE = 10000
# Uploaded files arrive in parts of this size; S3 allows 10,000 parts, so files up to 640 GiB.
MODEL_UPLOAD_PART_SIZE = 64 * 2**20
# A file's path inside an uploaded model directory; no part starts with a dot, so none is `..`.
MODEL_FILE_PATH = re.compile(r"[\w-][\w.-]*(/[\w-][\w.-]*)*")
# Lists which shard holds each tensor of sharded *.safetensors weights.
SAFETENSORS_INDEX = "model.safetensors.index.json"
# Weight formats that can run code when loaded; uploads hold *.safetensors weights only.
UNSAFE_WEIGHT_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt", ".pkl", ".h5", ".msgpack", ".gguf")
# Besides tokenizer_config.json, an uploaded model needs one of these for its vocabulary.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "vocab.json")

# Smoke Test
# A strftime pattern; the name prefixes every Dataset and Registered Model a Smoke Test creates.
SMOKE_TEST_NAME = "smoketest-%y%m%d-%H%M%S"
SMOKE_TEST_BASE_MODEL = "hf:Qwen/Qwen2.5-0.5B-Instruct"
# A tiny full-weight model, uploaded the normal way by each Smoke Test and finetuned from.
SMOKE_TEST_UPLOADED_MODEL = "hf:trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
# The case that finetunes from it.
SMOKE_TEST_UPLOADED_MODEL_CASE = "uploaded-model"
# The Phases and backend of the cases that train once: from the tiny model, and on `@distill`. A
# Phase names its algorithm and method, and may add an `output` and `lora` settings.
SMOKE_TEST_TRAINING = (({"algorithm": "sft", "method": "lora"},), "hf")
# Trains an `sft` Phase, then a `dpo` Phase that continues its Adapter.
SMOKE_TEST_CHAIN_CASE = "sft-dpo-chain"
SMOKE_TEST_CHAIN = (
    ({"algorithm": "sft", "method": "lora"}, {"algorithm": "dpo", "method": "lora"}),
    "hf",
)
# The weight method cases: an rsLoRA Adapter, a QLoRA and a DoRA Adapter each merged into full
# weights, and a `full` Phase that merges the Adapter of the Phase before it first.
SMOKE_TEST_MERGED_CASE = "sft-qlora-merged-hf"
SMOKE_TEST_WEIGHT_CASES = {
    "sft-rslora-hf": (
        ({"algorithm": "sft", "method": "lora", "lora": {"use_rslora": True}},),
        "hf",
    ),
    SMOKE_TEST_MERGED_CASE: (({"algorithm": "sft", "method": "qlora", "output": "merged"},), "hf"),
    "sft-dora-merged-hf": (
        ({"algorithm": "sft", "method": "lora", "output": "merged", "lora": {"use_dora": True}},),
        "hf",
    ),
    "sft-lora-dpo-full-hf": (
        ({"algorithm": "sft", "method": "lora"}, {"algorithm": "dpo", "method": "full"}),
        "hf",
    ),
}
# Runs right after `fetch`: a Pipeline step sends snippets to the Sandbox and checks its limits.
SMOKE_TEST_SANDBOX_CASE = "sandbox"
# Each starts an Endpoint, passes once vLLM is ready, and stops it: serving the Base Model, the
# uploaded tiny full-weight model, the Adapter of the first finetune case that keeps one, and the
# merged QLoRA model.
SMOKE_TEST_SERVING_CASES = (
    "serve-base-model",
    "serve-full-weights",
    "serve-adapter",
    "serve-merged",
)
# Trains like the first finetune case, then its `serve` step starts an Endpoint; that Endpoint is
# stopped once the case has a result, so it never holds a GPU the other cases wait for.
SMOKE_TEST_SERVE_STAGE_CASE = "finetune-serve"
# Distills the bundled `distill` prompts with the Base Model as Teacher, offering it one tool, then
# trains on `@distill` with SMOKE_TEST_TRAINING.
SMOKE_TEST_DISTILL_CASE = "distill-tools"
# Enough for the bundled prompts' one-sentence answers; a cut-off reply is dropped as malformed.
SMOKE_TEST_DISTILL_MAX_TOKENS = 256
SMOKE_TEST_DISTILL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "The current weather in a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
# Finetune cases that train from the tiny model, end with an Endpoint, follow `distill`, chain
# Phases or try a weight method's options; the others train one Phase on the Base Model.
SMOKE_TEST_SPECIAL_FINETUNE_CASES = (
    SMOKE_TEST_UPLOADED_MODEL_CASE,
    SMOKE_TEST_SERVE_STAGE_CASE,
    SMOKE_TEST_DISTILL_CASE,
    SMOKE_TEST_CHAIN_CASE,
    *SMOKE_TEST_WEIGHT_CASES,
)
# Evaluate case -> its `evaluate` block without the model: a benchmark on a few samples, or a short
# performance run. The Base Model, and the Adapter of the first finetune case, run a small benchmark
# that scores log-likelihoods, the harder path through vLLM; the Base Model also runs a coding
# benchmark of each harness, scored in the Sandbox, and a GuideLLM run.
SMOKE_TEST_EVALUATE_CASES = {
    "evaluate-base-model": {"benchmarks": ["lm_eval:truthfulqa_mc2"], "limit": 5},
    "evaluate-adapter": {"benchmarks": ["lm_eval:truthfulqa_mc2"], "limit": 5},
    "evaluate-coding": {"benchmarks": ["lm_eval:humaneval"], "limit": 5},
    "evaluate-evalscope": {"benchmarks": ["evalscope:mbpp_plus"], "limit": 5},
    "evaluate-performance": {
        "performance": {"prompt_tokens": 64, "output_tokens": 32, "concurrency": 2, "requests": 10}
    },
}
SMOKE_TEST_ADAPTER_EVALUATE_CASE = "evaluate-adapter"
# One tiny Dataset per Phase algorithm, named after it, uploaded the normal way by each Smoke Test.
SMOKE_TEST_DATASETS_DIRECTORY = Path(__file__).parent / "smoke_test_datasets"
# Every finetune case trains a few steps; only that it runs matters, not what it learns.
SMOKE_TEST_PHASE = {
    "settings": {
        "learning_rate": 1e-4,
        "num_train_epochs": 1,
        "max_steps": 3,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "max_length": 256,
    },
    "lora": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.0, "target_modules": "all-linear"},
}
# Kubeflow task state -> case result; other states are pending.
SMOKE_TEST_CASE_RESULTS = {
    "SUCCEEDED": "passed",
    "CACHED": "passed",
    "FAILED": "failed",
    "SKIPPED": "failed",
    "CANCELED": "failed",
}
# How often `mlp smoke-test` asks for new case results.
SMOKE_TEST_POLL_INTERVAL = timedelta(seconds=30)

# Evaluate
# `harness:task` -> its category, one-line description, dataset licence and download size. Only
# benchmarks whose datasets passed the licence gate (no non-commercial or share-alike licence)
# are listed. A size is what fetch leaves in the Model Cache for the benchmark (its downloaded
# files, their Arrow cache and any task data), measured with `download_benchmark` when added.
BENCHMARKS = {
    "lm_eval:gsm8k": {
        "category": "maths",
        "description": "Grade-school maths word problems, solved step by step.",
        "licence": "MIT",
        "size_bytes": 7_414_523,
    },
    "lm_eval:hellaswag": {
        "category": "reasoning",
        "description": "Commonsense inference: picks the likeliest ending of an everyday scene.",
        "licence": "MIT",
        "size_bytes": 195_531_832,
    },
    "lm_eval:commonsense_qa": {
        "category": "reasoning",
        "description": "Multiple-choice questions that need everyday commonsense knowledge.",
        "licence": "MIT",
        "size_bytes": 4_319_132,
    },
    "lm_eval:bbh": {
        "category": "reasoning",
        "description": "BIG-Bench Hard: 27 hard multi-step reasoning tasks, with chain-of-thought.",
        "licence": "MIT",
        "size_bytes": 3_209_014,
    },
    "lm_eval:mmlu": {
        "category": "knowledge",
        "description": "Multiple-choice questions on 57 subjects, from law to physics.",
        "licence": "MIT",
        "size_bytes": 12_909_397,
    },
    "lm_eval:mmlu_pro": {
        "category": "knowledge",
        "description": "A harder, ten-option MMLU across 14 subjects, with chain-of-thought.",
        "licence": "MIT",
        "size_bytes": 13_098_759,
    },
    "lm_eval:truthfulqa_mc2": {
        "category": "knowledge",
        "description": "Whether the model avoids common misconceptions and falsehoods.",
        "licence": "Apache-2.0",
        "size_bytes": 892_555,
    },
    "lm_eval:ifeval": {
        "category": "instruction following",
        "description": "Verifiable instructions such as word counts, formats and keywords.",
        "licence": "Apache-2.0",
        "size_bytes": 15_707_103,
    },
    "lm_eval:humaneval": {
        "category": "coding",
        "description": "Completes Python functions from their docstrings; unit tests score them.",
        "licence": "MIT",
        "size_bytes": 7_433_045,
    },
    "lm_eval:mbpp": {
        "category": "coding",
        "description": "Short Python programming problems, each checked by three tests.",
        "licence": "CC-BY-4.0",
        "size_bytes": 7_867_255,
    },
    # Run by EvalScope. Its other coding tasks need libraries the Sandbox lacks, run languages
    # other than Python, or fail the licence gate.
    "evalscope:mbpp_plus": {
        "category": "coding",
        "description": "MBPP with corrected problems and many more tests per problem.",
        "licence": "Apache-2.0",
        "size_bytes": 10_818_570,
    },
    "evalscope:super_gpqa": {
        "category": "knowledge",
        "description": "Graduate-level multiple-choice questions across 285 disciplines.",
        "licence": "ODC-BY",
        "size_bytes": 64_001_903,
    },
}
# Where the Model Cache holds benchmark datasets, under `<harness>/<task>/`.
BENCHMARKS_DIRECTORY = "benchmarks"
# Written into a benchmark's directory once fetch downloaded all its datasets.
BENCHMARK_FETCHED_MARKER = "fetched"
# Requests a harness sends to vLLM at once.
EVALUATE_CONCURRENT_REQUESTS = 16
# How often `evaluate` checks whether its vLLM is ready.
VLLM_READY_POLL_INTERVAL = timedelta(seconds=5)
# Where Pipeline steps reach an Endpoint inside the cluster.
ENDPOINT_SERVICE_URL = "http://{object_name}.{namespace}.svc:{port}"
# Where `evaluate` reaches the vLLM it starts for a model that no Endpoint serves.
LOCAL_VLLM_URL = f"http://localhost:{VLLM_PORT}"

# Distill
# Requests the `distill` step sends the Teacher at once.
DISTILL_CONCURRENT_REQUESTS = 16
# How long the step waits for one Teacher reply.
DISTILL_REPLY_TIMEOUT = timedelta(minutes=10)
# The step fails if it drops more of the Teacher's replies than this, as malformed (#17).
DISTILL_MAX_DROPPED_FRACTION = 0.2
# Dropped replies logged to the MLflow Run, with why each was dropped.
DISTILL_DROPPED_SAMPLES = 5

# Sandbox
# Where the Sandbox takes a batch of snippets and answers each one's result, in order.
SANDBOX_PATH = "/snippets"
# Each of a snippet's stdout and stderr; a snippet writing more is stopped.
SANDBOX_OUTPUT_LIMIT_BYTES = 2**20
# `performance` defaults: requests of this many prompt and output tokens, this many at once.
PERFORMANCE_PROMPT_TOKENS = 256
PERFORMANCE_OUTPUT_TOKENS = 128
PERFORMANCE_CONCURRENCY = 1
PERFORMANCE_REQUESTS = 100
# Measures `performance`; prefixes its metric keys and tags the Run NA when it fails.
PERFORMANCE_TOOL = "guidellm"
# GuideLLM's metrics `evaluate` logs with `performance`, each as these statistics over the
# successful requests.
PERFORMANCE_METRICS = (
    "time_to_first_token_ms",
    "inter_token_latency_ms",
    "output_tokens_per_second",
)
PERFORMANCE_STATISTICS = ("mean", "median", "p99")
