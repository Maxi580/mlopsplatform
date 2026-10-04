import re
from datetime import timedelta
from pathlib import Path

from mlp_core import api_paths

# Pipeline Request
# JSON Schemas of the TRL/PEFT config classes, written by generate_trainer_configs.py.
TRAINER_CONFIGS_DIRECTORY = Path(__file__).parent / "pipeline_request" / "trainer_configs"
# Training backend -> the weight methods it supports.
BACKENDS = {"hf": ("lora",)}
# Phase algorithm -> its TRL trainer, Dataset row formats, blocked settings and defaults.
ALGORITHMS = {
    "sft": {
        "trainer": "SFTTrainer",
        "config": "SFTConfig",
        "row_formats": ("messages", "prompt_completion", "text"),
        "blocked_settings": (
            # Where outputs go, where they are logged and how they are checkpointed.
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
            # Could ask for remote code or swap the Base Model's chat template (#16).
            "model_init_kwargs",
            "chat_template_path",
        ),
        "defaults": {"report_to": ["mlflow"], "save_strategy": "no", "disable_tqdm": True},
    },
}
# LoraConfig settings vLLM can't serve, or that the platform sets.
LORA_CONFIG = "LoraConfig"
BLOCKED_LORA_SETTINGS = ("use_dora", "modules_to_save", "bias", "task_type")
LORA_DEFAULTS = {"task_type": "CAUSAL_LM"}
# The largest Adapter rank vLLM serves.
MAX_LORA_RANK = 512
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
# Secret slot -> the environment variable of the one step that receives it. The API adds
# `serve_token` itself, for the `serve` step to call the API with.
SECRET_ENV_VARS = {"hf_token": "HF_TOKEN", "serve_token": "MLP_SERVE_TOKEN"}
# How long the `serve` step waits for the API to start the Endpoint.
SERVE_REQUEST_TIMEOUT = timedelta(minutes=5)
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
# Marks a serve token: it only starts its Pipeline's Endpoint, and lives as long as Secrets do.
SERVE_TOKEN_CLAIM = "serve_pipeline"
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

# Storage
# Suffixes of the Kubernetes sizes the settings use, e.g. `object_store_size: 100Gi`.
QUANTITY_SUFFIXES = {"": 1, "Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40, "Pi": 2**50}

# Model Cache
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
# The case that finetunes from it, with its Phase algorithm, method and backend.
SMOKE_TEST_UPLOADED_MODEL_CASE = "uploaded-model"
SMOKE_TEST_UPLOADED_MODEL_TRAINING = ("sft", "lora", "hf")
# Runs right after `fetch`: a Pipeline step sends snippets to the Sandbox and checks its limits.
SMOKE_TEST_SANDBOX_CASE = "sandbox"
# Each starts an Endpoint, passes once vLLM is ready, and stops it: serving the Base Model, the
# uploaded tiny full-weight model, and the Adapter of the first finetune case.
SMOKE_TEST_SERVING_CASES = ("serve-base-model", "serve-full-weights", "serve-adapter")
# Trains like the first finetune case, then its `serve` step starts an Endpoint; that Endpoint is
# stopped once the case has a result, so it never holds a GPU the other cases wait for.
SMOKE_TEST_SERVE_STAGE_CASE = "finetune-serve"
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

# Sandbox
# Where the Sandbox takes a batch of snippets and answers each one's result, in order.
SANDBOX_PATH = "/snippets"
# Each of a snippet's stdout and stderr; a snippet writing more is stopped.
SANDBOX_OUTPUT_LIMIT_BYTES = 2**20
