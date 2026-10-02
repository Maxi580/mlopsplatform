import re
from datetime import timedelta
from pathlib import Path

from mlp_core import api_paths

# Pipeline Request
# JSON Schemas of the TRL/PEFT config classes, written by generate_trainer_configs.py.
TRAINER_CONFIGS_DIRECTORY = Path(__file__).parent / "pipeline_request" / "trainer_configs"
# Training backend -> the weight methods it supports.
BACKENDS = {"hf": ("lora",)}
# Always run in this order.
STAGES = ("distill", "sweep", "finetune", "quantize", "speculate", "evaluate", "serve")
# Secret slot -> the environment variable of the one step that receives it.
SECRET_ENV_VARS = {"hf_token": "HF_TOKEN"}
# Where steps mount the Model Cache; the Hugging Face cache lives inside it.
MODEL_CACHE_PATH = "/model-cache"

# API
MIN_PASSWORD_LENGTH = 12
SESSION_COOKIE = "mlp_session"
TOKEN_LIFETIME = timedelta(hours=12)
JWT_ALGORITHM = "HS256"
MAX_FAILED_LOGINS = 5
FAILED_LOGIN_WINDOW = timedelta(minutes=15)
PUBLIC_PATHS = {api_paths.HEALTH, api_paths.LOGIN}
DEFAULT_HF_REVISION = "main"
# Shorter Secret values would match ordinary request strings.
MIN_SECRET_LENGTH = 8
HF_TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{30,}")
DOWNLOAD_URL_LIFETIME = timedelta(hours=1)
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
