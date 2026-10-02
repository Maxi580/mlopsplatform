import re
from datetime import timedelta

from mlp_core import api_paths

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
