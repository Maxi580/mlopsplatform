import typer
from huggingface_hub import get_token


def collect_secrets(profile: dict) -> dict[str, str]:
    """Secret values by slot: the Profile's, else HF_TOKEN or `hf auth login`, else a prompt."""
    secrets = dict(profile.get("secrets") or {})
    if "hf_token" not in secrets:
        token = get_token() or typer.prompt(
            "Hugging Face token (optional, Enter to skip)",
            default="",
            hide_input=True,
            show_default=False,
        )
        if token:
            secrets["hf_token"] = token
    return secrets
