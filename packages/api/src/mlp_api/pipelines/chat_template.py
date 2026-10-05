import hashlib
import json
from collections.abc import Callable

from mlp_core import config


def assistant_mask_problem(model: str, read: Callable[[str], bytes | None]) -> str | None:
    """Why TRL can't limit the model's loss to the assistant's turns, or None if it can."""
    template = chat_template(read)
    if template is None:
        return f"{model} has no chat template, so it can't train on messages rows"
    if config.GENERATION_MARKER.search(template):
        return None
    # TRL trains a known template, e.g. Qwen2.5's, with a marked version of its own.
    if hashlib.sha256(template.encode()).hexdigest() in training_chat_template_hashes():
        return None
    return (
        f"{model}'s chat template doesn't mark the assistant's turns with `{{% generation %}}`, "
        "and TRL has no marked version of it; leave out `assistant_only_loss`"
    )


# transformers reads chat_template.jinja before tokenizer_config.json's `chat_template`.
def chat_template(read: Callable[[str], bytes | None]) -> str | None:
    template = read("chat_template.jinja")
    if template is not None:
        return template.decode()
    template = json.loads(read("tokenizer_config.json") or "{}").get("chat_template")
    # Several named templates; TRL trains with the default one.
    if isinstance(template, list):
        template = next((t["template"] for t in template if t["name"] == "default"), None)
    return template


def training_chat_template_hashes() -> set[str]:
    return set(json.loads(config.TRAINING_CHAT_TEMPLATES.read_text()))
