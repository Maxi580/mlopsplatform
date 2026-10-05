from pathlib import Path

import yaml


def mlp_directory() -> Path:
    return Path.home() / ".mlp"


def load_profile() -> dict:
    """The CLI Profile; `url` names the platform and `ca_cert` the platform CA to trust."""
    return yaml.safe_load((mlp_directory() / "profile.yaml").read_text())


def build_pipeline_request(
    profile: dict,
    name: str | None,
    finetune_phases: list[str],
    serve: bool = False,
    evaluate: bool = False,
    distill: bool = False,
    quantize: bool = False,
    speculate: bool = False,
) -> dict:
    """A Pipeline Request holding only the named Stages and Phases, in the order named."""
    request = {"name": name or profile.get("name")}
    if distill:
        request["distill"] = profile.get("distill") or {}
    if finetune_phases:
        finetune = dict(profile["finetune"])
        variants = finetune.pop("phases")
        missing = [phase for phase in finetune_phases if phase not in variants]
        if missing:
            raise ValueError(f"Phases missing from the CLI Profile: {', '.join(missing)}")
        # A variant's algorithm defaults to its name, so `sft:` needs no `algorithm: sft`.
        finetune["phases"] = [{"algorithm": phase, **variants[phase]} for phase in finetune_phases]
        request["finetune"] = finetune
    if quantize:
        request["quantize"] = profile.get("quantize") or {}
    if speculate:
        request["speculate"] = profile.get("speculate") or {}
    if evaluate:
        request["evaluate"] = profile.get("evaluate") or {}
    if serve:
        request["serve"] = profile.get("serve") or {}
    return request
