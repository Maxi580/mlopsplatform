import json
from collections.abc import Callable

from mlp_core import config


def uploaded_model_tags(
    paths: set[str], base: str | None, tool_parser: str | None, read: Callable[[str], bytes]
) -> dict[str, str]:
    """The lineage tags of the uploaded files; ValueError naming every check they fail."""
    # 1. What the file names show, as checked before the upload started.
    problems = file_list_problems(paths, base)

    # 2. What the files hold; an auto_map anywhere asks for remote code, which never runs here.
    contents = {}
    json_files = {"config.json", "tokenizer_config.json", "adapter_config.json"}
    for path in (json_files | {config.SAFETENSORS_INDEX}) & paths:
        try:
            contents[path] = json.loads(read(path))
        except ValueError:
            problems.append(f"{path} is not valid JSON")
    problems += [
        f"{path} has an auto_map, which needs remote code"
        for path in sorted(json_files & contents.keys())
        if "auto_map" in contents[path]
    ]
    if not problems and base:
        problems = adapter_problems(contents["adapter_config.json"])
    elif not problems:
        problems = missing_shards(paths, contents.get(config.SAFETENSORS_INDEX, {}))
    if problems:
        raise ValueError("; ".join(problems))

    # 3. The tags: an Adapter names its base, full weights the parser vLLM calls tools with.
    tags = {"source": "uploaded", "owner": config.OWNER}
    if base:
        return {"weights": "adapter", "base_model": base, **tags}
    model_type = contents["config.json"].get("model_type")
    parser = tool_parser or config.TOOL_PARSERS.get(model_type, "none")
    return {"weights": "full", **tags, "tool_parser": parser}


def file_list_problems(paths: set[str], base: str | None) -> list[str]:
    """What the file names alone show is wrong; an adapter_config.json makes an Adapter."""
    problems = [
        f"{path} is not a relative path inside the model directory"
        for path in sorted(paths)
        if not config.MODEL_FILE_PATH.fullmatch(path)
    ]
    problems += [
        f"{path}: only *.safetensors weights are accepted, as others can run code on load"
        for path in sorted(paths)
        if path.endswith(config.UNSAFE_WEIGHT_SUFFIXES)
    ]
    if not any(path.endswith(".safetensors") for path in paths):
        problems.append("there are no *.safetensors weights")
    is_adapter = "adapter_config.json" in paths
    if is_adapter and not base:
        problems.append("an Adapter needs a `base` (hf:org/name or model:name@version)")
    elif base and not is_adapter:
        problems.append("`base` is only for Adapters, which have an adapter_config.json")
    elif not is_adapter:
        if "config.json" not in paths:
            problems.append("config.json is missing")
        if "tokenizer_config.json" not in paths or not set(config.TOKENIZER_FILES) & paths:
            files = " or ".join(config.TOKENIZER_FILES)
            problems.append(f"the tokenizer files are missing (tokenizer_config.json and {files})")
        if config.SAFETENSORS_INDEX not in paths and "model.safetensors" not in paths:
            index = config.SAFETENSORS_INDEX
            problems.append(f"neither model.safetensors nor {index} (the shard index) is there")
    return problems


def missing_shards(paths: set[str], index: dict) -> list[str]:
    shards = set(index.get("weight_map", {}).values())
    listed = f"though {config.SAFETENSORS_INDEX} lists it"
    return [f"{shard} is missing, {listed}" for shard in sorted(shards - paths)]


def adapter_problems(adapter_config: dict) -> list[str]:
    """The Adapter options vLLM can't serve."""
    problems = []
    if adapter_config.get("use_dora"):
        problems.append("DoRA Adapters (use_dora) can't be served by vLLM")
    if adapter_config.get("modules_to_save"):
        problems.append("Adapters with modules_to_save can't be served by vLLM")
    if adapter_config.get("bias", "none") != "none":
        problems.append("Adapters with a trained bias can't be served by vLLM")
    if adapter_config.get("r", 0) > config.MAX_LORA_RANK:
        rank = adapter_config["r"]
        problems.append(f"rank {rank} is above {config.MAX_LORA_RANK}, which vLLM serves")
    return problems
