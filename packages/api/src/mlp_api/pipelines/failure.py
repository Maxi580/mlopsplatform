from mlp_core import config


def failure_reason(log: str, retry: str) -> str:
    """The plain message for a GPU memory error, saying to `retry`, else the log's last lines."""
    lines = log.strip().splitlines()
    searched = "\n".join(lines[-config.FAILURE_SEARCHED_LINES :])
    for error, message in config.GPU_MEMORY_FAILURES:
        if error.search(searched):
            return message.format(retry=retry)
    if not lines:
        return config.NO_LOG_FAILURE
    # A progress bar can make one line huge, and this is sent with every listing.
    return "\n".join(lines[-config.FAILURE_SHOWN_LINES :])[-config.FAILURE_SHOWN_CHARACTERS :]
