from mlp_core import config


def failure_reason(log: str) -> str:
    """The plain message for an out-of-memory error in the log, else the log's last lines."""
    lines = log.strip().splitlines()
    if config.OUT_OF_MEMORY_ERROR.search("\n".join(lines[-config.FAILURE_SEARCHED_LINES :])):
        return config.OUT_OF_MEMORY_FAILURE
    if not lines:
        return config.NO_LOG_FAILURE
    # A progress bar can make one line huge, and this is sent with every listing.
    return "\n".join(lines[-config.FAILURE_SHOWN_LINES :])[-config.FAILURE_SHOWN_CHARACTERS :]
