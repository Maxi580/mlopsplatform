from mlp_stages.sandbox import run_in_sandbox


def check_sandbox(sandbox_memory_mb: str) -> None:
    """The Smoke Test's sandbox case: fails unless the Sandbox runs snippets within its limits."""
    # 1. One batch: what each snippet checks, and the status and output that pass it.
    over_memory_limit = 2 * int(sandbox_memory_mb) * 2**20
    network_probe = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=5)\n"
        "    print('network')\n"
        "except OSError:\n"
        "    print('no network')\n"
    )
    checks = [
        (
            "runs a snippet",
            {"code": "print(sum(map(int, input().split())))", "input": "1 2"},
            "ok",
            "3\n",
        ),
        # gVisor reports its own kernel; without the RuntimeClass, every other check could pass.
        (
            "runs under gVisor",
            {"code": "print('gvisor' in open('/proc/version').read())"},
            "ok",
            "True\n",
        ),
        ("has no network", {"code": network_probe}, "ok", "no network\n"),
        (
            "stops a snippet at the memory limit",
            {"code": f"bytearray({over_memory_limit})"},
            "memory_limit",
            None,
        ),
        ("kills a snippet at the timeout", {"code": "while True: pass"}, "timeout", None),
    ]
    results = run_in_sandbox([snippet for _, snippet, _, _ in checks])

    # 2. Each check's result; any failed one fails the step.
    failed = []
    for (check, _, status, stdout), result in zip(checks, results, strict=True):
        passed = result["status"] == status and stdout in (None, result["stdout"])
        print(f"{check}: {'passed' if passed else f'failed with {result}'}")
        if not passed:
            failed.append(check)
    if failed:
        raise SystemExit(f"The Sandbox failed: {', '.join(failed)}")
