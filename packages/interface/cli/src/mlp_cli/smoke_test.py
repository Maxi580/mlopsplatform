import time
from typing import Annotated

import typer

from mlp_cli.api import api_client, exit_on_error
from mlp_cli.profile import load_profile
from mlp_core import api_paths, config

Cases = Annotated[str, typer.Option(help="Comma-separated; naming any runs a custom Smoke Test")]


def smoke_test(
    phases: Cases = "",
    methods: Cases = "",
    backends: Cases = "",
    uploaded_model: Annotated[
        bool, typer.Option(help="Upload a tiny model and finetune from it; runs a custom one")
    ] = False,
    serving: Annotated[
        bool,
        typer.Option(help="Serve models on Endpoints and run the serve Stage; runs a custom one"),
    ] = False,
) -> None:
    """Run the Smoke Test, every case or only the named ones, and print each result."""
    # 1. The complete Smoke Test, or a custom one when cases are named.
    named = {"phases": phases, "methods": methods, "backends": backends}
    finetune = {key: value.split(",") for key, value in named.items() if value}
    selection = {"finetune": finetune} if finetune else {}
    if uploaded_model:
        selection["uploaded_model"] = True
    if serving:
        selection["serving"] = True
    with api_client() as client:
        if selection:
            response = client.post(api_paths.SMOKE_TEST_CUSTOM, json=selection)
        else:
            response = client.post(api_paths.SMOKE_TEST_COMPLETE)
    started = exit_on_error(response).json()
    typer.echo(f"Started Smoke Test {started['name']}")
    typer.echo(f"Kubeflow: {load_profile()['url']}{started['kubeflow_run_url']}")

    # 2. Each case's result as it arrives, until the Smoke Test finished.
    reported = {}
    while True:
        with api_client() as client:
            pipelines = exit_on_error(client.get(api_paths.PIPELINES)).json()
        [found] = [p for p in pipelines if p["id"] == started["id"]]
        for case, result in found["cases"].items():
            if result != "pending" and case not in reported:
                reported[case] = result
                typer.echo(f"{case}: {result}")
        if found["status"] in config.FINISHED_STATUSES:
            break
        time.sleep(config.SMOKE_TEST_POLL_INTERVAL.total_seconds())

    # 3. The tally; any failed case fails the command.
    failed = [case for case, result in found["cases"].items() if result != "passed"]
    typer.echo(f"{len(found['cases']) - len(failed)} passed, {len(failed)} failed")
    raise typer.Exit(1 if failed else 0)
