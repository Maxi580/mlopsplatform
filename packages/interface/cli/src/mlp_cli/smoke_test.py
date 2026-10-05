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
    sandbox: Annotated[
        bool,
        typer.Option(help="Run snippets in the Sandbox and check its limits; runs a custom one"),
    ] = False,
    uploaded_model: Annotated[
        bool, typer.Option(help="Upload a tiny model and finetune from it; runs a custom one")
    ] = False,
    serving: Annotated[
        bool,
        typer.Option(help="Serve models on Endpoints and run the serve Stage; runs a custom one"),
    ] = False,
    evaluate: Annotated[
        bool, typer.Option(help="Evaluate the Base Model and an Adapter; runs a custom one")
    ] = False,
    distill: Annotated[
        bool, typer.Option(help="Distill with a tool, then train on it; runs a custom one")
    ] = False,
    sweep: Annotated[
        bool, typer.Option(help="Sweep two Trials, then train with the best; runs a custom one")
    ] = False,
    chain: Annotated[
        bool, typer.Option(help="Train sft, then dpo continuing its Adapter; runs a custom one")
    ] = False,
    weights: Annotated[
        bool,
        typer.Option(
            help="Train rsLoRA, merged QLoRA/DoRA and full after an Adapter; runs a custom one"
        ),
    ] = False,
    resume: Annotated[
        bool,
        typer.Option(help="Stop a Phase after a Checkpoint, then resume it; runs a custom one"),
    ] = False,
    quantize: Annotated[
        bool,
        typer.Option(help="Quantize with each scheme, and an Adapter; runs a custom one"),
    ] = False,
    speculate: Annotated[
        bool,
        typer.Option(help="Train each Speculator type; with --serving, serve each; custom"),
    ] = False,
) -> None:
    """Run the Smoke Test, every case or only the named ones, and print each result."""
    # 1. The complete Smoke Test, or a custom one when cases are named.
    named = {"phases": phases, "methods": methods, "backends": backends}
    finetune = {key: value.split(",") for key, value in named.items() if value}
    selection = {"finetune": finetune} if finetune else {}
    if sandbox:
        selection["sandbox"] = True
    if uploaded_model:
        selection["uploaded_model"] = True
    if serving:
        selection["serving"] = True
    if evaluate:
        selection["evaluate"] = True
    if distill:
        selection["distill"] = True
    if sweep:
        selection["sweep"] = True
    if chain:
        selection["chain"] = True
    if weights:
        selection["weights"] = True
    if resume:
        selection["resume"] = True
    if quantize:
        selection["quantize"] = True
    if speculate:
        selection["speculate"] = True
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
