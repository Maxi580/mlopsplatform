import tempfile
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select, update

from mlp_api.auth.session import issue_serve_token
from mlp_api.datasets.registry import delete_dataset_version, list_datasets, upload_dataset_version
from mlp_api.endpoints.lifecycle import list_endpoints, stop_endpoint
from mlp_api.model_cache.downloads import preview_downloads
from mlp_api.model_cache.janitor import make_room_for_downloads
from mlp_api.models.registry import delete_model_version
from mlp_api.models.uploads import upload_model_directory
from mlp_api.pipelines.compiler import compile_smoke_test
from mlp_api.pipelines.hugging_face import pin_base_model
from mlp_api.pipelines.lifecycle import (
    create_pipeline,
    kubeflow_run_url,
    pipeline,
    pipeline_secret_name,
    set_pipeline,
    unfinished,
)
from mlp_api.pipelines.pipeline_request import validate_pipeline_request
from mlp_api.smoke_tests.cases import (
    SmokeTestSelection,
    finetune_case_request,
    finetune_cases,
    serving_cases,
)
from mlp_core import config
from mlp_core.pipeline_request.references import model_reference, split_base_model_reference

is_smoke_test = pipeline.c.cases.is_not(None)


def start_smoke_test(state, selection: SmokeTestSelection) -> dict:
    """The started Smoke Test's ID, name and run link; ValueError while one runs."""
    engine = state.engine
    # 1. One Smoke Test at a time.
    with engine.connect() as connection:
        running = connection.execute(select(pipeline).where(is_smoke_test, unfinished)).first()
    if running is not None:
        raise ValueError(f"Smoke Test {running.name} (#{running.id}) is still running")

    # 2. Its Pipeline, whose name prefixes all it creates; the reconciler cleans up after it.
    name = datetime.now(UTC).strftime(config.SMOKE_TEST_NAME)
    trainings = finetune_cases(selection)
    sandbox = [config.SMOKE_TEST_SANDBOX_CASE] if selection.sandbox else []
    cases = dict.fromkeys(["fetch", *sandbox, *trainings], "pending")
    pipeline_id = create_pipeline(engine, name, {}, cases)

    try:
        # 3. The Base Model, pinned to a commit, with room for it in the Model Cache.
        base_model = pin_base_model(state.hugging_face, config.SMOKE_TEST_BASE_MODEL, None)
        make_room_for_downloads(state, preview_downloads(state, base_model, None)["download_bytes"])

        # 4. The bundled Datasets and the tiny model, uploaded the normal way.
        for phase in {phase for phase, _, _ in trainings.values()}:
            dataset = config.SMOKE_TEST_DATASETS_DIRECTORY / f"{phase}.jsonl"
            upload_dataset_version(engine, state.object_store, f"{name}-{phase}", dataset)
        starting_models = dict.fromkeys(trainings, {"base_model": base_model})
        uploaded = None
        if config.SMOKE_TEST_UPLOADED_MODEL_CASE in trainings or selection.serving:
            uploaded = upload_tiny_model(state, f"{name}-uploaded")
            starting_models[config.SMOKE_TEST_UPLOADED_MODEL_CASE] = {"from": uploaded}

        # 5. Each case's Pipeline Request, resolved the way users' are and stored for in-use checks.
        requests = {}
        for case, training in trainings.items():
            data = finetune_case_request(case, name, starting_models[case], *training)
            requests[case], errors = validate_pipeline_request(
                data, {}, state.hugging_face, engine, state.model_registry
            )
            if errors:
                raise RuntimeError(f"case {case} is invalid: {errors}")
        resolved = {case: request.model_dump(mode="json") for case, request in requests.items()}

        # 6. The serving cases, which the reconciler runs once their model exists.
        serving = serving_cases(selection, trainings, name, base_model, uploaded)
        set_pipeline(
            engine,
            pipeline_id,
            request={"fetch": base_model, **resolved, **serving},
            cases={**cases, **dict.fromkeys(serving, "pending")},
        )

        # 7. The run, one node per case; a serve step calls the API with its serve token.
        if config.SMOKE_TEST_SERVE_STAGE_CASE in requests:
            token = issue_serve_token(state.jwt_secret, pipeline_id)
            state.cluster.create_secret(pipeline_secret_name(pipeline_id), {"serve_token": token})
        spec = compile_smoke_test(
            pipeline_id,
            name,
            base_model,
            selection.sandbox,
            requests,
            state.cluster.steps,
            state.settings,
        )
        run_id = state.cluster.submit_run(name, spec)
    except Exception as error:
        set_pipeline(engine, pipeline_id, status="failed")
        raise RuntimeError(f"Smoke Test {name} did not start: {error}") from None
    set_pipeline(engine, pipeline_id, kubeflow_run_id=run_id)
    return {"id": pipeline_id, "name": name, "kubeflow_run_url": kubeflow_run_url(run_id)}


def upload_tiny_model(state, name: str) -> str:
    """The `model:` Reference of the Smoke Test's tiny model, downloaded and then uploaded."""
    repo, commit = split_base_model_reference(
        pin_base_model(state.hugging_face, config.SMOKE_TEST_UPLOADED_MODEL, None)
    )
    with tempfile.TemporaryDirectory() as directory:
        state.hugging_face.download_model(repo, commit, Path(directory))
        uploaded = upload_model_directory(
            state.object_store, state.model_registry, name, Path(directory)
        )
    return model_reference(uploaded["name"], uploaded["version"])


def clean_up_smoke_tests(state) -> None:
    """Stops and deletes what every finished Smoke Test created, except its Kubeflow run."""
    engine, object_store, model_registry = state.engine, state.object_store, state.model_registry
    with engine.connect() as connection:
        rows = connection.execute(
            select(pipeline).where(is_smoke_test, ~unfinished, ~pipeline.c.cleaned_up)
        ).all()
    for row in rows:
        prefix = f"{row.name}-"
        # 1. Its Endpoints, which would keep their models in use.
        for found in list_endpoints(engine):
            if found["name"].startswith(prefix) and found["status"] != "stopped":
                stop_endpoint(engine, state.cluster, found["name"])

        # 2. Its Datasets.
        for dataset in list_datasets(engine):
            if dataset["name"].startswith(prefix):
                for found in dataset["versions"]:
                    delete_dataset_version(engine, object_store, dataset["name"], found["version"])

        # 3. Its Model Versions with their files, then their Registered Models.
        names = set()
        for found in model_registry.model_versions():
            if found.name.startswith(prefix):
                delete_model_version(
                    engine, model_registry, object_store, found.name, found.version
                )
                names.add(found.name)
        for name in sorted(names):
            model_registry.delete_registered_model(name)

        # 4. Marked, as set_pipeline only changes unfinished Pipelines.
        with engine.begin() as connection:
            connection.execute(
                update(pipeline).where(pipeline.c.id == row.id).values(cleaned_up=True)
            )
