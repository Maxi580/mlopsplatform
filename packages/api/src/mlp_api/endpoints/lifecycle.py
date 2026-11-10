import logging
import threading
import uuid
from dataclasses import replace
from datetime import UTC, datetime

from sqlalchemy import Engine, delete, insert, select, update

from mlp_api.endpoints.endpoint_model import EndpointModel, add_drafter, find_endpoint_model
from mlp_api.endpoints.history_retention import delete_endpoint_history
from mlp_api.endpoints.keys import new_endpoint_key
from mlp_api.endpoints.manifests import endpoint_manifests
from mlp_api.endpoints.table import endpoint, not_stopped
from mlp_api.gpu_usage.machines import find_gpu
from mlp_api.model_cache.downloads import preview_downloads
from mlp_api.model_cache.janitor import make_room_for_downloads
from mlp_api.models.model_memory import find_model_memory
from mlp_api.pipelines.cluster import Cluster
from mlp_api.pipelines.failure import failure_reason
from mlp_core import config
from mlp_core.endpoint_spec import EndpointSpec, serving_share, without_stored_quantization
from mlp_core.gpu_share import with_gpu_share

logger = logging.getLogger(__name__)

# Start, stop and reconcile one at a time, so a name is never taken twice and a reconcile never
# sees an Endpoint between its row and its Deployment; the API runs as one replica.
endpoints_lock = threading.Lock()


class EndpointConflict(Exception):
    pass


def start_endpoint(state, name: str, spec: EndpointSpec, smoke_test: bool = False) -> dict:
    """The new Endpoint, pending until vLLM is ready; ValueError if its model can't be served."""
    with endpoints_lock:
        # 1. The name is free until the Endpoint with it is deleted.
        if find_endpoint(state.engine, name) is not None:
            raise EndpointConflict(f"Endpoint {name} already exists; restart it or delete it first")

        # 2. What vLLM loads, pinned, on a GPU Share that can fit, with room in the Model Cache.
        spec, model = pin_endpoint_model(state, spec)
        spec = with_calculated_gpu_share(state, spec)
        refuse_impossible_gpu_share(state, spec)
        make_room_for_base_models(state, model)

        # 3. Its row first, so in-use checks and the reconciler know every Deployment.
        endpoint_uuid = uuid.uuid4().hex
        with state.engine.begin() as connection:
            connection.execute(
                insert(endpoint).values(
                    name=name,
                    uuid=endpoint_uuid,
                    key=new_endpoint_key(),
                    owner=config.OWNER,
                    spec=spec.model_dump(mode="json", exclude_defaults=True),
                    uses=model.references,
                    status="pending",
                    created_at=datetime.now(UTC),
                )
            )

        # 4. Its Kubernetes objects.
        return create_endpoint_objects(state, name, endpoint_uuid, spec, model, smoke_test)


def restart_endpoint(state, name: str, gpu_share: float | None = None) -> dict:
    """The stopped or failed Endpoint at its URL and key, on a new GPU Share if given; pending."""
    with endpoints_lock:
        # 1. Only one that doesn't run; older databases may have several rows, the newest counts.
        found = find_endpoint(state.engine, name)
        if found is None:
            raise LookupError(f"No Endpoint {name}")
        if found.status in (*config.ENDPOINT_STARTING_STATUSES, "running"):
            raise EndpointConflict(f"Endpoint {name} is {found.status}; stop it first")

        # 2. What vLLM loads from the pinned spec, on the new or stored GPU Share, with cache room.
        stored = EndpointSpec.model_validate(without_stored_quantization(found.spec))
        if gpu_share is not None:
            stored = stored.model_copy(update={"gpu_memory_utilization": gpu_share})
        refuse_impossible_gpu_share(state, stored)
        spec, model = pin_endpoint_model(state, stored)
        make_room_for_base_models(state, model)

        # 3. Without a failed Endpoint's leftover Deployment, pending again.
        state.cluster.delete_endpoint(name)
        with state.engine.begin() as connection:
            connection.execute(
                update(endpoint)
                .where(endpoint.c.id == found.id)
                .values(
                    status="pending",
                    failure=None,
                    spec=spec.model_dump(mode="json", exclude_defaults=True),
                )
            )

        # 4. Its Kubernetes objects under the same UUID.
        return create_endpoint_objects(state, name, found.uuid, spec, model)


def pin_endpoint_model(state, spec: EndpointSpec) -> tuple[EndpointSpec, EndpointModel]:
    """The spec with its models pinned, and what vLLM loads: the served model and its drafter."""
    model = find_endpoint_model(
        spec.model, state.hugging_face, state.model_registry, state.object_store
    )
    pinned = {"model": model.references[0]}
    if spec.speculative:
        model, drafter = add_drafter(
            model, spec.speculative, state.hugging_face, state.model_registry
        )
        pinned["speculative"] = spec.speculative.model_copy(update={"model": drafter})
    return spec.model_copy(update=pinned), model


def with_calculated_gpu_share(state, spec: EndpointSpec) -> EndpointSpec:
    """The pinned spec with its GPU Share and length; Endpoints have no Hugging Face token."""
    memory = find_model_memory(
        spec.model, None, state.hugging_face, state.model_registry, state.object_store
    )
    return with_gpu_share(spec, memory, find_gpu(state.cluster), state.settings)


def refuse_impossible_gpu_share(state, spec: EndpointSpec) -> None:
    problem = state.settings.gpu_share_problem(serving_share(spec))
    if problem:
        raise ValueError(problem)


def make_room_for_base_models(state, model: EndpointModel) -> None:
    if model.base_models:
        downloads = preview_downloads(state, model.base_models, None)
        make_room_for_downloads(state, downloads["download_bytes"])


def preview_endpoint(state, spec: EndpointSpec) -> EndpointSpec:
    """The spec as a start would pin it, its GPU Share included; ValueError if it can't start."""
    spec = with_calculated_gpu_share(state, pin_endpoint_model(state, spec)[0])
    refuse_impossible_gpu_share(state, spec)
    return spec


def create_endpoint_objects(
    state,
    name: str,
    endpoint_uuid: str,
    spec: EndpointSpec,
    model: EndpointModel,
    smoke_test: bool = False,
) -> dict:
    """The Endpoint once Kubernetes has its objects; after a failure none, and it is stopped."""
    cluster = state.cluster
    environment = cluster.endpoint_environment
    # A Smoke Test's Endpoint waits for its GPU Share behind the Smoke Test's other GPU jobs.
    if smoke_test:
        environment = replace(environment, gpu_queue=environment.smoke_test_gpu_queue)
    # Only the API calls the Judge, so it has no route, and its answers aren't traced.
    judge = name == config.JUDGE_ENDPOINT_NAME
    manifests = endpoint_manifests(
        name,
        endpoint_uuid,
        spec,
        model,
        state.settings.gpu_request(serving_share(spec)),
        state.settings.model_cache_size,
        environment,
        None if judge else traces_experiment_id(state, name),
        routed=not judge,
    )
    try:
        cluster.create_endpoint(manifests)
    except Exception as error:
        # Delete any half-created objects, so nothing serves without its route.
        cluster.delete_endpoint(name)
        set_status(state.engine, name, "stopped")
        raise RuntimeError(f"Kubernetes did not create Endpoint {name}: {error}") from None
    return endpoint_summary(find_endpoint(state.engine, name), cluster.endpoint_environment.domain)


def traces_experiment_id(state, name: str) -> str | None:
    """The ID of the experiment the Endpoint's Request Traces go to; None if MLflow is down."""
    # Cached until its history is deleted, as the Endpoint Key check asks for each request.
    if name not in state.endpoint_experiments:
        try:
            experiment = config.ENDPOINT_HISTORY_EXPERIMENT.format(name=name)
            state.endpoint_experiments[name] = state.model_registry.experiment_id(experiment)
        except Exception:
            # Serving matters more than its traces.
            logger.warning("MLflow is unreachable, so Endpoint %s has no Request Traces", name)
            return None
    return state.endpoint_experiments[name]


def stop_endpoint(engine: Engine, cluster: Cluster, name: str) -> None:
    """Deletes the Endpoint's Kubernetes objects; LookupError if no Endpoint with the name runs."""
    with endpoints_lock:
        found = find_endpoint(engine, name)
        if found is None or found.status == "stopped":
            raise LookupError(f"No Endpoint {name} is running")
        cluster.delete_endpoint(name)
        set_status(engine, name, "stopped")


def delete_endpoint(state, name: str) -> None:
    """Deletes the name's stopped Endpoints and history; LookupError if none, ValueError if runs."""
    engine = state.engine
    # The history goes under the lock, so an Endpoint started meanwhile keeps a fresh experiment.
    with endpoints_lock:
        with engine.begin() as connection:
            named = endpoint.c.name == name
            deleted = connection.execute(delete(endpoint).where(named, ~not_stopped)).rowcount
        if deleted:
            delete_endpoint_history(state, name, datetime.now(UTC))
            return
    found = find_endpoint(engine, name)
    if found is None:
        raise LookupError(f"No Endpoint {name}")
    raise ValueError(f"Endpoint {name} is {found.status}; stop it first")


def list_endpoints(engine: Engine, domain: str) -> list[dict]:
    """Every Endpoint, stopped ones included, newest first; their URLs are under the domain."""
    with engine.connect() as connection:
        rows = connection.execute(select(endpoint).order_by(endpoint.c.id.desc())).all()
    return [endpoint_summary(row, domain) for row in rows]


def reconcile_endpoints(state) -> None:
    """Each Endpoint's status from its pods: gone is stopped, failed lasts until one is ready."""
    engine, cluster = state.engine, state.cluster
    with endpoints_lock:
        with engine.connect() as connection:
            rows = connection.execute(select(endpoint).where(not_stopped)).all()
        for row in rows:
            status = cluster.endpoint_state(row.name) or "stopped"
            if status == config.ENDPOINT_CRASHING:
                fail_crashed_endpoint(state, row.name)
            elif status == "running":
                set_status(engine, row.name, status, failure=None)
            # Kubernetes keeps retrying, and one between attempts hasn't stopped failing.
            elif row.status != "failed" or status == "stopped":
                set_status(engine, row.name, status)


def fail_crashed_endpoint(state, name: str) -> None:
    """Marks it failed and says why; a log that can't be read is asked for again next round."""
    values = {}
    try:
        values["failure"] = failure_reason(state.cluster.endpoint_logs(name), "Restart")
    except Exception:
        logger.exception("Reading the log of crashed Endpoint %s failed", name)
    set_status(state.engine, name, "failed", **values)


def endpoint_summary(row, domain: str) -> dict:
    return {
        "name": row.name,
        "owner": row.owner,
        "model": row.spec["model"],
        "spec": row.spec,
        # Endpoints stored before #116 left out the then default.
        "gpu_share": row.spec.get("gpu_memory_utilization", config.SERVING_GPU_MEMORY_UTILIZATION),
        "status": row.status,
        "failure": row.failure,
        "url": config.ENDPOINT_URL.format(domain=domain, uuid=row.uuid),
        "key": row.key,
        "created_at": row.created_at.isoformat(),
        "judge": row.name == config.JUDGE_ENDPOINT_NAME,
        "judging": {
            "judged": row.judged,
            "every": row.judge_every,
            "max_per_hour": row.judge_max_per_hour,
        },
    }


def find_endpoint(engine: Engine, name: str):
    """The Endpoint with the name, or None; the newest, where older databases have several."""
    with engine.connect() as connection:
        query = select(endpoint).where(endpoint.c.name == name).order_by(endpoint.c.id.desc())
        return connection.execute(query).first()


# A stopped Endpoint stays stopped; only a restart brings it back.
def set_status(engine: Engine, name: str, status: str, **values) -> None:
    with engine.begin() as connection:
        connection.execute(
            update(endpoint)
            .where(endpoint.c.name == name, not_stopped)
            .values(status=status, **values)
        )
