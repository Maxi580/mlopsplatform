import logging
import threading
import uuid
from datetime import UTC, datetime

from sqlalchemy import Engine, delete, insert, select, update

from mlp_api.endpoints.endpoint_model import EndpointModel, add_drafter, find_endpoint_model
from mlp_api.endpoints.history_retention import delete_endpoint_history
from mlp_api.endpoints.keys import new_endpoint_key
from mlp_api.endpoints.manifests import endpoint_manifests
from mlp_api.endpoints.table import endpoint, not_stopped
from mlp_api.model_cache.downloads import preview_downloads
from mlp_api.model_cache.janitor import make_room_for_downloads
from mlp_api.pipelines.cluster import Cluster
from mlp_core import config
from mlp_core.endpoint_spec import EndpointSpec

logger = logging.getLogger(__name__)

# Start, stop and reconcile one at a time, so a name is never taken twice and a reconcile never
# sees an Endpoint between its row and its Deployment; the API runs as one replica.
endpoints_lock = threading.Lock()


class EndpointConflict(Exception):
    pass


def start_endpoint(state, name: str, spec: EndpointSpec) -> dict:
    """The new Endpoint, pending until vLLM is ready; ValueError if its model can't be served."""
    with endpoints_lock:
        # 1. The name is free until the Endpoint with it is deleted.
        if find_endpoint(state.engine, name) is not None:
            raise EndpointConflict(f"Endpoint {name} already exists; restart it or delete it first")

        # 2. What vLLM loads, pinned.
        spec, model = pin_endpoint_model(state, spec)

        # 3. Its row first, so in-use checks and the reconciler know every Deployment.
        endpoint_uuid, gpus = uuid.uuid4().hex, state.settings.gpus_per_endpoint
        with state.engine.begin() as connection:
            connection.execute(
                insert(endpoint).values(
                    name=name,
                    uuid=endpoint_uuid,
                    key=new_endpoint_key(),
                    owner=config.OWNER,
                    spec=spec.model_dump(mode="json", exclude_defaults=True),
                    uses=model.references,
                    gpus=gpus,
                    status="pending",
                    created_at=datetime.now(UTC),
                )
            )

        # 4. Its Kubernetes objects.
        return create_endpoint_objects(state, name, endpoint_uuid, spec, model, gpus)


def restart_endpoint(state, name: str) -> dict:
    """The stopped or failed Endpoint at its URL and key again, pending until vLLM is ready."""
    with endpoints_lock:
        # 1. Only one that doesn't run; older databases may have several rows, the newest counts.
        found = find_endpoint(state.engine, name)
        if found is None:
            raise LookupError(f"No Endpoint {name}")
        if found.status in ("pending", "running"):
            raise EndpointConflict(f"Endpoint {name} is {found.status}; stop it first")

        # 2. What vLLM loads from the pinned spec, so a moving Reference stays where it was.
        spec, model = pin_endpoint_model(state, EndpointSpec.model_validate(found.spec))

        # 3. Without a failed Endpoint's leftover Deployment, pending again on today's GPU count.
        state.cluster.delete_endpoint(name)
        gpus = state.settings.gpus_per_endpoint
        with state.engine.begin() as connection:
            connection.execute(
                update(endpoint)
                .where(endpoint.c.id == found.id)
                .values(status="pending", gpus=gpus)
            )

        # 4. Its Kubernetes objects under the same UUID.
        return create_endpoint_objects(state, name, found.uuid, spec, model, gpus)


def pin_endpoint_model(state, spec: EndpointSpec) -> tuple[EndpointSpec, EndpointModel]:
    """The spec with its models pinned, and what vLLM loads, with room in the Model Cache."""
    # 1. The served model and what drafts for it.
    model = find_endpoint_model(
        spec.model, state.hugging_face, state.model_registry, state.object_store
    )
    pinned = {"model": model.references[0]}
    if spec.speculative:
        model, drafter = add_drafter(
            model, spec.speculative, state.hugging_face, state.model_registry
        )
        pinned["speculative"] = spec.speculative.model_copy(update={"model": drafter})

    # 2. Room for its Base Models, which may have been evicted since it last ran.
    if model.base_models:
        downloads = preview_downloads(state, model.base_models, None)
        make_room_for_downloads(state, downloads["download_bytes"])
    return spec.model_copy(update=pinned), model


def create_endpoint_objects(
    state, name: str, endpoint_uuid: str, spec: EndpointSpec, model: EndpointModel, gpus: int
) -> dict:
    """The Endpoint once Kubernetes has its objects; after a failure none, and it is stopped."""
    cluster = state.cluster
    manifests = endpoint_manifests(
        name,
        endpoint_uuid,
        spec,
        model,
        gpus,
        state.settings.model_cache_size,
        cluster.endpoint_environment,
        traces_experiment_id(state, name),
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
    """The ID of the experiment the Endpoint name's Request Traces go to; None if MLflow is down."""
    try:
        return state.model_registry.experiment_id(
            config.ENDPOINT_HISTORY_EXPERIMENT.format(name=name)
        )
    except Exception:
        # Serving matters more than its traces.
        logger.warning("MLflow is unreachable, so Endpoint %s starts without Request Traces", name)
        return None


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


def reconcile_endpoints(engine: Engine, cluster: Cluster) -> None:
    """Copies each Endpoint's pod state into its status; without a Deployment it is stopped."""
    with endpoints_lock:
        with engine.connect() as connection:
            rows = connection.execute(select(endpoint).where(not_stopped)).all()
        for row in rows:
            set_status(engine, row.name, cluster.endpoint_state(row.name) or "stopped")


def endpoint_summary(row, domain: str) -> dict:
    return {
        "name": row.name,
        "owner": row.owner,
        "model": row.spec["model"],
        "spec": row.spec,
        "gpus": row.gpus,
        "status": row.status,
        "url": config.ENDPOINT_URL.format(domain=domain, uuid=row.uuid),
        "key": row.key,
        "created_at": row.created_at.isoformat(),
    }


def find_endpoint(engine: Engine, name: str):
    """The Endpoint with the name, or None; the newest, where older databases have several."""
    with engine.connect() as connection:
        query = select(endpoint).where(endpoint.c.name == name).order_by(endpoint.c.id.desc())
        return connection.execute(query).first()


# A stopped Endpoint stays stopped; only a restart brings it back.
def set_status(engine: Engine, name: str, status: str) -> None:
    with engine.begin() as connection:
        connection.execute(
            update(endpoint).where(endpoint.c.name == name, not_stopped).values(status=status)
        )
