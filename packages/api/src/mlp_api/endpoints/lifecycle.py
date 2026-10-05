import threading
from datetime import UTC, datetime

from sqlalchemy import Engine, insert, select, update

from mlp_api.endpoints.endpoint_model import add_drafter, find_endpoint_model
from mlp_api.endpoints.manifests import endpoint_manifests
from mlp_api.endpoints.table import endpoint, not_stopped
from mlp_api.model_cache.downloads import preview_downloads
from mlp_api.model_cache.janitor import make_room_for_downloads
from mlp_api.pipelines.cluster import Cluster
from mlp_core import config
from mlp_core.endpoint_spec import EndpointSpec

# Start, stop and reconcile one at a time, so a name is never taken twice and a reconcile never
# sees an Endpoint between its row and its Deployment; the API runs as one replica.
endpoints_lock = threading.Lock()


class EndpointNameTaken(Exception):
    pass


def start_endpoint(state, name: str, spec: EndpointSpec) -> dict:
    """The new Endpoint, pending until vLLM is ready; ValueError if its model can't be served."""
    with endpoints_lock:
        # 1. The name is free while no other Endpoint with it runs.
        if find_endpoint(state.engine, name) is not None:
            raise EndpointNameTaken(f"Endpoint {name} is already running; stop it or pick a name")

        # 2. What vLLM loads, pinned, and what drafts for it, with room for the Base Models in the
        # Model Cache.
        model = find_endpoint_model(
            spec.model, state.hugging_face, state.model_registry, state.object_store
        )
        pinned = {"model": model.references[0]}
        if spec.speculative:
            model, drafter = add_drafter(
                model, spec.speculative, state.hugging_face, state.model_registry
            )
            pinned["speculative"] = spec.speculative.model_copy(update={"model": drafter})
        spec = spec.model_copy(update=pinned)
        if model.base_models:
            downloads = preview_downloads(state, model.base_models, None)
            make_room_for_downloads(state, downloads["download_bytes"])

        # 3. Its row first, so in-use checks and the reconciler know every Deployment.
        settings, cluster = state.settings, state.cluster
        gpus = settings.gpus_per_endpoint
        with state.engine.begin() as connection:
            connection.execute(
                insert(endpoint).values(
                    name=name,
                    owner=config.OWNER,
                    spec=spec.model_dump(mode="json", exclude_defaults=True),
                    uses=model.references,
                    gpus=gpus,
                    status="pending",
                    created_at=datetime.now(UTC),
                )
            )

        # 4. Its Kubernetes objects; after a failure, a partial set is left to stop.
        manifests = endpoint_manifests(
            name, spec, model, gpus, settings.model_cache_size, cluster.endpoint_environment
        )
        try:
            cluster.create_endpoint(manifests)
        except Exception as error:
            set_status(state.engine, name, "failed")
            raise RuntimeError(f"Kubernetes did not create Endpoint {name}: {error}") from None
        return endpoint_summary(find_endpoint(state.engine, name))


def stop_endpoint(engine: Engine, cluster: Cluster, name: str) -> None:
    """Deletes the Endpoint's Kubernetes objects; LookupError if no Endpoint with the name runs."""
    with endpoints_lock:
        if find_endpoint(engine, name) is None:
            raise LookupError(f"No Endpoint {name} is running")
        cluster.delete_endpoint(name)
        set_status(engine, name, "stopped")


def list_endpoints(engine: Engine) -> list[dict]:
    """Every Endpoint, stopped ones included, newest first."""
    with engine.connect() as connection:
        rows = connection.execute(select(endpoint).order_by(endpoint.c.id.desc())).all()
    return [endpoint_summary(row) for row in rows]


def reconcile_endpoints(engine: Engine, cluster: Cluster) -> None:
    """Copies each Endpoint's pod state into its status; without a Deployment it is stopped."""
    with endpoints_lock:
        with engine.connect() as connection:
            rows = connection.execute(select(endpoint).where(not_stopped)).all()
        for row in rows:
            set_status(engine, row.name, cluster.endpoint_state(row.name) or "stopped")


def endpoint_summary(row) -> dict:
    return {
        "name": row.name,
        "owner": row.owner,
        "model": row.spec["model"],
        "spec": row.spec,
        "gpus": row.gpus,
        "status": row.status,
        "url": config.ENDPOINT_URL.format(name=row.name),
        "created_at": row.created_at.isoformat(),
    }


def find_endpoint(engine: Engine, name: str):
    """The Endpoint with the name that isn't stopped, or None."""
    with engine.connect() as connection:
        query = select(endpoint).where(endpoint.c.name == name, not_stopped)
        return connection.execute(query).first()


# A stopped Endpoint stays stopped.
def set_status(engine: Engine, name: str, status: str) -> None:
    with engine.begin() as connection:
        connection.execute(
            update(endpoint).where(endpoint.c.name == name, not_stopped).values(status=status)
        )
