import pytest

from mlp_api.upgrade import running_work
from mlp_api.upgrade.running_work import print_running_work, stop_running_work

from .test_endpoints import endpoints, qwen_on_the_hub, start, stop  # noqa: F401
from .test_pipelines import finish_run, pipelines, reconcile, submit, submittable  # noqa: F401

# Every test may submit a Pipeline for the `chat` Dataset and start an Endpoint for Qwen.
pytestmark = pytest.mark.usefixtures("submittable", "qwen_on_the_hub")


@pytest.fixture
def upgrade_cluster(cluster, monkeypatch):
    """The commands build their own Cluster in the API pod; here it is the fake."""
    monkeypatch.setattr(running_work, "Cluster", lambda: cluster)
    return cluster


def test_running_work_lists_unfinished_pipelines_and_endpoints_that_are_not_stopped(
    logged_in_api, upgrade_cluster, capsys
):
    pipeline_id = submit(logged_in_api).json()["id"]
    start(logged_in_api, name="chat")
    start(logged_in_api, name="old")
    stop(logged_in_api, name="old")

    print_running_work()

    assert capsys.readouterr().out.splitlines() == [
        f"Pipeline {pipeline_id} qwen-sft (pending)",
        "Endpoint chat (pending)",
    ]


def test_nothing_running_prints_nothing(logged_in_api, upgrade_cluster, capsys):
    print_running_work()

    assert capsys.readouterr().out == ""


def test_stop_running_work_cancels_pipelines_and_stops_endpoints(logged_in_api, upgrade_cluster):
    submit(logged_in_api)
    finish_run(upgrade_cluster, "RUNNING")
    start(logged_in_api, name="chat")

    stop_running_work()

    assert upgrade_cluster.terminated == list(upgrade_cluster.runs)
    assert upgrade_cluster.secrets == {}
    assert upgrade_cluster.endpoints == {}
    assert pipelines(logged_in_api)[0]["status"] == "cancelled"
    assert endpoints(logged_in_api)[0]["status"] == "stopped"


def test_stop_running_work_keeps_finished_pipelines(logged_in_api, upgrade_cluster):
    submit(logged_in_api)
    finish_run(upgrade_cluster, "SUCCEEDED")
    reconcile(logged_in_api)

    stop_running_work()

    assert upgrade_cluster.terminated == []
    assert pipelines(logged_in_api)[0]["status"] == "succeeded"
