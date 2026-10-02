import re

from mlp_api.pipelines.lifecycle import find_pipeline
from mlp_api.web.pipeline_form import fields_of
from mlp_core import api_paths

from .conftest import PASSWORD
from .test_pipeline_request import pipeline_request
from .test_pipelines import HF_TOKEN, finish_run, pipelines, reconcile

BROWSER = {"accept": "text/html,application/xhtml+xml"}


def log_in(api, password=PASSWORD):
    return api.post(
        api_paths.WEB_LOGIN, data={"password": password}, headers=BROWSER, follow_redirects=False
    )


def form_values(node, prefix="") -> dict[str, str]:
    """The form fields a user fills in for a Pipeline Request, by dotted path."""
    items = node.items() if isinstance(node, dict) else enumerate(node)
    values = {}
    for key, value in items:
        name = f"{prefix}{key}"
        if isinstance(value, dict | list):
            values |= form_values(value, f"{name}.")
        else:
            values[name] = str(value)
    return values


def submit_form(api, values=None, **secrets):
    data = form_values(pipeline_request()) | (values or {}) | secrets
    return api.post(api_paths.WEB_PIPELINE_FORM, data=data)


def field_error(html: str, name: str) -> str:
    match = re.search(rf'id="{re.escape(name)}-error"[^>]*>([^<]*)<', html)
    return match and match.group(1)


def test_a_browser_without_login_is_sent_to_the_login_page(api):
    response = api.get(api_paths.WEB_PIPELINES, headers=BROWSER, follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == api_paths.WEB_LOGIN
    assert api.get(api_paths.WEB_LOGIN).status_code == 200


def test_the_kfp_and_mlflow_uis_send_a_browser_without_login_to_the_login_page(api):
    response = api.get(api_paths.VERIFY, headers=BROWSER, follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == api_paths.WEB_LOGIN


def test_an_expired_session_sends_htmx_to_the_login_page(api):
    response = api.get(api_paths.WEB_PIPELINE_LIST, headers={"hx-request": "true"})

    assert response.status_code == 401
    assert response.headers["hx-redirect"] == api_paths.WEB_LOGIN


def test_web_login_sets_the_cookie_that_also_opens_the_kfp_and_mlflow_uis(api):
    response = log_in(api)

    assert response.status_code == 303
    assert response.headers["location"] == api_paths.WEB_PIPELINES
    cookie = response.headers["set-cookie"]
    assert "httponly" in cookie.lower() and "max-age=43200" in cookie.lower()
    assert api.get(api_paths.VERIFY).status_code == 200


def test_a_wrong_password_shows_the_login_page_again(api):
    response = log_in(api, "wrong password")

    assert response.status_code == 401
    assert "Wrong password" in response.text
    assert api.get(api_paths.VERIFY).status_code == 401


def test_web_logins_share_the_rate_limit(api):
    for _ in range(5):
        log_in(api, "wrong password")

    assert log_in(api).status_code == 429


def test_the_pipelines_page_shows_the_gpu_count_and_links_the_kfp_and_mlflow_uis(
    logged_in_api, platform_settings
):
    response = logged_in_api.get(api_paths.WEB_PIPELINES)

    assert response.status_code == 200
    assert f"{platform_settings['gpu_count']} GPU" in response.text
    assert 'href="/pipeline/"' in response.text
    assert 'href="/mlflow/"' in response.text


def test_the_form_has_a_field_for_every_value_in_the_schema(logged_in_api):
    html = logged_in_api.get(api_paths.WEB_PIPELINES).text

    for name in form_values(pipeline_request()):
        assert f'name="{name}"' in html, name
    assert 'name="hf_token"' in html


def test_the_form_submits_the_same_pipeline_request_the_cli_does(
    logged_in_api, submittable, cluster
):
    response = submit_form(logged_in_api, hf_token=HF_TOKEN)

    assert response.status_code == 200, response.text
    [pipeline] = pipelines(logged_in_api)
    assert f"Submitted Pipeline {pipeline['id']}" in response.text
    assert response.headers["hx-trigger"] == "pipelines-changed"
    assert cluster.secrets == {f"pipeline-{pipeline['id']}": {"hf_token": HF_TOKEN}}
    assert HF_TOKEN not in response.text


def test_the_form_reads_lists_and_more_settings(logged_in_api, submittable, cluster):
    values = {
        "finetune.phases.0.lora.target_modules": "q_proj, v_proj",
        "finetune.phases.0.settings.*": "weight_decay: 0.01",
    }

    response = submit_form(logged_in_api, values)

    assert "Submitted Pipeline" in response.text, response.text
    [pipeline] = pipelines(logged_in_api)
    stored = find_pipeline(logged_in_api.app.state.engine, pipeline["id"]).request
    phase = stored["finetune"]["phases"][0]
    assert phase["lora"]["target_modules"] == ["q_proj", "v_proj"]
    assert phase["settings"]["weight_decay"] == 0.01


def test_the_form_shows_validation_errors_next_to_their_fields(logged_in_api, submittable, cluster):
    values = {
        "finetune.phases.0.settings.learning_rate": "",
        "finetune.phases.0.settings.max_length": "long",
    }

    response = submit_form(logged_in_api, values)

    html = response.text
    assert field_error(html, "finetune.phases.0.settings.learning_rate") == "Field required"
    assert "valid integer" in field_error(html, "finetune.phases.0.settings.max_length")
    assert pipelines(logged_in_api) == [] and cluster.runs == {}
    # The entered values survive, so the user only fixes what is wrong.
    assert 'value="long"' in html


def test_the_form_shows_errors_found_after_the_schema_check(logged_in_api, submittable):
    response = submit_form(logged_in_api, {"finetune.phases.0.dataset": "dataset:missing"})

    assert field_error(response.text, "finetune.phases.0.dataset") == "no Dataset `missing`"


def test_bad_more_settings_are_shown_inline(logged_in_api, submittable):
    response = submit_form(logged_in_api, {"finetune.phases.0.settings.*": "[not, a, mapping]"})

    assert "mapping" in field_error(response.text, "finetune.phases.0.settings.*")


def test_the_list_shows_owner_status_stages_and_links(logged_in_api, submittable, cluster):
    submit_form(logged_in_api)
    finish_run(cluster, "RUNNING", mlflow_run_url="https://mlflow.test/#/runs/abc")
    reconcile(logged_in_api)

    html = logged_in_api.get(api_paths.WEB_PIPELINE_LIST).text

    [run_id] = cluster.runs
    for text in ("qwen-sft", "shared", "running", "finetune"):
        assert text in html
    assert f'href="/pipeline/#/runs/details/{run_id}"' in html
    assert 'href="https://mlflow.test/#/runs/abc"' in html


def test_the_list_updates_live(logged_in_api):
    html = logged_in_api.get(api_paths.WEB_PIPELINES).text

    assert f'hx-get="{api_paths.WEB_PIPELINE_LIST}"' in html
    assert "every" in html and "pipelines-changed" in html


def test_cancel_from_the_list(logged_in_api, submittable, cluster):
    submit_form(logged_in_api, hf_token=HF_TOKEN)
    [pipeline] = pipelines(logged_in_api)
    cancel_path = api_paths.WEB_CANCEL_PIPELINE.format(id=pipeline["id"])
    assert cancel_path in logged_in_api.get(api_paths.WEB_PIPELINE_LIST).text

    response = logged_in_api.post(cancel_path)

    assert response.status_code == 200
    assert pipelines(logged_in_api)[0]["status"] == "cancelled"
    assert cluster.secrets == {}
    assert cancel_path not in response.text


def test_cancelling_a_finished_pipeline_says_why(logged_in_api, submittable, cluster):
    submit_form(logged_in_api)
    finish_run(cluster, "SUCCEEDED")
    reconcile(logged_in_api)
    [pipeline] = pipelines(logged_in_api)

    response = logged_in_api.post(api_paths.WEB_CANCEL_PIPELINE.format(id=pipeline["id"]))

    assert "already succeeded" in response.text


def test_a_value_with_several_allowed_choices_is_a_select():
    schema = {"type": "object", "properties": {"backend": {"enum": ["hf", "unsloth"]}}}
    [field] = fields_of(schema, {}, "", 0)

    assert (field.kind, field.choices) == ("choice", ("hf", "unsloth"))
