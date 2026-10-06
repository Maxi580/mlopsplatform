from mlp_core.endpoint_spec import EndpointSpec
from mlp_core.pipeline_request.schema import PipelineRequest


def undescribed(schema: dict) -> list[str]:
    """Every field of the schema and its definitions without a `description`."""
    objects = {"": schema, **schema.get("$defs", {})}
    return [
        f"{name}.{field}"
        for name, definition in objects.items()
        for field, field_schema in definition.get("properties", {}).items()
        if "description" not in field_schema
    ]


def test_every_field_of_the_pipeline_request_has_a_description():
    assert undescribed(PipelineRequest.model_json_schema()) == []


def test_every_serving_option_has_a_description():
    assert undescribed(EndpointSpec.model_json_schema()) == []
