from mlp_core import api_paths, config

QWEN_7B = {"name": "Qwen/Qwen2.5-7B-Instruct", "parameters": 7_615_616_512, "gated": False}
LLAMA_8B = {"name": "meta-llama/Llama-3.1-8B-Instruct", "parameters": 8_030_261_248, "gated": True}
QWEN_CODER = {"name": "Qwen/Qwen2.5-Coder-1.5B", "parameters": 1_543_714_304, "gated": False}


def search(api, words="", **headers):
    response = api.get(api_paths.BASE_MODELS, params={"search": words}, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def test_without_a_search_the_curated_models_are_offered(logged_in_api, hugging_face):
    found = search(logged_in_api)

    assert [model["name"] for model in found] == [m["name"] for m in config.CURATED_BASE_MODELS]
    assert found[0]["reference"] == f"hf:{config.CURATED_BASE_MODELS[0]['name']}"
    assert hugging_face.searches == []


def test_a_search_finds_text_generation_models_holding_every_word(logged_in_api, hugging_face):
    hugging_face.search_results = [QWEN_7B, LLAMA_8B, QWEN_CODER]

    found = search(logged_in_api, "qwen 7B")

    assert found == [{**QWEN_7B, "reference": "hf:Qwen/Qwen2.5-7B-Instruct"}]
    # The Hub matches one word; the API keeps the models that hold the others too.
    assert hugging_face.searches == [("qwen", None)]


def test_a_search_uses_the_users_token_so_their_gated_and_private_models_show(
    logged_in_api, hugging_face
):
    hugging_face.search_results = [LLAMA_8B]

    found = search(logged_in_api, "llama", **{"x-hf-token": "hf_users_token"})

    assert found[0]["gated"] is True
    assert hugging_face.searches == [("llama", "hf_users_token")]
