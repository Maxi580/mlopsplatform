from dataclasses import dataclass

import httpx2 as httpx
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError


@dataclass(frozen=True)
class HubModel:
    commit: str
    needs_remote_code: bool


class Cluster:
    """The one place the API talks to systems a test can't run; tests swap in a fake."""

    def find_hub_model(self, repo: str, revision: str, token: str | None) -> HubModel | None:
        hub = HfApi(token=token or False)
        try:
            info = hub.model_info(repo, revision=revision)
            if info.gated:
                hub.auth_check(repo)
        except (HfHubHTTPError, httpx.HTTPError):
            return None
        return HubModel(commit=info.sha, needs_remote_code="auto_map" in (info.config or {}))
