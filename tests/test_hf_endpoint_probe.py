"""CPU-only, offline tests for no-download mirror health probe."""
import pytest

from scripts import probe_hf_endpoint as module


class MockResponse:
    def __init__(self, status_code=200, location=None, model_id=module.REPO):
        self.status_code = status_code
        self.headers = {"Location": location} if location is not None else {}
        self._id = model_id

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def json(self):
        return {"id": self._id}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class MockSession:
    def __init__(self, response):
        self.response = response
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url, **kwargs):
        assert url == "https://hf-mirror.com/api/models/" + module.REPO
        assert kwargs["allow_redirects"] is False
        assert kwargs["stream"] is True
        return self.response


def test_public_model_metadata_validated_without_downloading(monkeypatch):
    monkeypatch.setattr(module.requests, "Session", lambda: MockSession(MockResponse()))
    module.probe("https://hf-mirror.com")


@pytest.mark.parametrize("response,expected", [
    (MockResponse(308, "https://huggingface.co/api/models/x"), "redirects back"),
    (MockResponse(200, model_id="other/model"), "unexpected model id"),
])
def test_bad_mirror_is_rejected(monkeypatch, response, expected):
    monkeypatch.setattr(module.requests, "Session", lambda: MockSession(response))
    with pytest.raises(RuntimeError, match=expected):
        module.probe("https://hf-mirror.com")


@pytest.mark.parametrize("endpoint", [
    "http://hf-mirror.com",
    "https://user:secret@hf-mirror.com",
    "hf-mirror.com",
])
def test_reject_non_tls_or_embedded_credentials(endpoint):
    with pytest.raises(ValueError):
        module.probe(endpoint)
