"""Offline unit tests for mirror file-level diagnostics. No HTTP requests."""
from pathlib import Path

from scripts import inspect_pi05_mirror_resolve as mod


def test_308_to_official_hub_is_blocking():
    assert mod.inspected_redirect(
        308, "https://huggingface.co/org/repo/resolve/main/file") == (
            "huggingface.co", True)
    assert mod.inspected_redirect(
        302, "https://cdn-lfs-us-1.hf.co/blob?signature=SECRET") == (
            "cdn-lfs-us-1.hf.co", False)
    assert mod.inspected_redirect(200, "") == ("", False)
    assert mod.inspected_redirect(307, "/api/resolve-cache/whatever") == (
        "(relative)", False)


def test_probe_file_reports_redirect_not_raw_signed_url(monkeypatch, tmp_path):
    class FakeResponse:
        status_code = 308
        headers = {
            "Location": "https://huggingface.co/other?token=DO_NOT_PRINT",
        }

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_head(url, **kwargs):
        assert url.endswith(
            "LogosRoboticsLab/roboreel_models/resolve/main/"
            "nd/vid_cond_pi05_uniform/ckpt/params/a.npy")
        assert kwargs["allow_redirects"] is False
        return FakeResponse()

    monkeypatch.setattr(mod.requests, "head", fake_head)
    result = mod.probe_file(
        "https://hf-mirror.com",
        "nd/vid_cond_pi05_uniform/ckpt/params/a.npy",
        tmp_path, 100)
    assert result["code"] == 308
    assert result["points_to_blocked_hub"] is True
    assert result["redirect"] == "huggingface.co"
    assert result["missing_commit"] is True
    assert "DO_NOT_PRINT" not in str(result)


def test_local_size_matching_is_not_fake_physical_result(monkeypatch, tmp_path):
    class FakeResponse:
        status_code = 200
        headers = {"x-repo-commit": "abc"}
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(mod.requests, "head", lambda *a, **kw: FakeResponse())
    target = tmp_path / "nd/vid_cond_pi05_uniform/ckpt/train_config.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"1234")
    result = mod.probe_file(
        "https://hf-mirror.com", "nd/vid_cond_pi05_uniform/ckpt/train_config.json",
        tmp_path, 4)
    assert result["local"] == "matches-metadata-size"
    assert result["missing_commit"] is False
