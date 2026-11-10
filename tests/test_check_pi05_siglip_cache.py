"""Local-only test suite for original SigLIP base asset safety gate."""
from pathlib import Path

from scripts import check_pi05_siglip_cache as sig


def test_uncached_original_siglip_returns_blocked_no_network(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENPI_DATA_HOME", str(tmp_path / "empty_cache"))
    status = sig.audit(tmp_path, tmp_path / "empty_cache")
    assert status["siglip_pickle_available"] is False
    assert status["no_cloud_access_attempted"] is True
    assert status["base_params_possible"] is False


def test_trusted_existing_siglip_pickle_is_detected(tmp_path):
    home = tmp_path / "cache"
    file = home / "pi05_vision_encoder/siglip_params.pkl"
    file.parent.mkdir(parents=True)
    file.write_bytes(b"not_a_real_pickle_for_test_only")
    status = sig.audit(tmp_path, home)
    assert status["siglip_pickle_available"] is True
    assert status["no_cloud_access_attempted"] is True


def test_partial_orbax_metadata_not_declared_full_base(tmp_path):
    home = tmp_path / "cache"
    path = home / "openpi-assets/checkpoints/pi05_base/params"
    path.mkdir(parents=True)
    (path / "_METADATA").write_text("test-only-not-a-checkpoint")
    status = sig.audit(tmp_path, home)
    assert status["base_params_possible"] is True
    assert status["siglip_pickle_available"] is False
