"""Offline validation for π0.5/JAX runtime precheck; never starts CUDA."""
import json
from scripts import check_pi05_runtime as check


def test_expected_source_import_paths():
    paths = check.import_roots()
    assert paths[0].endswith("third_party/roboreel_models")
    assert paths[1].endswith("third_party/roboreel_models/src")
    assert paths[-1].endswith("third_party/roboreel_models/roboreel_env/src")


def test_features_requires_original_siglip_and_openpi():
    assert "openpi.shared.image_tools" in check.FEATURE_IMPORTS
    assert "demonstration_vla_suite.shared.siglip_tokenizer" in check.FEATURE_IMPORTS
    assert "jax" in check.FEATURE_IMPORTS
    assert "orbax.checkpoint" in check.EVAL_IMPORTS


def test_missing_imports_fail_closed(monkeypatch, capsys):
    def fake_import(name):
        raise ImportError("not installed")
    monkeypatch.setattr(check.importlib, "import_module", fake_import)
    assert check.main(["--mode", "features"]) == 2
    line = next(x for x in capsys.readouterr().out.splitlines()
                if x.startswith("PI05_RUNTIME_CHECK "))
    report = json.loads(line.split(" ", 1)[1])
    assert report["ready_for_next_gate"] is False
    assert report["physical_rollouts"] == 0
    assert report["siglip_weights_restored"] == "NOT_TESTED"


def test_numpy2_is_rejected_even_when_all_imports_and_gpu_pass(monkeypatch, capsys, tmp_path):
    import sys
    import types

    jax_fake = types.ModuleType("jax")
    jax_fake.devices = lambda: [types.SimpleNamespace(platform="gpu", id=0)]
    np_fake = types.ModuleType("numpy")
    np_fake.__version__ = "2.4.6"
    monkeypatch.setitem(sys.modules, "jax", jax_fake)
    monkeypatch.setitem(sys.modules, "numpy", np_fake)
    monkeypatch.setattr(check, "OFFICIAL", tmp_path)
    monkeypatch.setattr(check.importlib, "import_module", lambda _name: object())

    assert check.main(["--mode", "features"]) == 2
    out = capsys.readouterr().out
    report = json.loads(out.split("PI05_RUNTIME_CHECK ", 1)[1].splitlines()[0])
    assert report["gpu_visible"] is True
    assert report["numpy_version"] == "2.4.6"
    assert "INCOMPATIBLE_NUMPY" in report["checks"]["numpy_openpi_compatibility"]
    assert report["ready_for_next_gate"] is False


def test_numpy1264_is_compatible_in_mocked_gpu_preflight(monkeypatch, capsys, tmp_path):
    import sys
    import types

    jax_fake = types.ModuleType("jax")
    jax_fake.devices = lambda: [types.SimpleNamespace(platform="gpu", id=0)]
    np_fake = types.ModuleType("numpy")
    np_fake.__version__ = "1.26.4"
    monkeypatch.setitem(sys.modules, "jax", jax_fake)
    monkeypatch.setitem(sys.modules, "numpy", np_fake)
    monkeypatch.setattr(check, "OFFICIAL", tmp_path)
    monkeypatch.setattr(check.importlib, "import_module", lambda _name: object())

    assert check.main(["--mode", "features"]) == 0
    out = capsys.readouterr().out
    report = json.loads(out.split("PI05_RUNTIME_CHECK ", 1)[1].splitlines()[0])
    assert report["ready_for_next_gate"] is True
    assert report["checks"]["numpy_openpi_compatibility"] == "OK"
