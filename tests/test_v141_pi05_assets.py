"""CPU-only selection tests. Never download checkpoints or execute physics here."""
import pytest

from agcofl.v141_pi05_assets import asset_paths, pending, main
from agcofl.v140_custom_pick_push_physics import crossing_evidence, scene_geometry


def test_exact_official_eval_cache_layout(tmp_path):
    video, feature, keyframe = asset_paths(tmp_path, 80)
    assert str(video).endswith("human_demos/eval/no_distraction/push_cube/0080/front.mp4")
    assert str(feature).endswith("video_demonstration_features/without_distraction/push_cube/0080/front.npy")
    assert str(keyframe).endswith("bpp_keyframe_annotations/without_distraction/push_cube/0080.json")


def test_missing_and_cached_asset_gates(tmp_path):
    miss = pending(tmp_path, "uniform")
    assert len(miss["video"]) == len(miss["feature"]) == 20
    assert miss["keyframe"] == []
    assert len(pending(tmp_path, "keyframes")["keyframe"]) == 20
    video, feature, keyframe = asset_paths(tmp_path, 80)
    for path in (video, feature):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"nonempty")
    keyframe.parent.mkdir(parents=True, exist_ok=True)
    keyframe.write_text('{"keyframe_indices": [0, 20]}')
    missing = pending(tmp_path, "keyframes")
    assert 80 not in missing["video"]
    assert 80 not in missing["feature"]
    assert 80 not in missing["keyframe"]
    assert 81 in missing["feature"]
    with pytest.raises(ValueError):
        pending(tmp_path, "invalid")


def test_observed_wall_crossings_are_not_intended_paths():
    start = [0.0, 0.0, .02]
    wall = scene_geometry(start)
    assert .0 < wall["slab_x_min"] < wall["slab_x_max"] < .15
    assert wall["wall_top_z"] == pytest.approx(.046)
    trace = [{"cube": [0.075, 0., 0.10]}]
    elevated = crossing_evidence(trace, start)
    assert elevated["overflight_observed"]
    assert not elevated["around_wall_observed"]
    ground = crossing_evidence([{"cube": [0.075, .17, .02]}], start)
    assert not ground["overflight_observed"]
    assert ground["around_wall_observed"]
    no_cross = crossing_evidence([{"cube": [0.0, 0.0, .02]}], start)
    assert no_cross["slab_samples"] == 0
    assert not no_cross["around_wall_observed"]


def test_no_api_or_keyframe_run_in_current_phase():
    # Reject VLM-dependent branch before touching any file or network.
    with pytest.raises(SystemExit) as error:
        main(["--variant", "keyframes", "--extract-keyframes"])
    assert error.value.code == 2


def test_require_ready_blocks_official_eval_on_missing_assets(tmp_path):
    out = tmp_path / "report"
    with pytest.raises(SystemExit) as exc:
        main(["--variant", "uniform",
              "--official-root", str(tmp_path / "not_installed_upstream"),
              "--assets", str(tmp_path / "empty_assets"),
              "--out", str(out),
              "--require-ready"])
    assert exc.value.code == 2
    assert (out / "asset_gate.json").is_file()
    import json
    record = json.loads((out / "asset_gate.json").read_text())
    assert record["physics_executed"] is False
    assert record["official_preflight"]["ready_for_runtime"] is False
    assert "checkpoint_params_directory" in record["official_preflight"]["failures"]
