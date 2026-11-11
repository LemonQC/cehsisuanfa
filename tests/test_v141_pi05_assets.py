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


def test_feature_smoke_limit_requires_active_extraction(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["--variant", "uniform", "--max-feature-demos", "1"])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        main(["--variant", "uniform", "--extract-features",
              "--max-feature-demos", "0"])
    assert error.value.code == 2


def test_feature_smoke_slice_is_one_real_missing_demo(monkeypatch, tmp_path):
    from agcofl import v141_pi05_assets as mod

    # Fake only the extraction for a CPU scope test, never call a GPU tokenizer.
    root = tmp_path / "official"
    root.mkdir()
    monkeypatch.setattr(mod, "pinned_sha", lambda _: mod.PIN)
    # All 20 fixed EVAL video paths must exist for the frozen-data gate.
    for demo_id in mod.IDS:
        video, _, _ = mod.asset_paths(tmp_path / "assets/official_eval_data", demo_id)
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"offline-test-placeholder")
    observed = []
    monkeypatch.setattr(mod, "prepare_features",
                        lambda _root, _data, ids: observed.extend(ids))
    monkeypatch.setattr(mod, "preflight",
                        lambda *_: {"ready_for_runtime": False,
                                    "failures": ["checkpoint_params_directory"]})
    mod.main(["--variant", "uniform", "--official-root", str(root),
              "--assets", str(tmp_path / "assets"),
              "--out", str(tmp_path / "report"),
              "--extract-features", "--max-feature-demos", "1"])
    assert observed == [80]


def test_isolated_original_siglip_workers_one_video_each_and_resume(tmp_path,monkeypatch):
    """CPU-only fake child: parent selects only missing and restarts JAX each time."""
    from types import SimpleNamespace
    from agcofl import v141_pi05_assets as m
    root=tmp_path/"official";root.mkdir()
    assets=tmp_path/"assets"
    data=assets/"official_eval_data"
    for demo in [80,81,82]:
        video,feature,_=m.asset_paths(data,demo)
        video.parent.mkdir(parents=True,exist_ok=True)
        video.write_bytes(b"video-test")
    _,already,_=m.asset_paths(data,80)
    already.parent.mkdir(parents=True,exist_ok=True)
    already.write_bytes(b"author bundle already complete")
    seen=[]
    def fake_run(cmd,env,check):
        assert check is False
        assert cmd[:4] == [m.sys.executable,"-u","-m","agcofl.v141_pi05_assets"]
        assert env["XLA_PYTHON_CLIENT_PREALLOCATE"]=="false"
        assert "--single-feature-id" in cmd
        assert "--feature-process-mode" not in cmd  # child runs original shared encoder
        demo=int(cmd[cmd.index("--single-feature-id")+1])
        seen.append(demo)
        video,feature,_=m.asset_paths(data,demo)
        feature.parent.mkdir(parents=True,exist_ok=True)
        feature.write_bytes(b"atomically saved original bundle")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(m.subprocess,"run",fake_run)
    done=m.prepare_features_isolated(root,assets,[80,81,82])
    assert done == [81,82]
    assert seen == [81,82]
    assert already.read_bytes()==b"author bundle already complete"


def test_isolated_siglip_worker_sigkill_is_explicit_and_no_fake_bundle(
        tmp_path,monkeypatch):
    from types import SimpleNamespace
    from agcofl import v141_pi05_assets as m
    root=tmp_path/"official";root.mkdir()
    assets=tmp_path/"assets"
    video,feature,_=m.asset_paths(assets/"official_eval_data",87)
    video.parent.mkdir(parents=True,exist_ok=True)
    video.write_bytes(b"video-test")
    monkeypatch.setattr(m.subprocess,"run",
                        lambda *args,**kwargs: SimpleNamespace(returncode=-9))
    with pytest.raises(RuntimeError,match="signal 9"):
        m.prepare_features_isolated(root,assets,[87])
    assert not feature.exists()


def test_single_feature_internal_id_guards():
    from agcofl import v141_pi05_assets as m
    with pytest.raises(SystemExit) as error:
        m.main(["--variant","uniform","--extract-features",
                "--single-feature-id","79"])
    assert error.value.code==2
    with pytest.raises(SystemExit) as error:
        m.main(["--variant","uniform","--single-feature-id","87"])
    assert error.value.code==2
