"""Offline tests for strict official baseline source/data provenance."""
from pathlib import Path
from agcofl.official_robot_data import plan
from agcofl.official_baseline_audit import METHODS,OFFICIAL_COMMIT,audit


def _file(demo,name,size=1024):
    return dict(path=f"robot_trajectories/no_distraction/push_cube/{demo}/{name}",bytes=size)


def test_robot_data_plan_selects_only_complete_real_demo_triplets():
    files=[
        _file("0000","trajectory.h5"),_file("0000","base_camera.mp4"),
        _file("0000","hand_camera.mp4"),_file("0001","trajectory.h5"),
        _file("0001","base_camera.mp4"),  # incomplete: NO robot hand video
        _file("0002","trajectory.h5"),_file("0002","base_camera.mp4"),
        _file("0002","hand_camera.mp4"),
    ]
    d=plan(files,task="push_cube",max_demos=2,max_mb=1)
    assert d["status"]=="ROBOT_TRAIN_SUBSET_PLANNED"
    assert [x["demo"] for x in d["demos"]]==["0000","0002"]
    assert sum(len(x["files"]) for x in d["demos"])==6


def test_robot_data_plan_fails_hard_on_unknown_sizes_and_budget():
    files=[_file("0000",x,8_000_000) for x in (
        "trajectory.h5","base_camera.mp4","hand_camera.mp4")]
    import pytest
    with pytest.raises(ValueError,match="budget"):
        plan(files,max_demos=1,max_mb=1)
    files[0]["bytes"]=0
    with pytest.raises(ValueError,match="unknown"):
        plan(files,max_demos=1,max_mb=200)


def test_official_model_audit_never_calls_code_presence_success(tmp_path):
    root=tmp_path/"official_models";root.mkdir()
    (root/".git").mkdir()  # not real checkout
    out=tmp_path/"report.json"
    report=audit(root,tmp_path/"missing_data",out)
    assert out.is_file()
    assert report["pinned"] is False
    assert not report["checkpoint_downloaded"]
    assert not report["inference_performed"]
    assert len(report["methods"])>=5
    assert all(not x["official_rollout_executed"] for x in report["methods"])
    assert len(OFFICIAL_COMMIT)==40
