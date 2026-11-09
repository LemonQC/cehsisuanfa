import numpy as np
import pytest

from agcofl.partial_flow_study import (
    make_proxy, read_manifest, split_groups, FlowPredictor,
    rasterize, resample_polyline
)


def test_proxy_sources_share_identical_context(tmp_path):
    manifest=make_proxy(tmp_path,layouts=10)
    records=read_manifest(manifest)
    assert len(records)==80
    pairs={}
    for r in records:
        pairs.setdefault((r["layout_id"],r["condition"],r["phase"]),[]).append(r)
        assert np.load(tmp_path/r["mask_path"]).shape==(8,32,32)
    for pair in pairs.values():
        assert sorted(x["source"] for x in pair)==["human_proxy","robot"]
        assert pair[0]["context"]==pair[1]["context"]


def test_layout_splits_do_not_leak(tmp_path):
    records=read_manifest(make_proxy(tmp_path,layouts=12))
    split=split_groups(records,42)
    for x,y in (("train","val"),("train","test"),("val","test")):
        assert set(split[x]).isdisjoint(split[y])
    assert set(sum(split.values(),[]))=={r["layout_id"] for r in records}


def test_geometry_network_and_tiny_layout_guard(tmp_path):
    import torch
    assert rasterize([.1,-.1]).sum()>0
    assert np.allclose(resample_polyline([[0,0],[0,.1],[.2,.1]])[-1],[.2,.1])
    assert FlowPredictor(9)(torch.zeros(2,9)).shape==(2,8,32,32)
    records=read_manifest(make_proxy(tmp_path,layouts=5))
    with pytest.raises(ValueError,match="independent layouts"):
        split_groups(records,42)


def test_real_converter_cannot_upgrade_failure_to_success(tmp_path):
    import json
    from agcofl.prepare_real_segments import convert
    trace=tmp_path/"robot.npz"
    xy=np.column_stack([np.linspace(-.15,.15,30),np.zeros(30)])
    np.savez_compressed(trace,success=False,cube_before=xy,
                        cube_after=xy,visited=np.arange(30)>10)
    (tmp_path/"index.jsonl").write_text(json.dumps({
       "id":"robot0","layout_id":"L0","source":"robot",
       "condition":"useful","path":"robot.npz",
       "check_goal":[0,.15],"final_goal":[.15,0],"scene":[1,.15,0,1]
    })+"\n")
    out=tmp_path/"processed"
    manifest=convert(tmp_path/"index.jsonl",out)
    assert manifest.read_text()==""
    assert (out/"preparation_report.json").exists()


def test_training_proxy_explicit_guard(tmp_path):
    from agcofl.partial_flow_study import run_training
    manifest=make_proxy(tmp_path/"input",layouts=8)
    with pytest.raises(RuntimeError,match="allow-proxy"):
        run_training(manifest,tmp_path/"out",epochs=1,device="cpu")
