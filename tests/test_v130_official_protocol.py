"""Pure unit tests for 20×3 original RoboReel Vid2Robot official protocol.

Synthetic dicts are TEST FIXTURES, NOT recorded physical experiments.
"""
import json
import math
import pytest

from agcofl.official_v130_protocol import (
    build_config,validate_summary,aggregate_passes,EXPECTED_CELL
)


def fixture_config(tmp_path):
    p=tmp_path/"upstream.json"
    p.write_text(json.dumps({
        "model":"vid2robot","checkpoint_dir":"$"+"{ROBOREEL_CKPT}/nd/vid2robot",
        "scope":"last20","max_rollout_steps":800,"chunk_size":1,
        "human_demo_and_env_types":[
            ["without_distraction","without_distraction"],
            ["with_distraction","with_distraction"]],
        "save_video":False}))
    return p


def fixture_summary(offset=0,successes=4):
    rows=[dict(task="push_cube",demo_id=n,success=(n-80)<successes)
          for n in range(80,100)]
    return dict(model="vid2robot",runner="vid2robot",scope="last20",
        seed_offset_base=offset,tasks=["push_cube"],chunk_size=1,
        max_rollout_steps=800,
        human_demo_and_env_types=[["without_distraction","without_distraction"]],
        demo_ids_no_distract=list(range(80,100)),
        demo_ids_distract=list(range(80,100)),
        table={EXPECTED_CELL:dict(success=successes,total=20,rate=successes/20)},
        per_trial={EXPECTED_CELL:rows})


def test_identical_official_20_scene_protocol_except_cell_and_task(tmp_path):
    src=fixture_config(tmp_path)
    config=build_config(src,tmp_path/"seed0",0)
    assert config["model"]=="vid2robot"
    assert config["scope"]=="last20"
    assert config["tasks"]==["push_cube"]
    assert config["human_demo_and_env_types"]==[
        ["without_distraction","without_distraction"]]
    assert config["max_rollout_steps"]==800
    assert config["chunk_size"]==1
    assert config["seed_offset_base"]==0
    assert config["checkpoint_dir"]=="$"+"{ROBOREEL_CKPT}/nd/vid2robot"
    with pytest.raises(ValueError,match="seed protocol"):
        build_config(src,tmp_path/"bad",321)


def test_original_20_physics_trials_and_table_required():
    r=validate_summary(fixture_summary(),0)
    assert r["success"]==4
    assert r["total"]==20
    assert r["success_rate"]==.2
    assert len(r["per_trial"])==20
    for bad in (
        {"per_trial":{EXPECTED_CELL:fixture_summary()["per_trial"][EXPECTED_CELL][:-1]}},
        {"table":{EXPECTED_CELL:{"success":0,"total":20,"rate":0}}},
        {"seed_offset_base":100000},
        {"human_demo_and_env_types":[["with_distraction","with_distraction"]]},
        {"max_rollout_steps":200},
    ):
        x=fixture_summary()
        x.update(bad)
        with pytest.raises(ValueError):
            validate_summary(x,0)


def test_duplicate_or_string_success_does_not_count():
    x=fixture_summary()
    x["per_trial"][EXPECTED_CELL][-1]["demo_id"]=80
    with pytest.raises(ValueError,match="Missing or duplicate"):
        validate_summary(x,0)
    x=fixture_summary()
    x["per_trial"][EXPECTED_CELL][0]["success"]="true"
    with pytest.raises(ValueError,match="ambiguous"):
        validate_summary(x,0)


def test_three_actual_seed_passes_mean_sem_and_no_false_full_result():
    results={
       0:validate_summary(fixture_summary(0,4),0),
       100000:validate_summary(fixture_summary(100000,8),100000),
       200000:validate_summary(fixture_summary(200000,12),200000)}
    report=aggregate_passes(results)
    assert report["full_3pass_complete"] is True
    assert report["pooled_successes"]==24
    assert report["pooled_trials"]==60
    assert math.isclose(report["mean_success_rate"],.4)
    assert math.isclose(report["sem_across_seed_passes"],.2/math.sqrt(3))
    partial=aggregate_passes({0:results[0]})
    assert partial["full_3pass_complete"] is False
    assert partial["sem_across_seed_passes"] is None
    assert partial["pooled_trials"]==20
