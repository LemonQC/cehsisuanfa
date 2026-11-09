"""Pure-CPU protocol tests; never invoke official physical evaluator in CI."""
import json

import pytest

from agcofl.v140_official_pi05 import CELL, IDS, make_config, preflight, verify


def config():
    return {"model": "demo_pi05", "scope": "last20",
            "max_rollout_steps": 800, "checkpoint_dir": "/tmp/ckpt"}


def valid_summary(seed=0):
    rows = [{"task": "push_cube", "demo_id": i, "success": i == 95}
            for i in IDS]
    return {"model": "demo_pi05", "runner": "demo_pi05",
            "scope": "last20", "tasks": ["push_cube"],
            "max_rollout_steps": 800, "chunk_size": 20,
            "seed_offset_base": seed, "demo_ids_no_distract": IDS,
            "human_demo_and_env_types":
                [["without_distraction", "without_distraction"]],
            "per_trial": {CELL: rows},
            "table": {CELL: {"success": 1, "total": 20, "rate": .05}}}


def test_original_pi05_action_horizon_not_v130_chunk(tmp_path):
    cfg = make_config(config(), 100000, tmp_path)
    assert "chunk_size" not in cfg
    assert cfg["seed_offset_base"] == 100000
    assert cfg["tasks"] == ["push_cube"]
    with pytest.raises(ValueError):
        make_config({**config(), "chunk_size": 1}, 0, tmp_path)
    with pytest.raises(ValueError):
        make_config(config(), 600000, tmp_path)


def test_official_20_trial_summary_and_seed():
    row = verify(valid_summary(), 0)
    assert (row["success"], row["total"], row["success_rate"]) == (1, 20, .05)
    with pytest.raises(ValueError, match="incompatible"):
        verify(valid_summary(), 100000)


@pytest.mark.parametrize("mutate", [
    lambda x: x["per_trial"][CELL].pop(),
    lambda x: x["per_trial"][CELL].append(x["per_trial"][CELL][0].copy()),
    lambda x: x["per_trial"][CELL][0].update(success="False"),
    lambda x: x["per_trial"][CELL][0].update(demo_id=99),
    lambda x: x["table"][CELL].update(success=19),
    lambda x: x.update(chunk_size=1),
])
def test_reject_missing_duplicated_or_fabricated_physics(mutate):
    x = valid_summary()
    mutate(x)
    with pytest.raises(ValueError):
        verify(x, 0)


def test_no_checkpoint_and_no_cached_features_fails_closed(tmp_path):
    repo = tmp_path / "authors"
    assets = tmp_path / "assets"
    cfg = repo / "configs/eval/nd/vid_cond_pi05_uniform.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps(config()))
    out = preflight("uniform", repo, assets)
    assert out["ready_for_runtime"] is False
    assert out["physics_executed"] is False
    assert len(out["failures"]) >= 23
