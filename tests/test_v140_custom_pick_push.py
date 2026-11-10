"""CPU tests: no ManiSkill import, never claim physical success without rollouts."""
import pytest
from agcofl.v140_custom_pick_push_physics import route_via, source_valid


def test_waypoints_are_manual_oracle_not_claimed_learned():
    route = route_via([0.0, 0.0, 0.02])
    assert len(route) == 2
    assert route[0][1] > 0 and route[1][0] > route[0][0]


def test_actual_pick_contact_support_and_goal_are_all_required():
    valid = dict(env_success=True, ever_grasped=True, peak_lift_m=0.09,
                 released_on_table=True, final_goal_error_m=0.02,
                 wall_overflight_observed=True)
    assert source_valid(valid)
    for key, invalid in [("env_success", False), ("ever_grasped", False),
                         ("peak_lift_m", .02), ("released_on_table", False),
                         ("final_goal_error_m", .08),
                         ("wall_overflight_observed", False)]:
        d = dict(valid)
        d[key] = invalid
        assert not source_valid(d)


def test_measured_numpy_booleans_and_scalars_roundtrip_as_json():
    import json
    import numpy as np
    from agcofl.v140_custom_pick_push_physics import _json_dumps
    summary = {
        "type": "SIMULATED_PICK_SOURCE_PHYSICS_ONLY",
        "released_on_table": np.bool_(True),
        "observed_collision": np.bool_(False),
        "distance": np.float32(0.03125),
        "steps": np.int64(12),
        "position": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "valid_pick_source": False,
    }
    actual = json.loads(_json_dumps(summary))
    assert actual["released_on_table"] is True
    assert actual["observed_collision"] is False
    assert actual["distance"] == 0.03125
    assert actual["steps"] == 12
    assert len(actual["position"]) == 3
    assert actual["valid_pick_source"] is False


def test_physical_json_refuses_unhandled_and_nonfinite_values():
    import numpy as np
    from agcofl.v140_custom_pick_push_physics import _json_dumps
    with pytest.raises(TypeError):
        _json_dumps({"unexpected": object()})
    with pytest.raises(ValueError):
        _json_dumps({"invalid_measurement": np.float64(float("nan"))})


def test_torch_scalars_can_be_written_when_torch_is_available():
    import json
    import torch
    from agcofl.v140_custom_pick_push_physics import _json_dumps
    x = json.loads(_json_dumps({"ok": torch.tensor(True),
                                "steps": torch.tensor(17),
                                "state": torch.tensor([0.1, 0.2])}))
    assert x["ok"] is True
    assert x["steps"] == 17
    assert len(x["state"]) == 2


def test_pregrasp_targets_current_cube_not_frozen_seed_pose():
    import numpy as np
    from agcofl.v140_custom_pick_push_physics import (
        source_pregrasp_xyz, SOURCE_PREGRASP_CLEARANCE_M)
    current = np.array([0.0349, 0.0419, 0.02])
    target = source_pregrasp_xyz(current)
    assert np.allclose(target[:2], current[:2])
    assert np.isclose(target[2], current[2] + SOURCE_PREGRASP_CLEARANCE_M)
    assert target[2] > current[2]


def test_detect_displaced_cube_before_grasp_instead_of_chasing_old_pose():
    from agcofl.v140_custom_pick_push_physics import (
        source_cube_shifted, SOURCE_LATERAL_RESTART_M)
    initial = [0.0113, 0.0161, 0.02]
    assert not source_cube_shifted(
        [0.0113 + SOURCE_LATERAL_RESTART_M * 0.5, 0.0161, 0.02],
        initial)
    assert source_cube_shifted([0.0349, 0.0419, 0.02], initial)


def test_grasp_tracking_state_reapproaches_or_fails_with_explicit_reason():
    import numpy as np
    from agcofl.v140_custom_pick_push_physics import (
        ScriptedPickSource, SOURCE_DESCEND_TIMEOUT_STEPS)

    class Agent:
        def __init__(self):
            from types import SimpleNamespace
            import torch
            self.tcp_pose = SimpleNamespace(
                p=torch.tensor([[0.0, 0.0, 0.14]]))
        def is_grasping(self, obj):
            import torch
            return torch.tensor([False])

    class Body:
        def __init__(self):
            from types import SimpleNamespace
            import torch
            self.obj = SimpleNamespace(pose=SimpleNamespace(
                p=torch.tensor([[0.02, 0.02, 0.02]])))
            self.agent = Agent()
    class Env:
        def __init__(self):
            self.unwrapped = Body()
            self.action_space = type("Space", (), {"shape": (7,)})()

    env = Env()
    ctrl = ScriptedPickSource(env, np.array([0.02,0.02,0.02]),
                              np.array([0.17,0.02,0.02]))
    ctrl.phase = "descend"
    ctrl.approach_reference = np.array([0.02,0.02,0.02])
    ctrl.steps_in_phase = SOURCE_DESCEND_TIMEOUT_STEPS
    ctrl.act()
    assert ctrl.phase == "failed"
    assert ctrl.failure_reason == "pregrasp_descent_timeout"
    assert ctrl.trace[-1]["failure_reason"] == ctrl.failure_reason
