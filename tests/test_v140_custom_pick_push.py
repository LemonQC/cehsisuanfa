"""CPU tests: no ManiSkill import, never claim physical success without rollouts."""
import pytest
from agcofl.v140_custom_pick_push_physics import route_via, source_valid


def test_waypoints_are_manual_oracle_not_claimed_learned():
    route = route_via([0.0, 0.0, 0.02])
    assert len(route) == 2
    assert route[0][1] > 0 and route[1][0] > route[0][0]


def test_actual_pick_contact_support_and_goal_are_all_required():
    valid = dict(env_success=True, ever_grasped=True, peak_lift_m=0.09,
                 released_on_table=True, final_goal_error_m=0.02)
    assert source_valid(valid)
    for key, invalid in [("env_success", False), ("ever_grasped", False),
                         ("peak_lift_m", .02), ("released_on_table", False),
                         ("final_goal_error_m", .08)]:
        d = dict(valid)
        d[key] = invalid
        assert not source_valid(d)
