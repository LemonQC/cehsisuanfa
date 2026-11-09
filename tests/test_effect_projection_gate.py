"""No physics claims: numerical tests for effect geometry and data provenance."""
import numpy as np
import pytest
from agcofl.effect_projection_gate import (
    analyze_effect, astar_xy, _close_to_path, from_saved_robot_physics
)
from agcofl.train_data_discovery import split_kind, classify


def _source_pick():
    return dict(
        source_object_xyz=[
          [-.15,0,.02],[-.12,0,.04],[-.06,0,.11],
          [0,0,.15],[.08,0,.12],[.15,0,.02]
        ],
        source_mode="pick",
        source_provenance=dict(type="CALIBRATED_3D_REAL_REQUIRED"),
        table_center_z=.02,goal_xy=[.15,0],
        bounds_xy=[-.26,.26,-.24,.24],
        obstacles_xyxy=[[-.04,-.08,.04,.08]],
        required_via_xy=[[0,.15]],object_xy_radius=.025,
        grid_resolution=.012
    )


def test_pick_lift_over_barrier_not_directly_transferable_to_push():
    d=analyze_effect(_source_pick())
    assert d["support_violation_frames"]>0
    assert d["push_projected_xy_collision_frames"]>0
    assert not d["source_directly_compatible_by_geometry"]
    assert d["candidate_status"]=="GEOMETRIC_PROJECTABLE_PENDING_PHYSICS"
    candidate=d["proposed_geometric_candidate"]
    assert candidate["exists"] and candidate["collision_free"]
    assert candidate["meets_waypoints"]
    assert _close_to_path(candidate["path_xy"],[0,.15])
    # Strong oracle reminder: this is NOT a novel learned projection yet.
    assert d["identical_to_waypoint_oracle"]


def test_goal_only_and_waypoint_route_are_both_geometrically_feasible():
    d=analyze_effect(_source_pick())
    assert d["baselines"]["goal_only"]["exists"]
    assert d["baselines"]["goal_waypoint_oracle"]["exists"]
    assert d["hard_goal_constraint"]


def test_unknown_contact_mode_not_claimed_directly_reachable():
    src=_source_pick()
    src["source_object_xyz"]=[
        [-.15,-.17,.02],[-.1,-.17,.02],[.15,-.17,.02]
    ]
    src["required_via_xy"]=[]
    src["obstacles_xyxy"]=[]
    src["source_mode"]="unknown"
    d=analyze_effect(src)
    assert d["candidate_status"]=="SOURCE_MODE_UNKNOWN_PHYSICS_VALIDATION_REQUIRED"
    assert not d["source_directly_compatible_by_geometry"]


def test_reject_2d_projection_as_3d_physical_source():
    src=_source_pick()
    src["source_object_xyz"]=[[-.1,0],[0,0],[.1,0]]
    with pytest.raises(ValueError,match="\\[N>=3,3\\]"):
        analyze_effect(src)


def test_planner_rejects_target_inside_collision_box():
    path=astar_xy([-.15,0],[0,0],[[-.05,-.05,.05,.05]],
                  bounds=(-.2,.2,-.2,.2),radius=.025)
    assert path is None


def test_training_discovery_does_not_leak_official_eval_videos():
    assert split_kind("human_demos/eval/no_distraction/push_cube/0080/front.mp4")=="evaluation"
    assert classify("human_demos/eval/no_distraction/push_cube/0080/front.mp4") is None
    assert classify("human_demos/train/no_distraction/push_cube/0001/front.mp4")=="human_video"
    assert classify("robot_demos/train/push_cube/0001.npz")=="train_trajectory_or_annotation"
    assert classify("human_demos/no_distraction/push_cube/0001/front.mp4") is None
    assert classify("human_demos/val/push_cube/0002/front.mp4") is None
