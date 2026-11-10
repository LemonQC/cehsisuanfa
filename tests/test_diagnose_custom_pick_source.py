"""Offline tests of the physical source-state diagnosis (no sim)."""
import numpy as np
from scripts.diagnose_custom_pick_source import analyze,_longest_true


def test_longest_sustained_contact():
    assert _longest_true([False, True, True, False, True])==2
    assert _longest_true([False, False])==0


def test_failed_grasp_phase_does_not_conflate_ever_grasped():
    source={"seed":120,"end_phase":"failed","ever_grasped":True,
            "peak_lift_m":0.00008,"valid_pick_source":False}
    sample=[
      {"phase":"descend","cube":[0,0,0.02],"tcp":[0,0,0.1],
       "is_grasped":True,"gripper_command":-1.0},
      {"phase":"grasp","cube":[0,0,0.02],"tcp":[0,0,0.025],
       "is_grasped":False,"gripper_command":1.0},
      {"phase":"grasp","cube":[0.01,0,0.02],"tcp":[0.015,0,0.025],
       "is_grasped":False,"gripper_command":1.0},
      {"phase":"failed","cube":[0.01,0,0.02],"tcp":[0.015,0,0.025],
       "is_grasped":False,"gripper_command":-1.0}
    ]
    r=analyze(source,sample)
    assert r["reported_ever_grasped"] is True
    assert r["reported_lift_m"]<0.001
    assert r["phase_counts"]["grasp"]==2
    assert next(i for i in r["phase_diagnostics"] if i["phase"]=="grasp")[
        "grasp_detected_pre_action"]==0
    assert r["no_physics_executed"] is True
