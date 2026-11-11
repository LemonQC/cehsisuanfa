"""CPU-only M1-B2 checks: physical success cannot be inferred from plans."""
import json
import tempfile
import unittest
from pathlib import Path

from agcofl.m1b2_same_goal_effect_physics import (
    actual_gate, BLOCKER_X_OFFSET_M, MAX_STEPS, TracedEnv)
from scripts.m1b2_validate_paired_physics import summarize


def physical_record(mode,direct=False):
    source=mode=="source"
    data={
        "kind":"source_pick_to_same_goal" if source else mode,
        "seed":120,
        "same_final_goal_config":True,
        "state_before":[0.0,0.0,0.02],
        "goal":[0.15,0.0,0.001],
        "physically_verified_matched_success":True if source else (
            direct if mode=="target_direct" else True),
        "steps":12,
    }
    if source:
        data["gate"]={"valid":True}
    else:
        data["blocker_before"]=[0.077,0.0,0.025]
        data["push_only_success"]=True
    if mode=="target_direct":
        data["push_only_success"]=direct
    if mode=="target_clear":
        data["clear_and_push_success"]=True
        data["blocker_effect_physically_inserted"]=True
    return data


class PhysxEvidenceTests(unittest.TestCase):
    def test_source_needs_observed_contact_lift_release_and_grounded_goal(self):
        start=[0.0,0.0,0.02]
        goal=[0.15,0.0,0.001]
        good=[
           {"task_cube":[0.0,0.0,0.02],"is_grasping_task":False},
           {"task_cube":[0.01,0.0,0.12],"is_grasping_task":True},
           {"task_cube":[BLOCKER_X_OFFSET_M,0.0,0.15],
            "is_grasping_task":True},
           {"task_cube":[0.15,0.0,0.02],"is_grasping_task":False},
        ]
        gate=actual_gate(good,start,[0.15,0.,.02],goal,True)
        self.assertTrue(gate["valid"])
        no_grasp=[dict(r,is_grasping_task=False) for r in good]
        self.assertFalse(actual_gate(no_grasp,start,[.15,0,.02],goal,True)["valid"])
        not_released=[dict(r,is_grasping_task=True) for r in good]
        self.assertFalse(actual_gate(not_released,start,[.15,0,.02],goal,True)["valid"])
        not_goal=actual_gate(good,start,[.02,0,.02],goal,True)
        self.assertFalse(not_goal["valid"])

    def test_physics_aggregator_requires_matched_goals_and_real_three_cases(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            self.assertFalse(summarize(path)["passed"] if "passed" in summarize(path)
                             else False)
            for mode in ("source","target_direct","target_clear"):
                sub=path/mode
                sub.mkdir()
                (sub/"status.json").write_text(json.dumps(physical_record(mode)))
            ok=summarize(path)
            self.assertEqual(ok["status"],"PILOT_CONTRAST_OBSERVED_NOT_CAUSAL_PROOF")
            self.assertTrue(ok["pilot_contrast_pass"])
            self.assertFalse(ok["causal_necessity_proven"])
            self.assertFalse(ok["strong_capability_aware_tamp_evaluated"])
            changed=physical_record("target_clear")
            changed["goal"]=[0.20,0,0.001]
            (path/"target_clear/status.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,"different physical start/goal"):
                summarize(path)

    def test_direct_push_success_disproves_required_insert_contrast(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            for mode in ("source","target_direct","target_clear"):
                (path/mode).mkdir()
                (path/mode/"status.json").write_text(
                    json.dumps(physical_record(mode,direct=True)))
            result=summarize(path)
            self.assertFalse(result["pilot_contrast_pass"])
            self.assertFalse(result["causal_necessity_proven"])

    def test_traced_env_does_not_generate_success_without_actual_step(self):
        import numpy as np
        from types import SimpleNamespace as NS
        class Obj:
            pose=NS(p=np.asarray([[0.0,0.0,0.02]]))
        class Agent:
            tcp_pose=NS(p=np.asarray([[0.0,0.0,0.12]]))
            def is_grasping(self, obj):
                return np.asarray([False])
        class Fake:
            def __init__(self):
                self.unwrapped=NS(obj=Obj(),blocker=Obj(),agent=Agent())
                self.action_space=NS(shape=(7,))
            def step(self, action):
                return None,0,False,False,{"success":False}
        t=TracedEnv(Fake())
        self.assertEqual(t.action_space.shape,(7,))
        t.step(np.zeros(7))
        self.assertEqual(len(t.trace),1)
        self.assertFalse(t.trace[0]["env_success"])


if __name__=="__main__":
    unittest.main()
