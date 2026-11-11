"""Offline 450-step partial-stage recovery test; no ManiSkill / PhysX / GPU."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts import m1_current_closed_pusher_trace_audit as audit


def make_fixture(root):
    closed=root/"closed"
    reference=root/"open_direct"
    closed.mkdir(parents=True)
    reference.mkdir(parents=True)
    goal=[.21,0.,.001]
    n=450
    trace=[]
    for i in range(n):
        cube_x=0. if i<88 else -.05*(i-87)/(n-88)
        blocker_y=0. if i<6 else .129*(min(i-5,82)/82)
        row={
            "t":i,"task_cube":[cube_x,0.,.02],
            "blocker":[.105,blocker_y,.029],
            "tcp":[.0 if i<88 else -.005,0.,.02],
            "gripper_action":-1.,"finger_joint_qpos_m":[0.,0.],
            "is_grasping_task":False,"is_grasping_blocker":False,
            "env_success":False,
        }
        trace.append(row)
    case={
        "kind":"target_closed_mplib",
        "scene":"AGCOFLM1B3Corridor-v0",
        "version":"m1_current_target_mplib_closed_pusher_v2",
        "seed":120,"success":False,"steps":450,
        "error":"RuntimeError: M1B2_STEP_BUDGET_EXCEEDED: no unbounded expert planning",
        "goal":goal,"state_before":[0.,0.,.02],
        "blocker_before":[.1,0.,.029],
        "cube_final":trace[-1]["task_cube"],
        "blocker_final":trace[-1]["blocker"],
        "task_goal_error_m":math.hypot(trace[-1]["task_cube"][0]-.21,
                                      trace[-1]["task_cube"][1]),
        "phases":[{"name":"clear_blocker","actual_steps":82,
                   "actual_error_m":.006324}],
    }
    (closed/"status.json").write_text(json.dumps(case))
    (closed/"physical_trace.json").write_text(json.dumps(trace))
    old_trace=[]
    for i in range(35):
        old_trace.append({
            "t":i,"task_cube":[.21*i/34,0.,.02],
            "blocker":[.10,.125,.029],
            "tcp":[.21*i/34-.05,0.,.02],
            "gripper_action":-1.,"is_grasping_task":False,
        })
    old_status={
        "mode":"open_direct","scene":"AGCOFLM1B3Corridor-v0",
        "seed":120,"steps":35,
        "goal":goal,"state_before":[0.,0.,.02],
        "blocker_before":[.1,.125,.029],
        "gate":{"valid":True},
    }
    (reference/"status.json").write_text(json.dumps(old_status))
    (reference/"physical_trace.json").write_text(json.dumps(old_trace))
    return closed,reference


class ClosedPusherExistingTraceAudit(unittest.TestCase):
    def test_recovers_unfinished_stage_not_in_original_phases(self):
        with tempfile.TemporaryDirectory() as temp:
            c,r=make_fixture(Path(temp))
            result=audit.analyse(c,r)
            self.assertTrue(result["no_new_physics"])
            self.assertEqual(result["initial_close_steps"],6)
            self.assertEqual(result["completed_clear_phase_steps"],82)
            self.assertEqual(result["uncompleted_post_clear_attempt_steps"],362)
            self.assertEqual(result["observed_failure_signature"],
                "CUBE_MOVED_FARTHER_FROM_GOAL_AFTER_CLEARANCE")
            self.assertEqual(
                result["segments"]["after_clearance_attempt"]["steps"],362)
            self.assertAlmostEqual(
                result["segments"]["after_clearance_attempt"]["cube_final_goal_error_m"],.26)
            self.assertEqual(len(result["input_sha256"]["closed_mplib"]
                                     ["trace_sha256"]),64)
            self.assertEqual(result["critical_timeline_indices"]
                             ["first_post_clear_action_index"],88)

    def test_missing_actual_finger_or_blocker_grasp_flags_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            c,r=make_fixture(Path(temp))
            p=c/"physical_trace.json"
            rows=json.loads(p.read_text())
            del rows[100]["is_grasping_blocker"]
            p.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError,"two-object grasp flags"):
                audit.analyse(c,r)

    def test_inconsistent_end_pose_or_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            c,r=make_fixture(Path(temp))
            p=c/"status.json"
            data=json.loads(p.read_text())
            data["cube_final"]=[0.,0.,.02]
            p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"contradict trace"):
                audit.analyse(c,r)

    def test_wrong_source_goal_does_not_become_same_goal_result(self):
        with tempfile.TemporaryDirectory() as temp:
            c,r=make_fixture(Path(temp))
            p=r/"status.json"
            data=json.loads(p.read_text())
            data["goal"][0]=.15
            p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"DIFFERENT final goal"):
                audit.analyse(c,r)

    def test_original_and_output_files_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"analysis.json"
            audit.write_once(p,{"key":1})
            audit.write_once(p,{"key":1})
            with self.assertRaises(FileExistsError):
                audit.write_once(p,{"key":2})


if __name__=="__main__":
    unittest.main()
