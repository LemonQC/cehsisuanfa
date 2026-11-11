"""Offline M1 source-contact comparison tests, no ManiSkill/GPU/network."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts import m1_current_compare_grasp_contact as m


def fixture(case,old):
    case.mkdir(parents=True)
    trace=[]
    for i in range(9):
        trace.append({
            "t":i,
            "tcp":[-0.04+0.005*i,0.0,0.02],
            "task_cube":[0.,0.,0.02 if not old else 0.02+0.005*max(0,i-5)],
            "blocker":[0.1,0.,0.029],
            "gripper_action":1. if i<3 else -1.,
            "is_grasping_task": bool(old and i>=5),
            "env_success":False,
        })
    status={
        "kind":"source_pick_to_same_goal" if old else "source_side_grasp",
        "steps":9,
        "goal":[.15 if old else .21,0.,.001],
        "planner_phases":[] if old else [{
            "stage":"physical_grasp_check","actual_steps":9,
            "actual_is_grasping":False,
        }],
    }
    if old:
        status["gate"]={"valid":True}
    else:
        status.update(version="m1_current_task_official_executor_swap_v1",
                      original_geometry=True,seed=120,success=False)
    (case/"status.json").write_text(json.dumps(status))
    (case/"physical_trace.json").write_text(json.dumps(trace))


class OfflineContactComparison(unittest.TestCase):
    def test_compare_reports_real_close_with_no_contact_and_old_grasp(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            fixture(root/"new",old=False)
            fixture(root/"old",old=True)
            d=m.compare(root/"new",root/"old")
            self.assertEqual(d["new_failed_source"]["actual_grasp_steps"],0)
            self.assertEqual(d["new_failed_source"]["gripper_close_command_steps"],6)
            self.assertGreater(d["previously_successful_reference"]["actual_grasp_steps"],0)
            self.assertNotEqual(
                d["current_failed_scene_goal_x"],d["old_success_scene_goal_x"])
            self.assertEqual(d["physical_simulator_runs"],0)
            self.assertEqual(len(d["input_files"]["current"]["status_sha256"]),64)

    def test_missing_or_fake_success_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            fixture(root/"new",False)
            fixture(root/"old",True)
            p=root/"old/status.json"
            d=json.loads(p.read_text())
            d["gate"]["valid"]=False
            p.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,"not proven successful"):
                m.compare(root/"new",root/"old")

    def test_bad_physics_trace_fails_without_spawning_scenario(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            fixture(root/"new",False)
            fixture(root/"old",True)
            p=root/"new/physical_trace.json"
            t=json.loads(p.read_text())
            t[5]["t"]=100
            p.write_text(json.dumps(t))
            with self.assertRaisesRegex(ValueError,"inconsistent"):
                m.compare(root/"new",root/"old")

    def test_offline_output_never_overwrites_changed_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"audit.json"
            m.write_once(p,"A")
            m.write_once(p,"A")
            with self.assertRaises(FileExistsError):
                m.write_once(p,"B")


if __name__=="__main__":
    unittest.main()
