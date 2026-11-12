"""Test zero-simulator assessment of genuine 65-action M1 source probe."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts import m1_source_65_step_force_vs_planner_audit as a


def old_case(base,force=0.0):
    base.mkdir(parents=True,exist_ok=True)
    trace=[]
    for i in range(65):
        trace.append({
            "t":i,"execution_phase":"set_partial_open" if i<6 else "pregrasp",
            "tcp":[0.0,-0.014,.071],
            "task_cube":[0.0,0.0,.02],
            "finger_joint_qpos_m":[.023,.023],
            "tcp_quaternion_wxyz":[1.0,0.0,0.0,0.0],
            "is_grasping_task":False,
            "tool_static_wall_contact_force_max_n":force,
        })
    status={
       "version":"m1_source_geom_aligned_topdown_v1",
       "kind":"source_aligned_topdown","scene":"AGCOFLM1B3Corridor-v0",
       "seed":120,"success":False,"actual_steps":65,
       "error":"RuntimeError: SOURCE_UPSTREAM_grasp_pose_PLANNING_FAILED",
       "phases":[
         {"phase":"original_static_and_blocker_collision_points","point_count":792},
         {"phase":"set_partial_open","steps":6},
         {"phase":"pregrasp","actual_steps":59,"planner_result":True},
         {"phase":"grasp_pose","actual_steps":0,"planner_result":False}
       ]
    }
    (base/"status.json").write_text(json.dumps(status))
    (base/"physical_trace.json").write_text(json.dumps(trace))
    return base


class Saved65ActionAuditTests(unittest.TestCase):
    def test_old_frozen_planner_failure_separate_from_tool_contact(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=a.analyse(old_case(Path(tmp)/"data"))
            self.assertEqual(r["actual_physx_new_runs"],0)
            self.assertFalse(r["grasp_pose_planner_success"])
            self.assertEqual(r["grasp_pose_executed_actions"],0)
            self.assertEqual(r["physical_actions"],65)
            self.assertAlmostEqual(r["pregrasp_tcp_target_position_error_m"],
                                   (0.014**2+0.009**2)**.5)
            self.assertEqual(r["tool_static_wall_force_max_n"],0)
            self.assertEqual(r["failure_classification"],
                "HAND_FINGER_WALL_CONTACT_NOT_OBSERVED_IN_RECORDED_FORCE")

    def test_actual_tool_wall_contact_is_not_inferred_from_failed_planner(self):
        with tempfile.TemporaryDirectory() as tmp:
            case=old_case(Path(tmp)/"data",force=1.5)
            r=a.analyse(case)
            self.assertEqual(r["tool_static_wall_force_gt_0p5N_count"],65)
            self.assertEqual(r["failure_classification"],
                "OBSERVED_HAND_OR_FINGER_STATIC_WALL_CONTACT")
            self.assertEqual(r["first_significant_tool_wall_contact"]["t"],0)

    def test_missing_contact_forces_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            case=old_case(Path(tmp)/"data")
            path=case/"physical_trace.json"
            rows=json.loads(path.read_text())
            for row in rows:
                row["tool_static_wall_contact_force_max_n"]=None
            path.write_text(json.dumps(rows))
            r=a.analyse(case)
            self.assertIsNone(r["tool_static_wall_force_max_n"])
            self.assertEqual(r["tool_static_wall_contact_force_missing_count"],65)
            self.assertEqual(r["failure_classification"],
                "HAND_FINGER_WALL_CONTACT_FORCES_UNAVAILABLE")

    def test_changed_or_fabricated_result_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            case=old_case(Path(tmp)/"data")
            path=case/"status.json"
            st=json.loads(path.read_text());st["success"]=True
            path.write_text(json.dumps(st))
            with self.assertRaisesRegex(ValueError,"genuine 65-step"):
                a.analyse(case)

    def test_save_is_idempotent_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"summary.json"
            a.save_once(path,{"success":False})
            a.save_once(path,{"success":False})
            with self.assertRaises(FileExistsError):
                a.save_once(path,{"success":True})


if __name__=="__main__":
    unittest.main()
