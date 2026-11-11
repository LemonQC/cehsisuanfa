"""Offline M1 Panda sign semantics and true no-grasp push-only gates."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agcofl import m1_current_closed_pusher_ablation as m


class ClosedPusherInterfaceTests(unittest.TestCase):
    def test_published_panda_action_signs_are_not_inverted(self):
        contract=m.gripper_action_semantics()
        self.assertEqual(contract["panda_open_action"],1.0)
        self.assertEqual(contract["panda_closed_action"],-1.0)
        self.assertEqual(contract["previous_successful_clearance_original_command"],-1.0)
        self.assertEqual(contract["previous_failed_mplib_clearance_original_command"],1.0)
        self.assertFalse(contract["legacy_m1b3_gripper_never_closed_label_valid"])
        self.assertFalse(contract["legacy_m1b2_saw_gripper_close_label_valid"])

    def test_closed_panda_gripper_does_not_mean_object_grasp(self):
        trace=[
            {"gripper_action":-1.,"is_grasping_task":False,
             "is_grasping_blocker":False,"finger_joint_qpos_m":[0.003,0.003]},
            {"gripper_action":-1.,"is_grasping_task":False,
             "is_grasping_blocker":False,"finger_joint_qpos_m":[0.002,0.002]},
        ]
        result=m.observed_push_only(trace)
        self.assertTrue(result["valid"])
        self.assertEqual(result["closed_gripper_command_steps"],2)
        self.assertEqual(result["open_gripper_command_steps"],0)
        trace[-1]["is_grasping_blocker"]=True
        self.assertFalse(m.observed_push_only(trace)["valid"])

    def test_empty_observation_cannot_claim_success(self):
        self.assertFalse(m.observed_push_only([])["valid"])

    def test_output_is_separate_from_failed_open_and_old_cases(self):
        self.assertEqual(m.CASE,"target_closed_mplib")
        self.assertEqual(m.VERSION,"m1_current_target_mplib_closed_pusher_v2")
        self.assertIn("m1_current_closed_pusher_v2",m.OUTPUT)
        self.assertNotEqual(m.OUTPUT,"outputs/m1_current_executor_swap_smoke")
        with tempfile.TemporaryDirectory() as d:
            c=Path(d)/m.CASE
            c.mkdir()
            (c/"status.json").write_text("{}")
            with patch.object(m,"geometry_preflight",return_value={"pass":True}):
                with self.assertRaises(FileExistsError):
                    m.run_once("unused",d)


if __name__=="__main__":
    unittest.main()
