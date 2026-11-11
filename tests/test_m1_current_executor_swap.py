"""No simulator: enforce same frozen scene, original Panda action semantics."""
import json
import tempfile
import unittest
from pathlib import Path

from agcofl import m1_current_executor_swap as m


class CurrentTaskExecutorReplacement(unittest.TestCase):
    def test_swap_keeps_original_obstacle_scene_and_goal(self):
        contract=m.geometry_contract()
        self.assertTrue(contract["old_scene_geometry_changed"] is False)
        self.assertEqual(contract["scene"],"AGCOFLM1B3Corridor-v0")
        self.assertEqual(contract["source_approaching_xyz"],[1.0,0.0,0.0])
        self.assertEqual(contract["source_target_closing_axis"],[0.0,1.0,0.0])
        self.assertEqual(contract["source_goal_x_m"],.21)

    def test_target_jointspace_gripper_is_open_not_original_closed(self):
        self.assertEqual(m.VERSION,"m1_current_task_official_executor_swap_v1")
        self.assertEqual(m.MODES,("source_side_grasp","target_official_push"))
        self.assertEqual(m.geometry_contract()["target_gripper_action"],1.0)
        self.assertTrue(m.geometry_contract()["target_gripper_never_grasps"])

    def test_no_overwrite_of_any_existing_new_smoke(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            case=root/"source_side_grasp"
            case.mkdir()
            (case/"status.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                m.run_one("source_side_grasp",root,"unused_official_root")

    def test_wrong_mode_cannot_start_simulator(self):
        with self.assertRaises(ValueError):
            m.run_one("old_scene_rerun",Path("unused"),"unused")


if __name__=="__main__":
    unittest.main()
