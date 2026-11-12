"""CPU only: last existing-M1 target actuator substitution, not a new task."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from agcofl import m1_original_scene_cartesian_pusher_v5 as v


class OriginalSceneCartesianPusherTests(unittest.TestCase):
    def test_original_published_successful_action_semantics_reused(self):
        with patch.object(v,"geometry_preflight",return_value={"pass":True}):
            plan=v.contract()
        self.assertTrue(plan["never_change_original_task_or_geometry"])
        self.assertEqual(plan["source_control_mode"],"pd_ee_delta_pose")
        self.assertEqual(plan["closed_finger_command"],-1.)
        self.assertEqual(plan["goal_x_m"],.21)
        self.assertEqual(plan["strict_stall_max_steps"],15)
        self.assertEqual(v.MAX_STEPS,450)
        self.assertEqual(v.CONTACT_Z,.038)

    def test_original_7d_deltapose_action_no_custom_gain(self):
        class Robot:
            action_space=type("Shape",(),{"shape":(7,)})()
            tcp_pose=None
        # Preflight control shape can be checked without a simulator
        # by the original delta_pose_action directly.
        from agcofl.roboreel_bridge import delta_pose_action
        action=delta_pose_action(
            [-.06,0,.038],[.28,0,.038],(7,))
        self.assertEqual(action.shape,(7,))
        self.assertEqual(action[-1],-1.)
        self.assertGreater(action[0],0.)
        np.testing.assert_allclose(action[3:6],0)

    def test_one_factor_alternative_actuator_original_gripper_closed(self):
        self.assertEqual(v.CASE,"target_cartesian_pusher")
        self.assertNotEqual(v.OUT,"outputs/m1_current_target_reference_height_v4")
        self.assertEqual(v.MAX_PUSH_ACTIONS,120)
        self.assertLessEqual(v.MAX_CLEAR_ACTIONS+3*v.MAX_APPROACH_ACTIONS_PER_STAGE+
                             v.MAX_PUSH_ACTIONS,450)
        self.assertEqual(v.MAX_LATERAL_CUBE_DRIFT,.010)

    def test_immutable_outputs_and_no_physx_if_case_exists(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            d=root/v.CASE
            d.mkdir(parents=True)
            (d/"status.json").write_text("{}")
            with patch.object(v,"geometry_preflight",return_value={"pass":True}):
                with self.assertRaises(FileExistsError):
                    v.run_once(root,"unused")
        # Result is not a new method or validated M1 source.
        with patch.object(v,"geometry_preflight",return_value={"pass":True}):
            pre=v.contract()
            self.assertFalse(pre["strong_tamp_evaluated"])
            self.assertFalse(pre["learned_method_evaluated"])


if __name__=="__main__":
    unittest.main()
