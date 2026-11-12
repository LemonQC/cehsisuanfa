"""No simulator M1 v4 single-factor contact-height correction tests."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from agcofl import m1_current_target_reference_height_v4 as v
from agcofl import m1_current_target_safe_approach_v3 as previous


class ReferenceGroundedContactHeight(unittest.TestCase):
    def test_only_descend_target_height_changes_from_v3(self):
        cube=[0.,0.,.02]
        tcp=[.110,.086,.044]
        q=[1.,0.,0.,0.]
        old=previous.safe_approach_waypoints(cube,tcp,q)
        new=v.safe_approach_waypoints(cube,tcp,q)
        self.assertEqual([p[0] for p in old],[p[0] for p in new])
        for i in (0,1):
            np.testing.assert_allclose(old[i][1],new[i][1])
        self.assertAlmostEqual(old[2][1][0],new[2][1][0])
        self.assertAlmostEqual(old[2][1][1],new[2][1][1])
        self.assertAlmostEqual(old[2][1][2],.020)
        self.assertAlmostEqual(new[2][1][2],.038)
        self.assertGreater(v.HOVER_Z,.145)
        self.assertEqual(v.PUSH_SEGMENT_TCP_X,previous.PUSH_SEGMENT_TCP_X)

    def test_measured_success_and_failed_tcp_heights_in_same_physical_envelope(self):
        z_success=.037377677857875824
        z_v3=.04128366708755493
        for z in (z_success,z_v3):
            self.assertLessEqual(v.CONTACT_TCP_ACCEPT_Z_MIN,z)
            self.assertLessEqual(z,v.CONTACT_TCP_ACCEPT_Z_MAX)
        self.assertTrue(v.CONTACT_TCP_ACCEPT_Z_MIN <= .038 <= v.CONTACT_TCP_ACCEPT_Z_MAX)
        self.assertFalse(v.CONTACT_TCP_ACCEPT_Z_MIN<=.020<=v.CONTACT_TCP_ACCEPT_Z_MAX)

    def test_physx_backward_and_lateral_guards_unchanged(self):
        start=[0.,0.,.02]
        self.assertEqual(v.phase_violation(
            "push_1_x_0.110",[-.050044,.00167,.02],start,0),
            "TASK_PUSH_MOVED_REVERSE_X")
        self.assertEqual(v.phase_violation(
            "push_2_x_0.180",[.110,.011,.02],start,.110),
            "TASK_PUSH_DEVIATED_LATERALLY")
        self.assertEqual(v.phase_violation(
            "push_2_x_0.180",[.110,.007,.02],start,.110),
            "TASK_OFF_CENTER_BEFORE_NARROW_THROAT")

    def test_same_goal_scene_and_immutable_version(self):
        with patch.object(v,"geometry_preflight",return_value={"pass":True}):
            pre=v.direction_contract()
            self.assertEqual(pre["goal_x_m"],.21)
            self.assertTrue(pre["original_geometry_unchanged"])
            self.assertEqual(pre["reference_based_tcp_target_z_m"],.038)
            self.assertTrue(pre["original_gripper_closed"])
            with tempfile.TemporaryDirectory() as tmp:
                case=Path(tmp)/v.CASE
                case.mkdir()
                (case/"status.json").write_text("{}")
                with self.assertRaises(FileExistsError):
                    v.run_once(tmp,"unused")
        self.assertNotEqual(v.CASE,previous.CASE)
        self.assertNotEqual(v.OUT,previous.OUT)
        self.assertEqual(v.VERSION,"m1_current_target_reference_height_v4")


if __name__=="__main__":
    unittest.main()
