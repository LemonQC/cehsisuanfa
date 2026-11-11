"""No-PhysX regression of existing-M1 target measured +X safe handoff."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from agcofl import m1_current_target_safe_approach_v3 as v


class SafeRearApproachTests(unittest.TestCase):
    def test_hover_rear_descend_instead_of_crossing_cube_at_table_height(self):
        cube=np.array([0.,0.,.02])
        tcp=np.array([.105,.129,.029])
        q=np.array([1.,0.,0.,0.])
        wp=v.safe_approach_waypoints(cube,tcp,q)
        self.assertEqual([w[0] for w in wp],[
            "retreat_vertical","overhead_to_back","descend_behind_cube"])
        self.assertAlmostEqual(wp[0][1][2],.160)
        self.assertAlmostEqual(wp[1][1][2],.160)
        self.assertAlmostEqual(wp[1][1][0],-.065)
        self.assertAlmostEqual(wp[2][1][0],-.065)
        self.assertAlmostEqual(wp[2][1][2],.02)
        self.assertTrue(all(w[1][0]<0 for w in wp[1:]))
        self.assertGreater(wp[0][1][2],.075+.02+.02)

    def test_actual_backward_50mm_rejected_during_precontact(self):
        before=np.array([0.,0.,.02])
        backwards=np.array([-.050044,.00167,.02])
        self.assertEqual(v.phase_violation(
            "overhead_to_back",backwards,before,0),
            "APPROACH_MOVED_CUBE_BACKWARDS_BEFORE_PUSH")
        self.assertEqual(v.phase_violation(
            "push_1_x_0.110",backwards,before,0),
            "TASK_PUSH_MOVED_REVERSE_X")

    def test_valid_centered_forward_progress_and_off_center_are_distinct(self):
        start=[0.,0.,.02]
        self.assertIsNone(v.phase_violation(
            "push_1_x_0.110",[.07,.002,.02],start,.07))
        self.assertEqual(v.phase_violation(
            "push_2_x_0.180",[.11,.011,.02],start,.11),
            "TASK_PUSH_DEVIATED_LATERALLY")
        self.assertEqual(v.phase_violation(
            "push_2_x_0.180",[.11,.007,.02],start,.11),
            "TASK_OFF_CENTER_BEFORE_NARROW_THROAT")
        self.assertEqual(v.phase_violation(
            "push_3_x_0.260",[.11,.0,.02],start,.13),
            "TASK_PUSH_LOST_FORWARD_PROGRESS")

    def test_refuse_nonphysical_jump_or_changed_original_goal(self):
        self.assertEqual(v.phase_violation(
            "descend_behind_cube",[.02,0.,.02],[0.,0.,.02],0),
            "APPROACH_DISTURBED_TASK_BEFORE_CONTACT")
        self.assertEqual(v.phase_violation(
            "push_1_x_0.110",[.05,0.,.06],[0.,0.,.02],.05),
            "TASK_CUBE_LIFTED_DISALLOWED_PUSH_ONLY")
        with self.assertRaisesRegex(ValueError,"original final goal"):
            v.safe_approach_waypoints(
                [0.,0.,.02],[.1,.125,.02],[1.,0.,0.,0.],
                final_goal_x=.15)

    def test_original_corridor_geometry_and_all_outputs_immutable(self):
        with patch.object(v,"geometry_preflight",return_value={"pass":True}):
            plan=v.direction_contract()
            self.assertTrue(plan["precontact_matches_old_wrong_way_cube_x_within_1mm"])
            self.assertTrue(plan["original_geometry_unchanged"])
            self.assertEqual(plan["goal_x_m"],.21)
            self.assertEqual(plan["new_action_sequence"][-1],"push_tcp_x_.260")
            with tempfile.TemporaryDirectory() as temp:
                case=Path(temp)/v.CASE
                case.mkdir(parents=True)
                (case/"status.json").write_text("{}")
                with self.assertRaises(FileExistsError):
                    v.run_once(temp,"unused")


if __name__=="__main__":
    unittest.main()
