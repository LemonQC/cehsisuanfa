"""No PhysX test: v6 fixes ONLY proved premature rear-contact stop."""
import inspect
import unittest
import numpy as np
from agcofl import m1_original_scene_cartesian_pusher_v5 as old
from agcofl import m1_original_scene_cartesian_pusher_v6 as fixed


class M1OriginalScenePrematureDescentTests(unittest.TestCase):
    def test_measured_v5_descent_error_is_accepted_by_v5_not_v6(self):
        wanted=np.array([-.060, 0.0, .038])
        actual=np.array([-.059219107031822205,
                         .00019498061737976968,
                         .056888557970523834])
        error=float(np.linalg.norm(actual-wanted))
        self.assertLess(error,.022)             # v5 incorrectly ended
        self.assertGreater(error,fixed.DESCEND_REAR_TOLERANCE_M)
        self.assertAlmostEqual(error,.01891,delta=.0005)
        self.assertGreater(actual[2],.052)      # push_to_goal gate rejected

    def test_v6_reuses_original_successful_push_controller_and_task(self):
        self.assertEqual(fixed.CONTACT_Z,old.CONTACT_Z)
        self.assertEqual(fixed.CONTACT_Z,.038)
        self.assertEqual(fixed.MAX_STEPS,old.MAX_STEPS)
        self.assertEqual(fixed.MAX_PUSH_ACTIONS,old.MAX_PUSH_ACTIONS)
        self.assertEqual(fixed.BEHIND_X_OFFSET,old.BEHIND_X_OFFSET)
        self.assertEqual(fixed.MAX_CLEAR_ACTIONS,old.MAX_CLEAR_ACTIONS)
        self.assertIs(fixed.ActorPush,old.ActorPush)
        self.assertIs(fixed.delta_pose_action,old.delta_pose_action)
        self.assertEqual(fixed.CASE,"target_cartesian_contact_fixed")
        self.assertNotEqual(fixed.OUT,old.OUT)

    def test_descent_stage_uses_stricter_tolerance_but_others_unchanged(self):
        old_source=inspect.getsource(old.execute)
        new_source=inspect.getsource(fixed.execute)
        self.assertIn('tolerance=.022',old_source)
        self.assertIn("tolerance=DESCEND_REAR_TOLERANCE_M",new_source)
        self.assertEqual(fixed.DESCEND_REAR_TOLERANCE_M,.008)
        self.assertIn('motion_to(traced,"retreat_vertical"',new_source)
        self.assertIn('motion_to(traced,"hover_rear"',new_source)
        self.assertIn('phases.append(push_to_goal(traced,start,goal))',new_source)

    def test_approach_failed_result_not_credited_as_success(self):
        # Thresholds retain the ORIGINAL physical checks,
        # cannot pass by merely reducing the acceptance tolerance.
        self.assertEqual(fixed.GOAL_RADIUS,.020)
        self.assertEqual(fixed.VERSION,"m1_original_scene_cartesian_pusher_v6")
        self.assertTrue("pd_ee_delta_pose" in inspect.getsource(fixed.execute))
        self.assertIn('goal_error<GOAL_RADIUS and _actual_satisfied(b)',
                      inspect.getsource(fixed.execute))


if __name__=="__main__":
    unittest.main()
