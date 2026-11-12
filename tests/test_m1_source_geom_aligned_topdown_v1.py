"""CPU-only original M1-B3 source gripper corridor geometry and stop gates."""
import ast
import inspect
import math
import unittest
import numpy as np
from unittest.mock import patch

from agcofl import m1_source_geom_aligned_topdown_v1 as mod


class GripperGeometrySourceGate(unittest.TestCase):
    def test_real_urdf_screw_mount_exceeds_104mm_corridor_if_fully_open_y(self):
        with patch.object(mod,"geometry_preflight",return_value={"pass":True}):
            p=mod.geometry_contract()
        self.assertAlmostEqual(p["rail_inner_y_m"],.052)
        self.assertAlmostEqual(p["fully_open_screw_mount_outer_radius_m"],.066)
        self.assertAlmostEqual(p["fully_open_if_y_aligned_rail_intrusion_m"],.014)
        self.assertAlmostEqual(p["candidate_screw_mount_outer_radius_m"],.049)
        self.assertAlmostEqual(p["candidate_if_y_aligned_rail_clearance_m"],.003)
        self.assertGreater(p["candidate_tip_opening_m"],p["cube_width_m"])
        self.assertLess(p["candidate_tip_opening_m"],.052)
        self.assertEqual(p["closing_axis_parallel_corridor_longitudinal_x"],[1.,0.,0.])

    def test_panda_actual_normalized_gripper_action_matches_23mm_each(self):
        # Official Panda config is lower=-.010 and upper=.040 meters.
        # q = lower + (a+1)/2*(upper-lower)
        q = -.010 + (mod.PREGRASP_ACTION+1)/2*(.050)
        self.assertAlmostEqual(q,.023)
        self.assertAlmostEqual(mod.PREGRASP_ACTION,.32)
        self.assertLess(mod.MAX_TCP_CUBE_CENTER_DIST_M,.015)
        self.assertEqual(mod.PRECONTACT_ACTION_MAX,155)
        self.assertEqual(mod.FULL_ACTION_MAX,450)

    def test_seeded_collision_surface_point_cloud_has_original_geometry(self):
        a=mod.original_obstacle_points()
        b=mod.original_obstacle_points()
        self.assertEqual(a.ndim,2)
        self.assertEqual(a.shape[1],3)
        np.testing.assert_array_equal(a,b)
        self.assertGreater(len(a),400)
        self.assertLess(len(a),5000)
        self.assertTrue(np.isfinite(a).all())
        self.assertGreaterEqual(a[:,2].min(),0.)
        self.assertLessEqual(a[:,2].max(),.075)
        self.assertAlmostEqual(a[:,0].max(),.297,delta=.001)

    def test_guard_is_physical_not_planner_return(self):
        src=inspect.getsource(mod.execute)
        self.assertIn('planner.use_point_cloud=True',src)
        self.assertIn('MAX_TCP_CUBE_CENTER_DIST_M',src)
        self.assertIn('SOURCE_ACTUAL_TCP_NOT_AT_GRASP_BEFORE_CLOSE',src)
        self.assertLess(src.index('SOURCE_ACTUAL_TCP_NOT_AT_GRASP_BEFORE_CLOSE'),
                        src.index('planner.close_gripper(t=8)'))
        self.assertIn('SOURCE_NO_REAL_BILATERAL_GRASP_ABORT_BEFORE_LIFT',src)
        self.assertIn('SOURCE_CUBE_NOT_LIFTED_ABOVE_ORIGINAL_WALLS',src)
        self.assertIn('SOURCE_CUBE_NOT_CARRIED_TO_SAME_GOAL',src)
        self.assertIn('blocker_shift<.025',src)
        self.assertIn('_actual_satisfied(b)',src)

    def test_new_version_does_not_replace_old_results_or_target_v6(self):
        self.assertEqual(mod.VERSION,"m1_source_geom_aligned_topdown_v1")
        self.assertEqual(mod.CASE,"source_aligned_topdown")
        self.assertNotEqual(mod.OUT,"outputs/m1b3_corridor_dev")
        self.assertNotEqual(mod.OUT,"outputs/m1_current_executor_swap_smoke")
        self.assertNotEqual(mod.OUT,"outputs/m1_original_scene_cartesian_pusher_v6")
        self.assertEqual(mod.SEED,120)


if __name__=="__main__":
    unittest.main()
