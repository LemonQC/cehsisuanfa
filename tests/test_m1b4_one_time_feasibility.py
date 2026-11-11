"""CPU-only gate tests for one and only M1B4 rescue, no PhysX."""
import json
import tempfile
import unittest
from pathlib import Path

from agcofl import m1b4_one_time_feasibility as v1


def diagnosis(root, source_held=True, stall=True):
    rows = {
        "blocked_source":{
            "error":"RuntimeError: official_mplib_lower_failed",
            "measured":{
                "task_grasped_steps": 30 if source_held else 0,
                "task_max_z_m": .16 if source_held else .02,
                "task_total_xy_m": .22 if source_held else .00001,
                "task_net_xy_m": .04,
            }
        },
        "blocked_clear":{
            "measured":{
                "task_final_xy_m":[.12527310848236084,-.030821576714515686]
                if stall else [.08,.0],
            },
            "physical_phase_segments":[
                {"phase":"clear_blocker","phase_success":True},
                {"phase":"push_cube","phase_success":False},
            ]
        },
    }
    old={
        "type":"M1B3_EXISTING_PHYSICS_READ_ONLY_DIAGNOSIS",
        "scene":"AGCOFLM1B3Corridor-v0","seed":120,
        "ran_simulator":False,
        "cases":rows,
    }
    root.mkdir(parents=True,exist_ok=True)
    (root/"diagnosis.json").write_text(json.dumps(old))


class OneTimeFeasibilityGate(unittest.TestCase):
    def test_real_saved_throat_stall_geometry_has_justified_margin(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            diagnosis(root)
            report=v1.measured_mechanism_gate(root)
            self.assertTrue(report["stuck_xy_consistent_with_throat_collision"])
            self.assertAlmostEqual(
                report["wall_front_cube_center_x_limit_m"],.126)
            self.assertAlmostEqual(
                report["free_cube_center_half_band_y_m"],.005)
            self.assertTrue(report["geometry_unchanged"])
            self.assertTrue(report["source_lower_planner_fallback_is_justified"])
            self.assertGreater(report["source_actual_grasp_steps"],0)

    def test_if_original_pick_never_lifted_then_no_more_physx(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            diagnosis(root,source_held=False)
            report=v1.measured_mechanism_gate(root)
            self.assertFalse(report["source_lower_planner_fallback_is_justified"])
            # No programmatically launched robot in this negative condition.
            with self.assertRaises(SystemExit) as error:
                v1.main(["--old-root",str(root),
                         "--require-physical-source-evidence"])
            self.assertEqual(error.exception.code,6)

    def test_unmatched_throat_stall_refuses_controller_rescue(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            diagnosis(root,stall=False)
            with self.assertRaisesRegex(ValueError,"NOT close"):
                v1.measured_mechanism_gate(root)

    def test_unaligned_throat_guard_matches_measured_geometry(self):
        self.assertEqual(v1.pre_throat_lateral_alignment_needed(
            [.1252731,-.0308216,.02]),"MISALIGNED_AT_WALL_FRONT")
        self.assertFalse(v1.pre_throat_lateral_alignment_needed(
            [.105,.002,.02]))
        self.assertTrue(v1.pre_throat_lateral_alignment_needed(
            [.105,.015,.02]))


if __name__=="__main__":
    unittest.main()
