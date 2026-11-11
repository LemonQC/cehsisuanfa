"""stdlib-only M1-B3 checks; no ManiSkill, CUDA, models, or PhysX."""
import json
import tempfile
import unittest
from pathlib import Path

from agcofl.m1b3_corridor_physics import (
    MODES, BLOCKER_X, PARKING_Y, GOAL_X, GOAL_RADIUS,
    static_walls, _blocked_xy, cspace_reachable, geometry_preflight,
)
from scripts.m1b3_validate_corridor import summarize


def make_row(mode):
    source=mode=="blocked_source"
    return {
        "kind":"source_pick_to_same_goal" if source else mode,
        "mode":mode,"scene":"AGCOFLM1B3Corridor-v0",
        "same_goal_scene_version":"m1b3_corridor_v0",
        "seed":120,
        "state_before":[0.0,0.0,0.02],
        "goal":[GOAL_X,0.0,0.001],
        "final":[GOAL_X,0.0,0.02],
        "blocker_before":[BLOCKER_X,PARKING_Y if mode=="open_direct" else 0.0,0.029],
        "steps":44,
        "gate":{
            "valid":mode!="blocked_direct",
            "goal_error_m":0.003,
            "blocker_moved_m":0.12 if mode=="blocked_clear" else 0.0,
            "blocker_cleared_to_side_pocket":mode=="blocked_clear",
        },
    }


class CorridorPreflightTests(unittest.TestCase):
    def test_fixed_walls_block_stationary_blocker_and_open_passage(self):
        pre=geometry_preflight()
        self.assertTrue(pre["pass"],pre)
        self.assertTrue(all(pre["checks"].values()))
        self.assertFalse(pre["blocked_cspace"]["reachable"])
        self.assertTrue(pre["open_cspace"]["reachable"])
        self.assertEqual(pre["scene"],"AGCOFLM1B3Corridor-v0")
        self.assertEqual(pre["grid_step_m"],.003)

    def test_actual_placement_and_throat_are_size_selective(self):
        names={w["name"] for w in static_walls()}
        self.assertIn("throat_1",names)
        self.assertIn("throat_-1",names)
        self.assertIn("fence_left",names)
        self.assertIn("fence_right",names)
        self.assertTrue(_blocked_xy(BLOCKER_X,0,True))
        self.assertFalse(_blocked_xy(BLOCKER_X,0,False))
        self.assertFalse(_blocked_xy(0,0,True))
        self.assertFalse(_blocked_xy(GOAL_X,0,True))
        self.assertLess(GOAL_RADIUS,.05)

    def test_causal_gate_never_claims_paper_novelty_from_synthetic_records(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            partial=summarize(root)
            self.assertEqual(partial["status"],"INCOMPLETE_REAL_PHYSICS_CASES")
            self.assertFalse(partial["causal_necessity_proven"])
            for mode in MODES:
                file=root/mode/"status.json"
                file.parent.mkdir(parents=True)
                file.write_text(json.dumps(make_row(mode)))
            result=summarize(root)
            self.assertTrue(result["pilot_contrast_pass"])
            self.assertFalse(result["causal_necessity_proven"])
            self.assertFalse(result["strong_tamp_evaluated"])
            self.assertFalse(result["source_video_information_value_measured"])
            self.assertFalse(result["paper_ready"])
            # Counterfactual missing its true parked-blocker state: reject.
            fake=make_row("open_direct")
            fake["blocker_before"]=[BLOCKER_X,0.0,0.029]
            (root/"open_direct/status.json").write_text(json.dumps(fake))
            with self.assertRaisesRegex(ValueError,"counterfactual"):
                summarize(root)

    def test_direct_success_falsifies_insertion_contrast(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for mode in MODES:
                file=root/mode/"status.json"
                file.parent.mkdir(parents=True)
                data=make_row(mode)
                if mode=="blocked_direct":
                    data["gate"]["valid"]=True
                file.write_text(json.dumps(data))
            result=summarize(root)
            self.assertFalse(result["pilot_contrast_pass"])
            self.assertEqual(result["status"],"PHYSICS_CONTRAST_INCONCLUSIVE_OR_FAILED")

    def test_real_execution_failure_never_passes_novelty(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for mode in MODES:
                file=root/mode/"status.json"
                file.parent.mkdir(parents=True)
                data=make_row(mode)
                if mode=="blocked_source":
                    data["status"]="M1B3_EXECUTION_ERROR_INVALID_PHYSICS"
                    data["error"]="some real solver exception"
                    data["gate"]={"valid":False}
                    data["steps"]=0
                file.write_text(json.dumps(data))
            result=summarize(root)
            self.assertFalse(result["pilot_contrast_pass"])
            self.assertFalse(result["causal_necessity_proven"])


if __name__=="__main__":
    unittest.main()
