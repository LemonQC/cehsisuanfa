"""Stdlib-only M1-B3 real-file diagnosis tests: no ManiSkill or PhysX."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts import m1b3_diagnose_existing as diag


def fixtures(root):
    for mode, n in (("blocked_source",3), ("blocked_direct",2),
                    ("blocked_clear",4), ("open_direct",2)):
        case = root / mode
        case.mkdir(parents=True)
        row = {
            "scene":"AGCOFLM1B3Corridor-v0",
            "mode":mode,"seed":120,
            "same_goal_scene_version":"m1b3_corridor_v0",
            "state_before":[0.0,0.0,0.02],
            "goal":[0.21,0.0,0.001],
            "final":[0.02,0.0,0.02],
            "blocker_before":[0.10,0.125 if mode=="open_direct" else 0.0,0.029],
            "steps":n,"gate":{"valid":mode=="open_direct"},
        }
        if mode == "blocked_source":
            row["error"] = "RuntimeError: official_mplib_lower_failed"
            row["planner_phases"] = [{"name":"lower","planner_success":False}]
        elif mode == "blocked_clear":
            row["gate"]["blocker_cleared_to_side_pocket"] = True
            row["phases"] = [
                {"name":"clear_blocker","steps":2,"success":True,
                 "failure":None,"exhausted":False},
                {"name":"push_cube","steps":2,"success":False,
                 "failure":None,"exhausted":True},
            ]
        else:
            row["phases"] = [
                {"name":"push_cube","steps":n,"success":mode=="open_direct",
                 "failure":None,"exhausted":False},
            ]
        trace = []
        for t in range(n):
            trace.append({
                "t":t,"task_cube":[0.01*t,0.0,0.02],
                "blocker":[0.1,0.04*t if mode=="blocked_clear" else 0.0,0.029],
                "tcp":[0.01*t,0.0,0.12],
                "is_grasping_task":False,"env_success":False,
            })
        (case/"status.json").write_text(json.dumps(row))
        (case/"physical_trace.json").write_text(json.dumps(trace))


class FrozenExistingM1B3Diagnosis(unittest.TestCase):
    def test_existing_real_format_four_case_reads_and_phase_segment_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            fixtures(root)
            report=diag.analyze(root)
            self.assertTrue(report["same_start_and_goal"])
            self.assertEqual(report["cases"]["blocked_clear"]
                             ["physical_phase_segments"][1]["phase"],"push_cube")
            self.assertEqual(report["cases"]["blocked_clear"]
                             ["physical_phase_segments"][1]["measured"]["steps"],2)
            self.assertIn("lower",report["cases"]["blocked_source"]["error"])
            self.assertFalse(report["ran_simulator"])
            self.assertFalse(report["causal_necessity_proven"])
            self.assertIn("blocked_clear",diag.markdown_report(report))
            self.assertEqual(len(report["cases"]["blocked_direct"]
                                 ["original_files"]["trace_sha256"]),64)

    def test_corrupt_or_unmatched_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            fixtures(root)
            path=root/"blocked_direct"/"status.json"
            data=json.loads(path.read_text())
            data["steps"]=9
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"count"):
                diag.analyze(root)

    def test_no_implicit_rerun_or_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"diagnosis.json"
            diag.save_once(path,"first")
            diag.save_once(path,"first")
            with self.assertRaises(FileExistsError):
                diag.save_once(path,"changed")


if __name__ == "__main__":
    unittest.main()
