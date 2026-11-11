"""CPU-only tests for official ManiSkill PickCube expert evidence wrapper."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import m1b_run_official_pick_reference as m


class PickReferenceTests(unittest.TestCase):
    def test_genuine_success_metadata_and_failure_separation(self):
        meta={
            "env_id":"PickCube-v1","source_type":"motionplanning",
            "episodes":[
                {"episode_id":0,"info":{"success":True},"elapsed_steps":80},
                {"episode_id":1,"info":{"success":[False]},"elapsed_steps":110},
            ]}
        rows=m.extract_episodes(meta)
        self.assertEqual([x["success"] for x in rows],[True,False])
        self.assertEqual([x["episode_id"] for x in rows],[0,1])
        self.assertFalse(m.success_scalar("True"))
        meta["episodes"][1]["episode_id"]=0
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            m.extract_episodes(meta)

    def test_not_original_motionplanner_or_missing_label_rejected(self):
        meta={"env_id":"PushCube-v1","source_type":"motionplanning",
              "episodes":[{"episode_id":0,"info":{"success":True}}]}
        with self.assertRaisesRegex(ValueError,"Not the original"):
            m.extract_episodes(meta)
        meta["env_id"]="PickCube-v1"
        meta["source_type"]="custom scripted"
        with self.assertRaisesRegex(ValueError,"official motionplanning"):
            m.extract_episodes(meta)
        meta["source_type"]="motionplanning"
        meta["episodes"][0]["info"]={}
        with self.assertRaisesRegex(ValueError,"missing success"):
            m.extract_episodes(meta)

    def test_subprocess_launch_bounded_no_unlimited_retries_and_no_fake_same_goal(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            calls=[]
            evidence={
                "type":"M1B_OFFICIAL_MANISKILL_PICK_REFERENCE_ONLY",
                "attempts":3,"successful_reference_episodes":2,
                "same_goal_pick_push_pair_validated":False,
                "at_least_one_rgb_video_saved":True,
            }
            def fake_subprocess(command, **kw):
                calls.append(command)
                self.assertNotIn("--only-count-success",command)
                self.assertIn("--save-video",command)
                self.assertEqual(command[command.index("-n")+1],"3")
                self.assertEqual(command[command.index("-b")+1],"cpu")
                self.assertEqual(kw["timeout"],480)
                return mock.Mock(returncode=0)
            with mock.patch.object(m.subprocess,"run",side_effect=fake_subprocess):
                with mock.patch.object(m,"validate_record",return_value=evidence):
                    report=m.run(root,3,480)
                    self.assertFalse(report["same_goal_pick_push_pair_validated"])
            self.assertEqual(len(calls),1)
            self.assertTrue((root/"reference_verification.json").is_file())
            # Same output should only be validated, never re-executed.
            with mock.patch.object(m.subprocess,"run",side_effect=AssertionError("rerun")):
                with mock.patch.object(m,"validate_record",return_value=evidence):
                    m.run(root,3,480)

    def test_bounded_attempts_and_time_budget(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                m.run(Path(d),0,480)
            with self.assertRaises(ValueError):
                m.run(Path(d),3,9999)


if __name__=="__main__":
    unittest.main()
