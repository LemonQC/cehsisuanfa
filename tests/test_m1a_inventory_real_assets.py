"""No-GPU, no-network unittest for M1-A one-shot evidence inventory."""
import json
import tempfile
import unittest
from pathlib import Path
from scripts.m1a_inventory_real_assets import audit, local_videos, local_robot_actions


class M1AssetInventoryTests(unittest.TestCase):
    def test_only_genuine_local_train_video_not_eval(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            train=root/"outputs/next_gate_v100/train_data/hf_cache/human_demos/train/no_distraction/pick_cube/0000/front.mp4"
            eval_file=root/"outputs/next_gate_v100/train_data/hf_cache/human_demos/eval/no_distraction/pick_cube/0080/front.mp4"
            train.parent.mkdir(parents=True)
            eval_file.parent.mkdir(parents=True)
            train.write_bytes(b"v"*2048)
            eval_file.write_bytes(b"e"*2048)
            found=local_videos(root,"pick_cube")
            self.assertEqual(len(found),1)
            self.assertIn("/train/", "/"+found[0]["path"])
            self.assertNotIn("/eval/","/"+found[0]["path"])

    def test_actual_local_robot_files_required_not_just_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            manifest=root/"outputs/v110_baselines/robot_data_manifest.json"
            manifest.parent.mkdir(parents=True)
            items=[f"robot_trajectories/no_distraction/push_cube/0000/{name}"
                   for name in ("trajectory.h5","base_camera.mp4","hand_camera.mp4")]
            manifest.write_text(json.dumps({"downloaded":[{
                "demo":"0000","files":[{"relative_path":i} for i in items]}]}))
            self.assertEqual(local_robot_actions(root),[])
            for item in items:
                path=root/"outputs/v110_baselines/official_dataset"/item
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b"real-but-not-h5-inspected-in-this-test")
            found=local_robot_actions(root)
            self.assertEqual(len(found),1)
            self.assertEqual(found[0]["demo"],"0000")


    def test_v120_executed_robot_train_dir_not_official_eval_dir(self):
        """Regression: actual v120 run stored TRAIN H5 in robot_train/."""
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            manifest=root/"outputs/v120_baselines/robot_data_manifest.json"
            manifest.parent.mkdir(parents=True)
            records=[]
            for i in (0,1):
                paths=[f"robot_trajectories/no_distraction/push_cube/000{i}/{name}"
                       for name in ("raw.h5","base_camera.mp4","hand_camera.mp4")]
                records.append({"demo":f"000{i}",
                                "files":[{"relative_path":p} for p in paths]})
                for item in paths:
                    full=root/"outputs/v120_baselines/robot_train"/item
                    full.parent.mkdir(parents=True,exist_ok=True)
                    full.write_bytes(b"raw data stub")
            manifest.write_text(json.dumps({"status":"REAL_ROBOT_TRAIN_DATA_ACQUIRED",
                                            "downloaded":records}))
            candidates=local_robot_actions(root)
            self.assertEqual(len(candidates),2)
            self.assertEqual({v["demo"] for v in candidates},{"0000","0001"})
            self.assertTrue(all(v["data_root"]=="outputs/v120_baselines/robot_train"
                                for v in candidates))

    def test_m0_success_does_not_fabricate_m1_same_goal_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            official=root/"research_reports/2026-11-11_m0_pi05_uniform/verified_summary.json"
            official.parent.mkdir(parents=True)
            official.write_text(json.dumps({
               "type":"M0_AUTHOR_PI05_UNIFORM_ND_PUSHCUBE_VERIFIED_60_TRIALS",
               "pooled_successes":46,"pooled_trials":60}))
            report=audit(root)
            self.assertTrue(report["m0_official_pi05_archived_and_verified"])
            self.assertFalse(report["source_pick_to_same_final_goal_physically_verified"])
            self.assertFalse(report["push_only_target_blocker_insertion_physically_verified"])
            self.assertFalse(report["matched_goal_policy_baselines_executed"])
            self.assertEqual(report["status"],
                             "M1_SAME_GOAL_PAIR_AND_INSERTION_STILL_NOT_VERIFIED")


if __name__=="__main__":
    unittest.main()
