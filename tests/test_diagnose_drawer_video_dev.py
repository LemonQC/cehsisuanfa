"""Regression tests of immutable 2x2 ObjectInDrawer read-only evidence. No physics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.diagnose_drawer_video_dev import (
    EXPECTED_TASK, validate_source, analyze, range_metrics, main,
)


class DrawerReadOnlyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/"outputs/roboreel_video_dependency_dev/drawer_hf_v1"
        self.root.mkdir(parents=True)
        self.trials=[]
        self.pairs=[]
        for ep in (0, 1):
            lengths={0:800,1:800} if ep==0 else {0:634,1:650}
            pair_actions=[]
            for source in (0,1):
                steps=lengths[source]
                a=np.zeros((steps,8),dtype=np.float32)
                a[:,:7]=0.7
                a[:,7]=0.01
                a[0:20,0]=0.700+source*.001
                a[20:,0]=(0.7+source*.1+ep*.15)
                folder=self.root/f"episode_{ep:02d}"/f"source_{source:04d}"
                folder.mkdir(parents=True)
                f=folder/"actions.npy"
                np.save(f,a)
                pair_actions.append(a)
                self.trials.append({
                    "episode":ep,"source_train_demo_id":source,
                    "actions_path":str(f.resolve()),
                    "actions_sha256":hashlib.sha256(f.read_bytes()).hexdigest(),
                    "steps":steps,"success":ep==1,
                    "reset_seed":310000+ep,"policy_rng_seed":560000+ep,
                    "video_error":None,
                })
            n=min(len(pair_actions[0]),len(pair_actions[1]))
            dist=float(np.linalg.norm(pair_actions[0][:n]-pair_actions[1][:n],axis=1).mean())
            self.pairs.append({
                "episode":ep,"source_ids":[0,1],
                "successes":[ep==1,ep==1],
                "mean_action_l2_first_common_steps":dist,
            })
        self.doc={
            "status":"DEV_COMPLETE_NOT_FORMAL",
            "task":"toys_in_drawer","real_rollouts":4,"paired_episodes":2,
            "trials":self.trials,"pairs":self.pairs,
        }
        (self.root/"results.json").write_text(json.dumps(self.doc))

    def test_accepts_frozen_drawer_pairs_mixed_success_and_lengths(self):
        doc,actions,pairs=validate_source(self.root)
        self.assertEqual(doc["task"],EXPECTED_TASK)
        self.assertEqual(len(actions),4)
        self.assertEqual(len(actions[(1,0)]),634)
        self.assertEqual(len(actions[(1,1)]),650)
        self.assertEqual(len(pairs),2)

    def test_successes_and_first20_are_reported_separately(self):
        doc,actions,pairs=validate_source(self.root)
        r=analyze(doc,actions,pairs)
        self.assertEqual(r["successes"],2)
        self.assertEqual(r["real_rollouts"],4)
        self.assertEqual(r["pairs"][0]["successes_by_source"],[False,False])
        self.assertEqual(r["pairs"][1]["successes_by_source"],[True,True])
        for p in r["pairs"]:
            self.assertIn("1",p["prefix_metrics"])
            self.assertIn("20",p["prefix_metrics"])
            self.assertIn("400",p["prefix_metrics"])
            self.assertGreater(p["prefix_metrics"]["400"]["mean_8d_action_l2"],
                               p["prefix_metrics"]["20"]["mean_8d_action_l2"])

    def test_full_l2_uses_shortest_real_rollout(self):
        doc,actions,pairs=validate_source(self.root)
        r=analyze(doc,actions,pairs)
        row=r["pairs"][1]
        self.assertIn("634",row["prefix_metrics"])
        self.assertEqual(row["prefix_metrics"]["634"]["steps_compared"],634)
        self.assertAlmostEqual(row["mean_8d_action_l2_original"],
                               self.pairs[1]["mean_action_l2_first_common_steps"],places=5)

    def test_bad_action_hash_fails_closed(self):
        target=self.root/"episode_01/source_0001/actions.npy"
        target.write_bytes(target.read_bytes()+b"extra")
        with self.assertRaisesRegex(ValueError,"ACTION_HASH_MISMATCH"):
            validate_source(self.root)

    def test_renamed_task_not_accepted(self):
        doc=self.doc.copy()
        doc["task"]="stack_cups"
        (self.root/"results.json").write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError,"NOT_THE_FROZEN_OFFICIAL"):
            validate_source(self.root)

    def test_mismatched_policy_rng_rejected(self):
        doc=json.loads(json.dumps(self.doc))
        doc["trials"][3]["policy_rng_seed"]=17
        (self.root/"results.json").write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError,"PAIR_SEED_OR_RNG_NOT_MATCHED"):
            validate_source(self.root)

    def test_readonly_cli_does_not_write_into_original(self):
        out=Path(self.tmp.name)/"outputs/roboreel_video_dependency_diagnosis/drawer_hf_v1"
        before={p.relative_to(self.root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.root.rglob("*") if p.is_file()}
        rc=main(["--root",str(self.root),"--out",str(out),"--no-montage"])
        self.assertEqual(rc,0)
        self.assertTrue((out/"diagnosis.json").is_file())
        self.assertTrue((out/"SUMMARY.md").is_file())
        self.assertEqual(json.loads((out/"diagnosis.json").read_text())["successes"],2)
        after={p.relative_to(self.root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
               for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before,after)

    def test_cli_refuses_overwrite_of_completed_diagnosis(self):
        out=Path(self.tmp.name)/"outputs/roboreel_video_dependency_diagnosis/drawer_hf_v1"
        out.mkdir(parents=True)
        (out/"diagnosis.json").write_text("preserved")
        with self.assertRaisesRegex(FileExistsError,"REFUSE_TO_OVERWRITE"):
            main(["--root",str(self.root),"--out",str(out),"--no-montage"])

    def test_l2_metric_separates_gripper_vs_arm(self):
        a=np.zeros((20,8),dtype=np.float32)
        b=a.copy()
        b[:,7]=1
        x=range_metrics(a,b,20)
        self.assertEqual(x["mean_7d_arm_l2"],0)
        self.assertAlmostEqual(x["mean_gripper_abs_difference"],1)
        self.assertAlmostEqual(x["mean_8d_action_l2"],1)


if __name__=="__main__":
    unittest.main()
