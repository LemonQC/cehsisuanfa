"""No-GPU/no-network regression tests for M1-C original benchmark pivot."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import m1c_long_horizon_official_dev as m


class OriginalLongHorizonTests(unittest.TestCase):
    def test_only_published_multistage_tasks_allowed(self):
        self.assertEqual(
            m.rel_video("toys_in_drawer"),
            "human_demos/train/no_distraction/toys_in_drawer/0000/front.mp4")
        self.assertNotIn("/eval/",m.rel_video("toys_in_basket"))
        with self.assertRaises(ValueError):
            m.rel_video("pick_push_custom")

    def test_task_checkpoint_must_have_trained_for_requested_task(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            official=root/"official"
            assets=root/"assets"
            ckpt=assets/"released_checkpoints/nd/vid_cond_pi05_uniform/ckpt"
            ckpt.joinpath("params").mkdir(parents=True)
            config=ckpt.parent/"train_config.json"
            config.write_text(json.dumps({"tasks":["push_cube","toys_in_basket",
                                                   "toys_in_drawer"]}))
            for rel in ("src/evaluation/demo_pi05_runner.py",
                        "scripts/evaluation.py",
                        "scripts/compute_vision_features_for_demo.py",
                        "roboreel_env/src/envs/push_cube.py",
                        "roboreel_env/init_states/without_distraction/toys_in_drawer.pkl"):
                f=official/rel
                f.parent.mkdir(parents=True,exist_ok=True)
                f.write_bytes(b"original-code-fixture")
            _,video,feature,_=m.paths(assets,"toys_in_drawer")
            for p in (video,feature):
                p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(b"authentic-content-fixture")
            with patch.object(m,"pinned_sha",return_value=m.PIN):
                record=m.frozen_gate("toys_in_drawer",official,assets)
                self.assertTrue(record["ready_to_develop_on_original_task"])
                self.assertTrue(record["old_m1b_custom_hard_stopped"])
                self.assertFalse(record["physics_executed"])
                self.assertFalse(record["learned_new_method"])
                config.write_text(json.dumps({"tasks":["push_cube"]}))
                rejected=m.frozen_gate("toys_in_drawer",official,assets)
                self.assertIn(
                    "existing_policy_was_trained_for_requested_task",
                    rejected["missing"])

    def test_missing_real_task_assets_never_claim_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch.object(m,"pinned_sha",return_value="badsha"):
                pre=m.frozen_gate("toys_in_basket",root/"none",root/"assets")
            self.assertFalse(pre["ready_to_develop_on_original_task"])
            self.assertIn("one_authentic_human_train_video",pre["missing"])
            self.assertIn("one_original_siglip_feature_bundle",pre["missing"])
            self.assertIn("pinned_official_author_source",pre["missing"])

    def test_original_demo_training_id_does_not_leak_heldout(self):
        self.assertEqual(m.TRAIN_ID,0)
        self.assertIn("/train/",m.rel_video("toys_in_drawer"))
        self.assertFalse(any(f"/{i:04d}/" in m.rel_video("toys_in_drawer")
                             for i in range(80,100)))
        self.assertEqual(m.RELEASED,"vid_cond_pi05_uniform")


if __name__=="__main__":
    unittest.main()
