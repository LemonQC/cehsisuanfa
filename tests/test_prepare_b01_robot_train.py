"""Offline regressions: B01/B02 shared authenticity/split guard, no GPU or sim."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("b01_prep", Path(__file__).resolve().parents[1] / "scripts/prepare_b01_robot_train.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TrainDataGateTests(unittest.TestCase):
    def _demo(self, root, demo_id, actions=True, video=True):
        import h5py
        import numpy as np
        folder = root / demo_id
        folder.mkdir(parents=True)
        with h5py.File(folder / "trajectory.h5", "w") as f:
            f.create_dataset("traj_0/actions" if actions else "traj_0/observations", data=np.ones((3, 7)))
        for name in ("base_camera.mp4", "hand_camera.mp4"):
            (folder / name).write_bytes(b"\x00\x00\x00\x18ftypisomTEST") if video else (folder / name).write_bytes(b"NOT MP4")

    def test_one_run_with_64_demos_is_deterministic_51_13(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for i in range(64): self._demo(root, f"{i:04d}")
            report = mod.inspect_training_data(root)
            self.assertEqual(report["status"], "TRAIN_SCHEMA_AND_SPLIT_READY_NOT_POLICY_TRAINED")
            self.assertEqual((len(report["train"]), len(report["validation"])), (51, 13))
            self.assertEqual(report["train"], mod.inspect_training_data(root)["train"])
            self.assertFalse(set(report["train"]) & set(report["validation"]))
            path = mod.write_immutable(root, root / "results", report)
            self.assertEqual(path, mod.write_immutable(root, root / "results", report))

    def test_holdout_is_never_read_and_missing_actions_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._demo(root, "0000")
            self._demo(root, "0001", actions=False)
            self._demo(root, "0080")
            report = mod.inspect_training_data(root, required_demos=2)
            self.assertEqual(report["valid_demos"], 1)
            self.assertEqual(report["status"], "INSUFFICIENT_AUTHENTIC_TRAIN_TRIPLETS")
            self.assertEqual({x["reason"] for x in report["rejected"]}, {"ACTION_DATASET_NOT_FOUND", "NOT_TRAIN_ID"})
            self.assertNotIn("0080", report["train"] + report["validation"])

    def test_symlinked_triplets_and_broken_video_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._demo(root, "0000")
            self._demo(root, "0001", video=False)
            (root / "0002").symlink_to(root / "0000", target_is_directory=True)
            report = mod.inspect_training_data(root, required_demos=2)
            self.assertEqual(report["valid_demos"], 1)
            self.assertIn("INVALID_MP4_HEADER", [x["reason"] for x in report["rejected"]])

    def test_missing_root_and_bad_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "missing"
            self.assertEqual(mod.inspect_training_data(root)["status"], "TRAIN_ROOT_MISSING_OR_UNSAFE")
            with self.assertRaises(ValueError): mod.inspect_training_data(root, required_demos=1)


if __name__ == "__main__": unittest.main()
