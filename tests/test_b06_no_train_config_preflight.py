"""Offline B06 preflight regression: official UniSkill has no train_config.json."""
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import roboreel_uniskill_video_dev as runner


class UniSkillNoTrainConfigPreflightTests(unittest.TestCase):
    def test_can_prepare_skill_extraction_without_train_config_json(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td)
            upstream = project / runner.UPSTREAM_REL
            upstream.mkdir(parents=True)
            for rel in runner.REQUIRED_FILES:
                path = upstream / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x" * 200)
            data = project / "outputs/data"
            for demo in ("0000", "0001"):
                video = data / "human_demos/train/no_distraction/stack_cups" / demo / "front.mp4"
                video.parent.mkdir(parents=True, exist_ok=True)
                video.write_bytes(b"v" * 2048)
            policy = project / "outputs/checkpoints/nd/uniskill_policy/ckpt.pth"
            policy.parent.mkdir(parents=True, exist_ok=True)
            with policy.open("wb") as fh:
                fh.truncate(1024 * 1024 + 1)
            self.assertFalse((policy.parent / "train_config.json").exists())
            idm = project / "outputs/uniskill_assets/idm.pth"
            idm.parent.mkdir(parents=True, exist_ok=True)
            with idm.open("wb") as fh:
                fh.truncate(10 * 1024 * 1024 + 1)
            config = upstream / "configs/uniskill.json"
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text('{"policy": {}}')
            torch_fake = ModuleType("torch")
            torch_fake.cuda = SimpleNamespace(
                is_available=lambda: True,
                get_device_name=lambda i: "CPU fake CUDA for unit test",
            )
            def fake_git(_repo, *args):
                return runner.PINNED_UPSTREAM if args[0] == "rev-parse" else "local-git"
            with patch.object(runner, "git_value", side_effect=fake_git), \
                 patch.object(runner, "_package_versions",
                              return_value={name: "installed" for name in runner.REQUIRED_PACKAGES}), \
                 patch.object(runner.sys, "version_info", (3, 12, 0)), \
                 patch.dict(sys.modules, {"torch": torch_fake}):
                result = runner.preflight(
                    project=project,
                    data_dir=data,
                    policy_checkpoint=policy,
                    idm_checkpoint=idm,
                    skill_dir=data / "uniskill_skills",
                    config_path=config,
                    task="stack_cups",
                    source_ids=[0, 1],
                )
            self.assertEqual(result["status"], "READY_FOR_SKILL_EXTRACTION")
            self.assertFalse(result["base_checks"]["policy_train_config_required"])
            self.assertTrue(result["base_checks"]["policy_architecture_config_present"])
            self.assertFalse(result["base_checks"]["skill_arrays"][0]["present"])


if __name__ == "__main__":
    unittest.main()
