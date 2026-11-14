import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class B12PreflightContractTests(unittest.TestCase):
    def test_download_is_pinned_and_only_fetches_flow_checkpoint(self):
        script = (ROOT / "scripts/download_b12_flow_checkpoint.sh").read_text()
        self.assertIn("7949df58f6ef302622c9d957b38177c18372bbc8", script)
        self.assertIn("third_party/EC-Flow/ckpt/flow.pt", script)
        self.assertIn("MIN_BYTES=9000000000", script)
        self.assertIn("--continue-at -", script)
        self.assertNotIn("goal_img.pt", script)
        self.assertNotIn("sam2.1_hiera_large.pt", script)

    def test_memory_preflight_is_fp32_and_requires_headroom(self):
        script = (ROOT / "scripts/preflight_b12_flow_model.py").read_text()
        self.assertIn('"precision": "FP32"', script)
        self.assertIn('default=4.0', script)
        self.assertIn('DiT-XL-NoPosEmb-Lang', script)
        self.assertIn('model.load_state_dict(state_dict, strict=True)', script)
        self.assertIn('torch.inference_mode()', script)
        self.assertNotIn('from sam_and_track.utils', script)
        self.assertNotIn('import metaworld', script)

    def test_environment_is_isolated_and_refuses_existing_name(self):
        script = (ROOT / "scripts/create_b12_preflight_env.sh").read_text()
        self.assertIn('ecflow_b12', script)
        self.assertIn('torch==2.4.0 torchvision==0.19.0', script)
        self.assertIn('if conda env list', script)
        self.assertIn('Refusing to modify an existing environment', script)


if __name__ == "__main__":
    unittest.main()
