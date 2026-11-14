import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "agcofl" / "baseline_asset_audit.py"
SPEC = importlib.util.spec_from_file_location("baseline_asset_audit", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class BaselineAssetAuditTests(unittest.TestCase):
    def test_missing_source_is_reported_not_created(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = audit.inspect_method("B11", root, None)
            self.assertEqual(result["source_status"], "SOURCE_NOT_PRESENT_LOCALLY")
            self.assertIsNone(result["local_root"])
            self.assertFalse((root / "third_party").exists())

    def test_required_file_and_asset_checks_are_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for rel in audit.SPECS["B12"]["required"]:
                p = root / rel
                if rel == "metaworld":
                    p.mkdir(parents=True)
                else:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text("stub")
            result = audit.inspect_method("B12", root, root)
            self.assertEqual(result["source_status"], "SOURCE_PRESENT_COMMIT_UNVERIFIED")
            self.assertEqual(result["missing_required"], [])
            self.assertFalse(result["asset_groups_ready"]["flow_checkpoint"])
            self.assertIn("python_modules", result)

    def test_empty_required_file_is_still_present_but_asset_not_ready(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for rel in audit.SPECS["B11"]["required"]:
                p = root / rel
                if rel.endswith("inference.py") or rel.endswith("evaluation.py"):
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text("stub")
                elif rel == "LICENSE" or rel.endswith(".yml"):
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text("stub")
                else:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text("stub")
            result = audit.inspect_method("B11", root, root)
            self.assertEqual(result["source_status"], "SOURCE_PRESENT_COMMIT_UNVERIFIED")
            self.assertFalse(result["asset_groups_ready"]["camera_calibration"])


if __name__ == "__main__":
    unittest.main()
