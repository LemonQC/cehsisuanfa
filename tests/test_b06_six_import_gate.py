"""Static offline regression for B06 runtime import gate; no GPU/network."""
from pathlib import Path
import ast
import unittest

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "scripts/setup_b06_uniskill_stackcups.sh"
FAILURE = ROOT / "scripts/b06_failure_record.py"


class B06SixImportGateTests(unittest.TestCase):
    def test_exact_project_env_is_used_and_older_pins_unmodified(self):
        sh = SETUP.read_text()
        self.assertIn('"six==1.17.0"', sh)
        self.assertIn('if ! "$CONDA_ENV/bin/python" -c \'import six\'', sh)
        self.assertIn('"$CONDA_ENV/bin/python" -m pip install --disable-pip-version-check', sh)
        self.assertIn('"torch==2.10.0"', sh)
        self.assertIn('"transformers==5.3.0"', sh)
        self.assertNotIn("conda remove", sh)
        self.assertNotIn("pip install --upgrade torch", sh)

    def test_import_gate_before_weight_download_and_physx(self):
        sh = SETUP.read_text()
        mark = sh.index('STAGE="uniskill_policy_import_gate"')
        self.assertLess(sh.index('export PYTHONPATH='), mark)
        self.assertLess(mark, sh.index('STAGE="pin_hf_revisions_and_disk_gate"'))
        # This stage string also appears in the early partial-result guard;
        # compare with the final launch assignment, not the guard.
        self.assertLess(mark, sh.rindex('STAGE="real_physx_bounded_dev_and_publish"'))
        self.assertIn('from robomimic.config import config_factory', sh)
        self.assertIn('B06_POLICY_CONFIG_IMPORT_OK', sh)
        # The import is always checked even if environment marker is present.
        self.assertLess(sh.index('fi\n\nexport PYTHONPATH='), mark)

    def test_failure_evidence_has_pre_physx_stage(self):
        failure = FAILURE.read_text()
        ast.parse(failure)
        self.assertIn("'uniskill_policy_import_gate'", failure)
        self.assertIn("'uniskill_runner_init_pre_physx'", failure)
        self.assertIn("if stage == 'real_physx_bounded_dev_and_publish'", failure)
        setup = SETUP.read_text()
        self.assertIn('if [[ "$STAGE" == "real_physx_bounded_dev_and_publish" && ! -e "$RESULT_DIR" ]]', setup)
        self.assertIn('STAGE="uniskill_runner_init_pre_physx"', setup)


if __name__ == "__main__":
    unittest.main()
