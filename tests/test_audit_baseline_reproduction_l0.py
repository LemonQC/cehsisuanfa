"""No network, no GPU, no simulator tests for L0 resource inventory."""
import json
from pathlib import Path
import tempfile
import types
import sys
import unittest
from unittest.mock import patch

from scripts.audit_baseline_reproduction_l0 import (
    ROWS, inventory, official_evidence, write_once, _existing_regular, main)


class BaselineL0Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def test_all_14_unique_in_order(self):
        self.assertEqual([r[0] for r in ROWS], [f"B{i:02d}" for i in range(1,15)])

    def test_all_missing_safe(self):
        r=inventory(self.root)
        self.assertEqual(r["methods_count"],14)
        self.assertEqual(r["existing_official_repro_method_ids"],[])
        self.assertTrue(r["training_or_eval_was_not_executed"])
        self.assertEqual(r["physx_steps"],0)
        self.assertFalse(any(x["all_expected_local_entrypoints_present"] for x in r["methods"]))

    def test_real_entrypoint_presence_not_actual_reproduction(self):
        for x in ROWS[2][3]:
            p=self.root/x
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text("# fake native executable")
        r=inventory(self.root)
        item=next(x for x in r["methods"] if x["id"]=="B03")
        self.assertTrue(item["all_expected_local_entrypoints_present"])
        self.assertFalse(item["native_forward_executed_by_this_audit"])
        self.assertFalse(item["weights_sha_verified_in_this_audit"])

    def test_does_not_escape_project_paths(self):
        self.assertFalse(_existing_regular(self.root,"../../etc/passwd"))
        self.assertFalse(_existing_regular(self.root,"/etc/passwd"))
        (self.root/"inside").symlink_to("/etc")
        self.assertFalse(_existing_regular(self.root,"inside/passwd"))

    def test_strict_official_anchor_positive(self):
        p=self.root/"research_reports/2026-11-11_m0_pi05_uniform/verified_summary.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"status":"ACTUAL_OFFICIAL_EVALUATED_ARCHIVE_READY",
                "pooled_successes":46,"pooled_trials":60,"official_protocol":{"task":"push_cube"}}))
        q=self.root/"research_reports/2026-11-09_v130/official_aggregated_summary.json"
        q.parent.mkdir(parents=True)
        q.write_text(json.dumps({"full_3pass_complete":True,"task":"push_cube",
                "pooled_successes":3,"pooled_trials":60}))
        self.assertEqual([x for x,v in official_evidence(self.root).items() if v],["B03","B05"])
        self.assertEqual(inventory(self.root)["existing_official_repro_method_ids"],["B03","B05"])

    def test_wrong_anchor_cannot_claim_official(self):
        q=self.root/"research_reports/2026-11-09_v130/official_aggregated_summary.json"
        q.parent.mkdir(parents=True)
        q.write_text(json.dumps({"full_3pass_complete":False,"task":"push_cube",
                "pooled_successes":3,"pooled_trials":60}))
        self.assertIsNone(official_evidence(self.root)["B05"])

    def test_idempotent_file_once(self):
        r=inventory(self.root)
        p,data=write_once(self.root,r)
        self.assertEqual(p.read_bytes(),data)
        q,_=write_once(self.root,r)
        self.assertEqual(p,q)

    def test_no_network_or_file_download_default_main(self):
        self.assertEqual(main(["--project",str(self.root)]),0)
        out=list((self.root/"outputs/research_runs/baseline_readiness_l0").glob("*.json"))
        self.assertEqual(len(out),1)
        self.assertEqual(json.loads(out[0].read_text())["physx_steps"],0)

    def test_publisher_is_opt_in_and_uses_evidence_branch(self):
        stub = types.ModuleType("scripts.publish_research_evidence")
        from unittest.mock import Mock
        stub.assert_evidence_branch_exists = Mock(return_value=None)
        stub.write_immutable_report = Mock(return_value="PUBLISHED")
        with patch.dict(sys.modules, {"scripts.publish_research_evidence":stub}):
            self.assertEqual(main(["--project",str(self.root),"--publish"]),0)
        stub.assert_evidence_branch_exists.assert_called_once()
        self.assertEqual(stub.write_immutable_report.call_count,1)
        self.assertIn("evidence/baseline_l0/",stub.write_immutable_report.call_args.args[0])
        self.assertTrue(stub.write_immutable_report.call_args.args[0].endswith("/inventory.json"))


if __name__ == "__main__": unittest.main()
