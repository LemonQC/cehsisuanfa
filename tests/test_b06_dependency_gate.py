"""B06 dependency gate: offline pip-check classification only, no GPU/net."""
import unittest
from scripts.b06_dependency_gate import classify_pip_check

class B06DependencyGateTests(unittest.TestCase):
    def test_user_actual_pip_check_warnings_are_nonfatal(self):
        actual="\n".join([
            "uniskill-policy 0.1.0 has requirement torch==2.0.1, but you have torch 2.10.0+cu128.",
            "uniskill-policy 0.1.0 has requirement torchvision==0.15.2, but you have torchvision 0.25.0+cu128.",
            "uniskill-policy 0.1.0 has requirement transformers==4.36.2, but you have transformers 5.3.0.",
            "decord 0.6.0 is not supported on this platform",
        ])
        tolerated,blocked=classify_pip_check(actual,1)
        self.assertEqual(len(tolerated),4)
        self.assertEqual(blocked,[])
    def test_legacy_optional_dependencies_do_not_fail(self):
        tolerated,blocked=classify_pip_check(
            "uniskill-policy 0.1.0 requires libero, which is not installed.",1)
        self.assertEqual(len(tolerated),1)
        self.assertEqual(blocked,[])
    def test_other_packages_still_block(self):
        tolerated,blocked=classify_pip_check(
            "scipy 1.17.0 requires numpy>=2.3, but you have numpy 1.0.",1)
        self.assertEqual(tolerated,[])
        self.assertEqual(len(blocked),1)
    def test_empty_error_is_blocked(self):
        self.assertEqual(classify_pip_check("",1)[1],
                         ["PIP_CHECK_FAILED_WITHOUT_DIAGNOSTIC"])
    def test_healthy_pip_check(self):
        self.assertEqual(classify_pip_check("No broken requirements found.\n",0),
                         ([],[]))

if __name__=="__main__":
    unittest.main()
