import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.roboreel_uniskill_video_dev import (
    action_difference,
    safe_regular,
    save_json_once,
    save_progress_json,
    validate_skill_array,
    validate_train_ids,
)


class UniSkillVideoDevTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_only_two_distinct_train_ids(self):
        self.assertEqual(validate_train_ids([0, 1]), [0, 1])
        for bad in ([0, 0], [0, 80], [80, 81], [0], [0, 1, 2]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_train_ids(bad)

    def test_skill_shape_and_finiteness(self):
        meta = validate_skill_array(np.zeros((100, 1, 64), dtype=np.float32))
        self.assertEqual(meta["shape"], [100, 1, 64])
        with self.assertRaises(ValueError):
            validate_skill_array(np.zeros((100, 64), dtype=np.float32))
        arr = np.zeros((100, 1, 64), dtype=np.float32)
        arr[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_skill_array(arr)

    def test_action_difference(self):
        a = np.zeros((20, 8), dtype=np.float32).tolist()
        b = np.ones((20, 8), dtype=np.float32).tolist()
        self.assertAlmostEqual(action_difference(a, b), np.sqrt(8), places=6)
        self.assertIsNone(action_difference([], b))
        with self.assertRaises(ValueError):
            action_difference([[0.0] * 7], [[0.0] * 7])

    def test_save_json_once_is_immutable(self):
        p = self.root / "once.json"
        save_json_once(p, {"a": 1})
        save_json_once(p, {"a": 1})
        with self.assertRaises(FileExistsError):
            save_json_once(p, {"a": 2})

    def test_progress_file_only_appends(self):
        p = self.root / "partial.json"
        save_progress_json(p, {"trials": [{"episode": 0}]})
        save_progress_json(p, {"trials": [{"episode": 0}, {"episode": 1}]})
        with self.assertRaises(FileExistsError):
            save_progress_json(p, {"trials": [{"episode": 1}]})

    def test_progress_refuses_symlink(self):
        target = self.root / "target.json"
        target.write_text('{"trials": []}\n')
        link = self.root / "link.json"
        link.symlink_to(target)
        with self.assertRaises(FileExistsError):
            save_progress_json(link, {"trials": [{"episode": 0}]})

    def test_safe_regular_rejects_symlink_and_tiny(self):
        f = self.root / "file.bin"
        f.write_bytes(b"abc")
        self.assertTrue(safe_regular(f, 3))
        self.assertFalse(safe_regular(f, 4))
        link = self.root / "link.bin"
        link.symlink_to(f)
        self.assertFalse(safe_regular(link, 1))


if __name__ == "__main__":
    unittest.main()
