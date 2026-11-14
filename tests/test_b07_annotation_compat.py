import importlib
import sys
import types
import unittest
from unittest.mock import patch


class B07DecoderCompatibilityTests(unittest.TestCase):
    def test_existing_decord_is_left_untouched(self):
        fake = types.ModuleType("decord")
        with patch("importlib.util.find_spec", return_value=object()):
            with patch.dict(sys.modules, {"decord": fake}):
                module = importlib.import_module("agcofl.run_b07_annotation_with_cv2")
                self.assertFalse(module.install_cv2_decord_shim())
                self.assertIs(sys.modules["decord"], fake)

    def test_opencv_shim_exposes_decord_video_reader_api(self):
        class FakeCapture:
            def __init__(self, path):
                self.path = path
                self.pos = 0
            def isOpened(self):
                return True
            def get(self, key):
                return 3
            def set(self, key, value):
                self.pos = value
            def read(self):
                return True, self.pos

        fake_cv2 = types.ModuleType("cv2")
        fake_cv2.CAP_PROP_FRAME_COUNT = 1
        fake_cv2.CAP_PROP_POS_FRAMES = 2
        fake_cv2.COLOR_BGR2RGB = 3
        fake_cv2.VideoCapture = FakeCapture
        fake_cv2.cvtColor = lambda frame, code: frame + 10

        with patch("importlib.util.find_spec", return_value=None):
            with patch.dict(sys.modules, {"cv2": fake_cv2}):
                sys.modules.pop("decord", None)
                module = importlib.import_module("agcofl.run_b07_annotation_with_cv2")
                self.assertTrue(module.install_cv2_decord_shim())
                import decord
                reader = decord.VideoReader("synthetic.mp4", ctx=decord.cpu(0))
                self.assertEqual(len(reader), 3)
                self.assertEqual(reader[2].asnumpy(), 12)
                with self.assertRaises(IndexError):
                    reader[3]


if __name__ == "__main__":
    unittest.main()
