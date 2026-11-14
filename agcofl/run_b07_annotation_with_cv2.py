"""Run the pinned RoboReel local Qwen annotation script.

If Decord is unavailable in the project-local Python (e.g. Python 3.12),
provide a minimal OpenCV-backed VideoReader compatibility shim. This changes
only video decoding, not frame sampling, prompts, model, or annotation logic.
"""
from __future__ import annotations

import importlib.util
import runpy
import sys
import types
from pathlib import Path


def install_cv2_decord_shim() -> bool:
    if importlib.util.find_spec("decord") is not None:
        return False
    import cv2
    import numpy as np

    class _Frame:
        def __init__(self, array):
            self._array = array

        def asnumpy(self):
            return self._array

    class _VideoReader:
        def __init__(self, path, ctx=None):
            self.path = str(path)
            self._cap = cv2.VideoCapture(self.path)
            if not self._cap.isOpened():
                raise RuntimeError(f"OPENCV_VIDEO_OPEN_FAILED: {self.path}")
            self._count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self._count <= 0:
                raise RuntimeError(f"OPENCV_VIDEO_EMPTY_OR_UNKNOWN_FRAME_COUNT: {self.path}")

        def __len__(self):
            return self._count

        def __getitem__(self, index):
            index = int(index)
            if index < 0 or index >= self._count:
                raise IndexError(index)
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = self._cap.read()
            if not ok:
                raise RuntimeError(f"OPENCV_FRAME_DECODE_FAILED: {self.path} frame={index}")
            return _Frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    module = types.ModuleType("decord")
    module.VideoReader = _VideoReader
    module.cpu = lambda device_id=0: None
    sys.modules["decord"] = module
    return True


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m agcofl.run_b07_annotation_with_cv2 <official-script> [args...]")
    script = Path(sys.argv[1]).resolve()
    if not script.is_file():
        raise SystemExit(f"OFFICIAL_ANNOTATION_SCRIPT_MISSING: {script}")
    used_shim = install_cv2_decord_shim()
    print("TRO1_B07_VIDEO_DECODER", "OPENCV_COMPAT_SHIM" if used_shim else "OFFICIAL_DECORD", flush=True)
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
