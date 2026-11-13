"""Offline tests of B06 frozen IDM downloader. No internet, CUDA or simulator."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from scripts.b06_download_pinned_idm import (
    IDM_FILE, IDM_REPO, run, verify_lock,
)

REV = "a" * 40
SIZE = 10 * 1024 * 1024 + 17


class B06IDMDownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dest = self.root / "uniskill_assets"
        self.lock = self.root / "lock.json"
        self.obj = {
            "schema": "B06_HF_SOURCE_LOCK_V1",
            "idm_repo": IDM_REPO,
            "idm_revision": REV,
            "idm_file": IDM_FILE,
            "idm_bytes": SIZE,
        }
        self.lock.write_text(json.dumps(self.obj))
        self.calls = []

    def download(self, **kwargs):
        self.calls.append(kwargs)
        self.assertEqual(kwargs["repo_id"], IDM_REPO)
        self.assertEqual(kwargs["revision"], REV)
        self.assertEqual(kwargs["filename"], IDM_FILE)
        output = Path(kwargs["local_dir"]) / IDM_FILE
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as fh:
            fh.truncate(SIZE)
        return output

    def test_lock_verification(self):
        self.assertEqual(verify_lock(self.obj), (REV, SIZE))

    def test_missing_idm_downloads_only_one_pinned_file(self):
        result = run(lock=self.lock, dest=self.dest, preferred=None,
                     downloader=self.download, sleep=lambda _: None)
        self.assertEqual(result, "downloaded")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual((self.dest / "idm.pth").stat().st_size, SIZE)
        self.assertEqual((self.dest / IDM_FILE).stat().st_size, SIZE)

    def test_existing_idm_is_reused_without_network(self):
        self.dest.mkdir()
        with (self.dest / "idm.pth").open("wb") as fh:
            fh.truncate(SIZE)
        result = run(lock=self.lock, dest=self.dest,
                     preferred="https://hf-mirror.com",
                     downloader=Mock(side_effect=AssertionError("unexpected network")))
        self.assertEqual(result, "reused")

    def test_mirror_503_falls_back_to_official(self):
        def download(**kwargs):
            if kwargs["endpoint"] == "https://hf-mirror.com":
                raise TimeoutError("mirror 503")
            return self.download(**kwargs)
        result = run(lock=self.lock, dest=self.dest,
                     preferred="https://hf-mirror.com",
                     downloader=download, sleep=lambda _: None)
        self.assertEqual(result, "downloaded")
        self.assertEqual(self.calls[0]["endpoint"], "https://huggingface.co")

    def test_incomplete_existing_target_never_overwritten(self):
        self.dest.mkdir()
        (self.dest / "idm.pth").write_bytes(b"preserve user file")
        with self.assertRaises(FileExistsError):
            run(lock=self.lock, dest=self.dest, preferred=None,
                downloader=self.download, sleep=lambda _: None)
        self.assertEqual((self.dest / "idm.pth").read_bytes(), b"preserve user file")

    def test_bad_lock_is_rejected(self):
        for key, value in [
            ("idm_repo", "untrusted/UniSkill"),
            ("idm_file", "../idm.pth"),
            ("idm_revision", "main"),
            ("idm_bytes", 123),
        ]:
            with self.subTest(key=key):
                broken = dict(self.obj)
                broken[key] = value
                with self.assertRaises(ValueError):
                    verify_lock(broken)

    def test_partial_copy_stops_without_removing_it(self):
        self.dest.mkdir()
        partial = self.dest / "idm.pth.b06-copying"
        partial.write_bytes(b"preserve interrupted copy")
        with self.assertRaises(FileExistsError):
            run(lock=self.lock, dest=self.dest, preferred=None,
                downloader=self.download, sleep=lambda _: None)
        self.assertEqual(partial.read_bytes(), b"preserve interrupted copy")


if __name__ == "__main__":
    unittest.main()
