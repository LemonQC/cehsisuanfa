"""Network-free tests of B06 exact revision, per-file retries and safe resume."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import Mock

from scripts.b06_download_pinned_policy import (
    REPO, PREFIX, check_manifest, endpoints, fetch_one, is_complete, run,
)

REV = 'f' * 40


def mk_lock(files):
    return {'schema': 'B06_HF_SOURCE_LOCK_V1',
            'policy_repo': REPO, 'policy_revision': REV, 'policy_files': files}


def entry(name, data):
    return {'filename': PREFIX + name, 'bytes': len(data)}


class B06DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dest = self.root / 'released_checkpoints'
        self.config = b'{"model": "unit"}\n'
        self.train = b'{"train": {}}\n'
        self.weight = b'weight bytes' * 120
        self.contents = {
            PREFIX + 'config.json': self.config,
            PREFIX + 'train_config.json': self.train,
            PREFIX + 'ckpt.pth': self.weight,
        }
        self.lock_path = self.root / 'lock.json'
        self.lock_path.write_text(json.dumps(mk_lock([
            entry(k[len(PREFIX):], v) for k, v in self.contents.items()
        ])))

    def download(self, **kw):
        self.assertEqual(kw['revision'], REV)
        self.assertEqual(kw['repo_id'], REPO)
        name = kw['filename']
        output = Path(kw['local_dir']) / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(self.contents[name])
        return output

    def test_small_json_first_and_pinned_only(self):
        calls = []
        def fn(**kw):
            calls.append(kw['filename'])
            return self.download(**kw)
        result = run(lock=self.lock_path, dest=self.dest, preferred='https://hf-mirror.com',
                     downloader=fn, sleep=lambda _: None)
        self.assertEqual(len(result), 3)
        self.assertEqual(calls[-1], PREFIX + 'ckpt.pth')

    def test_completed_weight_and_json_skipped(self):
        (self.dest / PREFIX).mkdir(parents=True)
        for k, v in self.contents.items():
            (self.dest / k).write_bytes(v)
        result = run(lock=self.lock_path, dest=self.dest, preferred=None,
                     downloader=Mock(side_effect=AssertionError('redownloaded')),
                     sleep=lambda _: None)
        self.assertTrue(all(x == 'reused' for x in result.values()))

    def test_mirror_503_fallback_without_replacing_existing_weight(self):
        # Pretend the large checkpoint was saved before the tiny config failed.
        p = self.dest / PREFIX / 'ckpt.pth'
        p.parent.mkdir(parents=True)
        p.write_bytes(self.weight)
        calls = []
        def fn(**kw):
            calls.append((kw['filename'], kw['endpoint']))
            if kw['filename'].endswith('config.json') and kw['endpoint'].endswith('hf-mirror.com'):
                raise TimeoutError('mirror 503')
            return self.download(**kw)
        result = run(lock=self.lock_path, dest=self.dest, preferred='https://hf-mirror.com',
                     downloader=fn, sleep=lambda _: None)
        self.assertEqual(result[PREFIX + 'ckpt.pth'], 'reused')
        self.assertTrue(any('huggingface.co' in endpoint for _, endpoint in calls))
        self.assertFalse(any(name.endswith('ckpt.pth') for name, _ in calls))

    def test_missing_checkpoint_not_counted_as_complete(self):
        self.assertFalse(is_complete(self.dest / PREFIX / 'ckpt.pth', len(self.weight)))

    def test_wrong_size_final_file_never_overwritten(self):
        p = self.dest / PREFIX / 'ckpt.pth'
        p.parent.mkdir(parents=True)
        p.write_bytes(b'bad')
        with self.assertRaises(FileExistsError):
            run(lock=self.lock_path, dest=self.dest, preferred=None,
                downloader=self.download, sleep=lambda _: None)
        self.assertEqual(p.read_bytes(), b'bad')

    def test_corrupt_same_size_json_never_trusted(self):
        p = self.dest / PREFIX / 'config.json'
        p.parent.mkdir(parents=True)
        p.write_bytes(b'x' * len(self.config))
        self.assertFalse(is_complete(p, len(self.config)))
        with self.assertRaises(FileExistsError):
            run(lock=self.lock_path, dest=self.dest, preferred=None,
                downloader=self.download, sleep=lambda _: None)

    def test_manifest_rejects_wrong_revision_and_traversal(self):
        m = mk_lock([entry('ckpt.pth', self.weight), entry('train_config.json', self.train)])
        m['policy_revision'] = 'main'
        with self.assertRaises(ValueError):
            check_manifest(m)
        m['policy_revision'] = REV
        m['policy_files'][0]['filename'] = PREFIX + '../evil.pth'
        with self.assertRaises(ValueError):
            check_manifest(m)

    def test_official_uniskill_has_no_sibling_train_config(self):
        # The pinned RoboReel UniSkillRunner explicitly documents that
        # its policy checkpoint has no train_config.json sibling.
        # Checkpoint shape/normalization metadata is embedded in ckpt.pth.
        files = [entry('ckpt.pth', self.weight),
                 entry('config.json', self.config)]
        revision, accepted = check_manifest(mk_lock(files))
        self.assertEqual(revision, REV)
        self.assertEqual({x['filename'] for x in accepted},
                         {PREFIX + 'ckpt.pth', PREFIX + 'config.json'})

    def test_missing_actual_policy_checkpoint_is_blocked(self):
        with self.assertRaisesRegex(ValueError, 'B06_POLICY_CHECKPOINT_NOT_IN_LOCK'):
            check_manifest(mk_lock([entry('config.json', self.config)]))

    def test_endpoints_allowlist(self):
        self.assertEqual(endpoints('https://hf-mirror.com'),
                         ('https://hf-mirror.com', 'https://huggingface.co'))
        with self.assertRaises(ValueError):
            endpoints('https://example.com')

    def test_all_sources_fail_preserves_metadata(self):
        p = self.dest / PREFIX / 'config.json'
        def fail(**kw):
            raise TimeoutError('offline')
        with self.assertRaises(RuntimeError):
            run(lock=self.lock_path, dest=self.dest, preferred='https://hf-mirror.com',
                downloader=fail, sleep=lambda _: None)
        self.assertFalse(p.exists())
        self.assertTrue(self.lock_path.exists())

if __name__ == '__main__':
    unittest.main()
