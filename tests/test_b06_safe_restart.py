"""Only CPU tests: no HF, no GPU, no actual GitHub writes."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'scripts'

def load(name):
    path=SOURCE/(name+'.py')
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class B06SafeRestartTests(unittest.TestCase):
    def test_project_local_install_and_bounded_source(self):
        text=(SOURCE/'setup_b06_uniskill_stackcups.sh').read_text()
        self.assertIn('CONDA_ENV="$BASE/conda_envs/uniskill"',text)
        self.assertIn('export CONDA_PKGS_DIRS="$BASE/conda_pkgs"',text)
        self.assertIn('export PIP_CACHE_DIR="$BASE/pip_cache"',text)
        self.assertIn('export TMPDIR="$BASE/tmp"',text)
        self.assertIn('conda create -p "$CONDA_ENV"',text)
        self.assertIn('--data-root "$SUBSET"',text)
        self.assertIn('ln -s "$original" "$shortcut"',text)
        self.assertNotIn('conda create -n uniskill',text)
        self.assertNotIn('--data-root "$DATA" --output-root "$DATA/uniskill_skills"',text)
        self.assertIn('--revision "$ROBO_REV"',text)
        self.assertIn('revision=os.environ["IDM_REV"]',text)
        self.assertIn('republish_completed_without_physx',text)
        self.assertIn('scripts.b06_failure_record',text)

    def test_blocked_record_is_sanitized_and_immutable(self):
        mod=load('b06_failure_record')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            a=mod.make_record('download_pinned_author_idm',12)
            p,raw,digest=mod.save_local(root,a)
            self.assertEqual(json.loads(raw)['stage'],'download_pinned_author_idm')
            self.assertEqual(mod.save_local(root,a)[0],p)
            self.assertNotIn(str(root),raw.decode())
            self.assertTrue(digest in p.name)
            self.assertEqual(a['status'],'SETUP_BLOCKED_NO_PHYSX_STARTED_IN_THIS_ATTEMPT')
            b=mod.make_record('real_physx_bounded_dev_and_publish',3)
            self.assertEqual(b['status'],'PARTIAL_PHYSX_POSSIBLE_DO_NOT_RERUN')

    def test_unknown_stage_and_exit_code_rejected(self):
        mod=load('b06_failure_record')
        for stage,code in [('unknown',1),('project_gate',0),('project_gate',300)]:
            with self.assertRaises(ValueError):mod.make_record(stage,code)

    def test_publish_existing_rejects_outside_path(self):
        runner=load('roboreel_uniskill_video_dev')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):runner.publish_existing(Path(d))

    def test_publish_existing_fails_without_complete_files(self):
        runner=load('roboreel_uniskill_video_dev')
        with patch.object(runner,'PROJECT',Path('/tmp/test_project')):
            with self.assertRaises(FileNotFoundError):
                runner.publish_existing(Path('/tmp/test_project/outputs/research_runs/empty'))

if __name__=='__main__': unittest.main()
