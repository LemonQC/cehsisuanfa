"""CPU-only regression tests for immutable post-hoc StackCups diagnosis."""
from __future__ import annotations
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.diagnose_stack_video_dev import (validate_source, analyze, main, range_metrics)


class StackReadonlyDiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.project=Path(self.temp.name)
        self.root=self.project/'outputs/roboreel_video_dependency_dev/stack_v1'
        self.root.mkdir(parents=True)
        self.trials=[]
        self.pairs=[]
        self.original_actions={}
        for ep in (0,1):
            item=[]
            for source in (0,1):
                file=self.root/f'episode_{ep:02d}'/f'source_{source:04d}'/'actions.npy'
                file.parent.mkdir(parents=True)
                a=np.ones((800,8),dtype=np.float32)*0.7
                a[:,source]=1.5+0.1*ep
                np.save(file,a)
                self.original_actions[(ep,source)]=a
                trial={'episode':ep,'source_train_demo_id':source,'actions_path':str(file.resolve()),
                       'actions_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
                       'steps':800,'success':False,'reset_seed':310000+ep,
                       'policy_rng_seed':560000+ep,'video_error':None}
                self.trials.append(trial)
                item.append(trial)
            x=self.original_actions[(ep,0)]
            y=self.original_actions[(ep,1)]
            self.pairs.append({'episode':ep,'mean_action_l2_first_common_steps':
                               float(np.linalg.norm(x-y,axis=1).mean()),
                               'source_ids':[0,1],'successes':[False,False]})
        self.result={'status':'DEV_COMPLETE_NOT_FORMAL','task':'stack_cups',
                     'real_rollouts':4,'paired_episodes':2,'trials':self.trials,'pairs':self.pairs}
        (self.root/'results.json').write_text(json.dumps(self.result))

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_frozen_real_evidence_is_accepted(self):
        doc,actions,pairs=validate_source(self.root)
        self.assertEqual(len(actions),4)
        self.assertEqual(len(pairs),2)

    def test_metrics_first_and_horizon(self):
        a=self.original_actions[(0,0)]
        b=self.original_actions[(0,1)]
        m=range_metrics(a,b,20)
        self.assertEqual(m['steps_compared'],20)
        self.assertAlmostEqual(m['mean_8d_action_l2'],np.sqrt(2)*0.8,places=6)
        self.assertAlmostEqual(m['mean_gripper_abs_difference'],0)

    def test_summary_reproduces_original_mean_l2(self):
        doc,actions,pairs=validate_source(self.root)
        report=analyze(doc,actions,pairs)
        self.assertEqual(report['successes'],0)
        self.assertEqual(report['stage_failure_classification_without_video'],
                         'UNKNOWN_REQUIRES_VISUAL_REVIEW')
        self.assertAlmostEqual(report['pairs'][0]['mean_8d_action_l2_original'],
                               self.pairs[0]['mean_action_l2_first_common_steps'])

    def test_fails_on_action_hash_change(self):
        a=self.root/'episode_00/source_0000/actions.npy'
        a.write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError,'ACTION_HASH_MISMATCH'):
            validate_source(self.root)

    def test_fails_on_missing_trial(self):
        self.result['trials'].pop()
        (self.root/'results.json').write_text(json.dumps(self.result))
        with self.assertRaisesRegex(ValueError,'EXPECTED_EXACTLY_FOUR'):
            validate_source(self.root)

    def test_fails_on_mismatched_pair_rng(self):
        self.result['trials'][1]['policy_rng_seed']+=1
        (self.root/'results.json').write_text(json.dumps(self.result))
        with self.assertRaisesRegex(ValueError,'PAIR_SEED_OR_RNG_NOT_MATCHED'):
            validate_source(self.root)

    def test_writes_separate_diagnosis_never_original(self):
        before={str(x.relative_to(self.root)):hashlib.sha256(x.read_bytes()).hexdigest()
                for x in self.root.rglob('*') if x.is_file()}
        dest=self.project/'outputs/roboreel_video_dependency_diagnosis/stack_v1'
        self.assertEqual(main(['--root',str(self.root),'--out',str(dest),'--no-montage']),0)
        self.assertTrue((dest/'diagnosis.json').exists())
        self.assertEqual(before,{str(x.relative_to(self.root)):hashlib.sha256(x.read_bytes()).hexdigest()
                                 for x in self.root.rglob('*') if x.is_file()})
        with self.assertRaisesRegex(FileExistsError,'REFUSE_TO_OVERWRITE'):
            main(['--root',str(self.root),'--out',str(dest),'--no-montage'])

    def test_fails_when_output_inside_frozen_root(self):
        with self.assertRaisesRegex(ValueError,'OUTPUT_MUST_BE_SEPARATE'):
            main(['--root',str(self.root),'--out',str(self.root/'new'),'--no-montage'])


if __name__=='__main__':
    unittest.main()
