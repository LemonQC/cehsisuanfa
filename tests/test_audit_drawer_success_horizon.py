"""CPU regression of the frozen per-step successful-prefix auditor."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from scripts.audit_drawer_success_horizon import audit,main

class DrawerSuccessHorizonTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'outputs/roboreel_video_dependency_dev/drawer_hf_v1'
        self.root.mkdir(parents=True)
        self.rows=[]
        for ep in (0,1):
            for src in (0,1):
                steps=800 if ep==0 else (634 if src==0 else 650)
                tr=[False]*steps
                if ep==1:tr[-1]=True
                folder=self.root/f'episode_{ep:02d}'/f'source_{src:04d}'
                folder.mkdir(parents=True)
                actions=folder/'actions.npy';actions.write_bytes(b'original_binary_8D_action_array')
                trace=folder/'success_trace.json';trace.write_text(json.dumps(tr))
                self.rows.append({'episode':ep,'source_train_demo_id':src,'steps':steps,
                                  'success':ep==1,'last_success':ep==1,
                                  'reset_seed':310000+ep,'policy_rng_seed':560000+ep,
                                  'terminated_events':int(ep==1),'actions_path':str(actions.resolve()),
                                  'actions_sha256':hashlib.sha256(actions.read_bytes()).hexdigest()})
        (self.root/'results.json').write_text(json.dumps({
            'status':'DEV_COMPLETE_NOT_FORMAL','task':'toys_in_drawer',
            'paired_episodes':2,'real_rollouts':4,'trials':self.rows}))

    def test_400_and_800_are_distinct(self):
        a=audit(self.root)
        self.assertEqual(a['success_by_400'],0)
        self.assertEqual(a['success_by_800'],2)
        self.assertEqual([x['first_success_step'] for x in a['trials']],[None,None,634,650])

    def test_early_transient_success_counted_as_success_by_400(self):
        path=self.root/'episode_01/source_0001/success_trace.json'
        tr=json.loads(path.read_text());tr[299]=True;path.write_text(json.dumps(tr))
        a=audit(self.root)
        self.assertEqual(a['success_by_400'],1)
        self.assertEqual([x['first_success_step'] for x in a['trials']],[None,None,634,300])

    def test_success_flag_with_no_true_trace_rejected(self):
        path=self.root/'episode_01/source_0001/success_trace.json'
        path.write_text(json.dumps([False]*650))
        with self.assertRaisesRegex(ValueError,'LAST_SUCCESS_TRACE_CONTRADICTION'):
            audit(self.root)

    def test_bad_length_rejected(self):
        path=self.root/'episode_00/source_0000/success_trace.json'
        path.write_text('[]')
        with self.assertRaisesRegex(ValueError,'WRONG_LENGTH'):
            audit(self.root)

    def test_string_looks_like_bool_rejected(self):
        path=self.root/'episode_00/source_0000/success_trace.json'
        path.write_text(json.dumps(['false']*800))
        with self.assertRaisesRegex(ValueError,'NON_BOOLEAN'):
            audit(self.root)

    def test_action_hash_must_match_original(self):
        path=self.root/'episode_01/source_0000/actions.npy'
        path.write_bytes(b'TAMPERED')
        with self.assertRaisesRegex(ValueError,'ACTION_HASH_MISMATCH'):
            audit(self.root)

    def test_pair_rng_mismatch_refused(self):
        results=self.root/'results.json'
        x=json.loads(results.read_text());x['trials'][1]['policy_rng_seed']=0
        results.write_text(json.dumps(x))
        with self.assertRaisesRegex(ValueError,'UNMATCHED_PAIRED_SCENE'):
            audit(self.root)

    def test_duplicate_episode_source_refused(self):
        results=self.root/'results.json'
        x=json.loads(results.read_text());x['trials'][1]['source_train_demo_id']=0
        results.write_text(json.dumps(x))
        with self.assertRaisesRegex(ValueError,'DUPLICATED_EPISODE_SOURCE'):
            audit(self.root)

    def test_main_only_writes_outside_frozen_root(self):
        before={p.relative_to(self.root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.root.rglob('*') if p.is_file()}
        out=Path(self.tmp.name)/'outputs/roboreel_video_dependency_diagnosis/drawer_horizon_v1'
        self.assertEqual(main(['--root',str(self.root),'--out',str(out)]),0)
        self.assertTrue((out/'SUMMARY.md').is_file())
        self.assertEqual(json.loads((out/'horizon_audit.json').read_text())['new_physx_rollouts'],0)
        after={p.relative_to(self.root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
               for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before,after)

    def test_main_refuses_replace_existing_evidence(self):
        out=Path(self.tmp.name)/'outputs/roboreel_video_dependency_diagnosis/drawer_horizon_v1'
        out.mkdir(parents=True)
        with self.assertRaisesRegex(FileExistsError,'DO_NOT_OVERWRITE'):
            main(['--root',str(self.root),'--out',str(out)])

if __name__=='__main__':unittest.main()
