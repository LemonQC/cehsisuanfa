"""CPU-only publisher regression; no GitHub network and no PhysX."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest

from scripts.publish_research_evidence import read_result, render_readme


class HorizonPublisherTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'drawer_horizon_v1'
        self.root.mkdir()
        self.doc={
            'status':'READ_ONLY_DRAWER_HORIZON_AUDIT','task':'toys_in_drawer',
            'default_registered_limit':400,'experimental_limit':800,
            'rollout_count':4,'success_by_400':0,'success_by_800':2,
            'secret_token':'hf_DO_NOT_EVER_PUBLISH',
            'trials':[
                {'episode':0,'source_train_demo_id':0,'steps':800,'first_success_step':None,'success_by_400':False,'success_by_800':False},
                {'episode':0,'source_train_demo_id':1,'steps':800,'first_success_step':None,'success_by_400':False,'success_by_800':False},
                {'episode':1,'source_train_demo_id':0,'steps':634,'first_success_step':634,'success_by_400':False,'success_by_800':True},
                {'episode':1,'source_train_demo_id':1,'steps':650,'first_success_step':650,'success_by_400':False,'success_by_800':True},
            ],
        }
        self.write()

    def write(self):
        (self.root/'horizon_audit.json').write_text(json.dumps(self.doc))

    def test_sanitized_counts_and_summary(self):
        ret=read_result(self.root,'outputs/roboreel_video_dependency_diagnosis/drawer_horizon_v1',None)
        self.assertEqual(ret['status'],'READ_ONLY_DRAWER_HORIZON_AUDIT')
        self.assertEqual(ret['horizon_audit']['success_by_400'],0)
        self.assertEqual(ret['horizon_audit']['success_by_800'],2)
        self.assertEqual(len(ret['horizon_audit']['trials']),4)
        self.assertIn('source_file_hashes',ret)
        self.assertIn('horizon_sha256',ret['source_file_hashes'])
        self.assertNotIn('hf_DO_NOT_EVER_PUBLISH',json.dumps(ret))
        self.assertIn('NOT official heldout',render_readme(ret,False))

    def test_counts_mismatch_rejected(self):
        self.doc['success_by_800']=4;self.write()
        with self.assertRaisesRegex(ValueError,'INCONSISTENT_HORIZON_COUNTS'):
            read_result(self.root,'outputs/roboreel_video_dependency_diagnosis/drawer_horizon_v1',None)

    def test_nonboolean_success_rejected(self):
        self.doc['trials'][0]['success_by_400']='false';self.write()
        with self.assertRaisesRegex(ValueError,'INVALID_HORIZON_PAIR_OR_BOOLEAN'):
            read_result(self.root,'outputs/roboreel_video_dependency_diagnosis/drawer_horizon_v1',None)

    def test_bad_initial_horizon_rejected(self):
        self.doc['default_registered_limit']=600;self.write()
        with self.assertRaisesRegex(ValueError,'INVALID_FROZEN_DRAWER_HORIZON_AUDIT'):
            read_result(self.root,'outputs/roboreel_video_dependency_diagnosis/drawer_horizon_v1',None)

    def test_first_success_must_match_by400(self):
        self.doc['trials'][2]['first_success_step']=250;self.write()
        with self.assertRaisesRegex(ValueError,'INCONSISTENT_HORIZON_METRICS'):
            read_result(self.root,'outputs/roboreel_video_dependency_diagnosis/drawer_horizon_v1',None)


if __name__=='__main__':unittest.main()
