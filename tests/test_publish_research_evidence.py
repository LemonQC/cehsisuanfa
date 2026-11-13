"""Offline only: no GitHub API, JAX, or PhysX in unit tests."""
import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.publish_research_evidence import (
    build_evidence,project_source,read_result,safe_trial,write_immutable_report,
    assert_evidence_branch_exists,REPO,BRANCH,
)


class PublishEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup)
        self.root=Path(self.t.name)
        self.src=self.root/'outputs/roboreel_video_dependency_dev/drawer_hf_v1'
        self.src.mkdir(parents=True)
        self.payload={
            'status':'DEV_COMPLETE_NOT_FORMAL','task':'toys_in_drawer',
            'real_rollouts':4,'paired_episodes':2,'any_pair_success_difference':False,
            'pairs':[{'episode':0,'source_ids':[0,1],'successes':[False,True],
                      'mean_action_l2_first_common_steps':0.14,
                      'same_reset_seed':True,'same_rng_seed':True}],
            'trials':[{'episode':0,'source_train_demo_id':0,'steps':800,'success':False,
                       'error':None,'video_path':'/private/home/foo.mp4',
                       'source_video_paths':['/gated/train/foo.mp4']},
                      {'episode':0,'source_train_demo_id':1,'steps':230,'success':True,
                       'error':'RuntimeError: private token hf_supersecret details'}],
            'unexpected_secret':'hf_SUPER_SECRET',
        }
        (self.src/'results.json').write_text(json.dumps(self.payload))

    def test_sanitize_results_excludes_secrets_and_video_paths(self):
        d=read_result(self.src,'outputs/roboreel_video_dependency_dev/drawer_hf_v1',None)
        content=json.dumps(d)
        self.assertEqual(d['physical_rollouts_reported'],4)
        self.assertEqual(d['pairs'][0]['mean_action_l2_first_common_steps'],0.14)
        self.assertNotIn('hf_SUPER_SECRET',content)
        self.assertNotIn('hf_supersecret',content)
        self.assertNotIn('/private/',content)
        self.assertNotIn('/gated/',content)
        self.assertEqual(d['trials'][1]['error_type'],'RuntimeError')

    def test_detached_branch_only_and_summary(self):
        dest,files,obj=build_evidence(self.root,self.src,None,False)
        self.assertTrue(dest.startswith('evidence/runs/drawer_hf_v1_'))
        self.assertEqual(set(files),{'evidence.json','README.md'})
        self.assertNotIn(b'private',b''.join(files.values()))
        self.assertIn(b'DEV_COMPLETE_NOT_FORMAL',files['README.md'])

    def test_same_input_same_output_name(self):
        a,_,_=build_evidence(self.root,self.src,None,False)
        b,_,_=build_evidence(self.root,self.src,None,False)
        self.assertEqual(a,b)

    def test_snapshots_change_when_real_metric_changes(self):
        a,_,_=build_evidence(self.root,self.src,None,False)
        self.payload['trials'][0]['success']=True
        (self.src/'results.json').write_text(json.dumps(self.payload))
        b,_,_=build_evidence(self.root,self.src,None,False)
        self.assertNotEqual(a,b)

    def test_reject_hf_model_dataset_source(self):
        p=self.root/'outputs/roboreel_partnet_single_44962/data/dataset/44962';p.mkdir(parents=True)
        with self.assertRaises(ValueError):project_source(self.root,p)

    def test_reject_external_source(self):
        p=self.root.parent/'unlikely_external_outputs'
        with self.assertRaises((ValueError,FileNotFoundError)):project_source(self.root,p)

    def test_partial_results_supported_without_new_physx(self):
        (self.src/'results.json').unlink()
        (self.src/'per_trial.partial.json').write_text(json.dumps(self.payload['trials']))
        d=read_result(self.src,'outputs/roboreel_video_dependency_dev/drawer_hf_v1',None)
        self.assertEqual(d['status'],'PARTIAL')
        self.assertEqual(len(d['trials']),2)

    def test_b06_append_only_partial_object_with_complete_results(self):
        # B06 actually persists {"trials": [...]} while the older publisher
        # expected a bare list and refused to create the video preview.
        b06=self.root/'outputs/research_runs/b06_uniskill_stackcups_dev_v1'
        b06.mkdir(parents=True)
        result={
            'status':'DEV_COMPLETE_NOT_FORMAL',
            'task':'stack_cups',
            'rollouts':4,'episodes':2,
            'any_pair_success_difference':False,
            'trials':[{'episode':e,'source_train_demo_id':sid,
                       'success':False,'steps':800,'error_class':None}
                      for e in range(2) for sid in range(2)],
            'pairs':[{'episode':e,'source_ids':[0,1],
                      'successes':[False,False]} for e in range(2)],
        }
        (b06/'results.json').write_text(json.dumps(result))
        (b06/'per_trial.partial.json').write_text(
            json.dumps({'trials':result['trials']}))
        obj=read_result(b06,'outputs/research_runs/b06_uniskill_stackcups_dev_v1',None)
        self.assertEqual(obj['status'],'DEV_COMPLETE_NOT_FORMAL')
        self.assertEqual(obj['physical_rollouts_reported'],4)
        self.assertEqual(obj['paired_episodes_reported'],2)
        self.assertEqual(len(obj['trials']),4)
        self.assertEqual(len(obj['pairs']),2)
        dest,files,summary=build_evidence(self.root,b06,None,False)
        self.assertTrue(dest.startswith('evidence/runs/b06_uniskill_stackcups_dev_v1_'))
        self.assertEqual(set(files),{'evidence.json','README.md'})
        self.assertIn(b'DEV_COMPLETE_NOT_FORMAL',files['README.md'])

    def test_b06_malformed_partial_object_is_refused(self):
        b06=self.root/'outputs/research_runs/b06_uniskill_stackcups_dev_v1'
        b06.mkdir(parents=True)
        (b06/'results.json').write_text(json.dumps({'status':'DEV_COMPLETE_NOT_FORMAL',
                                                    'task':'stack_cups','trials':[]}))
        (b06/'per_trial.partial.json').write_text(json.dumps({'trials':{},
                                                               'token':'secret'}))
        with self.assertRaisesRegex(ValueError,'PARTIAL_TRIALS_INVALID_OBJECT'):
            read_result(b06,'outputs/research_runs/b06_uniskill_stackcups_dev_v1',None)

    def test_pipeline_status_is_sanitized(self):
        p=self.root/'outputs/experiment_publisher/pipeline_status/drawer_hf_v1/status.json'
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({'phase':'scene_smoke','exit_code':3,
                                 'stage_result':'FAILED','token':'hf_SECRET'}))
        _,files,obj=build_evidence(self.root,self.src,p,False)
        self.assertEqual(obj['pipeline']['phase'],'scene_smoke')
        self.assertNotIn(b'hf_SECRET',files['evidence.json'])

    def test_status_file_outside_allowlist_rejected(self):
        p=self.root/'private.json';p.write_text('{}')
        with self.assertRaises(ValueError):build_evidence(self.root,self.src,p,False)

    @patch('scripts.publish_research_evidence._run_gh')
    def test_remote_create_on_dedicated_branch(self,mock):
        mock.side_effect=[(False,None),(True,{'content':{'sha':'ok'}})]
        result=write_immutable_report('evidence/runs/test/README.md',b'abc')
        self.assertEqual(result,'PUBLISHED')
        self.assertEqual(mock.call_args[1]['payload']['branch'],BRANCH)
        self.assertNotIn('main',json.dumps(mock.call_args[1]['payload']))
        self.assertIn('-X',mock.call_args_list[0].args[0])
        self.assertEqual(mock.call_args_list[0].args[0][1],'GET')

    @patch('scripts.publish_research_evidence._run_gh')
    def test_existing_identical_remote_is_reused(self,mock):
        mock.return_value=(True,{'type':'file','content':base64.b64encode(b'abc').decode()})
        self.assertEqual(write_immutable_report('evidence/runs/test/README.md',b'abc'),'EXISTS_SAME_BYTES')
        self.assertEqual(mock.call_count,1)

    @patch('scripts.publish_research_evidence._run_gh')
    def test_existing_different_remote_refuses_overwrite(self,mock):
        mock.return_value=(True,{'type':'file','content':base64.b64encode(b'diff').decode()})
        with self.assertRaises(FileExistsError):write_immutable_report('evidence/runs/test/README.md',b'abc')
        self.assertEqual(mock.call_count,1)

    @patch('scripts.publish_research_evidence._run_gh')
    def test_missing_branch_fails(self,mock):
        mock.return_value=(False,None)
        with self.assertRaises(RuntimeError):assert_evidence_branch_exists()

    def test_repo_constant(self):
        self.assertEqual(REPO,'LemonQC/cehsisuanfa')
        self.assertEqual(BRANCH,'tro1-experiment-evidence')

if __name__=='__main__':unittest.main()

class MorePublisherCases(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def test_readonly_diagnosis_publishes_only_metrics(self):
        src=self.root/'outputs/roboreel_video_dependency_diagnosis/stack_v1'
        src.mkdir(parents=True)
        (src/'diagnosis.json').write_text(json.dumps({'task':'stack_cups',
            'status':'READ_ONLY_DIAGNOSIS_NOT_NEW_ROLLOUT','successes':0,
            'root':'/sensitive/local/path',
            'pairs':[{'episode':0,'mean_8d_action_l2_original':0.75,
                'prefix_metrics':{'1':{'first_8d_action_l2':0.0013},
                                  '20':{'mean_8d_action_l2':0.0016}}}]}))
        dest,files,obj=build_evidence(self.root,src,None,False)
        self.assertEqual(obj['diagnosis']['pairs'][0]['full_mean_8d_l2'],0.75)
        self.assertEqual(obj['status'],'READ_ONLY_DIAGNOSIS_NOT_NEW_ROLLOUT')
        self.assertNotIn(b'sensitive/local/path',files['evidence.json'])
        self.assertIn(b'first20 mean=0.0016',files['README.md'])

    def test_g0_reports_do_not_invent_completed_goals(self):
        src=self.root/'outputs/g0_goal_pair_audit/v1';src.mkdir(parents=True)
        (src/'g0_audit_0123.json').write_text(json.dumps({
            'decision':'G0_NOT_PASSED',
            'human_goal_pair_audit':{'status':'G0_BLOCKED'},
            'secret':'hf_a_t_o_k_e_n'}))
        _,files,obj=build_evidence(self.root,src,None,False)
        self.assertEqual(obj['status'],'G0_NOT_PASSED')
        self.assertEqual(obj['g0_audit']['human_goal_pair_status'],'G0_BLOCKED')
        self.assertNotIn(b'hf_a_t_o_k_e_n',files['evidence.json'])
