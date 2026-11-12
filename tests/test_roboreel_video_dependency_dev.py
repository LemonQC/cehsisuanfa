"""Zero simulator/model/network regression fixtures for the small TRAIN-only DEV gate."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.roboreel_video_dependency_dev import (
    preflight, require_plan, check_features, video_inventory,
    save_json_once, AuditedEnv, action_displacement, DEFAULT_TASK)


class VideoSensitivityDevTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.project=Path(self.tmp.name)
        self.config=self.project/'outputs/v140_pi05_uniform/config_seed_offset_0.json'
        self.config.parent.mkdir(parents=True)
        self.ckpt=self.project/'ckpt/29999'
        (self.ckpt/'params').mkdir(parents=True)
        (self.ckpt.parent/'train_config.json').write_text(json.dumps({
            'data_dir':str(self.project/'dataset'),
            'tasks':['toys_in_drawer','push_cube'],
            'demo_cameras':['front'],
            'demonstration_config':'perceptual-framesamp-context.yaml'}))
        self.config.write_text(json.dumps({'model':'demo_pi05','checkpoint_dir':str(self.ckpt)}))
        self.upstream=self.project/'third_party/roboreel_models/src/evaluation'
        self.upstream.mkdir(parents=True)
        (self.upstream/'demo_pi05_runner.py').write_text('# synthetic fixture only\n')

    def tearDown(self):
        self.tmp.cleanup()

    def add_train(self, id, *, media=True, feats=True):
        d=f'{id:04d}'
        root=self.project/'dataset'
        v=root/'human_demos/train/no_distraction/toys_in_drawer'/d/'front.mp4'
        f=root/'video_demonstration_features/without_distraction/toys_in_drawer'/d/'front.npy'
        if media:
            v.parent.mkdir(parents=True,exist_ok=True); v.write_bytes(b'x'*2048)
        if feats:
            f.parent.mkdir(parents=True,exist_ok=True); f.write_bytes(b'x'*2048)

    def test_no_config_no_run(self):
        self.config.unlink()
        r=preflight(self.project,None,DEFAULT_TASK)
        self.assertEqual(r['reason'],'MISSING_EVAL_CONFIG')

    def test_missing_train_assets_is_blocked(self):
        r=preflight(self.project,None,DEFAULT_TASK)
        self.assertEqual(r['reason'],'NEED_TWO_GENUINE_TRAIN_VIDEOS_AND_FEATURES')
        self.assertEqual(r['train_demos_usable'],0)
        self.assertEqual(r['new_physx_actions'],0)

    def test_full_train_assets_allows_two_distinct_demos_only(self):
        self.add_train(1);self.add_train(2)
        r=preflight(self.project,None,DEFAULT_TASK)
        self.assertEqual(r['status'],'READY_FOR_BOUNDED_DEV')
        self.assertEqual(require_plan(r,2,800),[1,2])
        self.assertEqual(r['usable_train_demo_ids'],[1,2])

    def test_holdout_eval_media_cannot_enable_dev(self):
        self.add_train(80);self.add_train(81)
        r=preflight(self.project,None,DEFAULT_TASK)
        self.assertEqual(r['train_demos_usable'],0)
        self.assertEqual(r['status'],'BLOCKED')

    def test_feature_only_does_not_count_as_human_source(self):
        self.add_train(1,media=False)
        self.add_train(2,media=False)
        r=preflight(self.project,None,DEFAULT_TASK)
        self.assertEqual(r['train_feature_demos'],2)
        self.assertEqual(r['train_demos_usable'],0)
        self.assertEqual(r['status'],'BLOCKED')

    def test_media_only_does_not_count_as_policy_tokens(self):
        self.add_train(1,feats=False)
        self.add_train(2,feats=False)
        r=preflight(self.project,None,DEFAULT_TASK)
        self.assertEqual(r['train_video_demos_with_media'],2)
        self.assertEqual(r['train_demos_usable'],0)

    def test_checkpoint_not_trained_for_requested_task(self):
        t=self.ckpt.parent/'train_config.json'
        t.write_text(json.dumps({'data_dir':str(self.project/'dataset'),
          'tasks':['push_cube'], 'demonstration_config':'perceptual-framesamp-context.yaml'}))
        r=preflight(self.project,None,'toys_in_drawer')
        self.assertEqual(r['reason'],'CHECKPOINT_NOT_TRAINED_FOR_TASK')

    def test_planner_bounded_not_eval_or_unlimited(self):
        x={'status':'READY_FOR_BOUNDED_DEV','usable_train_demo_ids':[3,5]}
        for ep,steps in [(0,200),(5,200),(1,801),(1,0)]:
            with self.assertRaises(ValueError):require_plan(x,ep,steps)
        with self.assertRaises(ValueError):require_plan({**x,'usable_train_demo_ids':[80,81]},2,200)
        with self.assertRaises(ValueError):require_plan({**x,'usable_train_demo_ids':[101,102]},2,200)
        with self.assertRaises(RuntimeError):require_plan({**x,'status':'BLOCKED','reason':'MISSING'},1,50)

    def test_audited_wrapper_counts_only_true_actions(self):
        class Faux:
            spec='fake'
            def step(self,action):
                return None, 0., False, False, {'success':False}
        w=AuditedEnv(Faux())
        self.assertEqual(w.spec,'fake')
        w.step([1,0,0,0,0,0,0,0]);w.step([2,0,0,0,0,0,0,0]);self.assertEqual(w.n_steps,2)
        self.assertFalse(w.last_success)
        self.assertEqual(len(w.action_trace),2)

    def test_non_8d_action_is_rejected_before_step(self):
        class Faux:
            def step(self,action): raise AssertionError('should not physically step')
        w=AuditedEnv(Faux())
        with self.assertRaisesRegex(ValueError,'finite 8D'):
            w.step([1,2])
        self.assertEqual(w.n_steps,0)

    def test_action_difference_metric(self):
        self.assertIsNone(action_displacement([], [[0]*8]))
        self.assertAlmostEqual(action_displacement([[0]*8], [[1]+[0]*7]),1.0)

    def test_official_m0_placeholder_paths_use_existing_v120_layout(self):
        from scripts.roboreel_video_dependency_dev import official_path_vars, project_path
        root=self.project/'outputs/v120_baselines'
        ckpt=root/'released_checkpoints/nd/vid_cond_pi05_uniform/ckpt'
        (ckpt/'params').mkdir(parents=True)
        (ckpt.parent/'train_config.json').write_text(json.dumps({
            'data_dir':'${ROBOREEL_DATA}', 'tasks':['toys_in_drawer'],
            'demo_cameras':['front'],
            'demonstration_config':'perceptual-framesamp-context.yaml'}))
        self.config.write_text(json.dumps({
            'model':'demo_pi05',
            'checkpoint_dir':'${ROBOREEL_CKPT}/nd/vid_cond_pi05_uniform/ckpt'}))
        with patch.dict(os.environ,{'ROBOREEL_CKPT':'','ROBOREEL_DATA':''}):
            env=official_path_vars(self.project)
            self.assertEqual(project_path(self.project,'${ROBOREEL_CKPT}/nd/vid_cond_pi05_uniform/ckpt',env),ckpt.resolve())
            self.assertEqual(project_path(self.project,'$ROBOREEL_DATA',env),
                             (root/'official_eval_data').resolve())
            r=preflight(self.project,None,'toys_in_drawer')
        self.assertEqual(r['checkpoint_dir'],str(ckpt.resolve()))
        self.assertEqual(r['data_dir'],str((root/'official_eval_data').resolve()))
        self.assertEqual(r['reason'],'NEED_TWO_GENUINE_TRAIN_VIDEOS_AND_FEATURES')
        self.assertNotIn('UNEXPANDED_PATH_VARIABLE',str(r))

    def test_operator_override_has_precedence_over_v120_defaults(self):
        from scripts.roboreel_video_dependency_dev import official_path_vars, project_path
        override=self.project/'special/checked_checkpoint_root'
        with patch.dict(os.environ,{'ROBOREEL_CKPT':str(override),'ROBOREEL_DATA':''}):
            paths=official_path_vars(self.project)
        self.assertEqual(project_path(self.project,'${ROBOREEL_CKPT}/nd/policy',paths),
                         (override/'nd/policy').resolve())
        self.assertEqual(paths['ROBOREEL_DATA'],
                         str(self.project/'outputs/v120_baselines/official_eval_data'))

    def test_unrecognised_placeholder_still_fails_closed(self):
        from scripts.roboreel_video_dependency_dev import project_path
        with patch.dict(os.environ,{'UNKNOWN_CHECKPOINT_VAR':''}):
            with self.assertRaisesRegex(ValueError,'UNEXPANDED_PATH_VARIABLE'):
                project_path(self.project,'${UNKNOWN_CHECKPOINT_VAR}/model',
                             {'ROBOREEL_CKPT':'/known'})

    def test_result_file_immutable(self):
        p=self.project/'out/report.json'
        save_json_once(p,{'a':1});save_json_once(p,{'a':1})
        with self.assertRaises(FileExistsError):save_json_once(p,{'a':2})


if __name__=='__main__': unittest.main()
