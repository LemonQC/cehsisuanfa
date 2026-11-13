"""No-PartNet, CPU-only safety tests for official RoboReel StackCups.

No network, simulator, checkpoint or GPU needed. These tests are not
evidence of physical rollout success.
"""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

from scripts import prepare_roboreel_train_drawer as prep
from scripts.roboreel_video_dependency_dev import (
    verify_no_partnet_stack_assets, ALLOWED_TASKS, preflight, require_plan)
from agcofl.resumable_hf_assets import _valid


class StackNoPartNetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.project=Path(self.tmp.name)
        prep.configure_task('stack_cups')

    def tearDown(self):
        prep.configure_task('toys_in_drawer')
        self.tmp.cleanup()

    def build_cups(self):
        root=self.project/'third_party/roboreel_models/roboreel_env/assets/scenes/stack_cups'
        for cup in ('dark_blue_cup','red_cup','light_blue_cup'):
            for name in ('visual.glb','collision_vhacd.obj','orientation_result.json'):
                path=root/cup/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b'local fake mesh fixture'*14)
        return root

    def test_pinned_stack_env_id_and_no_partnet(self):
        self.assertEqual(ALLOWED_TASKS['stack_cups'],'StackCups-v1')
        self.assertEqual(prep.TASK,'stack_cups')
        with self.assertRaises(ValueError):prep.configure_task('close_drawer')
        self.assertEqual(prep.TASK,'stack_cups')

    def test_exact_train_paths(self):
        self.assertEqual(prep.media_rel(0),
            'human_demos/train/no_distraction/stack_cups/0000/front.mp4')
        self.assertEqual(prep.feature_rel(1),
            'video_demonstration_features/without_distraction/stack_cups/0001/front.npy')
        for idx in (-1,80,99,100):
            with self.assertRaises(ValueError):prep.media_rel(idx)

    def test_train_metadata_same_task_only(self):
        rows=[{'path':prep.media_rel(i),'bytes':10240} for i in (12,1)]
        rows+= [
            {'path':'human_demos/train/no_distraction/toys_in_drawer/0000/front.mp4','bytes':10240},
            {'path':'human_demos/eval/no_distraction/stack_cups/0080/front.mp4','bytes':10240},
        ]
        selected=prep.select_two(rows)
        self.assertEqual([item['id'] for item in selected['video_files']],[1,12])
        self.assertEqual(selected['task'],'stack_cups')
        self.assertTrue(selected['no_eval_data_or_robot_actions'])

    def test_hf_transport_train_allowlist(self):
        valid=prep.media_rel(0)
        self.assertEqual(_valid({'path':valid,'bytes':20000},prep.DATASET,
                 'dataset','https://hf-mirror.com')[1],valid)
        invalid=[
            valid.replace('train','eval'),valid.replace('0000','0080'),
            valid.replace('front.mp4','left.mp4'),
            valid.replace('stack_cups','close_drawer'),
        ]
        for bad in invalid:
            with self.assertRaises(ValueError):
                _valid({'path':bad,'bytes':20000},prep.DATASET,
                       'dataset','https://hf-mirror.com')

    def test_cup_asset_gate_missing(self):
        status=verify_no_partnet_stack_assets(self.project)
        self.assertFalse(status['partnet_mobility_required'])
        self.assertEqual(status['status'],'MISSING_PINNED_STACK_CUPS_ASSETS')
        self.assertEqual(len(status['missing']),9)

    def test_only_packaged_three_cup_assets_are_required(self):
        self.build_cups()
        status=verify_no_partnet_stack_assets(self.project)
        self.assertEqual(status['status'],'READY_REPO_STACK_CUPS')
        self.assertEqual(len(status['files']),9)
        self.assertTrue(all(x['sha256'] for x in status['files']))

    def test_missing_one_packaged_mesh_blocks(self):
        root=self.build_cups()
        (root/'red_cup/collision_vhacd.obj').unlink()
        status=verify_no_partnet_stack_assets(self.project)
        self.assertEqual(status['status'],'MISSING_PINNED_STACK_CUPS_ASSETS')

    def test_legacy_drawer_source_remains_separate(self):
        prep.configure_task('toys_in_drawer')
        self.assertIn('/toys_in_drawer/',prep.media_rel(0))
        prep.configure_task('stack_cups')
        self.assertIn('/stack_cups/',prep.media_rel(0))

    def test_download_budget_still_enforced(self):
        rows=[{'path':prep.media_rel(i),'bytes':200*1024*1024} for i in (0,1)]
        with self.assertRaisesRegex(RuntimeError,'EXCEEDS_BUDGET'):
            prep.select_two(rows)

    def test_stack_asset_preflight_fails_before_train_download(self):
        cfg=self.project/'outputs/v140_pi05_uniform/config_seed_offset_0.json'
        cfg.parent.mkdir(parents=True)
        checkpoint=self.project/'ckpt/29999'
        (checkpoint/'params').mkdir(parents=True)
        (checkpoint.parent/'train_config.json').write_text(json.dumps({
            'data_dir':str(self.project/'dataset'),
            'tasks':['stack_cups','push_cube'],
            'demo_cameras':['front'],
            'demonstration_config':'perceptual-framesamp-context.yaml'}))
        cfg.write_text(json.dumps({'model':'demo_pi05','checkpoint_dir':str(checkpoint)}))
        runner=self.project/'third_party/roboreel_models/src/evaluation/demo_pi05_runner.py'
        runner.parent.mkdir(parents=True)
        runner.write_text('# no hardware fixture')
        status=preflight(self.project,None,'stack_cups')
        self.assertEqual(status['reason'],'MISSING_PINNED_STACK_CUPS_ASSETS')
        self.build_cups()
        status2=preflight(self.project,None,'stack_cups')
        self.assertEqual(status2['reason'],'NEED_TWO_GENUINE_TRAIN_VIDEOS_AND_FEATURES')
        data=self.project/'dataset'
        for i in (0,1):
            media=data/prep.media_rel(i)
            feature=data/prep.feature_rel(i)
            media.parent.mkdir(parents=True,exist_ok=True)
            feature.parent.mkdir(parents=True,exist_ok=True)
            media.write_bytes(b'not a real video, fixture only'*60)
            feature.write_bytes(b'not a real npy, fixture only'*50)
        status3=preflight(self.project,None,'stack_cups')
        self.assertEqual(status3['status'],'READY_FOR_BOUNDED_DEV')
        self.assertEqual(require_plan(status3,2,800),[0,1])
        self.assertTrue(status3['heldout_eval_ids_excluded'])

    def test_cli_local_only_does_not_touch_network_or_outputs(self):
        from unittest.mock import patch
        with patch.object(prep,'list_remote',side_effect=AssertionError('must not call network')):
            ret=prep.main(['--task','stack_cups','--project',str(self.project),
                  '--data-root',str(self.project/'dataset')])
        self.assertEqual(ret,0)
        self.assertFalse((self.project/'outputs/roboreel_train_stack_cups_assets/v1').exists())


if __name__=='__main__':
    unittest.main()
