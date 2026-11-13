"""No data-network, no PhysX, no human-video semantics fabrication."""
from __future__ import annotations
import hashlib
import tempfile
import sys
import types
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch
from agcofl.video_goal_push_spec import CounterfactualPushSpec, assert_no_goal_leak
from scripts.prepare_g0_self_recorded_pair import (
    read_clips, build_draft, write_immutable, project_file, main as ingest_main)
from scripts.check_g1_goal_design import build_design,run

class GoalPairDesignTests(TestCase):
    def setUp(self):
        self.ctx=tempfile.TemporaryDirectory();self.addCleanup(self.ctx.cleanup)
        self.proj=Path(self.ctx.name)

    def _videos(self):
        raw=self.proj/'outputs/g0_human_goal_pairs/raw';raw.mkdir(parents=True)
        for f in ('left.mp4','right.mp4'):(raw/f).write_bytes((f*500).encode())
        return raw

    def test_opposing_goals_disjoint(self):
        spec=CounterfactualPushSpec()
        goals=spec.targets((0,0,0.02))
        self.assertEqual(goals['-x'],(-0.15,0.0,0.02))
        self.assertEqual(goals['+x'],(0.15,0.0,0.02))
        self.assertFalse(spec.goal_reached(goals['+x'],goals['-x']))
        self.assertTrue(spec.goal_reached(goals['+x'],goals['+x']))

    def test_spawn_bounds_and_no_false_reachability(self):
        spec=CounterfactualPushSpec()
        self.assertEqual(spec.paired_design((0,0,0.02),310000)['physics_reachability_established'],False)
        with self.assertRaises(ValueError):spec.targets((1.,0.,0.02))

    def test_target_regions_cannot_overlap(self):
        with self.assertRaises(ValueError):CounterfactualPushSpec(distance_m=0.05,radius_m=0.05)

    def test_anti_leak_rejects_goal_keys_nested(self):
        for obs in ({'state':{'goal_pos':[1,0,0]}}, {'target_xyz':[0,0,0]},
                    {'reward':5},{'task_direction':'-x'}):
            with self.assertRaisesRegex(ValueError,'GOAL_INFORMATION_LEAK'):
                assert_no_goal_leak(obs)
        assert_no_goal_leak({'rgb':{'front':None},'robot_joint_state':None})

    def test_geometry_determinism_and_equal_obs(self):
        self.assertEqual(build_design(),build_design())
        self.assertTrue(all(x['policy_input_equal_across_branches_by_construction'] for x in build_design()['design_goal_pairs']))
        self.assertFalse(build_design()['physical_reachability_verified'])

    def test_geometry_report_immutable(self):
        a=run(self.proj,self.proj/'out')
        self.assertEqual(a,run(self.proj,self.proj/'out'))
        a.write_text('corrupted')
        with self.assertRaises(FileExistsError):run(self.proj,self.proj/'out')

    def test_real_human_pair_missing_returns_blocked(self):
        self.assertEqual(ingest_main(['--project',str(self.proj)]),2)

    def test_pair_sha_bytes_and_review_false(self):
        self._videos()
        rows=read_clips(self.proj,Path('outputs/g0_human_goal_pairs/raw/left.mp4'),
                         Path('outputs/g0_human_goal_pairs/raw/right.mp4'),
                         probe=lambda path:{'ok':True,'width':224,'height':224})
        draft=build_draft(rows)
        self.assertEqual([c['goal_robot_x'] for c in draft['clips']],['-x','+x'])
        self.assertNotEqual(draft['clips'][0]['sha256'],draft['clips'][1]['sha256'])
        self.assertTrue(all(not v['human_hand_demonstration'] for v in draft['clips']))
        self.assertTrue(all(v['rights']=='REQUIRES_HUMAN_REVIEW' for v in draft['clips']))

    def test_refuse_identical_video_bytes(self):
        raw=self._videos();(raw/'right.mp4').write_bytes((raw/'left.mp4').read_bytes())
        with self.assertRaisesRegex(ValueError,'IDENTICAL_HUMAN_VIDEO'):
            read_clips(self.proj,raw/'left.mp4',raw/'right.mp4',probe=lambda path:{'ok':True})

    def test_outside_project_input_blocked(self):
        with self.assertRaises(ValueError):project_file(self.proj,Path('/tmp/not-in-project.mp4'))

    def test_video_decode_failure_blocked(self):
        raw=self._videos()
        with self.assertRaisesRegex(ValueError,'DECODE_FAILED'):
            read_clips(self.proj,raw/'left.mp4',raw/'right.mp4',
                       probe=lambda path:{'ok':False,'reason':'mock_bad'})

    def test_first_stage_is_human_review_not_pass(self):
        self._videos()
        def fake_probe(f):return {'ok':True,'width':320,'height':240}
        with patch.dict(sys.modules,{'scripts.audit_goal_contrast_g0':types.SimpleNamespace(probe_video=fake_probe)}), \
             patch('scripts.prepare_g0_self_recorded_pair.review_strip',return_value='REVIEW_STRIP_CREATED_NOT_SEMANTICS_CERTIFIED'):
            self.assertEqual(ingest_main(['--project',str(self.proj)]),0)
        rows=list((self.proj/'outputs/g0_human_goal_pairs/review').glob('*.json'))
        self.assertEqual(len(rows),1)
        self.assertIn('REQUIRES_HUMAN_REVIEW',rows[0].read_text())

if __name__=='__main__':main()
