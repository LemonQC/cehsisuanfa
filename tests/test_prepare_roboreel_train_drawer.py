"""Offline/train-only safety regression for two real RoboReel drawer videos."""
from __future__ import annotations
import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_roboreel_train_drawer import (
    MAX_BYTES, media_rel, feature_rel, select_two, existing_local,
    cache_home, save_once, REMOTE, assert_video_sources,
)


def row(i, size=10240):
    return {'path': media_rel(i), 'bytes': size}


class PrepareDrawerTests(unittest.TestCase):
    def test_only_train_ids(self):
        self.assertIn('/0000/front.mp4', media_rel(0))
        self.assertIn('/0079/front.mp4', media_rel(79))
        self.assertIn('/without_distraction/',feature_rel(79))
        for n in (-1, 80, 99, 100):
            with self.assertRaises(ValueError): media_rel(n)

    def test_exact_two_smallest_real_train(self):
        result=select_two([row(23),row(3),row(0),
                        {'path':'human_demos/eval/no_distraction/toys_in_drawer/0080/front.mp4','bytes':10240}])
        self.assertEqual([x['id'] for x in result['video_files']], [0,3])
        self.assertEqual(result['file_count'], 2)
        self.assertTrue(result['no_eval_data_or_robot_actions'])

    def test_dont_accidentally_use_other_task_or_camera(self):
        r=select_two([row(1),row(2),
                      {'path':'human_demos/train/no_distraction/push_cube/0000/front.mp4','bytes':10240},
                      {'path':'human_demos/train/no_distraction/toys_in_drawer/0003/left.mp4','bytes':10240}])
        self.assertEqual([v['id'] for v in r['video_files']], [1,2])

    def test_two_are_required(self):
        with self.assertRaisesRegex(RuntimeError,'NEED_TWO'):
            select_two([row(0)])

    def test_zero_sizes_block(self):
        with self.assertRaises(ValueError): select_two([row(1),row(2,0)])
        with self.assertRaises(ValueError): select_two([row(1),row(2,-1)])

    def test_budget(self):
        with self.assertRaisesRegex(RuntimeError,'EXCEEDS_BUDGET'):
            select_two([row(1,MAX_BYTES//2+1),row(2,MAX_BYTES//2+1)])
        with self.assertRaises(ValueError): select_two([row(1),row(2)], MAX_BYTES+1)

    def test_ignore_eval_even_forged_four_digit(self):
        x=[row(1),row(2),{'path':'human_demos/train/no_distraction/toys_in_drawer/0080/front.mp4','bytes':9999}]
        self.assertEqual([x['id'] for x in select_two(x)['video_files']],[1,2])

    def test_local_inventory_absent_and_present(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.assertEqual(existing_local(root),[])
            video=root/media_rel(1)
            video.parent.mkdir(parents=True)
            video.write_bytes(b'abcd')
            v=existing_local(root)
            self.assertEqual(v[0]['id'],1)
            self.assertEqual(v[0]['video_bytes'],4)
            self.assertIsNone(v[0]['feature_bytes'])

    def test_save_immutable(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a/plan.json'
            save_once(p,{'a':1});save_once(p,{'a':1})
            with self.assertRaises(FileExistsError):save_once(p,{'a':2})

    def test_invalid_existing_video_cannot_be_claimed_decodable(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/media_rel(0);p.parent.mkdir(parents=True)
            p.write_bytes(b'garbage' * 2000)
            with self.assertRaisesRegex(RuntimeError,'INVALID_OFFICIAL_TRAIN_MP4'):
                assert_video_sources({'video_files':[row(0,len(b'garbage'*2000))]},Path(d))

    def test_no_cache_prevents_gcs(self):
        with tempfile.TemporaryDirectory() as d:
            stub=types.ModuleType('scripts.check_pi05_siglip_cache')
            stub.trusted_mirror_install=lambda project, home: False
            with patch.dict(sys.modules, {'scripts.check_pi05_siglip_cache': stub}):
                with patch.dict('os.environ', {'OPENPI_DATA_HOME': d}):
                    self.assertIsNone(cache_home(Path(d)))

    def test_existing_hf_transport_allowlist_is_train_only_for_new_task(self):
        from agcofl.resumable_hf_assets import _valid
        good=media_rel(0)
        self.assertEqual(_valid({'path':good,'bytes':10240},
            'LogosRoboticsLab/roboreel','dataset','https://hf-mirror.com')[1],good)
        for bad in ('human_demos/train/no_distraction/toys_in_drawer/0080/front.mp4',
                    'human_demos/train/no_distraction/push_cube/0000/front.mp4',
                    'human_demos/train/no_distraction/toys_in_drawer/0000/left.mp4'):
            with self.assertRaises(ValueError):
                _valid({'path':bad,'bytes':10240},
                       'LogosRoboticsLab/roboreel','dataset','https://hf-mirror.com')

    def test_train_path_regex_not_eval(self):
        self.assertTrue(REMOTE.fullmatch(media_rel(0)))
        self.assertFalse(REMOTE.fullmatch('human_demos/eval/no_distraction/toys_in_drawer/0080/front.mp4'))


if __name__=='__main__':unittest.main()
