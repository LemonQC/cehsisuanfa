"""Real tensor forward/backward and tiny raw H5+video integration (CPU-only)."""
import importlib.util
from pathlib import Path
import json
import tempfile
import unittest

import h5py
import imageio.v2 as imageio
import numpy as np
import torch

MODULE=Path(__file__).resolve().parents[1]/'agcofl/robot_only_reimpl.py'
spec=importlib.util.spec_from_file_location('tro1_robot_only',MODULE)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class TestRobotOnly(unittest.TestCase):
    def test_default_root_matches_acquired_v120_robot_dataset(self):
        self.assertEqual(m.DATA_ROOT, m.ROOT/'outputs/v120_baselines/robot_train/robot_trajectories/no_distraction/push_cube')

    def test_articulation_offset_and_live_rgb(self):
        a=np.zeros((5,31),dtype=np.float32);a[:,13:22]=np.arange(9)
        s=m.pack_state(a)
        self.assertEqual(s.shape,(5,8));self.assertEqual(float(s[0,-1]),7.5)
        obs={'agent':{'qpos':torch.tensor([[0.,1.,2.,3.,4.,5.,6.,7.,8.]])},
             'sensor_data':{'base_camera':{'Color':torch.zeros((1,48,48,4),dtype=torch.uint8)}}}
        self.assertEqual(m.live_state(obs)[-1],7.5)
        self.assertEqual(m.live_rgb(obs).shape,(48,48,3))

    def test_bc_diffusion_and_infer(self):
        torch.set_num_threads(2)
        rgb=torch.rand(2,2,3,48,48);s=torch.rand(2,8);a=torch.rand(2,3,8)
        bc=m.new_model('b01',3);loss=((bc(rgb,s)-a)**2).mean();loss.backward()
        self.assertTrue(torch.isfinite(loss))
        dp=m.new_model('b02',3,steps=4);dpl=dp.loss(rgb,s,a);dpl.backward()
        self.assertTrue(torch.isfinite(dpl));self.assertEqual(tuple(dp.sample(rgb,s).shape),(2,3,8))
        stats={k:np.ones(8,dtype=np.float32) for k in ('state_std','action_std')}
        stats.update(state_mean=np.zeros(8,dtype=np.float32),action_mean=np.zeros(8,dtype=np.float32))
        ck={'method':'b01','horizon':3,'side':48,'stats':stats}
        chunk=m.infer(bc,ck,[np.zeros((48,48,3),dtype=np.uint8)],np.zeros(8,dtype=np.float32))
        self.assertEqual(chunk.shape,(3,8))

    def _demo(self,root,id,with_video=True):
        d=root/id;d.mkdir(parents=True)
        with h5py.File(d/'trajectory.h5','w') as f:
            f.create_dataset('traj_0/actions',data=np.ones((6,8),np.float32)*float(int(id)+1))
            st=np.zeros((7,31),np.float32);st[:,13:22]=np.arange(9)
            f.create_dataset('traj_0/env_states/articulations/panda_wristcam_hd',data=st)
        imgs=np.stack([np.full((48,48,3),t*10,dtype=np.uint8) for t in range(6)],axis=0)
        for cam in ('base_camera','hand_camera'):
            if with_video:imageio.mimwrite(str(d/f'{cam}.mp4'),imgs,fps=10,macro_block_size=1)

    def test_raw_demo_cache_split_and_guard(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'raw';root.mkdir()
            self._demo(root,'0000');self._demo(root,'0001');self._demo(root,'0080')
            cache=Path(d)/'cache'
            report=m.preprocess(root,cache,side=48,min_demos=64,allow_small_dev=True)
            self.assertEqual(report['label'],'SMALL_DEV_DATA_ONLY')
            self.assertEqual(len(report['demos']),2)
            self.assertEqual(sorted(report['train']+report['validation']),['0000','0001'])
            self.assertFalse((cache/'0080').exists())
            self.assertEqual(len(m.RobotDataset(cache,'train',3)),4)
            with self.assertRaises(FileExistsError):m.preprocess(root,cache,side=48,min_demos=2)
            train_data=m.RobotDataset(cache,'train',3)
            stats=m.train_stats(train_data)
            self.assertTrue(np.isfinite(stats['action_std']).all())
            with self.assertRaises(ValueError):m.eval_bank('x','y','z',mode='formal',episodes=2)
            out=Path(d)/'run'
            path=m.train('b01',cache,out,horizon=3,steps=1,batch=2,allow_small_dev=True,device='cpu')
            model,ck=m.load_policy(path)
            self.assertEqual(ck['demos'],2)
            self.assertTrue((out/'train_summary.json').is_file())
            with self.assertRaises(ValueError):m.eval_bank(path,'missing',Path(d)/'eval',mode='formal',episodes=20)
            dp_path=m.train('b02',cache,Path(d)/'dp_run',horizon=3,steps=1,batch=2,allow_small_dev=True,device='cpu')
            dp,dp_ck=m.load_policy(dp_path)
            self.assertEqual(dp_ck['method'],'b02')
            action=m.infer(dp,dp_ck,[np.zeros((48,48,3),np.uint8)],np.zeros(8,np.float32))
            self.assertEqual(action.shape,(3,8))

    def test_bad_frame_alignment_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'raw';root.mkdir()
            self._demo(root,'0000')
            d0=root/'0000'
            imageio.mimwrite(str(d0/'base_camera.mp4'),[np.zeros((48,48,3),dtype=np.uint8)]*4,fps=10,macro_block_size=1)
            with self.assertRaisesRegex(ValueError,'FRAME_COUNT_MISMATCH'):
                m.preprocess(root,Path(d)/'cache',side=48,min_demos=2,allow_small_dev=True)


if __name__=='__main__':unittest.main()
