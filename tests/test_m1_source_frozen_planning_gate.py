"""No simulator imports: validate conservative gates and frozen H5 handling."""
import json
import tempfile
import unittest
import sys
import types
from unittest import mock
from pathlib import Path

from scripts import m1_source_frozen_planning_gate as g


class OfflinePlanningGateTests(unittest.TestCase):
    def test_mplib_0_1_1_ik_uses_flat7_without_pose_class(self):
        import numpy as np
        fake_mplib = types.ModuleType('mplib')
        fake_mplib.__path__ = []
        fake_mplib.__version__ = '0.1.1'
        with mock.patch.dict(sys.modules, {
            'mplib': fake_mplib, 'mplib.pymp': None,
        }):
            value, contract = g.make_mplib_ik_pose([.1, .2, .3, 1, 0, 0, 0])
        self.assertIsInstance(value, np.ndarray)
        self.assertEqual(value.shape, (7,))
        np.testing.assert_array_equal(value, [.1, .2, .3, 1, 0, 0, 0])
        self.assertEqual(contract, 'mplib_0.1.1_flat_7d_xyz_wxyz')

    def test_mplib_0_1_1_collision_uses_explicit_qpos_with_point_cloud(self):
        import numpy as np
        q = np.arange(9, dtype=float)
        class LegacyPlanner:
            def __init__(self):
                self.calls=[]
            def check_for_self_collision(self, articulation=None, qpos=None):
                if articulation is not None:
                    raise ValueError('Wrong articulation argument')
                np.testing.assert_array_equal(qpos, q)
                self.calls.append('self_qpos')
                return []
            def check_for_env_collision(self, articulation=None, qpos=None,
                                        with_point_cloud=False, use_attach=False):
                if articulation is not None:
                    raise ValueError('Wrong articulation argument')
                np.testing.assert_array_equal(qpos, q)
                self.assertion = (with_point_cloud is True and use_attach is False)
                self.calls.append('env_qpos_cloud')
                return ['known_wall_contact']
        planner=LegacyPlanner()
        r=g.collision_query(planner,q,'0.1.1')
        self.assertEqual(planner.calls,['self_qpos','env_qpos_cloud'])
        self.assertTrue(planner.assertion)
        self.assertEqual(r['self']['observed'],False)
        self.assertEqual(r['environment']['observed'],True)
        self.assertEqual(r['environment']['count'],1)
        self.assertEqual(r['environment']['api'],'mplib_0.1.1_explicit_qpos')

    def test_missing_legacy_collision_api_cannot_be_read_as_clearance(self):
        import numpy as np
        class Stub:
            def check_for_self_collision(self, articulation=None, qpos=None):
                return []
        r=g.collision_query(Stub(),np.zeros(9),'0.1.1')
        self.assertIsNone(r['environment']['observed'])
        self.assertEqual(r['environment']['reason'],'API_UNAVAILABLE')
        self.assertEqual(g.verdict('Success',False,None,'Success'),
                         'INCONCLUSIVE_COLLISION_API')

    def test_mplib_ik_pose_uses_pymp_when_root_has_no_pose(self):
        import numpy as np
        fake_mplib = types.ModuleType('mplib')
        fake_mplib.__path__ = []
        fake_mplib.__version__ = 'test-pinned'
        fake_pymp = types.ModuleType('mplib.pymp')
        class FakePose:
            def __init__(self, p, q):
                self.p, self.q = p, q
        fake_pymp.Pose = FakePose
        with mock.patch.dict(sys.modules, {'mplib': fake_mplib,
                                           'mplib.pymp': fake_pymp}):
            obj, source = g.make_mplib_ik_pose([.1, .2, .3, 1, 0, 0, 0])
        self.assertIsInstance(obj, FakePose)
        self.assertEqual(source, 'mplib.pymp.Pose')
        np.testing.assert_array_equal(obj.p, [.1, .2, .3])
        np.testing.assert_array_equal(obj.q, [1., 0., 0., 0.])

    def test_mplib_ik_pose_reports_missing_binding_not_fake_success(self):
        fake_mplib = types.ModuleType('mplib')
        fake_mplib.__path__ = []
        fake_mplib.__version__ = 'test-missing'
        with mock.patch.dict(sys.modules, {'mplib': fake_mplib,
                                           'mplib.pymp': None}):
            with self.assertRaisesRegex(RuntimeError, 'MPLIB_IK_POSE_API_UNAVAILABLE'):
                g.make_mplib_ik_pose([.1, .2, .3, 1, 0, 0, 0])

    def test_mplib_ik_pose_invalid_quaternion_rejected_without_planning(self):
        with self.assertRaisesRegex(ValueError, 'QUATERNION_NOT_UNIT'):
            g.make_mplib_ik_pose([.1, .2, .3, 2, 0, 0, 0])

    def test_collision_and_path_have_distinct_outcomes(self):
        self.assertEqual(g.verdict('IK Failed', None, None, 'Failed'),
                         'INCONCLUSIVE_IK_OR_COLLISION_FAILURE')
        self.assertEqual(g.verdict('Success', True, False, 'Failed'),
                         'GOAL_COLLISION_REPORTED_BY_PLANNER')
        self.assertEqual(g.verdict('Success', False, None, 'Success'),
                         'INCONCLUSIVE_COLLISION_API')
        self.assertEqual(g.verdict('Success', False, False, 'Failed'),
                         'IK_CANDIDATE_BUT_SCREW_REJECTED')
        self.assertEqual(g.verdict('Success', False, False, 'Success'),
                         'PLANNING_CANDIDATE_RECONSTRUCTED_POSE_ONLY')

    def test_no_physical_step_wrapper(self):
        class FauxEnv:
            action_space = 'fake'
            def __init__(self):
                self.unwrapped = self
            def step(self, action):
                raise AssertionError('Danger: physical step invoked')
        with self.assertRaisesRegex(RuntimeError, 'FORBIDDEN_NEW_PHYSX_STEP'):
            g.NoPhysicalStep(FauxEnv()).step([1])

    def test_refuses_to_overwrite_changed_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'a.json'
            g.write_once(p, {'decision': 'INCONCLUSIVE'})
            g.write_once(p, {'decision': 'INCONCLUSIVE'})
            with self.assertRaises(FileExistsError):
                g.write_once(p, {'decision': 'SUCCESS'})

    def test_h5_qpos_last_frame_and_ambiguous_detection(self):
        try:
            import h5py
            import numpy as np
        except ImportError:
            self.skipTest('h5py and numpy required for H5 fixture')
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            p=d/'source.h5'
            arr=np.zeros((66,9),dtype=np.float64)
            arr[-1,-2:]=.023
            with h5py.File(p,'w') as f:
                f.create_dataset('traj_0/obs/agent/qpos',data=arr)
                f.create_dataset('traj_0/actions', data=np.zeros((65, 8), dtype=np.float32))
            r=g.extract_snapshot(d)
            self.assertEqual(r['qpos_snapshot_index'],65)
            self.assertEqual(r['qpos_9'][-1],.023)
            with h5py.File(p,'a') as f:
                other=arr.copy();other[-1,0]=2
                f.create_dataset('traj_0/env_states/robot/qpos',data=other)
            with self.assertRaisesRegex(ValueError, 'AMBIGUOUS_QPOS'):
                g.extract_snapshot(d)
            # Explicit selection must not bypass conflicting recorded qpos.
            with self.assertRaisesRegex(ValueError, 'AMBIGUOUS_QPOS'):
                g.extract_snapshot(d,dataset='traj_0/obs/agent/qpos')

    def test_recovers_real_65_step_style_packed_panda_31d(self):
        import h5py
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source.h5'
            packed=np.zeros((66,31), dtype=np.float32)
            packed[:,3]=1.0  # root quaternion wxyz
            packed[:,13:20]=np.array([0.,-.79,0.,-2.36,0.,1.57,.79],dtype=np.float32)
            packed[:,20:22]=.023
            packed[:,22:31]=.005
            with h5py.File(p,'w') as f:
                f.create_dataset('traj_0/actions',data=np.zeros((65,8),dtype=np.float32))
                f.create_dataset('traj_0/env_states/articulations/panda',data=packed)
                cube=np.zeros((66,13),dtype=np.float32)
                cube[:,3]=1.0
                cube[:,2]=.02
                f.create_dataset('traj_0/env_states/actors/cube',data=cube)
            r=g.extract_snapshot(Path(tmp))
            self.assertEqual(r['dataset'],g.PACKED_PANDA_PATH)
            self.assertEqual(r['layout'],'MANISKILL_ARTICULATION_ROOT13_QPOS9_QVEL9')
            self.assertEqual(len(r['packed_articulation_state_31']),31)
            self.assertEqual(len(r['qpos_9']),9)
            self.assertAlmostEqual(r['qpos_9'][-1],.023,places=6)
            self.assertAlmostEqual(r['packed_qvel_9'][0],.005,places=6)
            self.assertEqual(r['qpos_snapshot_index'],65)
            self.assertEqual(r['actor_last_xyz']['cube'],[0.,0.,float(cube[-1,2])])

    def test_rejects_misidentified_packed_states_not_well_formed(self):
        import h5py
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source.h5'
            bad=np.zeros((66,31),dtype=np.float32)
            with h5py.File(p,'w') as f:
                f.create_dataset('traj_0/actions',data=np.zeros((65,8)))
                f.create_dataset(g.PACKED_PANDA_PATH,data=bad)
            with self.assertRaisesRegex(ValueError,'ROOT_QUATERNION_NOT_UNIT'):
                g.extract_snapshot(Path(tmp))
            with h5py.File(p,'a') as f:
                f[g.PACKED_PANDA_PATH][:,3]=1.
                f[g.PACKED_PANDA_PATH][-1,21]=.08
            with self.assertRaisesRegex(ValueError,'JOINT_SANITY_FAILED'):
                g.extract_snapshot(Path(tmp))

    def test_named_and_packed_joint_values_must_agree(self):
        import h5py
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source.h5'
            packed=np.zeros((66,31),dtype=np.float32)
            packed[:,3]=1.0
            with h5py.File(p,'w') as f:
                f.create_dataset('traj_0/actions',data=np.zeros((65,8)))
                f.create_dataset(g.PACKED_PANDA_PATH,data=packed)
                named=np.zeros((66,9),dtype=np.float32)
                named[-1,0]=1.0
                f.create_dataset('traj_0/obs/agent/qpos',data=named)
            with self.assertRaisesRegex(ValueError,'AMBIGUOUS_QPOS'):
                g.extract_snapshot(Path(tmp))

    def test_packed_articulation_requires_original_named_path(self):
        import h5py
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source.h5'
            with h5py.File(p,'w') as f:
                f.create_dataset('traj_0/actions',data=np.zeros((65,8)))
                f.create_dataset('traj_0/obs',data=np.zeros((66,31)))
            with self.assertRaisesRegex(ValueError,'NO_QPOS_DATASET'):
                g.extract_snapshot(Path(tmp))

    def test_rejects_flat_observation_not_identified_as_qpos(self):
        try:
            import h5py
            import numpy as np
        except ImportError:
            self.skipTest('h5py/numpy unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source.h5'
            with h5py.File(p,'w') as f:
                f.create_dataset('traj_0/obs',data=np.zeros((66,65)))
                f.create_dataset('traj_0/actions', data=np.zeros((65, 8), dtype=np.float32))
            with self.assertRaisesRegex(ValueError,'NO_QPOS_DATASET'):
                g.extract_snapshot(Path(tmp))


if __name__=='__main__':
    unittest.main()
