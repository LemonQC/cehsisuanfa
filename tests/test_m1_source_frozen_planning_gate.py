"""No simulator imports: validate conservative gates and frozen H5 handling."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts import m1_source_frozen_planning_gate as g


class OfflinePlanningGateTests(unittest.TestCase):
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
