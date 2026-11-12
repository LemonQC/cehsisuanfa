"""Stdlib fixture tests: physical evidence and source videos stay distinct."""
import json
import tempfile
import unittest
from pathlib import Path
from scripts import m1_existing_asset_decision as d


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data)+'\n')


def make_b2(root, source=True, direct=True, clear=True, raw=True, video=False):
    base=root/'outputs/m1b2_same_goal_dev'
    for mode, ok, steps in [('source',source,170),('target_direct',direct,31),('target_clear',clear,51)]:
        status={
            'kind': 'source_pick_to_same_goal' if mode=='source' else mode,
            'seed': 120,'state_before':[0.,0.,.02], 'goal':[.15,0.,.02],
            'same_final_goal_config':True, 'physically_verified_matched_success':ok,
            'error': None, 'steps' if mode=='source' else 'actual_rollout_steps':steps,
            'gate':{'valid':source} if mode=='source' else {},
            'blocker_before':[.077,0.,.025],
            'push_only_success':direct if mode=='target_direct' else clear,
            'clear_and_push_success':clear if mode=='target_clear' else False,
            'blocker_effect_physically_inserted':mode=='target_clear',
        }
        write(base/mode/'status.json',status)
        if raw:
            write(base/mode/'physical_trace.json',[{'t':i} for i in range(steps)])
    if video:
        (base/'source'/'source.mp4').write_bytes(b'v'*2048)
        (base/'source'/'source.h5').write_bytes(b'h'*2048)


def make_b3(root, report=True):
    src=root/d.B3_SOURCE
    tgt=root/d.B3_TARGET
    write(src/'status.json',{
        'version':'m1_source_geom_aligned_topdown_v1','scene':'AGCOFLM1B3Corridor-v0',
        'seed':120,'success':False,'actual_steps':65})
    write(src/'physical_trace.json',[{'t':i} for i in range(65)])
    write(tgt/'status.json',{
        'scene':'AGCOFLM1B3Corridor-v0','seed':120,'success':True,'steps':66,
        'clearance_valid':True,'goal':[.21,0,.02], 'goal_error_m':.00819,
        'push_only':{'valid':True}})
    write(tgt/'physical_trace.json',[{'t':i} for i in range(66)])
    if report:
        write(root/d.B3_PLAN,{
            'decision':'INCONCLUSIVE_IK_OR_COLLISION_FAILURE','new_physx_actions':0,
            'authorizes_physics':False,'planning':{'ik_solutions_returned':0,'screw_status':'screw plan failed'}})


class ExistingAssetsTests(unittest.TestCase):
    def test_b2_direct_success_refutes_insert_despite_source_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);make_b2(root,video=True)
            r=d.audit(root)
            self.assertTrue(r['b2']['physics_pair_from_existing_statuses'])
            self.assertTrue(r['b2']['raw_traces_complete'])
            self.assertTrue(r['b2']['source_robot_video_on_disk'])
            self.assertFalse(r['b2']['source_video_human_provenance'])
            self.assertEqual(r['b2']['causal_insert_result'],'REFUTED_BY_SUCCESSFUL_DIRECT_PUSH')
            self.assertEqual(r['decision'],'NO_GO_FOR_CURRENT_M1_INSERT_METHOD_START')
            self.assertEqual(r['new_physx_actions'],0)

    def test_b2_direct_fail_clear_success_only_conditional(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);make_b2(root,direct=False,video=True)
            r=d.audit(root)
            self.assertEqual(r['b2']['causal_insert_result'],
                             'SCRIPTED_CONTRAST_ONLY_NEEDS_STRONG_TAMP_AND_CAUSAL_PROOF')
            self.assertEqual(r['decision'],
                             'CANDIDATE_REQUIRES_VIDEO_INFORMATION_AND_STRONG_TAMP_GATE')
            self.assertFalse(r['b2']['video_information_gain_tested'])
            self.assertFalse(r['b2']['equal_information_strong_tamp_executed'])

    def test_missing_source_video_never_fabricates_video_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);make_b2(root,direct=False)
            r=d.audit(root)
            self.assertTrue(r['b2']['physics_pair_from_existing_statuses'])
            self.assertFalse(r['b2']['source_robot_video_on_disk'])
            self.assertEqual(r['decision'],'NO_GO_FOR_CURRENT_M1_INSERT_METHOD_START')

    def test_mismatched_goal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);make_b2(root)
            status=root/'outputs/m1b2_same_goal_dev/target_direct/status.json'
            v=json.loads(status.read_text());v['goal'][0]=.21;write(status,v)
            with self.assertRaisesRegex(ValueError,'different physical start/goal'):
                d.audit(root)

    def test_b3_target_is_not_same_goal_successful_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);make_b2(root);make_b3(root)
            r=d.audit(root)
            self.assertTrue(r['b3']['target_success'])
            self.assertFalse(r['b3']['source_success'])
            self.assertFalse(r['b3']['physics_pair_from_existing_statuses'])
            self.assertEqual(r['b3']['planning_audit'],'IK_0_SCREW_FAILED_NO_PHYSICS_AUTHORIZATION')
            self.assertEqual(r['decision'],'NO_GO_FOR_CURRENT_M1_INSERT_METHOD_START')
            self.assertIn('| B3 x=.210 | NO | YES (v6) |',d.markdown(r))

    def test_b3_corrupted_planning_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);make_b3(root)
            write(root/d.B3_PLAN,{'decision':'PLANNING_CANDIDATE_RECONSTRUCTED_POSE_ONLY',
                                  'planning':{'ik_solutions_returned':1},
                                  'new_physx_actions':0,'authorizes_physics':False})
            with self.assertRaisesRegex(ValueError,'planning report differs'):
                d.b3_review(root)

    def test_trace_length_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);make_b2(root)
            write(root/'outputs/m1b2_same_goal_dev/source/physical_trace.json',[{'t':0}])
            with self.assertRaisesRegex(ValueError,'Trace/status action mismatch'):
                d.b2_review(root)

    def test_write_once_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'output.json'
            d.write_once(p,'a');d.write_once(p,'a')
            with self.assertRaises(FileExistsError):d.write_once(p,'b')

    def test_absent_data_no_physics_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=d.audit(Path(tmp))
            self.assertFalse(r['b2']['physics_pair_from_existing_statuses'])
            self.assertFalse(r['b3']['target_success'])
            self.assertEqual(r['decision'],'NO_GO_FOR_CURRENT_M1_INSERT_METHOD_START')


if __name__ == '__main__':
    unittest.main()
