"""G0 is a conservative source-evidence gate, not a video semantics classifier."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest
from scripts.audit_goal_contrast_g0 import (
    make_report,audit_pair,check_authored_task,save_immutable,probe_video,CATALOG
)


class G0AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.project=Path(self.tmp.name)
        self.scenes=self.project/'third_party/roboreel_models/roboreel_env/src/envs'
        self.scenes.mkdir(parents=True)
        (self.scenes/'push_cube.py').write_text('PUSH_DISTANCE = 0.15\ntarget_xyz[..., 0] += self.PUSH_DISTANCE\n')
        (self.scenes/'stack_cups.py').write_text('StackCups-v1 dark_blue_cup red_cup light_blue_cup')

    def test_true_author_task_constants_are_fixed(self):
        x=check_authored_task(self.project)
        self.assertTrue(x['push_source_verified'])
        self.assertTrue(x['cups_source_verified'])
        self.assertFalse(x['target_contrast_provided'])

    def test_without_manifest_is_blocked_without_download(self):
        x=make_report(self.project,None)
        self.assertEqual(x['decision'],'G0_NOT_PASSED')
        self.assertEqual(x['physical_rollouts'],0)
        self.assertEqual(x['remote_downloads'],0)

    def test_official_labels_only_not_a_verified_video_pair(self):
        q=[x for x in CATALOG if x['dataset']=='Something-Something V2'][0]
        self.assertEqual(q['opposing_goals_verified'],'class_labels_only')
        self.assertFalse(q['actual_human_video_pair_verified'])

    def _manifest(self,**changes):
        clips=[]
        for goal,name in [('-x','left.mp4'),('+x','right.mp4')]:
            f=self.project/'inputs/g0_human'/name;f.parent.mkdir(parents=True,exist_ok=True)
            f.write_bytes((name*300).encode())
            item={'goal_robot_x':goal,'path':str(f.relative_to(self.project)),
                  'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),
                  'source_type':'self_recorded','rights':'self_recorded_consent_confirmed',
                  'human_hand_demonstration':True,
                  'initial_scene_equivalence_attested':True,
                  'camera_to_robot_x_mapping_documented':True,
                  'visible_target_region':'both target zones marked'}
            clips.append(item)
        payload={'target_task':'video_goal_push','clips':clips}
        for key,value in changes.items():payload[key]=value
        path=self.project/'inputs/g0_human/manifest.json'
        path.write_text(json.dumps(payload))
        return path,payload

    def test_bad_task_blocked(self):
        path,payload=self._manifest(target_task='RoboReel PushCube')
        self.assertEqual(audit_pair(path,self.project)['status'],'G0_BLOCKED')

    def test_good_files_reach_manual_review_not_claim_success(self):
        path,payload=self._manifest()
        with patch('scripts.audit_goal_contrast_g0.probe_video',return_value={'ok':True,'width':224,'height':224}):
            report=audit_pair(path,self.project)
        self.assertEqual(report['status'],'G0_CANDIDATE_REQUIRES_MANUAL_VISUAL_REVIEW')
        self.assertFalse(report['video_semantics_independently_verified'])

    def test_identical_media_bytes_are_rejected(self):
        path,payload=self._manifest()
        a=self.project/payload['clips'][0]['path']
        b=self.project/payload['clips'][1]['path']
        b.write_bytes(a.read_bytes())
        payload['clips'][1]['sha256']=hashlib.sha256(b.read_bytes()).hexdigest()
        path.write_text(json.dumps(payload))
        with patch('scripts.audit_goal_contrast_g0.probe_video',return_value={'ok':True}):
            report=audit_pair(path,self.project)
        self.assertIn('SAME_VIDEO_FILE_REUSED',report['reason_codes'])

    def test_same_goal_cannot_pass(self):
        path,payload=self._manifest()
        payload['clips'][1]['goal_robot_x']='-x'
        path.write_text(json.dumps(payload))
        with patch('scripts.audit_goal_contrast_g0.probe_video',return_value={'ok':True}):
            report=audit_pair(path,self.project)
        self.assertIn('OPPOSITE_GOALS_NOT_BOTH_PRESENT',report['reason_codes'])

    def test_unlicensed_video_blocked(self):
        path,payload=self._manifest()
        payload['clips'][0]['rights']='unverified'
        path.write_text(json.dumps(payload))
        with patch('scripts.audit_goal_contrast_g0.probe_video',return_value={'ok':True}):
            report=audit_pair(path,self.project)
        self.assertIn('RIGHTS_UNVERIFIED',report['reason_codes'])

    def test_missing_coordinate_proof_blocked(self):
        path,payload=self._manifest()
        payload['clips'][0]['camera_to_robot_x_mapping_documented']=False
        path.write_text(json.dumps(payload))
        with patch('scripts.audit_goal_contrast_g0.probe_video',return_value={'ok':True}):
            report=audit_pair(path,self.project)
        self.assertIn('NO_CAMERA_TO_ROBOT_MAPPING',report['reason_codes'])

    def test_outside_project_is_blocked(self):
        path,payload=self._manifest()
        payload['clips'][0]['path']='../../../../outside.mp4'
        path.write_text(json.dumps(payload))
        with patch('scripts.audit_goal_contrast_g0.probe_video',return_value={'ok':True}):
            report=audit_pair(path,self.project)
        self.assertIn('INVALID_PATH_OUTSIDE_PROJECT',report['reason_codes'])

    def test_heldout_eval_video_path_blocked(self):
        path,payload=self._manifest()
        old=self.project/payload['clips'][0]['path']
        eval_dir=self.project/'human_demos/eval/no_distraction/push_cube/0080';eval_dir.mkdir(parents=True)
        f=eval_dir/'front.mp4';f.write_bytes(old.read_bytes())
        payload['clips'][0]['path']=str(f.relative_to(self.project))
        path.write_text(json.dumps(payload))
        with patch('scripts.audit_goal_contrast_g0.probe_video',return_value={'ok':True}):
            report=audit_pair(path,self.project)
        self.assertIn('HELDOUT_EVAL_SOURCE_FORBIDDEN',report['reason_codes'])

    def test_immutable_results_deduplicate(self):
        out=self.project/'outputs/g0'
        payload={'decision':'G0_NOT_PASSED','physical_rollouts':0}
        f=save_immutable(out,payload)
        self.assertEqual(save_immutable(out,payload),f)
        f.write_text('tampered')
        with self.assertRaises(FileExistsError):save_immutable(out,payload)

    def test_no_manifest_files_are_not_silently_accepted(self):
        path,payload=self._manifest()
        (self.project/payload['clips'][0]['path']).unlink()
        with patch('scripts.audit_goal_contrast_g0.probe_video',return_value={'ok':True}):
            report=audit_pair(path,self.project)
        self.assertIn('VIDEO_NOT_PRESENT_OR_BAD_EXTENSION',report['reason_codes'])

    def test_video_decode_failure_blocks(self):
        path,payload=self._manifest()
        with patch('scripts.audit_goal_contrast_g0.probe_video',return_value={'ok':False,'reason':'INVALID'}):
            report=audit_pair(path,self.project)
        self.assertIn('VIDEO_CANNOT_DECODE:INVALID',report['reason_codes'])

    def test_first_first_second_goal_logic(self):
        path,payload=self._manifest()
        payload['clips'][1]['goal_robot_x']=None
        path.write_text(json.dumps(payload))
        with patch('scripts.audit_goal_contrast_g0.probe_video',return_value={'ok':True}):
            report=audit_pair(path,self.project)
        self.assertIn('OPPOSITE_GOALS_NOT_BOTH_PRESENT',report['reason_codes'])

if __name__=='__main__':unittest.main()
