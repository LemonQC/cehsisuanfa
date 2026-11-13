"""Offline tests: no GPU, network or ManiSkill dependencies are imported."""
from pathlib import Path
import json
import tempfile
import unittest
from scripts.b06_prephysx_resume_gate import check_unstarted_run


class B06ResumeGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project = Path(self.tmp.name)
        self.out = self.project / 'outputs/research_runs/b06_uniskill_stackcups_dev_v1'
        self.out.mkdir(parents=True)
        self.base = {
            'upstream_revision': 'f'*40,
            'policy_checkpoint_sha256': 'a'*64,
            'idm_checkpoint_sha256': 'b'*64,
            'train_videos': [{'id': i, 'sha256': str(i)*64} for i in [0, 1]],
            'skill_arrays': [{'id': i, 'sha256': str(i+2)*64} for i in [0, 1]],
        }
        self.plan = {'schema': 'TRO1_B06_UNISKILL_STACKCUPS_PREFLIGHT_V1',
                     'status': 'READY_FOR_DEV', 'task': 'stack_cups',
                     'source_train_demo_ids': [0, 1], 'new_physx_actions': 0,
                     'base_checks': self.base}
        self.manifest = {
            'schema': 'TRO1_B06_UNISKILL_STACKCUPS_DEV_V1',
            'status': 'RUNNING_BOUNDED_DEV_NOT_FORMAL',
            'task': 'stack_cups', 'episodes': 2, 'max_steps': 800,
            'source_train_demo_ids': [0, 1], 'train_videos_only': True,
            'no_heldout_eval_videos': True, 'no_policy_training': True,
            'official_upstream_revision': 'f'*40,
            'policy_checkpoint_sha256': 'a'*64,
            'idm_checkpoint_sha256': 'b'*64,
            'source_video_sha256': {'0': '0'*64, '1': '1'*64},
            'skill_sha256': {'0': '2'*64, '1': '3'*64},
        }
        self.write()

    def write(self):
        (self.out/'preflight.json').write_text(json.dumps(self.plan))
        (self.out/'manifest.json').write_text(json.dumps(self.manifest))

    def test_prephysx_metadata_only_accepted_without_mutation(self):
        originals = {p.name: p.read_bytes() for p in self.out.iterdir()}
        check_unstarted_run(self.out, self.project, plan=self.plan)
        self.assertEqual(originals, {p.name:p.read_bytes() for p in self.out.iterdir()})

    def test_with_trial_folder_must_block(self):
        (self.out/'episode_00').mkdir()
        with self.assertRaises(ValueError): check_unstarted_run(self.out, self.project)

    def test_partial_progress_file_must_block(self):
        (self.out/'per_trial.partial.json').write_text('{}')
        with self.assertRaises(ValueError): check_unstarted_run(self.out, self.project)

    def test_unknown_file_blocks(self):
        (self.out/'unknown.tmp').write_text('x')
        with self.assertRaises(ValueError): check_unstarted_run(self.out, self.project)

    def test_symlink_blocks(self):
        (self.out/'symlink').symlink_to(self.out/'manifest.json')
        with self.assertRaises(ValueError): check_unstarted_run(self.out, self.project)

    def test_wrong_checkpoint_sha_blocks(self):
        self.manifest['policy_checkpoint_sha256'] = 'c'*64
        self.write()
        with self.assertRaises(ValueError): check_unstarted_run(self.out, self.project)

    def test_changed_current_plan_blocks(self):
        current = json.loads(json.dumps(self.plan))
        current['base_checks']['skill_arrays'][0]['sha256']='e'*64
        with self.assertRaises(ValueError): check_unstarted_run(self.out, self.project, current)

    def test_different_out_path_blocks(self):
        renamed=self.out.parent/'other'
        self.out.rename(renamed)
        with self.assertRaises(ValueError):check_unstarted_run(renamed,self.project)

    def test_registration_import_is_ahead_of_policy_init(self):
        root=Path(__file__).resolve().parents[1]
        source=(root/'scripts/roboreel_uniskill_video_dev.py').read_text()
        block=source[source.index('    # Pin the headless render'):source.index('    runner = UniSkillRunner(')]
        self.assertIn('import mani_skill.envs',block)
        self.assertIn('import envs',block)
        self.assertIn('gym.spec(TASKS[plan["task"]])',block)
        self.assertLess(block.index('import envs'),block.index('from evaluation.uniskill_runner'))

if __name__=='__main__': unittest.main()
