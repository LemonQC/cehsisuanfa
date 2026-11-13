"""Offline exact-SHA B06 Git submodule recovery: no network, no GPU."""
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/b06_recover_submodule_git.sh'

def git(*args, cwd=None, check=True):
    return subprocess.run(['git', *map(str, args)], cwd=cwd,
        text=True, capture_output=True, check=check)

class B06SubmoduleRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.origin = self.root / 'original'
        self.target = self.root / 'partial'
        git('init', '-q', self.origin)
        git('-C', self.origin, 'config', 'user.email', 'test@example.org')
        git('-C', self.origin, 'config', 'user.name', 'Test')
        (self.origin / 'entrypoint.py').write_text('pass\n')
        git('-C', self.origin, 'add', 'entrypoint.py')
        git('-C', self.origin, 'commit', '-qm', 'pin')
        self.sha = git('-C', self.origin, 'rev-parse', 'HEAD').stdout.strip()
        git('clone', '-q', '--no-checkout', self.origin, self.target)
        # Simulate corrupted/unborn HEAD in already-created submodule worktree.
        git('-C', self.target, 'symbolic-ref', 'HEAD', 'refs/heads/absent')
    def run_repair(self):
        return subprocess.run(['bash', str(SCRIPT), str(self.target),
              self.sha, str(self.origin), 'entrypoint.py'],
              text=True, capture_output=True)
    def test_recover_unborn_head(self):
        result=self.run_repair()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(git('-C',self.target,'rev-parse','HEAD').stdout.strip(),self.sha)
        self.assertTrue((self.target/'entrypoint.py').is_file())
    def test_idempotent(self):
        self.assertEqual(self.run_repair().returncode,0)
        self.assertEqual(self.run_repair().returncode,0)
    def test_dirty_submodule_must_not_be_overwritten(self):
        (self.target/'private.txt').write_text('do not change')
        r=self.run_repair()
        self.assertNotEqual(r.returncode,0)
        self.assertTrue((self.target/'private.txt').is_file())
    def test_not_independent_git_worktree(self):
        invalid=self.root/'invalid'
        invalid.mkdir()
        r=subprocess.run(['bash',str(SCRIPT),str(invalid),self.sha,
                           str(self.origin),'entrypoint.py'],text=True,capture_output=True)
        self.assertNotEqual(r.returncode,0)
    def test_wrong_sha_refused(self):
        r=subprocess.run(['bash',str(SCRIPT),str(self.target),'0'*40,
                          str(self.origin),'entrypoint.py'],text=True,capture_output=True)
        self.assertNotEqual(r.returncode,0)
    def test_other_committed_head_must_not_be_replaced(self):
        git('-C',self.target,'checkout','-q','--detach',self.sha)
        (self.target/'entrypoint.py').write_text('changed\n')
        git('-C',self.target,'add','.')
        git('-C',self.target,'-c','user.name=Test','-c','user.email=test@example.org',
            'commit','-qm','different')
        r=self.run_repair()
        self.assertNotEqual(r.returncode,0)
        self.assertEqual((self.target/'entrypoint.py').read_text(),'changed\n')

if __name__=='__main__':
    unittest.main()
