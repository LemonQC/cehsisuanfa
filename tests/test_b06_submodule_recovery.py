"""Pure offline Git tests; no GPU, network or user files touched."""
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/b06_recover_submodule_git.sh'

def git(*args, check=True):
    return subprocess.run(['git', *map(str,args)], text=True, capture_output=True, check=check)

class B06SubmoduleRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.origin=self.root/'original'
        self.target=self.root/'partial'
        git('init','-q',self.origin)
        git('-C',self.origin,'config','user.email','test@example.org')
        git('-C',self.origin,'config','user.name','Test')
        (self.origin/'entrypoint.py').write_text('pass\n')
        git('-C',self.origin,'add','entrypoint.py')
        git('-C',self.origin,'commit','-qm','pin')
        self.sha=git('-C',self.origin,'rev-parse','HEAD').stdout.strip()
        git('clone','-q','--no-checkout',self.origin,self.target)
        # A --no-checkout clone has a populated index, so 'HEAD -> nonexistent'
        # alone misleadingly looks dirty due to tracked files absent on disk.
        # Use an EMPTY index as an authentic interrupted submodule checkout.
        git('-C',self.target,'read-tree','--empty')
        git('-C',self.target,'symbolic-ref','HEAD','refs/heads/absent')
    def repair(self, sha=None, target=None):
        return subprocess.run(['bash',str(SCRIPT),str(target or self.target),
            sha or self.sha,str(self.origin),'entrypoint.py'],text=True,capture_output=True)
    def assert_ok(self,r):
        self.assertEqual(r.returncode,0, r.stdout+'\n'+r.stderr)
    def test_recover_unborn_head(self):
        r=self.repair();self.assert_ok(r)
        self.assertEqual(git('-C',self.target,'rev-parse','HEAD').stdout.strip(),self.sha)
        self.assertTrue((self.target/'entrypoint.py').is_file())
    def test_idempotent(self):
        self.assert_ok(self.repair())
        self.assert_ok(self.repair())
    def test_dirty_submodule_rejected(self):
        (self.target/'private.txt').write_text('protected')
        self.assertNotEqual(self.repair().returncode,0)
        self.assertEqual((self.target/'private.txt').read_text(),'protected')
    def test_not_git_worktree(self):
        p=self.root/'invalid';p.mkdir()
        self.assertNotEqual(self.repair(target=p).returncode,0)
    def test_unknown_sha_refused(self):
        self.assertNotEqual(self.repair(sha='0'*40).returncode,0)
    def test_clean_other_commit_restored_to_pinned_with_backup(self):
        self.assert_ok(self.repair())
        (self.target/'entrypoint.py').write_text('different\n')
        git('-C',self.target,'add','.')
        git('-C',self.target,'-c','user.email=test@example.org','-c','user.name=Test','commit','-qm','different')
        different=git('-C',self.target,'rev-parse','HEAD').stdout.strip()
        self.assertNotEqual(different,self.sha)
        r=self.repair();self.assert_ok(r)
        self.assertEqual(git('-C',self.target,'rev-parse','HEAD').stdout.strip(),self.sha)
        self.assertEqual(git('-C',self.target,'rev-parse',f'refs/b06-recovery/preexisting-{different}').stdout.strip(),different)
    def test_different_commit_with_dirty_file_refused(self):
        self.assert_ok(self.repair())
        (self.target/'entrypoint.py').write_text('changed\n')
        git('-C',self.target,'add','.')
        git('-C',self.target,'-c','user.email=test@example.org','-c','user.name=Test','commit','-qm','different')
        (self.target/'entrypoint.py').write_text('uncommitted\n')
        self.assertNotEqual(self.repair().returncode,0)
        self.assertEqual((self.target/'entrypoint.py').read_text(),'uncommitted\n')

if __name__=='__main__': unittest.main()
