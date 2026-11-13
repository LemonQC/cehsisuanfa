"""Offline tests: exact authorized 44962 archive, no secrets, no unsafe extraction."""
from __future__ import annotations
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

from scripts.download_roboreel_partnet44962_hf import (
    REPO_ID, ARCHIVE, MODEL_ID, OFFICIAL, MIRROR, safe_archive_entries,
    extract_single_model, authorized_hf_single, _dataset_zip_info,
)
from scripts.prepare_roboreel_drawer_partnet import locations, inspect

URDF='<robot name="drawer"><link name="base"><visual><geometry><mesh filename="textured_objs/base.obj"/></geometry></visual></link></robot>\n'+(' '*140)


def archive(path, files, prefix='44962/'):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,payload in files.items():
            z.writestr(prefix+name,payload)
    return path


class TestAuthorizedHfPartnet(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project=Path(self.tmp.name)
        self.loc=locations(self.project)
        source=self.loc['upstream_metadata']
        source.parent.mkdir(parents=True)
        source.write_text(json.dumps({'44962':{'partnet_mobility_id':44962,'scale':0.3}}))
        self.zp=self.project/'44962.zip'
        archive(self.zp,{'mobility.urdf':URDF,'textured_objs/base.obj':'v 0 0 0\n'})
        self.params=[]

    def info(self,size=None):
        return SimpleNamespace(sha='a'*40,siblings=[SimpleNamespace(
            rfilename=ARCHIVE,size=self.zp.stat().st_size if size is None else size)])

    def fake_api(self,endpoint,token):
        self.params.append(('api',endpoint,token))
        return SimpleNamespace(dataset_info=lambda repo_id,files_metadata,token:
                               self.info())

    def fake_download(self,**kwargs):
        self.params.append(('download',kwargs))
        return str(self.zp)

    def test_authorized_model_install_only_one_archive(self):
        ret=authorized_hf_single(self.project,token='hf_TEST_SECRET',
            api_cls=self.fake_api,file_downloader=self.fake_download)
        self.assertEqual(ret['status'],'READY_FOR_SCENE_INIT')
        self.assertTrue(self.loc['model'].joinpath('mobility.urdf').exists())
        self.assertTrue(self.loc['metadata'].exists())
        self.assertEqual(ret['hf_provenance']['dataset'],REPO_ID)
        self.assertEqual(ret['hf_provenance']['filename'],ARCHIVE)
        self.assertEqual(ret['hf_provenance']['endpoint'],MIRROR)
        self.assertEqual(self.params[-1][1]['filename'],ARCHIVE)
        self.assertEqual(self.params[-1][1]['repo_type'],'dataset')
        self.assertEqual(self.params[-1][1]['revision'],'a'*40)
        self.assertTrue(self.params[-1][1]['cache_dir'].startswith(str(self.loc['home'])))

    def test_idempotent_no_second_remote_download(self):
        authorized_hf_single(self.project,token='hf_TEST',
            api_cls=self.fake_api,file_downloader=self.fake_download)
        def fail(**kwargs):raise AssertionError('unexpected remote download')
        self.assertEqual(authorized_hf_single(self.project, token='hf_TEST',
            api_cls=self.fake_api,file_downloader=fail)['status'],'READY_FOR_SCENE_INIT')

    def test_official_fallback_after_mirror_auth_error(self):
        def api(endpoint,token):
            if endpoint==MIRROR:
                raise PermissionError('HF_GATE_BLOCKED')
            return self.fake_api(endpoint,token)
        ret=authorized_hf_single(self.project,token='hf_SECRET',api_cls=api,
            file_downloader=self.fake_download)
        self.assertEqual(ret['hf_provenance']['endpoint'],OFFICIAL)

    def test_auth_required_without_token(self):
        with self.assertRaisesRegex(RuntimeError,'HF_AUTH_REQUIRED'):
            authorized_hf_single(self.project,token='INVALID',
                api_cls=self.fake_api,file_downloader=self.fake_download)
        self.assertFalse(self.loc['model'].exists())

    def test_no_fallback_to_wrong_file_or_full_snapshot(self):
        def api(endpoint,token):return SimpleNamespace(dataset_info=lambda *a,**k:
            SimpleNamespace(sha='a'*40,siblings=[SimpleNamespace(rfilename='all_models.tar',size=1024)]))
        with self.assertRaisesRegex(RuntimeError,'AUTHENTICATED_HF_44962_FETCH_FAILED'):
            authorized_hf_single(self.project,token='hf_XYZ',api_cls=api,
                file_downloader=self.fake_download)
        self.assertFalse(self.loc['model'].exists())
        self.assertFalse(any(x[0]=='download' for x in self.params))

    def test_no_token_printed_on_failed_download(self):
        secret='hf_DONT_PRINT_ME'
        def downloader(**kwargs):
            raise RuntimeError('https://hf-mirror.com/?token='+secret)
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaises(RuntimeError) as err:
                authorized_hf_single(self.project,token=secret,api_cls=self.fake_api,
                    file_downloader=downloader)
        self.assertNotIn(secret,str(err.exception)+output.getvalue())

    def test_zip_rejects_traversal(self):
        archive(self.zp,{'mobility.urdf':URDF,'../secret':'malware'})
        with self.assertRaisesRegex(ValueError,'UNSAFE_ZIP_FILENAME'):
            extract_single_model(self.zp,self.project/'unpacked')

    def test_zip_rejects_different_model(self):
        archive(self.zp,{'mobility.urdf':URDF})
        with zipfile.ZipFile(self.zp,'a') as z:z.writestr('103498/mobility.urdf',URDF)
        with self.assertRaisesRegex(ValueError,'ZIP_CONTAINS_OTHER_MODEL'):
            extract_single_model(self.zp,self.project/'unpacked')

    def test_zip_rejects_symlink(self):
        archive(self.zp,{'mobility.urdf':URDF})
        with zipfile.ZipFile(self.zp,'a') as z:
            s=zipfile.ZipInfo('44962/escape')
            s.create_system=3
            s.external_attr=(0o120777<<16)
            z.writestr(s,'/etc/passwd')
        with self.assertRaisesRegex(ValueError,'SYMLINK_ZIP_MEMBER'):
            extract_single_model(self.zp,self.project/'unpacked')

    def test_zip_rejects_over_budget_without_materializing(self):
        with patch('scripts.download_roboreel_partnet44962_hf.MAX_MODEL_BYTES',128):
            with self.assertRaisesRegex(ValueError,'OVERSIZED_ARCHIVE_MEMBER|ARCHIVE_UNPACK_BUDGET'):
                extract_single_model(self.zp,self.project/'unpacked')
        self.assertFalse((self.project/'unpacked/44962').exists())

    def test_zip_flat_layout(self):
        archive(self.zp,{'mobility.urdf':URDF,'textured_objs/base.obj':'v 0 0 0\n'},prefix='')
        r=extract_single_model(self.zp,self.project/'unpacked')
        self.assertTrue(Path(r['model']).joinpath('mobility.urdf').is_file())

    def test_existing_partial_asset_refuses_overwrite(self):
        self.loc['model'].mkdir(parents=True)
        with self.assertRaisesRegex(FileExistsError,'EXISTING_PARTIAL_MODEL'):
            authorized_hf_single(self.project,token='hf_T',
                api_cls=self.fake_api,file_downloader=self.fake_download)

    def test_archive_size_pinned_before_download(self):
        def api(endpoint,token):return SimpleNamespace(dataset_info=lambda *a,**k:
            self.info(size=600*1024*1024))
        with self.assertRaisesRegex(RuntimeError,'AUTHENTICATED_HF_44962_FETCH_FAILED'):
            authorized_hf_single(self.project,token='hf_XYZ',api_cls=api,
                file_downloader=self.fake_download)
        self.assertFalse(any(x[0]=='download' for x in self.params))

    def test_disallow_untrusted_host(self):
        with self.assertRaisesRegex(ValueError,'UNAPPROVED_HF_ENDPOINT'):
            authorized_hf_single(self.project,token='hf_XYZ',
                api_cls=self.fake_api,file_downloader=self.fake_download,
                endpoints=('https://example.com',))

    def test_zip_bad_urdf_preserved_without_install(self):
        archive(self.zp,{'mobility.urdf':'<notrobot/>'+' '*200})
        with self.assertRaisesRegex(RuntimeError,'AUTHENTICATED_HF_44962_FETCH_FAILED'):
            authorized_hf_single(self.project,token='hf_XYZ',
                api_cls=self.fake_api,file_downloader=self.fake_download)
        self.assertFalse(self.loc['model'].exists())


if __name__=='__main__':unittest.main()
