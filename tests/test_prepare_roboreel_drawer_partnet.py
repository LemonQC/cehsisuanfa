import json
import io
import os
import sys
import types
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.prepare_roboreel_drawer_partnet import (
    download_authorized, configure_for_robot_dev,
    locations, inspect, validate_model, install_from_authorized_source,
    verified_upstream_metadata, ensure_metadata, MODEL_ID, MAX_MODEL_BYTES)

METADATA={'44817':{'partnet_mobility_id':44817,'scale':0.3},
          MODEL_ID:{'partnet_mobility_id':44962,'scale':0.3}}

class TestSingleDrawerAsset(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.loc=locations(self.root)
        self.loc['upstream_metadata'].parent.mkdir(parents=True)
        self.loc['upstream_metadata'].write_text(json.dumps(METADATA))
        self.source=self.root/'authorized/44962'
        self.source.mkdir(parents=True)
        (self.source/'mobility.urdf').write_text('<robot name="drawer"><link name="base" /></robot> '+(' '*160))
    def tearDown(self):self.temp.cleanup()
    def test_missing_not_ready(self):
        r=inspect(self.root)
        self.assertEqual(r['status'],'NEED_OFFICIAL_PARTNET_44962')
        self.assertEqual(r['new_physx_actions'],0)
        self.assertEqual(r['ms_asset_dir'],str(self.loc['home']))
    def test_metadata_pinned_id_and_scale(self):
        self.assertEqual(verified_upstream_metadata(self.loc['upstream_metadata'])[MODEL_ID]['scale'],0.3)
        self.loc['upstream_metadata'].write_text('{}')
        with self.assertRaises(ValueError):verified_upstream_metadata(self.loc['upstream_metadata'])
    def test_import_real_model_folder_to_maniskill_expected_layout(self):
        r=install_from_authorized_source(self.root,self.source)
        self.assertEqual(r['status'],'READY_FOR_SCENE_INIT')
        self.assertTrue(self.loc['model'].joinpath('mobility.urdf').is_file())
        self.assertEqual(r['mani_skill_asset_dir'],str(self.loc['data']))
        self.assertTrue(r['metadata_installed'])
    def test_import_parent_folder(self):
        r=install_from_authorized_source(self.root,self.source.parent)
        self.assertEqual(r['status'],'READY_FOR_SCENE_INIT')
    def test_import_rejects_wrong_id_or_partial(self):
        (self.source/'mobility.urdf').unlink()
        with self.assertRaises(ValueError):install_from_authorized_source(self.root,self.source)
        self.assertFalse(self.loc['model'].exists())
    def test_urdf_must_be_robot(self):
        (self.source/'mobility.urdf').write_text('<foo>' + (' '*160) + '</foo>')
        self.assertFalse(validate_model(self.source)['valid'])
    def test_existing_metadata_differing_refused(self):
        self.loc['metadata'].parent.mkdir(parents=True)
        self.loc['metadata'].write_text('not original')
        with self.assertRaises(FileExistsError):ensure_metadata(self.loc)
    def test_scene_asset_is_checked_before_maniskill_import(self):
        with patch.dict(os.environ, {'MS_ASSET_DIR':''}):
            with self.assertRaisesRegex(RuntimeError,'ROBOREEL_DRAWER_PARTNET_44962_NOT_READY'):
                configure_for_robot_dev(self.root,'toys_in_drawer')
            self.assertEqual(os.environ.get('MS_ASSET_DIR'), '')
            install_from_authorized_source(self.root,self.source)
            result=configure_for_robot_dev(self.root,'toys_in_drawer')
            self.assertEqual(result['status'],'READY_FOR_SCENE_INIT')
            self.assertEqual(os.environ['MS_ASSET_DIR'],str(self.loc['home']))

    def test_scene_asset_conflicting_MS_ASSET_DIR_is_refused(self):
        install_from_authorized_source(self.root,self.source)
        with patch.dict(os.environ, {'MS_ASSET_DIR':'/some/other/path'}):
            with self.assertRaisesRegex(RuntimeError,'MS_ASSET_DIR_POINTS_TO_DIFFERENT_LOCATION'):
                configure_for_robot_dev(self.root,'toys_in_drawer')

    def test_other_task_does_not_change_MS_ASSET_DIR(self):
        with patch.dict(os.environ, {'MS_ASSET_DIR':'/existing'}):
            self.assertEqual(configure_for_robot_dev(self.root,'push_cube')['status'],
                             'NOT_NEEDED_FOR_TASK')
            self.assertEqual(os.environ['MS_ASSET_DIR'], '/existing')

    def test_no_symlinked_source_models(self):
        (self.source/'link').symlink_to('/etc/passwd')
        self.assertFalse(validate_model(self.source)['valid'])
    def test_only_exact_single_model_id(self):
        p=self.root/'authorized/103498';p.mkdir(parents=True)
        (p/'mobility.urdf').write_text((self.source/'mobility.urdf').read_text())
        with self.assertRaises(ValueError):install_from_authorized_source(self.root,p)
    def test_does_not_accept_large_model_folder(self):
        with patch('scripts.prepare_roboreel_drawer_partnet.MAX_MODEL_BYTES',8):
            self.assertFalse(validate_model(self.source)['valid'])
    def test_sapien_token_bearing_url_never_reaches_stdout_or_stderr(self):
        import contextlib
        fake_sapien = types.ModuleType('sapien')
        fake_sapien.__path__ = []
        fake_asset = types.ModuleType('sapien.asset')
        def mocked_download(model_id, token, directory):
            print('https://sapien.ucsd.edu/api/download/compressed/44962.zip?token='+token)
            print('very sensitive token: '+token,file=sys.stderr)
            path=Path(directory)/str(model_id)
            path.mkdir()
            (path/'mobility.urdf').write_text((self.source/'mobility.urdf').read_text())
            return str(path/'mobility.urdf')
        fake_asset.download_partnet_mobility = mocked_download
        fake_sapien.asset = fake_asset
        output=io.StringIO()
        error=io.StringIO()
        with patch.dict(sys.modules, {'sapien':fake_sapien,'sapien.asset':fake_asset}):
            with contextlib.redirect_stdout(output),contextlib.redirect_stderr(error):
                r=download_authorized(self.root, 'SENSITIVE_SECRET_DO_NOT_OUTPUT')
        self.assertEqual(r['status'],'READY_FOR_SCENE_INIT')
        self.assertNotIn('SENSITIVE_SECRET_DO_NOT_OUTPUT',output.getvalue()+error.getvalue())

    def test_sapien_authorized_failure_does_not_expose_token(self):
        fake_sapien=types.ModuleType('sapien')
        fake_sapien.__path__=[]
        fake_asset=types.ModuleType('sapien.asset')
        secret='SENSITIVE_SECRET_DO_NOT_OUTPUT'
        def fail(model_id, token, directory):
            raise ValueError('URL?token='+token)
        fake_asset.download_partnet_mobility=fail
        fake_sapien.asset=fake_asset
        with patch.dict(sys.modules, {'sapien':fake_sapien,'sapien.asset':fake_asset}):
            with self.assertRaises(RuntimeError) as cm:
                download_authorized(self.root,secret)
        self.assertNotIn(secret,str(cm.exception))
        self.assertEqual(len(list(self.loc['model'].parent.glob('44962'))),0)

if __name__ == '__main__':unittest.main()
