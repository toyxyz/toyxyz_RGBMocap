import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mocap import model_manager as manager, models
from mocap.storage import atomic_json


class InstalledModelTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve()
        self.root = base / 'installation'
        self.root.mkdir()
        self.external = base / 'original.ckpt'
        self.external.write_bytes(b'Keep the original')
        paths = dict(HOME=self.root, ROOT=self.root, MODELS=self.root / 'models',
                     UPSTREAM=self.root / 'third_party')
        for target in (patch.multiple(manager, **paths), patch.object(models, 'MODELS', paths['MODELS'])):
            target.start()
            self.addCleanup(target.stop)
        self.model = models.BY_KEY['flow_sam']
        self.payload = self.model.path
        self.payload.parent.mkdir(parents=True)
        self.payload.write_bytes(b'Example installed model')
        self.receipt = self.model.receipt
        self.receipt.write_bytes(b'Record')
        self.keys = [self.model.key]
        self.alias = next(path for path, _ in manager.runtime_aliases(self.model)
                          if 'FlowHMR' in path.parts)
        self.alias.parent.mkdir(parents=True)

    def test_shared_workflows_use_actual_dependencies(self):
        users = manager.model_consumers(manager.workflow_catalog())
        self.assertEqual(users['flow_sam'], ('flowhmr', 'exmr'))
        self.assertEqual(users['flowhmr'], ('flowhmr',))
        self.assertEqual(set(users['flow_vggt']), {'flowhmr', 'mamma', 'pear'})
        self.assertEqual(set(users['sam31_people']), {'flowhmr', 'exmr', 'gvhmr'})

    def test_removes_known_links_but_keeps_import_source_and_unrelated_files(self):
        atomic_json(self.receipt, {'source': str(self.external)})
        os.link(self.payload, self.alias)
        held_by_user = self.external.with_name('outside-hardlink.ckpt')
        os.link(self.payload, held_by_user)
        other = self.payload.with_name('custom.ckpt')
        other.write_bytes(b'Other model')
        plan = manager.removal_plan(self.keys)
        self.assertEqual({item.path for item in plan}, {self.payload, self.receipt, self.alias})
        self.assertEqual(manager.physical_bytes(plan), self.payload.stat().st_size + self.receipt.stat().st_size)
        self.assertEqual(manager.physical_bytes(plan, reclaim=True), self.receipt.stat().st_size)
        manager.remove_models(self.keys, plan)
        self.assertTrue(all(not item.path.exists() for item in plan))
        self.assertEqual(held_by_user.read_bytes(), b'Example installed model')
        self.assertEqual(self.external.read_bytes(), b'Keep the original')
        self.assertEqual(other.read_bytes(), b'Other model')
        self.assertFalse(manager.journal_path().exists())
        self.payload.write_bytes(b'Reimported model')
        self.assertIn(self.model, manager.installed_models())

    def test_refuses_a_model_changed_after_confirmation(self):
        plan = manager.removal_plan(self.keys)
        self.payload.write_bytes(b'New version')
        with self.assertRaisesRegex(ValueError, 'changed after confirmation'):
            manager.remove_models(self.keys, plan)
        self.assertEqual(self.payload.read_bytes(), b'New version')
        self.assertFalse(manager.journal_path().exists())

    def test_interrupted_removal_rolls_back_from_journal(self):
        plan = manager.removal_plan(self.keys)
        rename = Path.rename

        def interrupt(path, destination):
            if path == self.receipt:
                raise KeyboardInterrupt
            return rename(path, destination)

        with patch.object(Path, 'rename', interrupt), self.assertRaises(KeyboardInterrupt):
            manager.remove_models(self.keys, plan)
        self.assertFalse(self.payload.exists())
        self.assertTrue(manager.journal_path().exists())
        self.assertTrue(manager.recover_removal())
        self.assertEqual(self.payload.read_bytes(), b'Example installed model')
        self.assertEqual(self.receipt.read_bytes(), b'Record')
        self.assertFalse(manager.journal_path().exists())

    def test_recovery_purges_committed_staging_without_touching_replacement(self):
        plan = manager.removal_plan(self.keys)
        with patch.object(manager, 'recover_removal', side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            manager.remove_models(self.keys, plan)
        self.payload.write_bytes(b'Replacement')
        self.assertTrue(manager.recover_removal())
        self.assertEqual(self.payload.read_bytes(), b'Replacement')
        self.assertFalse(self.receipt.exists())

    def test_pear_removes_active_bundle_and_preserves_old_generation(self):
        from mocap import pear_models

        pear = models.BY_KEY['pear_geometry'].path
        active = pear.parent / 'geometry-active' / 'assets'
        old = pear.parent / 'geometry-old' / 'assets'
        relative = 'body/geometry.bin'
        for directory in (active, old):
            (directory / relative).parent.mkdir(parents=True)
        payload = active / relative
        payload.write_bytes(b'Geometry')
        os.link(payload, old / relative)
        stamp = payload.stat()
        manifest = dict(bundle_sha256='test-bundle', files=[dict(path=relative, bytes=stamp.st_size)])
        atomic_json(pear, dict(folder='geometry-active/assets', bundle_sha256='test-bundle',
                               files={relative:[stamp.st_size, stamp.st_mtime_ns]}))
        with patch.object(pear_models, 'MODELS', manager.MODELS), patch.object(pear_models, 'spec', return_value=manifest):
            plan = manager.removal_plan(['pear_geometry'])
            self.assertEqual(manager.physical_bytes(plan, reclaim=True), pear.stat().st_size)
            manager.remove_models(['pear_geometry'], plan)
            self.assertFalse(pear_models.geometry_ready())
        self.assertFalse(active.parent.exists())
        self.assertEqual((old / relative).read_bytes(), b'Geometry')

    def test_relocated_models_folder_keeps_its_own_removal_journal(self):
        relocated = self.root.parent / 'model-storage'
        with patch.object(manager, 'MODELS', relocated), patch.object(models, 'MODELS', relocated):
            payload = self.model.path
            payload.parent.mkdir(parents=True)
            payload.write_bytes(b'Relocated model')
            self.model.receipt.write_bytes(b'Relocated receipt')
            plan = manager.removal_plan(self.keys)
            self.assertEqual({item.path for item in plan}, {payload, self.model.receipt})
            self.assertEqual(manager.journal_path().parent, relocated)
            manager.remove_models(self.keys, plan)
            self.assertFalse(payload.exists())
            self.assertFalse(manager.journal_path().exists())
        self.assertTrue(self.payload.is_file())

    def test_in_place_registration_protects_a_file_in_the_models_folder(self):
        with patch.dict(manager.MODEL_FILES, {self.model.key: self.payload}, clear=True):
            self.assertIn(self.model, manager.installed_models())
            self.assertEqual(manager.installed_files(self.model), ())
            with self.assertRaisesRegex(ValueError, 'registered in place'):
                manager.removal_plan(self.keys)
        self.assertEqual(self.payload.read_bytes(), b'Example installed model')

    def test_another_in_place_registration_protects_its_link_target(self):
        self.external.unlink()
        try:
            self.external.symlink_to(self.payload)
        except OSError as error:
            self.skipTest(f'File symlinks unavailable on this system: {error}')
        with patch.dict(manager.MODEL_FILES, {'flow_config': self.external}, clear=True):
            with self.assertRaisesRegex(ValueError, 'shares its storage path'):
                manager.removal_plan(self.keys)
        self.assertTrue(self.external.is_symlink())
        self.assertEqual(self.payload.read_bytes(), b'Example installed model')

    def test_rejects_unrecognized_path_in_recovery_record(self):
        custom = manager.MODELS / 'my-model.ckpt'
        custom.write_bytes(b'Separate model')
        atomic_json(manager.journal_path(), dict(
            version=1,id='0'*32,committed=True,keys=self.keys,files=[
                dict(path=str(custom),fingerprint=manager.fingerprint(manager.entry(custom, manager.HOME)))
            ]))
        with self.assertRaisesRegex(ValueError, 'leaves managed storage'):
            manager.recover_removal()
        self.assertEqual(custom.read_bytes(), b'Separate model')

    def test_external_symlink_cannot_be_removed_as_a_managed_model(self):
        self.payload.unlink()
        try:
            self.payload.symlink_to(self.external)
        except OSError as error:
            self.skipTest(f'File symlinks unavailable on this system: {error}')
        self.assertIn(self.model, manager.installed_models())
        with self.assertRaisesRegex(ValueError, 'stored externally'):
            manager.removal_plan(self.keys)
        self.assertEqual(self.external.read_bytes(), b'Keep the original')


if __name__ == '__main__':
    unittest.main()
