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
        self.receipt = manager.receipt_path(self.model)
        self.receipt.write_bytes(b'Record')
        self.keys = [self.model.key]
        self.alias = next(path for path, _ in manager.runtime_aliases(self.model)
                          if 'FlowHMR' in path.parts)
        self.alias.parent.mkdir(parents=True)

    def test_shared_workflows_use_actual_dependencies(self):
        catalog = manager.workflow_catalog()
        users = manager.model_consumers(catalog)
        self.assertEqual(users['flow_sam'], ('flowhmr', 'exmr'))
        self.assertEqual(users['flowhmr'], ('flowhmr',))
        self.assertEqual(set(users['flow_vggt']), {'flowhmr', 'mamma', 'vidihand2'})
        self.assertEqual(users['vidihand2_vdm'], ('vidihand2',))
        self.assertEqual(users['vidihand2_projector'], ('vidihand2',))
        self.assertIn('vidihand2', users['mano_right'])
        self.assertEqual(catalog['vidihand2'], {
            'vidihand2_vdm', 'vidihand2_projector', 'mano_right', 'flow_vggt'})
        self.assertEqual(users['pear_geocalib'], ('mamma',))
        self.assertTrue({'pear', 'pear_geometry', 'pear_yolo'}.isdisjoint(users))
        self.assertEqual(set(users['sam31_people']), {'flowhmr', 'exmr', 'gvhmr'})

    def test_vidihand2_official_checkpoints_require_verified_import(self):
        from dataclasses import replace
        from hashlib import sha256
        from mocap.first_person.runtime import command

        key = "vidihand2_projector"
        model = models.BY_KEY[key]
        payload = b"example projector"
        fixture = replace(model, minimum_bytes=len(payload),
                          expected_sha256=sha256(payload).hexdigest())
        source = self.external
        source.write_bytes(payload)
        settings = dict(mode="native", python="python", upstream=str(self.root))
        with patch.dict(models.BY_KEY, {key: fixture}):
            self.assertNotIn("--managed-models", command(settings, "input", None, "out", "cancel"))
            models.import_model_file(key, source)
            self.assertTrue(models.ready(fixture))
            args = command(settings, "input", None, "out", "cancel")
            import json
            self.assertEqual(json.loads(args[args.index("--managed-models")+1]),
                             {key: str(fixture.path)})
            fixture.path.write_bytes(payload+b"!")
            self.assertFalse(models.ready(fixture))
        self.assertIn("/blob/", models.manual_download_page(key))
        self.assertIn("projector.safetensors", models.manual_instructions_html(key))

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

    def test_recovers_interrupted_legacy_pear_removal(self):
        from mocap.legacy_pear_removal import GEOMETRY_ASSETS

        pear = manager.MODELS / 'pear'
        relative = sorted(GEOMETRY_ASSETS)[0]
        payload = pear / 'geometry-active' / 'assets' / relative
        pointer = pear / 'geometry.json'
        checkpoint = pear / 'pear_model.pt'
        untouched = pear / 'geometry-old' / 'assets' / relative
        untouched.parent.mkdir(parents=True)
        untouched.write_bytes(b'Old generation')
        payload.parent.mkdir(parents=True)
        txid = 'a' * 32
        files = []
        for i, path in enumerate((payload, pointer, checkpoint)):
            stage = path.with_name(f'.rgbmocap-remove-{txid}-{i}')
            stage.write_bytes(b'Interrupted removal')
            files.append(dict(path=str(path), fingerprint=manager.fingerprint(manager.entry(stage, manager.HOME))))
        atomic_json(manager.journal_path(), dict(
            version=1, id=txid, committed=False, keys=['pear_geometry', 'pear'], files=files))

        self.assertTrue(manager.recover_removal())
        self.assertEqual(payload.read_bytes(), b'Interrupted removal')
        self.assertEqual(pointer.read_bytes(), b'Interrupted removal')
        self.assertEqual(checkpoint.read_bytes(), b'Interrupted removal')
        self.assertEqual(untouched.read_bytes(), b'Old generation')
        self.assertFalse(manager.journal_path().exists())

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
