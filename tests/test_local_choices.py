"""Remembered export choices and start model in config/local.json, without projects or models."""
from importlib import import_module
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QWidget

from mocap import local_choices, paths, resource_safety
from mocap.character_export_dialog import CharacterExportDialog
from mocap.source_export_dialog import SourceExportDialog

APP = QApplication.instance() or QApplication([])
# Today's source export request when nothing was saved.
DEFAULT_OPTIONS = dict(format='fbx', bvh_profile='generic', prepend_tpose=False, take_scope='current',
                       export_face=False, face_format='txt', smooth_face=True, fbx_profile='generic',
                       fbx_compatibility=False, fbx_rest_pose='t', place_at_origin=False, symmetric_rest=False)
SAVED = dict(format='bvh', bvh_profile='biped', fbx_profile='ue5_mannequin', ue5_target='quinn',
             ue5_in_place=True, ue5_include_mesh=True, fbx_rest_pose='a', fbx_compatibility=True,
             prepend_tpose=True, place_at_origin=True, symmetric_rest=True, export_face=True,
             face_format='livelink_csv', smooth_face=False)


def refuse(options):
    raise ValueError('Take 1 has no saved motion.')


class LocalChoicesTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        home = patch.object(paths, 'HOME', Path(tmp.name))
        home.start()
        self.addCleanup(home.stop)
        self.settings = Path(tmp.name) / 'config/local.json'

    def write(self, value):
        self.settings.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, bytes):self.settings.write_bytes(value)
        else:self.settings.write_text(value if isinstance(value, str) else json.dumps(value), encoding='utf-8')

    def read(self):
        return json.loads(self.settings.read_text(encoding='utf-8'))

    def dialog(self, cls=SourceExportDialog, *args, **kwargs):
        dialog = cls(*args, **kwargs)
        self.addCleanup(dialog.deleteLater)
        return dialog

    def gemx_window(self):
        parent = QWidget()
        self.addCleanup(parent.deleteLater)
        parent.model_choice = QComboBox()
        parent.model_choice.addItem('GEM-X', 'gemx')
        return parent

    def values(self, dialog):
        return {name: control.currentData() if hasattr(control, 'findData') else control.isChecked()
                for name, control in dialog.remembered_controls().items()}

    def test_nothing_saved_keeps_todays_defaults(self):
        self.assertEqual(local_choices.start_model(), 'exmr')
        self.assertEqual(self.dialog().options(), DEFAULT_OPTIONS)
        self.assertEqual(self.dialog(CharacterExportDialog, 'pmx').options(),
                         dict(format='fbx', include_mesh=True, export_face=False, face_format='txt', smooth_face=True))
        self.assertFalse(self.settings.exists())

    def test_round_trip_preserves_resource_safety(self):
        resource_safety.save(False)
        local_choices.remember_model('flowhmr')
        dialog = self.dialog()
        dialog.format.setCurrentIndex(dialog.format.findData('bvh'))
        dialog.accept()
        saved = self.read()
        self.assertIs(saved['resource_safety'], False)
        self.assertEqual(saved['motion_model'], 'flowhmr')
        self.assertEqual(saved['source_export']['format'], 'bvh')
        self.assertFalse(resource_safety.load())
        self.assertEqual(local_choices.start_model(), 'flowhmr')
        self.assertEqual(list(self.settings.parent.iterdir()), [self.settings])

    def test_unreadable_file_uses_defaults_and_is_not_replaced(self):
        for content in (b'{"source_export": {"format": "bvh"', b'\xff\xfe\x00', '[1, 2]',
                        ' ' * (resource_safety.MAX_BYTES + 1)):
            with self.subTest(content=content[:20]):
                self.write(content)
                before = self.settings.read_bytes()
                self.assertEqual(local_choices.start_model(), 'exmr')
                dialog = self.dialog()
                self.assertEqual(dialog.options(), DEFAULT_OPTIONS)
                dialog.accept()
                self.assertEqual(dialog.result(), QDialog.Accepted)
                with self.assertRaises(ValueError):
                    local_choices.remember_model('gvhmr')
                self.assertEqual(self.settings.read_bytes(), before)

    def test_unknown_values_fall_back_to_defaults(self):
        self.write(dict(motion_model='retired', source_export=dict(
            format='obj', bvh_profile=7, fbx_profile=None, fbx_rest_pose='z', prepend_tpose='yes',
            export_face=1, ue5_in_place='true', smooth_face='off')))
        self.assertEqual(local_choices.start_model(), 'exmr')
        dialog = self.dialog()
        self.assertEqual(dialog.options(), DEFAULT_OPTIONS)
        self.assertIs(dialog.ue5_root.currentData(), False)
        for value in (7, None, ['exmr'], 'pear'):
            self.write(dict(motion_model=value, source_export=['bvh']))
            self.assertEqual(local_choices.start_model(), 'exmr')
            self.assertEqual(self.dialog().options(), DEFAULT_OPTIONS)

    def test_optional_model_needs_its_runtime(self):
        for model in local_choices.OPTIONAL_RUNTIMES:
            self.write(dict(motion_model=model))
            runtime = import_module('mocap.' + model + '_runtime')
            with patch.object(runtime, 'configuration', side_effect=RuntimeError('Runtime is not installed.')):
                self.assertEqual(local_choices.start_model(), 'exmr')
            with patch.object(runtime, 'configuration', return_value={}):
                self.assertEqual(local_choices.start_model(), model)
        self.write(dict(motion_model='gvhmr'))
        self.assertEqual(local_choices.start_model(), 'gvhmr')

    def test_invalid_optional_runtime_registration_falls_back(self):
        root = self.settings.parents[1]
        for model in ("mamma", "prompthmr"):
            self.write({"motion_model": model})
            runtime = import_module("mocap." + model + "_runtime")
            record = root / ".setup" / (model + "-runtime.json")
            record.parent.mkdir(exist_ok=True)
            configure = runtime.configuration
            for data in ([], None, 1):
                with self.subTest(model=model, data=data):
                    record.write_text(json.dumps(data), encoding="utf-8")
                    with patch.object(runtime, "configuration", lambda: configure(root)):
                        self.assertEqual(local_choices.start_model(), "exmr")
    def test_only_offline_picks_are_written(self):
        local_choices.remember_model('pear')
        self.assertFalse(self.settings.exists())
        local_choices.remember_model('gvhmr')
        self.assertEqual(self.read(), dict(motion_model='gvhmr'))
        with patch.object(resource_safety, 'update') as update:
            local_choices.remember_model('gvhmr')
            local_choices.remember_model('pear')
        update.assert_not_called()

    def test_selector_remembers_user_picks_only(self):
        from mocap.gui import MainWindow
        window = SimpleNamespace(model_choice=QComboBox(), log=[])
        for model in ('exmr', 'flowhmr', 'gvhmr', 'pear'):
            window.model_choice.addItem(model, model)
        window.model_choice.setCurrentIndex(1)
        MainWindow.remember_model_pick(window, 1)
        self.assertEqual(self.read()['motion_model'], 'flowhmr')
        # select_model refused the GVHMR pick and restored FlowHMR before this slot ran.
        MainWindow.remember_model_pick(window, 2)
        window.model_choice.setCurrentIndex(3)
        MainWindow.remember_model_pick(window, 3)
        self.assertEqual(self.read()['motion_model'], 'flowhmr')
        self.write('{broken')
        window.model_choice.setCurrentIndex(0)
        MainWindow.remember_model_pick(window, 0)
        self.assertEqual(len(window.log), 1)
        self.assertTrue(window.log[0].startswith('Cannot save the AI Motion Model choice: '))

    def test_dialog_restores_each_choice_but_not_take_scope(self):
        self.write(dict(source_export=dict(SAVED, take_scope='selected', selected_takes=['take'])))
        dialog = self.dialog(show_take_scope=True)
        self.assertEqual(self.values(dialog), SAVED)
        self.assertEqual(dialog.take_scope.currentData(), 'current')
        self.assertTrue(dialog.face_export.format.isVisibleTo(dialog))
        options = dialog.options()
        self.assertEqual({key: options[key] for key in ('format', 'bvh_profile', 'prepend_tpose', 'place_at_origin',
                                                         'symmetric_rest', 'export_face', 'face_format', 'smooth_face')},
                         dict(format='bvh', bvh_profile='biped', prepend_tpose=True, place_at_origin=True,
                              symmetric_rest=True, export_face=True, face_format='livelink_csv', smooth_face=False))
        dialog.format.setCurrentIndex(dialog.format.findData('fbx'))
        self.assertEqual({key: dialog.options()[key] for key in ('fbx_profile', 'ue5_target', 'ue5_in_place', 'ue5_include_mesh')},
                         dict(fbx_profile='ue5_mannequin', ue5_target='quinn', ue5_in_place=True, ue5_include_mesh=True))

    def test_continue_saves_and_cancel_does_not(self):
        dialog = self.dialog()
        dialog.format.setCurrentIndex(dialog.format.findData('bvh'))
        dialog.face_export.checkbox.setChecked(True)
        dialog.reject()
        self.assertFalse(self.settings.exists())
        dialog = self.dialog(show_take_scope=True)
        self.assertEqual((dialog.format.currentData(), dialog.face_export.checkbox.isChecked()), ('fbx', False))
        dialog.format.setCurrentIndex(dialog.format.findData('bvh'))
        dialog.face_export.checkbox.setChecked(True)
        dialog.take_scope.setCurrentIndex(dialog.take_scope.findData('all'))
        dialog.accept()
        saved = self.read()['source_export']
        self.assertEqual((saved['format'], saved['export_face']), ('bvh', True))
        self.assertEqual(set(saved), set(SAVED))
        dialog = self.dialog(show_take_scope=True)
        self.assertEqual((dialog.format.currentData(), dialog.face_export.checkbox.isChecked()), ('bvh', True))
        self.assertEqual(dialog.take_scope.currentData(), 'current')

    def test_refused_export_saves_nothing(self):
        dialog = self.dialog(validate=refuse)
        dialog.format.setCurrentIndex(dialog.format.findData('bvh'))
        dialog.accept()
        self.assertNotEqual(dialog.result(), QDialog.Accepted)
        self.assertFalse(self.settings.exists())

    def test_choice_another_model_offers_is_kept(self):
        self.write(dict(source_export=dict(format='bvh', bvh_profile='biped')))
        dialog = self.dialog(SourceExportDialog, self.gemx_window())
        self.assertEqual(dialog.profile.currentData(), 'generic')
        self.assertEqual(dialog.options()['bvh_profile'], 'generic')
        dialog.accept()
        self.assertEqual(self.read()['source_export']['bvh_profile'], 'biped')
        self.assertEqual(self.dialog().profile.currentData(), 'biped')

    def test_source_defaults_keep_selected_takes_and_save_on_continue(self):
        parent = QWidget()
        self.addCleanup(parent.deleteLater)
        parent.model_choice = QComboBox(parent)
        parent.model_choice.addItem('FlowHMR', 'flowhmr')
        parent.camera_tools = SimpleNamespace(
            background_options=lambda: {'flowhmr': True},
            scale=SimpleNamespace(value=lambda: 2.5),
        )
        defaults = self.values(self.dialog(SourceExportDialog, parent))
        original = dict(source_export=SAVED, motion_model='flowhmr', resource_safety=False)
        self.write(original)
        dialog = self.dialog(SourceExportDialog, parent, show_take_scope=True)
        dialog.take_scope.setCurrentIndex(dialog.take_scope.findData('selected'))
        takes = {Path('selected-take')}
        dialog.selected_take_paths = takes
        dialog.format.setCurrentIndex(dialog.format.findData('blend'))
        dialog.blend_layout.setCurrentIndex(dialog.blend_layout.findData('combined'))
        dialog.animate_camera.setChecked(False)
        dialog.background_refine.setChecked(False)
        dialog.translation_scale.setValue(10)

        dialog.defaults_button.click()
        self.assertEqual(self.values(dialog), defaults)
        self.assertEqual(dialog.blend_layout.currentData(), 'separate')
        self.assertTrue(dialog.animate_camera.isChecked())
        self.assertTrue(dialog.background_refine.isChecked())
        self.assertEqual(dialog.translation_scale.value(), 2.5)
        self.assertEqual(dialog.options()['take_scope'], 'selected')
        self.assertEqual(dialog.options()['selected_takes'], takes)
        self.assertFalse(dialog.face_export.format.isVisibleTo(dialog))
        self.assertEqual(dialog.result(), QDialog.Rejected)
        dialog.reject()
        self.assertEqual(self.read(), original)

        dialog = self.dialog(validate=refuse)
        dialog.defaults_button.click()
        dialog.accept()
        self.assertEqual(dialog.result(), QDialog.Rejected)
        self.assertEqual(self.read(), original)

        dialog = self.dialog()
        dialog.defaults_button.click()
        dialog.accept()
        self.assertEqual(self.read(), dict(original, source_export=defaults))
        self.assertEqual(self.dialog().options(), DEFAULT_OPTIONS)

    def test_defaults_discard_a_bvh_choice_unavailable_to_gemx(self):
        self.write(dict(source_export=dict(format='bvh', bvh_profile='biped')))
        dialog = self.dialog(SourceExportDialog, self.gemx_window())
        dialog.defaults_button.click()
        dialog.accept()
        self.assertEqual(self.read()['source_export']['bvh_profile'], 'generic')
        self.assertEqual(self.dialog().options(), DEFAULT_OPTIONS)

    def test_character_defaults_replace_unavailable_vmd_on_save(self):
        original = dict(character_export=dict(format='vmd', include_mesh=False, export_face=True,
                                              face_format='livelink_csv', smooth_face=False))
        self.write(original)
        dialog = self.dialog(CharacterExportDialog, 'fbx')
        dialog.defaults_button.click()
        self.assertEqual(dialog.options(), dict(format='fbx', include_mesh=True, export_face=False,
                                                face_format='txt', smooth_face=True))
        dialog.reject()
        self.assertEqual(self.read(), original)

        dialog = self.dialog(CharacterExportDialog, 'fbx')
        dialog.defaults_button.click()
        dialog.accept()
        self.assertEqual(self.dialog(CharacterExportDialog, 'pmx').options(), dialog.options())

    def test_save_failure_does_not_block_export(self):
        parent = self.gemx_window()
        parent.log = []
        with patch.object(resource_safety, 'update', side_effect=OSError('Access is denied')):
            dialog = self.dialog(SourceExportDialog, parent)
            dialog.accept()
        self.assertEqual(dialog.result(), QDialog.Accepted)
        self.assertEqual(parent.log, ['Cannot save export options: Access is denied'])

    def test_character_export_choices(self):
        self.write(dict(character_export=dict(format='vmd', include_mesh=False, export_face=True)))
        dialog = self.dialog(CharacterExportDialog, 'fbx')
        self.assertEqual(dialog.options(), dict(format='fbx', include_mesh=False, export_face=True,
                                                face_format='txt', smooth_face=True))
        dialog.accept()
        self.assertEqual(self.read()['character_export']['format'], 'vmd')
        self.assertEqual(self.dialog(CharacterExportDialog, 'pmx').format.currentData(), 'vmd')
        dialog = self.dialog(CharacterExportDialog, 'pmx')
        dialog.mesh.setChecked(True)
        dialog.reject()
        self.assertIs(self.read()['character_export']['include_mesh'], False)


if __name__ == '__main__':
    unittest.main()
