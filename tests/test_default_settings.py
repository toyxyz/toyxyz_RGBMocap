"""Saved default settings for new work in config/local.json, without models, results or inference."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QMenu, QTreeWidget

from mocap import paths, project_workspace, resource_safety
from mocap.gui import MainWindow
from mocap.project_workspace_ui import DEFAULTS, DefaultSettingsDialog

APP = QApplication.instance() or QApplication([])
BUILT_IN = dict(flow_options=dict(camera_interval=0, fast_sam=True, sam_batch4=True, sam_gpu_crop=True,
                                  sam_hand_stride4=True, hamer=False),
                exmr_options=dict(static_camera=True, smoothing='global'),
                prompthmr_options=dict(people=2, interaction=True),
                correction=dict(auto_floor=True, floor=0.0, preset='normal', pelvis_offset=0.0, mode='flat',
                                foot_lock=True, hand_lock=False),
                track_face=False)


class DefaultSettingsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        home = patch.object(paths, 'HOME', Path(tmp.name))
        home.start()
        self.addCleanup(home.stop)
        self.settings = Path(tmp.name) / 'config/local.json'
        self.video = Path(tmp.name) / 'missing clip.mp4'

    def window(self, *, automatically_confirm=True):
        window = MainWindow()
        if automatically_confirm:
            confirmation = patch.object(window, 'confirm_default_settings', return_value=True)
            confirmation.start()
            self.addCleanup(confirmation.stop)
        def close():
            window.close()
            window.deleteLater()
            APP.processEvents()
        self.addCleanup(close)
        return window

    def project(self, models=None, view=None):
        """A model-folder project on ExMR with no results, as the app creates one from a video."""
        root = project_workspace.create(self.settings.parent.parent / 'You_project', None,
                                        dict(path=str(self.video), duration=5.0, fps=30.0, frames=150), 'exmr')
        state = project_workspace.read(root)
        state.update(models=models or {}, view=view or {})
        (root / 'project.json').write_text(json.dumps(state), encoding='utf-8')
        return root

    def write(self, value):
        self.settings.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, bytes):self.settings.write_bytes(value)
        else:self.settings.write_text(value if isinstance(value, str) else json.dumps(value), encoding='utf-8')

    def read(self):
        return json.loads(self.settings.read_text(encoding='utf-8'))

    def state(self, window):
        settings = window.workspace_settings()
        return dict({key: settings[key] for key in ('flow_options', 'exmr_options', 'prompthmr_options', 'correction')},
                    track_face=window.track_face.isChecked())

    def choose(self, window, combo, value):
        combo.setCurrentIndex(combo.findData(value))

    def test_file_menu_confirms_save_and_reset_with_the_values_that_apply(self):
        window = self.window(automatically_confirm=False)
        self.choose(window, window.flow_sampling, 5)
        self.choose(window, window.exmr_camera, False)
        window.exmr_smoothing.setChecked(False)
        window.camera_tools.restore_background({'exmr': True})
        window.restore_track_face(True)
        self.choose(window, window.compute_mode, 'low_memory')
        menu, = [menu for menu in window.menuBar().findChildren(QMenu) if menu.title() == 'File']
        actions = {action.text(): action for action in menu.actions()}
        save, reset = 'Save Settings as Default', 'Reset Default Settings'

        def click_confirmation(title, choice):
            result = []

            def click():
                dialog = APP.activeModalWidget()
                try:
                    self.assertIsInstance(dialog, DefaultSettingsDialog)
                    self.assertEqual(dialog.windowTitle(), title)
                    tree = dialog.findChild(QTreeWidget)
                    groups = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
                    result.append({group.text(0): {group.child(i).text(0): group.child(i).text(1)
                                                   for i in range(group.childCount())}
                                   for group in groups})
                    buttons = dialog.findChild(QDialogButtonBox)
                    self.assertTrue(buttons.button(QDialogButtonBox.StandardButton.Cancel).isDefault())
                    QTest.mouseClick(buttons.button(choice), Qt.MouseButton.LeftButton)
                except BaseException as error:
                    result.append(error)
                    if dialog is not None:
                        dialog.reject()

            QTimer.singleShot(0, click)
            actions[title].trigger()
            if result and isinstance(result[-1], BaseException):
                raise result[-1]
            self.assertEqual(len(result), 1)
            return result[0]

        status = window.status.text()
        preview = click_confirmation(save, QDialogButtonBox.StandardButton.Cancel)
        self.assertFalse(self.settings.exists())
        self.assertEqual(window.status.text(), status)
        self.assertEqual({group: len(values) for group, values in preview.items()},
                         {'GVHMR': 3, 'FlowHMR': 7, 'ExMR': 3, 'PromptHMR': 2,
                          'Motion Correction': 4, 'Processing Resources': 1, 'Face Tracking': 1})
        self.assertEqual(preview['FlowHMR']['Camera / Sampling'], 'Every 5 Frames')
        self.assertEqual(preview['ExMR'], {'Camera': 'Moving Camera / SimpleVO', 'Smooth Motion': 'Off',
                                           'Refine Camera Rotation': 'On'})
        self.assertEqual(preview['Processing Resources'], {'Resource Mode': 'Low Memory'})
        self.assertEqual(preview['Face Tracking']['Track Face'], 'On')
        self.assertEqual(preview, click_confirmation(save, QDialogButtonBox.StandardButton.Ok))
        self.assertEqual(window.status.text(), 'Default settings saved for new videos.')
        self.assertEqual(self.read()[DEFAULTS]['flow_options']['camera_interval'], 5)
        self.assertEqual(self.read()[DEFAULTS]['processing_options'], {'resource_mode': 'low_memory'})

        before_reset = self.settings.read_bytes()
        preview = click_confirmation(reset, QDialogButtonBox.StandardButton.Cancel)
        self.assertEqual(self.settings.read_bytes(), before_reset)
        self.assertFalse(window._reset_starting_settings)
        self.assertEqual(preview['FlowHMR']['Camera / Sampling'], 'Moving / Unknown - Auto')
        self.assertEqual(preview['ExMR'], {'Camera': 'Static Camera', 'Smooth Motion': 'On',
                                           'Refine Camera Rotation': 'Off'})
        self.assertEqual(preview['GVHMR'], {'HTD Refine': 'On', 'FootMR': 'On', 'HandFlow': 'On'})
        self.assertEqual(preview['Processing Resources'], {'Resource Mode': 'Automatic'})
        self.assertEqual(preview['PromptHMR'], {'Camera': 'Fixed Camera', 'Interaction Prompt': 'On'})
        self.assertEqual(preview['Motion Correction'], {'Correction Mode': 'Flat Ground', 'Lock Preset': 'Normal',
                                                         'Foot': 'On', 'Hand': 'Off'})
        self.assertEqual(preview['Face Tracking']['Track Face'], 'Off')
        self.assertEqual(preview, click_confirmation(reset, QDialogButtonBox.StandardButton.Ok))
        self.assertNotIn(DEFAULTS, self.read())
        self.assertEqual(window.status.text(), 'Default settings reset for new videos.')
        self.assertTrue(window.track_face.isChecked())
        self.assertEqual(window.compute_mode.currentData(), 'low_memory')
        window.load_video(self.video)
        self.assertEqual(self.state(window), BUILT_IN)
        self.assertEqual(window.compute_mode.currentData(), 'auto')

    def test_nothing_saved_is_todays_app(self):
        window = self.window()
        self.assertEqual(self.state(window), BUILT_IN)
        window.load_video(self.video)
        self.assertEqual(self.state(window), BUILT_IN)
        # Today Track Face and the PromptHMR options carry over to the next video of the session.
        window.restore_track_face(True)
        window.prompthmr_interaction.setChecked(False)
        window.load_video(self.video)
        self.assertEqual((window.track_face.isChecked(), window.prompthmr_interaction.isChecked()), (True, False))
        self.assertFalse(self.settings.exists())

    def test_save_then_new_window_and_new_video_start_from_saved(self):
        window = self.window()
        self.choose(window, window.flow_sampling, 5)
        window.track_face.setChecked(True)
        window.flow_hamer.setChecked(True)
        window.hand_lock.setChecked(True)
        window.save_default_settings()
        self.assertFalse(window.busy)
        self.assertEqual(window.log.toPlainText().splitlines()[-1],
                         'Default settings saved. New videos, the next start and models a project has no settings for use them.')
        expected = json.loads(json.dumps(BUILT_IN))
        expected['flow_options'].update(camera_interval=5, hamer=True)
        expected['correction']['hand_lock'] = True
        expected['track_face'] = True
        second = self.window()
        self.assertEqual(self.state(second), expected)
        self.assertFalse(second.busy)
        self.choose(second, second.flow_sampling, 10)
        second.track_face.setChecked(False)
        second.load_video(self.video)
        self.assertEqual(self.state(second), expected)
        self.assertFalse(second.busy)

    def test_reset_returns_to_built_in_values_and_keeps_other_settings(self):
        resource_safety.save(False)
        window = self.window()
        self.choose(window, window.exmr_camera, False)
        window.save_default_settings()
        self.assertEqual(self.read()[DEFAULTS]['exmr_options']['static_camera'], False)
        window.reset_default_settings()
        self.assertEqual(self.read(), dict(resource_safety=False))
        self.assertEqual(window.log.toPlainText().splitlines()[-1],
                         'Default settings reset. New videos and the next start use the built-in settings.')
        self.assertIs(window.exmr_camera.currentData(), False)  # the current work keeps its settings
        self.assertEqual(self.state(self.window()), BUILT_IN)
        window.load_video(self.video)
        self.assertEqual(self.state(window), BUILT_IN)

    def test_reset_applies_to_next_clip_without_editing_current_work(self):
        resource_safety.update({'ui_theme': 'light', 'ui_scale': 200, 'source_export': {'format': 'fbx'}})
        window = self.window()
        window.restore_track_face(True)
        self.choose(window, window.prompthmr_camera, 'moving')
        window.prompthmr_interaction.setChecked(False)
        window.prompthmr_people.setValue(5)
        window.save_default_settings()

        window.reset_default_settings()
        self.assertEqual(self.read(), {'ui_theme': 'light', 'ui_scale': 200, 'source_export': {'format': 'fbx'}})
        self.assertTrue(window.track_face.isChecked())
        self.assertEqual(window.prompthmr_camera.currentData(), 'moving')
        self.assertFalse(window.prompthmr_interaction.isChecked())

        window.load_video(self.video)
        self.assertFalse(window.track_face.isChecked())
        self.assertEqual(window.prompthmr_camera.currentData(), 'static')
        self.assertTrue(window.prompthmr_interaction.isChecked())
        self.assertEqual(window.prompthmr_people.value(), 5)

        window.restore_track_face(True)
        window.prompthmr_interaction.setChecked(False)
        window.load_video(self.video)
        self.assertTrue(window.track_face.isChecked())
        self.assertFalse(window.prompthmr_interaction.isChecked())

    def test_resource_mode_and_workflow_defaults_round_trip_without_entering_project_settings(self):
        window = self.window()
        self.switch(window, 'gvhmr')
        window.htd_refine.setChecked(False)
        window.handflow_refine.setChecked(False)
        self.switch(window, 'exmr')
        window.camera_tools.restore_background({'flowhmr': True, 'exmr': True})
        self.choose(window, window.compute_mode, 'low_memory')
        window.save_default_settings()

        saved = self.read()[DEFAULTS]
        self.assertEqual(saved['gvhmr_options'], dict(htd=False, footmr=True, handflow=False))
        self.assertEqual(saved['processing_options'], {'resource_mode': 'low_memory'})
        second = self.window()
        self.assertEqual(second.compute_mode.currentData(), 'low_memory')
        self.assertEqual(second.camera_tools.background_options(), {'flowhmr': True, 'exmr': True})
        self.switch(second, 'gvhmr')
        self.assertEqual(second.selected_result(), 'corrected_footmr_initial')

        self.choose(second, second.compute_mode, 'auto')
        second.camera_tools.restore_background({})
        second.load_video(self.video)
        self.assertEqual(second.compute_mode.currentData(), 'low_memory')
        self.assertEqual(second.camera_tools.background_options(), {'flowhmr': True, 'exmr': True})
        self.assertEqual(second.selected_result(), 'corrected_footmr_initial')

        project = self.project({'gvhmr': {'settings': {'result': 'corrected_handflow_footmr_refined'}}},
                               {'camera_background': {'flowhmr': False, 'exmr': False}})
        self.assertTrue(second.open_workspace(project))
        self.switch(second, 'gvhmr')
        self.assertEqual(second.selected_result(), 'corrected_handflow_footmr_refined')
        self.assertEqual(second.camera_tools.background_options(), {'flowhmr': False, 'exmr': False})
        self.assertEqual(second.compute_mode.currentData(), 'low_memory')
        self.assertNotIn('processing_options', second.workspace_settings())

    def test_prepared_clip_reload_keeps_working_defaults_until_a_result_is_saved(self):
        self.write({DEFAULTS: dict(gvhmr_options=dict(htd=False, footmr=False, handflow=False),
                                  flow_options=dict(refine_camera_rotation=True))})
        window = self.window()
        self.switch(window, 'gvhmr')
        self.assertEqual(window.selected_result(), 'corrected_initial')
        root = self.settings.parent.parent / 'prepared'
        root.mkdir()
        manifest = dict(source={'path': str(self.video)}, fps=30.0, frames=150,
                        start=0.0, end=5.0, width=640, height=480, pipeline={'model': 'gvhmr'})
        (root / 'project.json').write_text(json.dumps(manifest), encoding='utf-8')
        self.assertTrue(window.load_project(root, keep_current_options=True))
        self.assertEqual(window.selected_result(), 'corrected_initial')
        self.assertTrue(window.camera_tools.background_options()['flowhmr'])

        manifest['pipeline']['result'] = 'corrected_handflow_footmr_refined'
        (root / 'project.json').write_text(json.dumps(manifest), encoding='utf-8')
        self.assertTrue(window.load_project(root, keep_current_options=True))
        self.assertEqual(window.selected_result(), 'corrected_handflow_footmr_refined')

    def test_per_video_values_are_not_saved(self):
        window = self.window()
        window.auto_floor.setChecked(False)
        window.floor_height.setValue(.4)
        window.pelvis_offset.setValue(-.05)
        window.prompthmr_people.setValue(5)
        window.mamma_people.setValue(3)
        self.choose(window, window.prompthmr_camera, 'moving')
        window.save_default_settings()
        saved = self.read()[DEFAULTS]
        self.assertEqual(set(saved), {'gvhmr_options', 'flow_options', 'exmr_options', 'prompthmr_options',
                                      'correction', 'track_face', 'processing_options'})
        self.assertEqual(saved['correction'], dict(mode='flat', preset='normal', foot_lock=True, hand_lock=False))
        self.assertEqual(saved['prompthmr_options'], dict(interaction=True, camera='moving'))
        self.assertNotIn('result', json.dumps(saved))
        second = self.window()
        self.assertEqual(second.workspace_settings()['correction'], BUILT_IN['correction'])
        self.assertEqual(second.prompthmr_camera.currentData(), 'moving')

    def switch(self, window, model):
        window.model_choice.setCurrentIndex(window.model_choice.findData(model))
        self.assertEqual(window.model_choice.currentData(), model)

    def test_project_model_with_nothing_saved_starts_from_defaults(self):
        self.write({DEFAULTS: dict(flow_options=dict(camera_interval=5), correction=dict(preset='strong'))})
        window = self.window()
        self.assertTrue(window.open_workspace(self.project()))
        self.choose(window, window.flow_sampling, 10)  # changed while on ExMR, saved with ExMR
        self.switch(window, 'flowhmr')
        self.assertEqual((window.flow_sampling.currentData(), window.correction_preset.currentData()), (5, 'strong'))
        self.assertEqual(window.workspace_settings()['correction']['auto_floor'], True)
        for model in ('exmr', 'flowhmr'):
            self.switch(window, model)
        self.assertEqual(window.flow_sampling.currentData(), 5)
        self.switch(window, 'exmr')
        self.assertEqual(window.flow_sampling.currentData(), 10)
        self.assertFalse(window.busy)

    def test_project_group_saved_is_restored_exactly(self):
        self.write({DEFAULTS: dict(flow_options=dict(camera_interval=5, sam_batch4=False), correction=dict(preset='strong'))})
        # An older project that saved FlowHMR's group without two keys: those keys get built-in values, not defaults.
        window = self.window()
        window.open_workspace(self.project({'flowhmr': {'settings': {'flow_options': {'camera_interval': 10, 'fast_sam': False}}}}))
        self.switch(window, 'flowhmr')
        flow = window.workspace_settings()['flow_options']
        self.assertEqual((flow['camera_interval'], flow['fast_sam'], flow['sam_batch4']), (10, False, True))
        self.assertEqual(window.correction_preset.currentData(), 'strong')  # a group it saved nothing for

    def test_project_without_defaults_keeps_built_in_values(self):
        window = self.window()
        window.open_workspace(self.project())
        self.switch(window, 'flowhmr')
        self.assertEqual(self.state(window), BUILT_IN)

    def test_track_face_only_when_the_project_view_lacks_it(self):
        window = self.window()
        window.open_workspace(self.project())
        self.assertFalse(window.track_face.isChecked())
        self.write({DEFAULTS: dict(track_face=True)})
        for view, expected in (({}, True), ({'track_face': False}, False)):
            window = self.window()
            window.open_workspace(self.project(view=view))
            self.assertIs(window.track_face.isChecked(), expected)
            self.assertFalse(window.busy)

    def test_unreadable_files_keep_built_in_values(self):
        for content in (b'{"default_settings": {', b'\xff\xfe\x00', '[1, 2]', ' ' * (resource_safety.MAX_BYTES + 1),
                        {DEFAULTS: ['flow_options']}, {DEFAULTS: 'Every 5 Frames'}, {DEFAULTS: None}):
            with self.subTest(content=str(content)[:30]):
                self.write(content)
                before = self.settings.read_bytes()
                window = self.window()
                self.assertEqual(self.state(window), BUILT_IN)
                window.load_video(self.video)
                self.assertEqual(self.state(window), BUILT_IN)
                if isinstance(content, (bytes, str)):
                    window.save_default_settings()
                    self.assertTrue(window.log.toPlainText().splitlines()[-1].startswith('Cannot save default settings: '))
                    self.assertEqual(self.settings.read_bytes(), before)

    def test_each_unusable_value_falls_back_alone(self):
        self.write({DEFAULTS: dict(
            flow_options=dict(camera_interval=True, fast_sam='no', hamer=1, sam_batch4=False, sam_gpu_crop=None),
            exmr_options=dict(static_camera=1, smoothing=False),
            processing_options=dict(resource_mode='unavailable'),
            prompthmr_options=dict(camera='drone', interaction=False, people=7),
            correction=dict(mode='stairs', preset='strong', foot_lock=None, hand_lock=True, floor=2.0, auto_floor=False),
            track_face='yes', model_guide=dict(width=900))})
        expected = json.loads(json.dumps(BUILT_IN))
        expected['flow_options']['sam_batch4'] = False
        expected['prompthmr_options']['interaction'] = False
        expected['correction'].update(preset='strong', hand_lock=True)
        window = self.window()
        self.assertEqual(self.state(window), expected)
        self.assertEqual(window.prompthmr_camera.currentData(), 'static')
        self.assertEqual(window.compute_mode.currentData(), 'auto')
        self.assertEqual(window.log.toPlainText().count('Default settings not applied'), 0)

    def test_settings_of_another_model_wait_on_its_page(self):
        self.write(dict(motion_model='exmr', **{DEFAULTS: dict(flow_options=dict(camera_interval=5, hamer=True, fast_sam=False))}))
        window = self.window()
        self.assertEqual(window.model_choice.currentData(), 'exmr')
        self.assertEqual(window.workspace_settings()['exmr_options'], BUILT_IN['exmr_options'])
        window.model_choice.setCurrentIndex(window.model_choice.findData('flowhmr'))
        self.assertEqual((window.flow_sampling.currentData(), window.flow_hamer.isChecked()), (5, True))
        self.assertTrue(window.fast_sam.isChecked())  # HaMeR keeps Fast SAM locked on, as in the app today
        self.assertFalse(window.busy)


if __name__ == '__main__':
    unittest.main()
