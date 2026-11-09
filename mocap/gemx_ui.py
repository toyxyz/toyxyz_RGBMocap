"""Settings and launch for the optional GEM-X worker."""

import uuid

from PySide6.QtWidgets import QMessageBox

from .gemx_runtime import configuration


class GemXWorkspace:

    def sync_gemx_controls(self):
        if self.model_choice.currentData() != 'gemx':
            return
        self.find_person_button.hide()
        self.people_help_dialog.setText(
            'GEM-X automatically chooses the primary person from one fixed-camera video. '
            'Use a clip where the intended person is clearly visible. Check the row to show the reconstructed person.')
        self.run_button.setToolTip('Reconstruct one fixed-camera clip as a native SOMA surface and skeleton.')
        self.export_action.setToolTip('Export native SOMA as FBX, BVH or a portable Blender bundle. Select a retargeted character to export its motion.')
        self.run_button.setEnabled(not self.busy and not self.contact_saving)

    def show_gemx_start_error(self, reason):
        message = 'Cannot start GEM-X: ' + reason
        self.status.setText(message)
        self.log.append(message)
        QMessageBox.warning(self, 'GEM-X', reason)

    def run_gemx(self):
        if self.busy or self.contact_saving:
            return
        if not self.info or not self.video_path:
            self.show_gemx_start_error('Open a video before reconstructing.')
            return
        from .models import missing
        from .paths import HOME, PROJECTS
        from .storage import atomic_json, project_key

        try:
            configuration()
            if missing('gemx_assets'):
                self.status.setText('Prepare the missing GEM-X files in Model Manager.')
                self.open_model_manager()
                return
            start, end = self.processing_range()
            if end - start > 60.001 or round((end - start) * 30) < 4:
                raise ValueError('Trim to 4–1,800 frames at 30 fps (up to 60 seconds).')
            root = self.workspace_root or PROJECTS / (
                self.video_path.stem[:80] + '_' +
                project_key(self.video_path, 0, self.info['duration']) + '_project')
            folder = HOME / 'cache/worker_jobs'
            folder.mkdir(parents=True, exist_ok=True)
            request = folder / (uuid.uuid4().hex + '.json')
            atomic_json(request, dict(version=1, root=str(root), video=str(self.video_path),
                                      start=start, end=end, correct=self.motion_correction.isChecked(),
                                      correction=self.selected_correction_options()))
            self.start_jobs([('gemx_reconstruct', ['--request', str(request)])])
        except (OSError, ValueError, RuntimeError) as error:
            self.show_gemx_start_error(str(error))
