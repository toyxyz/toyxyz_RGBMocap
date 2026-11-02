"""Export the actual current viewport, paired with the selected decoded video."""
from pathlib import Path
import time

from PySide6.QtCore import QObject, QTimer, Qt, Signal
from PySide6.QtWidgets import QDialog, QFileDialog, QLabel, QMessageBox, QProgressBar, QPushButton, QVBoxLayout

from .preview_export import PreviewEncoder, output_layout


class ExportProgress(QDialog):
    cancel_requested = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle('Export Preview')
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumWidth(440)
        self.allow_close = False
        layout = QVBoxLayout(self)
        self.label = QLabel('Preparing preview export...')
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.cancel_button = QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.cancel_requested)
        layout.addWidget(self.cancel_button)

    def reject(self):
        if self.allow_close:
            super().reject()
        else:
            self.cancel_requested.emit()

    def closeEvent(self, event):
        if self.allow_close:
            event.accept()
        else:
            self.cancel_requested.emit()
            event.ignore()


class PreviewExport(QObject):
    def __init__(self, window, destination, resume=False):
        super().__init__(window)
        self.window, self.destination, self.resume = window, Path(destination), resume
        self.writer = None
        self.error = None
        self.cancelling = self.close_window = False
        self.phase = 'waiting'
        self.started = self.wait_started = time.monotonic()
        self.first, self.end = window.slider.clip_in, window.slider.clip_out
        self.target, self.rendered = self.first, 0
        self.fps = window.info['fps']
        self.decoder, self.generation = window.video_view.decoder, window.video_view.generation
        self.original_frame = window.slider.value()
        view = window.motion_view
        self.view_size = view.size()
        self.context = self.scene_context()
        self.camera = {name: getattr(view, name).copy() if hasattr(getattr(view, name), 'copy') else getattr(view, name)
                       for name in ('yaw', 'pitch', 'distance', 'pan', 'scene_center', 'auto_fit', 'view_name', 'follow')}
        self.window_limits = window.minimumSize(), window.maximumSize()
        self.shortcut_states = [s.isEnabled() for s in window.shortcuts]
        self.accept_drops = window.acceptDrops()
        self.audio, self.audio_start = self.audio_source()
        # Reuse one native progress window across exports (including its GPU
        # presentation resources) instead of repeatedly creating top-levels.
        self.dialog = getattr(window, '_preview_export_dialog', None)
        if self.dialog is None:
            self.dialog = window._preview_export_dialog = ExportProgress(window)
        self.dialog.allow_close = False
        self.dialog.cancel_button.setEnabled(True)
        self.dialog.progress.setValue(0)
        self.dialog.cancel_requested.connect(self.cancel)
        self.dialog.progress.setRange(0, self.end - self.first)
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.check)
        window.video_view.decode_error.connect(self.decode_failed)
        window.video_view.frame_presented.connect(self.video_ready)
        window.character_loader.failed.connect(self.character_failed)

    def scene_context(self):
        w = self.window
        v = w.motion_view
        return (w.project, w.workspace_root, w.active_result, w.model_choice.currentData(),
                id(v.joints), id(v.people_scene), id(v.target_frames), id(v.mesh_frames), id(v.native_vertices))

    def video_ready(self, index):
        if self.phase == 'waiting' and index == self.target:
            QTimer.singleShot(0, self.check)

    def character_failed(self, key, message):
        w = self.window
        if any(c.get('visible', True) and
               (key == str(w.character_path(c['asset'])) or
                (c.get('animation') and Path(key).is_relative_to(w.character_path(c['animation']))))
               for c in w.characters):
            self.fail('Cannot prepare the character preview: ' + message)

    def characters_ready(self):
        from bisect import bisect_right
        w = self.window
        expected = 0
        for character in w.characters:
            if not character.get('visible', True):
                continue
            takes = character.get('takes')
            if takes:
                starts = character.setdefault('_starts', [t['start'] for t in takes])
                i = bisect_right(starts, w.motion_view.scene_time) - 1
                if i < 0 or w.motion_view.scene_time >= takes[i]['start'] + takes[i]['frames'] / takes[i]['fps']:
                    continue
            expected += 1
        return len(list(w.character_draws())) == expected

    def audio_source(self):
        w = self.window
        inputs = getattr(w, 'mamma_inputs', [])
        if w.model_choice.currentData() == 'mamma' and inputs:
            path = Path(inputs[w.mamma_active_camera]['path'])
        else:
            path = Path(w.video_path)
        if path.is_file():
            return path, w.source_offset + self.first / self.fps
        video = w.video_view
        return Path(self.decoder.path), (self.first + getattr(video, 'timeline_start', 0)) / self.fps

    def start(self):
        w = self.window
        w.stop_playback()
        w.setFixedSize(w.size())
        w.setAcceptDrops(False)
        for shortcut in w.shortcuts:
            shortcut.setEnabled(False)
        self.dialog.show()
        self.timer.start()
        self.request_frame()

    def request_frame(self):
        self.phase = 'waiting'
        self.wait_started = time.monotonic()
        w = self.window
        w.slider.blockSignals(True)
        w.slider.setValue(self.target)
        w.slider.blockSignals(False)
        w.seek(self.target)
        QTimer.singleShot(0, self.check)

    def decode_failed(self, message):
        self.fail('Cannot export the selected video: ' + message)

    def fail(self, message):
        if self.error is None:
            self.error = message
        self.cancel()

    def cancel(self, *, close_window=False):
        self.close_window |= close_window
        self.cancelling = True
        self.dialog.cancel_button.setEnabled(False)
        self.dialog.label.setText('Cancelling preview export...')
        if self.writer is not None:
            self.writer.cancel()
        else:
            QTimer.singleShot(0, self.finish)

    def check(self):
        if self.cancelling or self.phase == 'done':
            return
        w = self.window
        elapsed = time.monotonic() - self.started
        total = self.end - self.first
        eta = f' · About {(total-self.rendered)*elapsed/self.rendered:.0f}s left' if self.rendered >= 3 and self.phase != 'finalizing' else ''
        message = (f'Finishing MP4 · {elapsed:.0f}s elapsed' if self.phase == 'finalizing' else
                   f'Rendering frame {self.rendered + 1}/{total} · {elapsed:.0f}s elapsed{eta}')
        self.dialog.label.setText(message)
        w.status.setText('Export Preview: ' + message)
        if self.phase != 'waiting':
            if self.phase == 'writing' and time.monotonic() - self.wait_started > 60:
                self.fail('The video encoder stopped responding. The destination was not changed.')
            return
        video, view = w.video_view, w.motion_view
        if video.decoder is not self.decoder or video.generation != self.generation:
            self.fail('The input video changed during export.');return
        if view.size() != self.view_size:
            self.fail('The viewport size changed during export. Please try again.');return
        if self.scene_context() != self.context:
            self.fail('The displayed motion changed during export. Please try again.');return
        if video.frame is None or video.index != self.target or view.index != self.target:
            if time.monotonic() - self.wait_started > 30:
                self.fail('Timed out waiting for the matching video and 3D frame.')
            return
        try:
            if not self.characters_ready():
                if time.monotonic() - self.wait_started > 30:
                    self.fail('Timed out preparing the character animation frame.')
                return
            self.phase = 'capturing'
            # Retain the chosen framing even if an asynchronous asset completes.
            for name, value in self.camera.items():
                setattr(view, name, value.copy() if hasattr(value, 'copy') else value)
            if hasattr(view, 'grabFramebuffer'):
                if not view.isValid():
                    raise RuntimeError('The 3D viewport is not ready.')
                image = view.grabFramebuffer()
            else:
                image = view.grab().toImage()
            if image.isNull():
                raise RuntimeError('Cannot capture the current 3D viewport.')
            if self.writer is None:
                self.writer = PreviewEncoder(self.destination, output_layout(video.frame, image), self.fps,
                    total, self.audio, self.audio_start, self)
                self.writer.frame_written.connect(self.written)
                self.writer.finished.connect(self.finish)
                self.writer.start()
                w.log.append(f'Export Preview: {total} frames at {self.fps:g} fps → {self.destination}')
            self.phase = 'writing'
            self.writer.submit(self.target, video.frame, image)
        except Exception as exc:
            self.fail(str(exc))

    def written(self, index):
        if self.cancelling:
            return
        if index != self.target:
            self.fail('Preview export received an out-of-order frame.');return
        self.rendered += 1
        self.dialog.progress.setValue(self.rendered)
        if self.target + 1 == self.end:
            self.phase = 'finalizing'
            self.dialog.progress.setRange(0, 0)
            self.writer.finish_frames()
        else:
            self.target += 1
            self.request_frame()

    def finish(self):
        if self.phase == 'done' or (self.writer is not None and self.writer.isRunning()):
            return
        self.phase = 'done'
        self.timer.stop()
        w = self.window
        try:
            w.video_view.decode_error.disconnect(self.decode_failed)
            w.video_view.frame_presented.disconnect(self.video_ready)
            w.character_loader.failed.disconnect(self.character_failed)
        except RuntimeError:
            pass
        view = w.motion_view
        for name, value in self.camera.items():
            setattr(view, name, value)
        view._paint_camera = None
        w.setMinimumSize(self.window_limits[0])
        w.setMaximumSize(self.window_limits[1])
        w.setAcceptDrops(self.accept_drops)
        for shortcut, enabled in zip(w.shortcuts, self.shortcut_states):
            shortcut.setEnabled(enabled)
        w.slider.blockSignals(True)
        w.slider.setValue(self.original_frame)
        w.slider.blockSignals(False)
        w.seek(self.original_frame)
        self.dialog.allow_close = True
        self.dialog.accept()
        self.dialog.cancel_requested.disconnect(self.cancel)
        error = self.error or (self.writer.error if self.writer is not None else None)
        success = self.writer is not None and self.writer.succeeded
        message = ('Preview saved: ' + str(self.destination) if success else
                   'Export Preview failed: ' + error if error else 'Preview export cancelled. The destination was not changed.')
        w.status.setText(message)
        w.log.append(message)
        w._preview_export = None
        w.update_readiness()
        if self.resume and not self.close_window:
            w.toggle_play()
        if self.close_window:
            QTimer.singleShot(0, w.close)
        self.deleteLater()


class PreviewExportWorkspace:
    def refresh_preview_export_action(self, *args):
        self.export_preview_action.setEnabled(self.preview_export_available())

    def preview_export_available(self):
        view = self.motion_view
        return (not getattr(self, '_pear_active', False) and self.model_choice.currentData() != 'pear'
                and not self.busy and not self.contact_saving and not getattr(self, '_preview_export', None)
                and not self._workspace_loading
                and (self.model_choice.currentData() != 'mamma' or (not self.mamma_pending and not self._mamma_switching))
                and self.info is not None and self.video_readable and self.video_view.frame is not None
                and (self.preview_signature is None or (self.preview_data is not None
                     and self.preview_data.project == self.project))
                and (not view.people_mode or (view.people_scene is not None and self.preview_data is not None
                     and self.preview_data.project == self.project))
                and (view.joints is not None or view.people_scene is not None or bool(getattr(self, 'character_assets', {}))))

    def save_preview(self):
        if not self.preview_export_available():
            return
        resume = self.timer.isActive()
        self.stop_playback()
        picker = QFileDialog(self, 'Export Preview')
        picker.setAcceptMode(QFileDialog.AcceptSave)
        picker.setNameFilter('MP4 video (*.mp4)')
        picker.setDefaultSuffix('mp4')
        picker.selectFile(Path(self.video_path).stem + '-preview.mp4')
        if not picker.exec():
            if resume:self.toggle_play()
            return
        try:
            self.start_preview_export(picker.selectedFiles()[0], resume=resume)
        except (OSError, ValueError, RuntimeError) as exc:
            self.status.setText('Cannot export preview: ' + str(exc))
            QMessageBox.warning(self, 'Export Preview', str(exc))
            if resume:self.toggle_play()

    def start_preview_export(self, destination, *, resume=None):
        if not self.preview_export_available():
            raise ValueError('Load an offline motion and its video before exporting a preview. PEAR is not supported.')
        path = Path(destination).resolve()
        if path.suffix.lower() != '.mp4' or not path.parent.is_dir():
            raise ValueError('Choose an MP4 file in an existing folder.')
        inputs = [self.video_path, self.video_view.decoder.path]
        inputs += [entry['path'] for entry in getattr(self, 'mamma_inputs', [])]
        if any(p and path == Path(p).resolve() for p in inputs):
            raise ValueError('The preview destination must not replace an input video.')
        if not 0 <= self.slider.clip_in < self.slider.clip_out <= self.info['frames']:
            raise ValueError('The timeline trim is invalid.')
        from .video import executable
        executable('ffmpeg')
        controller = PreviewExport(self, path, self.timer.isActive() if resume is None else resume)
        self._preview_export = controller
        self.export_preview_action.setEnabled(False)
        controller.start()
