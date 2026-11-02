from __future__ import annotations
from .tooltips import tooltip_text
from .ui_refresh import batch_readiness
from .validation_snapshot import read_only_checks

import json
import codecs
import math
import os
from collections import deque
from pathlib import Path
import time

from PySide6.QtCore import QElapsedTimer, QEvent, QProcess, QProcessEnvironment, QTimer, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QKeySequence, QShortcut, QTextOption
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFileDialog,
    QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QSplitter, QTableWidget, QTableWidgetItem, QTextBrowser, QTextEdit, QVBoxLayout, QWidget)

from .models import MODELS_LIST, import_body_model, missing, ready
from .paths import ROOT, PROJECTS, worker_command
from .video import select_preview
from .viewer import VideoView, MotionView
from .playback import playback_frame, PlaybackTimerResolution
from .timeline import TimelineSlider
from .preview_data import PreviewDataLoader, project_signature
from .motion_tools import MOTION_TOOLS, selected_export, RESULTS, result_kind, result_options, result_is_current, base_result, body_result, has_handflow

STAGES = {"prepare": "Prepare Video", "track": "Track Actor", "pva": "Analyze 2D", "gvhmr": "Initial 3D Reconstruction", "refine": "Refine Motion", "export": "Export", "download": "Model Manager", "diagnose": "Check System"}

from .theme import STYLE
STAGES["mesh"] = "Prepare SMPL-X Mesh"
STAGES['rig']='Prepare Bone Preview'
STAGES['save_fbx']='Save FBX'
STAGES['copy_bvh']='Save BVH'
STAGES['save_blend']='Save Blender Project'
STAGES['save_blend_scene']='Save MAMMA Blender Scene'
STAGES["footmr"] = "Refine Ankles / FootMR"
STAGES["handflow"] = "Infer Fingers / HandFlow"
STAGES['correct'] = 'Motion Correction'
from .workspace_ui import build_workspace
from .people_ui import PeopleWorkspace
from .project_save_ui import ProjectSaving
from .project_workspace_ui import ModelWorkspace
STAGES.update(mamma_prepare='Prepare Multi-view Videos',mamma_masks='MAMMA Person Masks',mamma_landmarks='MAMMA Surface Landmarks',mamma_camera='MAMMA Cameras',mamma_fit='MAMMA 3D Fit',mamma_publish='MAMMA Motion Correction and Preview')
STAGES.update(prompthmr_prepare='Prepare PromptHMR Video',prompthmr_track='PromptHMR People and Masks',prompthmr_pose='PromptHMR Camera, Keypoints and Motion',prompthmr_world='PromptHMR Shared World and Post-optimization',prompthmr_publish='PromptHMR Motion Correction and Preview')
STAGES['save_project']='Save Project As'
STAGES.update(people='Find People',person='Prepare Person Motion')
STAGES.update(flow_camera='Estimate Scene Camera / VGGT-Omega', flow_features='Extract Body and Hand Features / SAM', flowhmr='Reconstruct Motion / FlowHMR', flow_export='Prepare FlowHMR Preview and FBX', import_model='Import Model')
STAGES.update(exmr_preprocess='ExMR Body, Hand and Foot Keypoints',exmr_features='ExMR Body and Hand Features',exmr='Reconstruct Motion / ExMR')

STAGES.update(flow_batch='Process FlowHMR Motion',exmr_batch='Process ExMR Motion',gvhmr_batch='GVHMR Multi-person',features='GVHMR Image Observations')

class ModelDownloadDialog(QDialog):
    """Readable, selectable provider instructions with bounded scrolling."""
    def __init__(self, parent, key, manager=None):
        super().__init__(parent)
        self.manager = manager
        self.model_key = key
        from .models import BY_KEY, manual_download_page, manual_instructions_html
        self.setWindowTitle(BY_KEY[key].name + " - Manual Download")
        self.setMinimumSize(360, 260)
        area = self.screen().availableGeometry()
        self.resize(min(680, max(360, area.width() - 80)),
                    min(560, max(260, area.height() - 100)))
        layout = QVBoxLayout(self)
        self.browser = QTextBrowser()
        self.browser.setStyleSheet(
            'QTextBrowser { font-family:"' + QApplication.font().family() + '"; font-size:12px; padding:6px; }')
        self.browser.setOpenLinks(False)
        self.browser.setOpenExternalLinks(False)
        self.browser.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.browser.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.browser.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.browser.setHtml(manual_instructions_html(key))
        self.browser.anchorClicked.connect(QDesktopServices.openUrl)
        layout.addWidget(self.browser)
        self.import_status = QLabel()
        self.import_status.setWordWrap(True)
        self.import_status.setMaximumHeight(64)
        self.import_status.hide()
        layout.addWidget(self.import_status)
        actions = QHBoxLayout(); actions.addStretch()
        self.website = QPushButton("Open Download Page")
        self.website.setToolTip('Open the model provider website in your browser.')
        self.website.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(manual_download_page(key))))
        actions.addWidget(self.website)
        self.import_file = QPushButton("Import...")
        self.import_file.setToolTip('Select the downloaded model file to check and import it. These instructions stay open.')
        self.import_file.clicked.connect(self.import_download)
        actions.addWidget(self.import_file)
        close = QPushButton("Close"); close.clicked.connect(self.accept)
        close.setToolTip('Close these download instructions.')
        actions.addWidget(close); layout.addLayout(actions)
        self.status_timer = QTimer(self)
        self.status_timer.setInterval(100)
        self.status_timer.timeout.connect(self.sync_import_status)
        self.sync_import_status()

    def showEvent(self, event):
        super().showEvent(event)
        self.sync_import_status()
        self.status_timer.start()

    def hideEvent(self, event):
        self.status_timer.stop()
        super().hideEvent(event)

    def import_download(self):
        if self.manager is not None:
            self.manager.import_local_model(self.model_key, picker_parent=self)
            self.sync_import_status()

    def sync_import_status(self):
        manager = self.manager
        self.import_file.setEnabled(manager is not None and not manager.window.busy)
        if manager is None:
            return
        manager.sync_task_status()
        message = manager.task_status.toPlainText() if manager._observing_setup or manager.window.busy else ''
        if self.import_status.toolTip() != message:
            self.import_status.setTextFormat(Qt.PlainText)
            self.import_status.setText(message[:500])
            self.import_status.setToolTip(message)
            self.import_status.setVisible(bool(message))


class ModelsDialog(QDialog):
    def __init__(self, window, tool_key=None):
        super().__init__(window)
        self.window = window
        self.tool = MOTION_TOOLS[tool_key or window.model_choice.currentData()]
        from .models import STAGE_REQUIREMENTS
        keys = ({key for key in STAGE_REQUIREMENTS[self.tool.key+'_assets']} if self.tool.key in ('mamma','prompthmr','pear') else
                {key for stage in (*self.tool.stages(refine=True, footmr=True, handflow=True),'people') for key in STAGE_REQUIREMENTS.get(stage, ())})
        if self.tool.key=='pear':keys.update(STAGE_REQUIREMENTS['pear_align'])
        if self.tool.key=='prompthmr':keys.update(STAGE_REQUIREMENTS['prompthmr_camera_assets'])
        if self.tool.key=='flowhmr':
            keys.add('flow_metadata')
            if window.flow_hamer.isChecked():
                from .flow_hamer import MODEL_KEYS
                keys.update(MODEL_KEYS)
        # Keep public automatic downloads together, followed by manual-only files.
        # Stable sorting preserves the existing component order within each group.
        self.models = sorted((model for model in MODELS_LIST if model.key in keys),
                             key=lambda model: not bool(model.drive_id or model.download_url))
        self.setWindowTitle(f"{self.tool.label} Model Manager")
        self.resize(850, min(540, max(360, self.screen().availableGeometry().height() - 80)))
        layout = QVBoxLayout(self)
        description = QLabel(f"Models for {self.tool.label} and compatible add-ons.\nUse Auto Download for public files, or Manual Download and Import for any model.\nRegistered body model files are not redistributed by this app.")
        description.setWordWrap(True)
        layout.addWidget(description)
        if self.tool.key=='mamma':
            from .mamma_manager import runtime_panel
            layout.addWidget(runtime_panel(self))
        elif self.tool.key=='prompthmr':
            from .prompthmr_manager import runtime_panel
            layout.addWidget(runtime_panel(self))
        elif self.tool.key=='pear':
            from .pear_runtime import status
            note=QLabel(status()[1]+'\nPEAR support is included in install.cmd and update.cmd. Uses the existing shared runtime; Camera Align uses a small Flow-runtime support overlay. No CUDA compiler is needed. VGGT and GeoCalib are optional for multiple fixed views; model files are prepared separately below.')
            note.setWordWrap(True);layout.addWidget(note)
        self.table = QTableWidget(len(self.models), 3)
        self.table.setHorizontalHeaderLabels(["Model", "Status", "Action"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 300)
        self.table.setColumnWidth(1, 130)
        layout.addWidget(self.table)
        self.task_status = QTextBrowser()
        self.task_status.setFixedHeight(64)
        self.task_status.setOpenLinks(False)
        self.task_status.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.task_status.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.task_status.setStyleSheet(
            'QTextBrowser { font-family:"' + QApplication.font().family() + '"; font-size:12px; padding:4px; }')
        self.task_status.setPlainText("Ready to download or import models.")
        layout.addWidget(self.task_status)
        self.task_progress = QProgressBar()
        self.task_progress.setTextVisible(False)
        self.task_progress.setRange(0, 100); self.task_progress.setValue(0)
        layout.addWidget(self.task_progress)
        actions = QHBoxLayout()
        self.cancel_task = QPushButton("Cancel")
        self.cancel_task.setToolTip('Stop the current download or import queue. Completed files are kept.')
        self.cancel_task.setEnabled(False)
        self.cancel_task.clicked.connect(self.cancel_model_job)
        actions.addWidget(self.cancel_task)
        actions.addStretch()
        self.download_all = QPushButton("Download All")
        self.download_all.setToolTip(tooltip_text(
            "Download missing public model files. Files requiring registration must be downloaded manually and imported. Completed downloads are kept if you cancel or retry."))
        self.download_all.clicked.connect(self.start_download_all)
        actions.addWidget(self.download_all)
        close = QPushButton("Close")
        close.setToolTip(tooltip_text("Close Model Manager. Any running download or import continues in the main workspace."))
        close.clicked.connect(self.accept); actions.addWidget(close)
        layout.addLayout(actions)
        self._batch_total = 0
        self._observing_setup = False
        self._row_state = None
        self._status_state = None
        self.status_timer = QTimer(self)
        self.status_timer.setInterval(100)
        self.status_timer.timeout.connect(self.sync_task_status)
        self.refresh()

    def refresh(self):
        from .models import STAGE_REQUIREMENTS
        for i, model in enumerate(self.models):
            name = model.name + (" (Optional: HTD / 2D)" if self.tool.key == "gvhmr" and model.key == "pva" else "")
            if self.tool.key=='pear' and model.key in STAGE_REQUIREMENTS['pear_align']:name+=' (Optional: Camera Align)'
            available = ready(model)
            for column, text in ((0, name), (1, "Ready" if available else "Missing")):
                item = self.table.item(i, column)
                if item is None:
                    item = QTableWidgetItem(); self.table.setItem(i, column, item)
                item.setText(text)
                if column==0:item.setToolTip(str(model.path))
                item.setForeground(QColor("#d5d5d5" if available else "#ffb45c"))
                font = item.font(); font.setBold(not available); item.setFont(font)
            cell = self.table.cellWidget(i, 2)
            if cell is None or cell.property("model_key") != model.key:
                cell = QWidget(); cell.setProperty("model_key", model.key)
                row = QHBoxLayout(cell); row.setContentsMargins(0, 0, 0, 0)
                cell.automatic = None
                if model.drive_id or model.download_url:
                    cell.automatic = QPushButton("Auto Download")
                    cell.automatic.setToolTip(tooltip_text("Download this file automatically. If it fails, use Manual Download or Import an existing copy."))
                    cell.automatic.clicked.connect(lambda checked=False, key=model.key: self.start_download(key))
                    row.addWidget(cell.automatic)
                cell.manual = QPushButton("Manual Download...")
                cell.manual.setToolTip(tooltip_text("Show download links and instructions. Download in your browser, then use Import in the instructions or this row."))
                cell.manual.clicked.connect(lambda checked=False, key=model.key: self.manual_setup(key))
                row.addWidget(cell.manual)
                cell.local = QPushButton("Import...")
                cell.local.setToolTip(tooltip_text("Choose a model file you already downloaded. The app checks it before replacing any existing file."))
                cell.local.clicked.connect(lambda checked=False, key=model.key: self.import_local_model(key))
                row.addWidget(cell.local)
                self.table.setCellWidget(i, 2, cell)
            if cell.automatic is not None:
                cell.automatic.setEnabled(not self.window.busy and not available)
            cell.manual.setEnabled(True)
            cell.local.setEnabled(not self.window.busy)
        self.download_all.setEnabled(not self.window.busy and bool(self.pending_downloads()))
        self._row_state = (self.window.busy, id(self.window.process))

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh(); self.sync_task_status()
        self.status_timer.start()

    def hideEvent(self, event):
        self.status_timer.stop()
        super().hideEvent(event)

    def sync_task_status(self):
        # Poll only cheap GUI state; filesystem readiness is refreshed once per
        # worker transition, never on each progress update or every timer tick.
        w = self.window
        row_state = (w.busy, id(w.process))
        if row_state != self._row_state:
            self.refresh()
        setup = w.current_stage in {"download", "import_model"}
        if w.busy and setup:
            self._observing_setup = True
        state = (w.busy, setup, w.cancelled, w.failed, w.status.text(),w.task_info.text(),
                 w.progress.minimum(), w.progress.maximum(), w.progress.value(), len(w.jobs))
        if state == self._status_state:
            return
        self._status_state = state
        active = w.busy and setup
        self.cancel_task.setEnabled(active and not w.cancelled)
        if setup and self._observing_setup:
            message = w.status.text()
            if active:message+='\n'+w.task_info.text()
            if active and self._batch_total > 1 and not w.cancelled and not w.failed:
                index = min(self._batch_total, max(1, self._batch_total - len(w.jobs)))
                message = f"Model {index}/{self._batch_total}: " + message
            self.task_status.setPlainText(message if len(message) <= 4096 else message[:4096] + "\nSee Console for the complete message.")
            self.task_status.setStyleSheet(
                'QTextBrowser { font-family:"' + QApplication.font().family() +
                '"; font-size:12px; padding:4px;' +
                ('color:#ffb45c;' if w.failed else '') + ' }')
            if active:
                self.task_progress.setRange(w.progress.minimum(), w.progress.maximum())
                self.task_progress.setValue(w.progress.value())
            else:
                self.task_progress.setRange(0, 100)
                self.task_progress.setValue(0 if w.failed or w.cancelled else 100)
        elif w.busy:
            self.task_status.setPlainText("Another task is running. You can still open manual download instructions.")
            self.task_progress.setRange(0, 0)
        else:
            self.task_status.setPlainText("Ready to download or import models.")
            self.task_progress.setRange(0, 100); self.task_progress.setValue(0)

    def start_model_jobs(self, jobs):
        if self.window.busy:
            return
        self._batch_total = len(jobs)
        self._observing_setup = True
        self._status_state = None
        self.window.start_jobs(jobs)
        self.sync_task_status()

    def cancel_model_job(self):
        if self.window.busy and self.window.current_stage in {"download", "import_model"}:
            self.window.cancel_job()
            self.sync_task_status()

    def pending_downloads(self):
        return [model for model in self.models
                if (model.drive_id or model.download_url) and not ready(model)]

    def start_download_all(self):
        if self.window.busy:
            return
        jobs = [("download", ["--model", model.key]) for model in self.pending_downloads()]
        if not jobs:
            self.refresh()
            return
        self.start_model_jobs(jobs)

    def start_download(self, key):
        if key not in {model.key for model in self.models}:
            raise ValueError("This model is not part of the selected motion tool.")
        self.start_model_jobs([("download", ["--model", key])])

    def manual_setup(self, key):
        if key not in {model.key for model in self.models}:
            raise ValueError("This model is not part of the selected motion tool.")
        dialog = ModelDownloadDialog(self, key, manager=self)
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    def import_local_model(self, key, picker_parent=None):
        from .models import BY_KEY
        if self.window.busy:
            return
        if key not in {model.key for model in self.models}:
            raise ValueError("This model is not part of the selected motion tool.")
        model = BY_KEY[key]
        name, _ = QFileDialog.getOpenFileName(picker_parent or self, 'Import ' + model.name, '',
            ('PEAR assets (assets.zip);;All files (*)' if key=='pear_geometry' else f'Model file ({model.path.name});;All files (*)'))
        if name:
            self.start_model_jobs([('import_model', ['--model', key, '--video', name])])



from .character_ui import CharacterWorkspace
from .mamma_ui import MammaWorkspace
from .prompthmr_ui import PromptHMRWorkspace
from .pear_ui import PearWorkspace
from .preview_export_ui import PreviewExportWorkspace

class MainWindow(PreviewExportWorkspace,PearWorkspace,PromptHMRWorkspace,MammaWorkspace,CharacterWorkspace,ModelWorkspace,ProjectSaving,PeopleWorkspace,QMainWindow):
    def __init__(self, video: Path | None = None):
        super().__init__()
        self.setWindowTitle("RGB MoCap | Motion Workspace")
        self.resize(1440, 880)
        self.setMinimumSize(1100, 700)
        self.video_path = None
        self.init_mamma()
        self.init_pear()
        self.project = None
        self.workspace_root=None
        self._workspace_loading=False
        self._workspace_video_reference=False
        self._workspace_camera=None
        self.init_people()
        self.active_result = "corrected_footmr_refined"
        self.info = None
        self.source_offset = 0.0
        self.source_end = 0.0
        self.video_readable = False
        self.busy = False
        self.jobs = []
        self.process = None
        self.cancel_process = None
        self.buffer = ""
        self.failed = False
        self.cancelled = False
        self.hardware_summary = None
        self.current_stage = None
        self._batch_started_at = None
        self._batch_has_motion = False
        from .task_progress import TaskProgress
        self.task_state=TaskProgress()
        self._task_manifests={};self._queued_task_estimates=[]
        self.play_clock = QElapsedTimer()
        self.timer_resolution = PlaybackTimerResolution()
        self.play_anchor = 0
        self.playback_tick = False
        self.presented_times = deque()
        self.preview_data = None
        self.preview_signature = None
        from .contact_writer import ContactWriter
        self.contact_tracks={};self.contact_saving=False;self.contact_save_token=0
        self.contact_writer=ContactWriter(self)
        self.contact_writer.saved.connect(self.foot_contacts_saved)
        self.contact_writer.failed.connect(self.foot_contacts_failed)
        self.contact_writer.settled.connect(self.foot_contacts_settled)
        self.data_loader = PreviewDataLoader(self)
        self.data_loader.ready.connect(self.preview_ready)
        self.data_loader.failed.connect(self.preview_failed)
        self.data_loader.tracked_actor_ready.connect(self.tracked_actor_ready)
        build_workspace(self)
        self.init_characters()
        self.video_view.frame_presented.connect(self.refresh_preview_export_action)
        self.scene_loader.ready.connect(self.refresh_preview_export_action)
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.next_frame)
        self.shortcuts = []
        for key, action in [("Space", self.toggle_play), ("Left", lambda: self.step_frames(-1)),
                            ("Right", lambda: self.step_frames(1)),
                            ("Shift+Left", lambda: self.step_frames(-round(self.info['fps']) if self.info else -30)),
                            ("Shift+Right", lambda: self.step_frames(round(self.info['fps']) if self.info else 30)),
                            ("Home", lambda: self.step_frames(-self.slider.maximum())),
                            ("End", lambda: self.step_frames(self.slider.maximum()))]:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(action)
            self.shortcuts.append(shortcut)
        self.setAcceptDrops(True)
        self.data_loader.start()
        self.contact_writer.start()
        self.processing_guard = None
        self.guard_timer = QTimer(self)
        self.guard_timer.setInterval(500)
        self.guard_timer.timeout.connect(self.check_processing_resources)
        self.task_timer=QTimer(self)
        self.task_timer.setInterval(500)
        self.task_timer.timeout.connect(self.update_task_progress)
        # Initialize the same model-specific visibility used on selector changes.
        # Otherwise the empty GVHMR stack reserves the FlowHMR page height.
        self.select_model()
        if video:
            self.load_video(video)

    @read_only_checks
    def update_readiness(self):
        self.export_preview_action.setEnabled(self.preview_export_available())
        if self._pear_active:
            if hasattr(self,'camera_tools'):self.camera_tools.sync()
            self.pear_sync_controls();return
        if getattr(self,'_readiness_depth',0):
            self._readiness_pending=True
            return
        if hasattr(self,'character_loader'):self.sync_characters()
        self.update_people_readiness()
        tool = MOTION_TOOLS[self.model_choice.currentData()]
        self.compute_mode.setEnabled(not self.busy)
        self.resource_safety.setEnabled(not self.busy)
        stages = tool.stages(self.htd_refine.isChecked(), self.footmr_refine.isChecked(), handflow=self.handflow_refine.isChecked())
        from .models import STAGE_REQUIREMENTS, BY_KEY
        required = {key for stage in stages for key in STAGE_REQUIREMENTS.get(stage, ())}
        if self.flow_hamer.isChecked():
            for checkbox in (self.fast_sam,self.flow_hand_stride4):
                blocked=checkbox.blockSignals(True);checkbox.setChecked(True);checkbox.blockSignals(blocked)
            if tool.key=='flowhmr':
                from .flow_hamer import MODEL_KEYS
                required.update(MODEL_KEYS)
        if tool.key=='flowhmr' and self.flow_sampling.currentData()==-1:required.discard('flow_vggt')
        if tool.key=='mamma' and self.mamma_camera_mode.currentData()=='manual':required.difference_update(('flow_vggt','pear_geocalib'))
        if tool.key=='prompthmr' and self.prompthmr_camera.currentData()=='moving':required.update(STAGE_REQUIREMENTS['prompthmr_camera_assets'])
        unavailable = [BY_KEY[key] for key in sorted(required) if not ready(BY_KEY[key])]
        count = len(required) - len(unavailable)
        text = f"{tool.label}: {count}/{len(required)} models ready"
        if unavailable:
            noun = "model" if len(unavailable) == 1 else "models"
            text += f"\nWarning: {len(unavailable)} required {noun} missing.\nOpen Model Manager to download or import."
        self.readiness.setStyleSheet("color:#ffb45c; font-weight:600;" if unavailable else "")
        self.readiness.setToolTip(tooltip_text("Missing required models:\n" + "\n".join(model.name for model in unavailable) if unavailable else ""))
        self.readiness.setText(text)
        for button in (self.models_button, self.analyze_button, self.run_button):
            button.setEnabled(not self.busy)
        self.cancel_button.setEnabled(self.busy)
        self.check_system_action.setEnabled(not self.busy)
        self.model_choice.setEnabled(not self.busy and not self.contact_saving)
        self.htd_refine.setEnabled(not self.busy)
        self.footmr_refine.setEnabled(not self.busy)
        self.handflow_refine.setEnabled(not self.busy)
        self.motion_correction.setEnabled(not self.busy)
        self.correction_controls.setEnabled(not self.busy and self.motion_correction.isChecked())
        flat=self.correction_mode.currentData()=='flat'
        self.auto_floor.setEnabled(not self.busy and flat)
        self.floor_height.setEnabled(not self.busy and flat and not self.auto_floor.isChecked())
        self.correction_button.setEnabled(not self.busy and not self.contact_saving and self.project is not None and
            (self.project/RESULTS[base_result(self.active_result)][0]).exists())
        self.slider.trim_enabled = not self.busy and self.info is not None
        self.slider.contact_edit_enabled=not self.busy and not self.contact_saving and self.motion_correction.isChecked() and self.slider.contacts_available
        self.slider.contact_track_enabled=(self.foot_lock.isChecked(),)*2+(self.hand_lock.isChecked(),)*2
        self.slider.update()
        self.reset_footlocks_button.setEnabled(self.slider.contact_edit_enabled)
        self.reset_trim_button.setEnabled(self.slider.trim_enabled)
        character=self.selected_character()
        self.export_action.setText('Export Character...' if character else 'Export...')
        self.export_action.setEnabled(not self.busy and (bool(character.get('takes')) and not character.get('needs_retarget') if character else self.project is not None and self.current_export() is not None))
        self.export_action.setToolTip(tooltip_text("Choose a character export format and whether to include its mesh. PMX characters also support VMD motion." if character else "Export the displayed motion as FBX, BVH or a portable Blender project. Multiple people export separately; MAMMA also offers a whole-scene Blender bundle."))
        self.play_button.setEnabled(self.info is not None and self.video_readable)
        for action in self.open_actions:
            action.setEnabled(not self.busy)
        self.save_project_action.setEnabled(not self.busy and not self.contact_saving and self.video_path is not None and self.info is not None)
        self.mesh_button.setEnabled(self.preview_preparation_available())
        self.import_character_action.setEnabled(not self.busy and not self.contact_saving)
        self.flow_sampling.setEnabled(not self.busy)
        self.fast_sam.setEnabled(not self.busy and not self.flow_hamer.isChecked())
        self.flow_hand_stride4.setEnabled(not self.busy and not self.flow_hamer.isChecked())
        self.flow_hamer.setEnabled(not self.busy)
        self.flow_acceleration.setVisible(self.fast_sam.isChecked())
        self.flow_acceleration.setEnabled(not self.busy)
        self.exmr_camera.setEnabled(not self.busy)
        self.exmr_smoothing.setEnabled(not self.busy)
        self.flowhmr_background.setEnabled(not self.busy)
        self.exmr_background.setEnabled(not self.busy)
        self.analyze_button.setVisible(tool.key=='gvhmr')
        self.sync_correction_floor_preview()
        self.sync_mamma_controls()
        self.sync_prompthmr_controls()
        if hasattr(self,'camera_tools'):self.camera_tools.sync()

    def sync_correction_floor_preview(self):
        self.motion_view.correction_floor_applicable=self.correction_mode.currentData()=='flat'
        enabled=self.motion_correction.isChecked() and self.correction_mode.currentData()=='flat'
        pending=bool(self.project is not None and enabled and
                     not result_is_current(self.project,'corrected_'+base_result(self.active_result)))
        self.motion_view.set_correction_floor(self.floor_height.value() if self.project is not None else None,
            automatic=self.auto_floor.isChecked(),pending=pending,enabled=enabled)

    def show_model_guide(self):
        from .model_guide import ModelGuideDialog
        dialog = ModelGuideDialog(self, self.model_choice.currentData())
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    def select_model(self, *_):
        if self.pear_select_model():return
        tool = MOTION_TOOLS[self.model_choice.currentData()]
        if self.workspace_root is not None and not self._workspace_loading and tool.key!=getattr(self,'_selected_tool',None):
            if self.busy or self.contact_saving:
                self.model_choice.blockSignals(True)
                self.model_choice.setCurrentIndex(self.model_choice.findData(self._selected_tool))
                self.model_choice.blockSignals(False)
                return
            self.switch_workspace_model(tool.key)
            return
        previous=getattr(self,'_selected_tool','gvhmr')
        choices=getattr(self,'_tool_refinement',{'gvhmr':(True,True,True,True)})
        if previous!=tool.key:
            if previous=='gvhmr':choices['gvhmr']=(self.htd_refine.isChecked(),self.footmr_refine.isChecked(),self.motion_correction.isChecked(),self.handflow_refine.isChecked())
            else:choices[previous]=(False,False,self.motion_correction.isChecked(),False)
            values=choices.get(tool.key,(tool.key=='gvhmr',tool.key=='gvhmr',True,tool.key=='gvhmr'))
            for control,value in zip((self.htd_refine,self.footmr_refine,self.motion_correction,self.handflow_refine),values):
                control.blockSignals(True);control.setChecked(value);control.blockSignals(False)
        self._selected_tool=tool.key;self._tool_refinement=choices
        self.active_result=self.selected_result()
        if self.project is not None:
            state=json.loads((self.project/'project.json').read_text(encoding='utf-8'))
            if state.get('pipeline',{}).get('model','gvhmr')!=tool.key:
                # Keep source timing/camera but never show another engine's result.
                self.stop_playback();self.clear_preview();self.reset_people()
                self.project=None
                self.active_result=self.selected_result()
                self.status.setText(f'{tool.label} selected. Reconstruct this clip or open a {tool.label} project.')
        for index in range(self.model_pages.count()):
            if self.model_pages.widget(index).property("motion_tool") == tool.key:
                self.model_pages.setCurrentIndex(index)
                break
        self.model_pages.setVisible(tool.key!='gvhmr')
        self.htd_refine.setVisible(bool(tool.refinement))
        self.refine_description.setVisible(bool(tool.refinement))
        self.footmr_refine.setVisible(tool.key == "gvhmr")
        self.handflow_refine.setVisible(tool.key == "gvhmr")
        self.footmr_description.setVisible(tool.key == "gvhmr")
        if not tool.refinement:
            self.refine_preview_note.hide()
        self.models_group.setTitle(tool.label+' Models')
        self.correction_group.setVisible(True)
        self.slider.contacts_supported=True
        self.slider._cache_key=None
        self.slider.update()
        self.correction_description.setText('Only checked contact types are corrected. Uses the selected '+tool.label+' result.')
        self.analyze_button.setToolTip(tooltip_text('Find body points in the video for GVHMR refinement. Reconstruct 3D normally includes this step when needed.'))
        self.mamma_mode_changed(previous,tool.key)
        self.update_readiness()

    def selected_result(self, correction=None):
        gvhmr=self.model_choice.currentData()=='gvhmr'
        return result_kind(gvhmr and self.htd_refine.isChecked(),gvhmr and self.footmr_refine.isChecked(),
            self.motion_correction.isChecked() if correction is None else correction,
            handflow=gvhmr and self.handflow_refine.isChecked())

    def refinement_changed(self, enabled):
        self.set_active_result(self.selected_result())

    def current_export(self):
        if self._pear_active:return None
        info=self.contact_tracks.get(base_result(self.active_result))
        if self.active_result.startswith('corrected_') and (self.contact_saving or (info and info.get('unsaved'))):return None
        # selected_export checks source, correction and target receipts itself.
        if self.project is not None:
            return selected_export(self.project, self.active_result)
        return None

    def render_changed(self):
        self.motion_view.show_bones=self.bone_overlay.isChecked()
        self.motion_view.update()
        if hasattr(self,'camera_tools'):self.camera_tools.canvas.update()

    def preview_preparation_available(self):
        if self.project is None or self.busy or self.contact_saving:return False
        if not (self.project/RESULTS[base_result(self.active_result)][0]).is_file():return False
        info=self.contact_tracks.get(base_result(self.active_result),{})
        if self.active_result.startswith('corrected_') and info.get('unsaved'):return False
        if self.model_choice.currentData()=='flowhmr':
            from .flow_placement import needs_update
            if needs_update(self.project):return True
        return ((self.project/RESULTS[self.active_result][0]).is_file()
                and result_is_current(self.project,self.active_result))

    def prepare_target_preview(self):
        if self.project is None or self.busy or self.contact_saving:return
        from .mixamo_data import current
        projects=[self.project]
        if self.people_state:
            from .people_store import safe_child
            projects += [safe_child(self.people_root,take['project']) for person in self.people_state['people']
                if person.get('checked') for take in person.get('takes',[])]
        jobs=[]
        for project in dict.fromkeys(projects):
            from .flow_placement import needs_update
            placement_old=self.model_choice.currentData()=='flowhmr' and needs_update(project)
            if (project/RESULTS[base_result(self.active_result)][0]).is_file() and (placement_old or result_is_current(project,self.active_result) and not current(project,self.active_result)):
                jobs.append(('rig',['--project',str(project),'--result',self.active_result]))
        if jobs:self.start_jobs(jobs)

    def open_model_manager(self):
        dialog = ModelsDialog(self)
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    def set_active_result(self, kind):
        """Select the cached preview and sync the HTD option without inference."""
        htd, foot = result_options(kind)
        self.active_result = kind
        self.handflow_refine.blockSignals(True)
        self.handflow_refine.setChecked(has_handflow(kind))
        self.handflow_refine.blockSignals(False)
        self.htd_refine.blockSignals(True)
        self.htd_refine.setChecked(htd)
        self.htd_refine.blockSignals(False)
        self.footmr_refine.blockSignals(True)
        self.footmr_refine.setChecked(foot)
        self.footmr_refine.blockSignals(False)
        self.motion_correction.blockSignals(True)
        self.motion_correction.setChecked(kind.startswith('corrected_'))
        self.motion_correction.blockSignals(False)
        self.refresh_results()
        self.request_people_scene()
        self.update_readiness()

    def choose_video(self):
        if self._pear_active:
            names,_=QFileDialog.getOpenFileNames(self,'Open PEAR Camera Videos','', 'Videos (*.mp4 *.mov *.avi *.mkv *.webm)')
            if names:self.load_pear_videos(names)
            return
        if self.model_choice.currentData()=='mamma':
            names,_=QFileDialog.getOpenFileNames(self,'Open MAMMA Camera Videos','', 'Videos (*.mp4 *.mov *.avi *.mkv *.webm)')
            if names:self.load_mamma_videos(names)
            return
        name, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4 *.mov *.avi *.mkv *.webm)")
        if name:
            self.load_video(Path(name))

    def load_video(self, path):
        if self._pear_active:return self.load_pear_videos([path])
        if self.model_choice.currentData()=='mamma':
            return self.load_mamma_videos([path])
        try:
            if self.workspace_root is not None:self.remember_workspace_model()
            self.clear_mamma_inputs()
            self.workspace_root=None;self._workspace_video_reference=False;self._workspace_camera=None
            self.stop_playback()
            self.clear_preview()
            self.reset_people()
            self.motion_view.auto_fit = True
            self.video_path, self.project, self.info = path, None, None
            self.setWindowTitle('RGB MoCap | Motion Workspace')
            self.flow_sampling.setCurrentIndex(self.flow_sampling.findData(0))
            self.fast_sam.setChecked(True)
            self.flow_batch4.setChecked(True)
            self.flow_gpu_crop.setChecked(True)
            self.flow_hand_stride4.setChecked(True)
            self.flow_hamer.setChecked(False)
            self.exmr_camera.setCurrentIndex(0)
            self.exmr_smoothing.setChecked(True)
            self.camera_tools.restore_background({})
            self.handflow_refine.blockSignals(True)
            self.handflow_refine.setChecked(self.model_choice.currentData()=='gvhmr')
            self.handflow_refine.blockSignals(False)
            self._tool_refinement={'gvhmr':(True,True,True,True)}
            for option in (self.htd_refine, self.footmr_refine, self.motion_correction):
                option.blockSignals(True)
                option.setChecked(True)
                option.blockSignals(False)
            self.active_result = self.selected_result()
            self.restore_correction_options({})
            self.source_offset = self.source_end = 0.0
            self.video_readable = False
            self.slider.setRange(0, 0)
            self.filename.setText(f"{path.name} · Opening video...")
            self.video_view.open(path)
            self.update_readiness()
        except Exception as error:
            QMessageBox.warning(self, "Video Error", str(error))

    def video_ready(self, info):
        if self.pear_restore_media_ready():return
        if self.pear_video_ready(info):return
        if self.mamma_video_ready(info):return
        self.video_readable = True
        self.slider.set_media(self.info["fps"] if self.info else info["fps"])
        if self.project is None and not self._workspace_video_reference:
            self.info = info
            self.slider.setRange(0, max(0, info["frames"] - 1))
            self.source_offset, self.source_end = 0.0, info["duration"]
            self.slider.reset_trim()
            self.clip_changed(self.slider.clip_in, self.slider.clip_out)
            self.filename.setText(f"{self.video_path.name} · {info['width']}×{info['height']} · {info['fps']:.2f}fps · {info['duration']:.2f} s")
            self.fps_label.setText(f"Source {info['fps']:.2f} fps")
            self.seek(0)
            if self.workspace_root is not None:
                from .project_workspace import read
                self.source_offset=float(read(self.workspace_root).get('time_offset',0.))
                self.source_end=self.source_offset+info['duration']
                self.restore_workspace_view(getattr(self,'_workspace_pending_view',{}))
        self.update_readiness()

    def video_failed(self, message):
        self.video_readable = False
        self.stop_playback()
        self.status.setText("Video preview: " + message)
        self.update_readiness()

    def choose_project(self):
        name, _ = QFileDialog.getOpenFileName(self, "Open Project", str(PROJECTS), "Project (project.json)")
        if name:
            from .project_workspace import root_for
            try:
                project=Path(name).parent
                if root_for(project) is None:
                    raise ValueError('This project uses the previous folder format. Open its video to create a new model-folder project.')
                self.load_project(project)
            except (OSError,ValueError,KeyError) as error:self.status.setText('Cannot open project: '+str(error))

    @batch_readiness
    def load_project(self, project):
        if self._pear_active:
            self.pear_leave()
            self.model_choice.blockSignals(True);self.model_choice.setCurrentIndex(self.model_choice.findData(self._selected_tool));self.model_choice.blockSignals(False)
        from .project_workspace import is_workspace
        if is_workspace(project):return self.open_workspace(project)
        from .project_storage import project_to_open, source_video
        try:
            project, workspace = project_to_open(project)
        except (OSError, ValueError) as error:
            self.status.setText('Cannot open project: '+str(error));return False
        from .motion_tools import require_supported_project, UnsupportedMotionTool
        try:
            require_supported_project(project)
        except UnsupportedMotionTool as error:
            self.status.setText(str(error))
            return False
        manifest = json.loads((project / "project.json").read_text(encoding="utf-8"))
        if manifest.get('pipeline',{}).get('model')=='prompthmr':
            group=(project/manifest.get('people_root','.')).resolve()
            if (group/'prompthmr_pending.json').exists():
                self.status.setText('PromptHMR publication is incomplete. Reconstruct again; previous projects were preserved.');return False
            self.restore_prompthmr_options(manifest.get('prompthmr_options',{}))
        if manifest.get('pipeline',{}).get('model')=='mamma':
            group=(project/manifest.get('people_root','.')).resolve()
            if (group/'mamma_pending.json').exists():
                self.status.setText('MAMMA publication is incomplete. Reconstruct again to finish; previous projects were preserved.')
                return False
        self.stop_playback()
        self.clear_preview()
        self.motion_view.auto_fit = True
        self.project = project
        self.camera_tools.restore_background({})
        self.workspace_project_loaded(project,manifest)
        from .project_storage import storage_root
        saved_root=storage_root(project,manifest)
        self.setWindowTitle('RGB MoCap | '+(saved_root.name if saved_root else 'Motion Workspace'))
        pipeline = manifest.get("pipeline", {})
        from .flow_camera import saved_interval
        from .flow_sam import saved_options
        flow=saved_options(project,manifest)
        self.flow_hamer.setChecked(False)
        self.fast_sam.setChecked(flow['fast_sam'])
        self.flow_batch4.setChecked(flow['sam_batch4'])
        self.flow_gpu_crop.setChecked(flow['sam_gpu_crop'])
        self.flow_hand_stride4.setChecked(flow['sam_hand_stride4'])
        self.flow_hamer.setChecked(flow['hamer'])
        from .exmr_options import saved as saved_exmr
        exmr=saved_exmr(project,manifest)
        self.exmr_camera.setCurrentIndex(self.exmr_camera.findData(exmr['static_camera']))
        self.exmr_smoothing.setChecked(exmr['smoothing']=='global')
        self.flow_sampling.setCurrentIndex(max(0,self.flow_sampling.findData(saved_interval(project,manifest))))
        self.model_choice.blockSignals(True)
        self.model_choice.setCurrentIndex(self.model_choice.findData(pipeline.get('model','gvhmr')))
        self.model_choice.blockSignals(False)
        opening=self._workspace_loading;self._workspace_loading=True
        try:self.select_model()
        finally:self._workspace_loading=opening
        has_refined = any((project / name).exists() for name in ("opt_results.pt", "refined_joints.npz", "refined_mesh.json"))
        has_motion = has_refined or any((project / name).exists() for name in ("hmr4d_results.pt", "initial_joints.npz"))
        result = pipeline.get("result", "refined" if has_refined else "initial")
        saved_selection=bool(pipeline.get('result'))
        self.handflow_refine.blockSignals(True)
        self.handflow_refine.setChecked(has_handflow(result) if has_motion or saved_selection else pipeline.get('model','gvhmr')=='gvhmr')
        self.handflow_refine.blockSignals(False)
        self.htd_refine.blockSignals(True)
        self.htd_refine.setChecked(result_options(result)[0] if has_motion or saved_selection else True)
        self.htd_refine.blockSignals(False)
        self.footmr_refine.blockSignals(True)
        # Existing projects keep their saved selection; new reconstructions default all on.
        self.footmr_refine.setChecked(result_options(result)[1] if has_motion or saved_selection else True)
        self.footmr_refine.blockSignals(False)
        self.motion_correction.blockSignals(True)
        self.motion_correction.setChecked(result.startswith('corrected_') if has_motion or saved_selection else True)
        self.motion_correction.blockSignals(False)
        self.restore_correction_options(manifest.get('correction',{}))
        self.active_result = self.selected_result()
        self.video_path = source_video(project, manifest)
        self.video_readable = False
        fps = manifest.get("fps", 30)
        self.info = {"fps": fps, "frames": manifest["frames"], "duration": manifest["frames"] / fps}
        self.source_offset, self.source_end = manifest["start"], manifest["end"]
        self.slider.set_media(fps)
        # MAMMA restores its selected camera below. Opening the actor's default
        # clip first discards that decoder and its cache on every row selection.
        if not (pipeline.get('model')=='mamma' and self.mamma_inputs and self.mamma_thumbnail_buttons):
            self.video_view.open(select_preview(project), (manifest["width"], manifest["height"]))
        self.slider.setRange(0, manifest["frames"] - 1)
        self.filename.setText(f"{self.video_path.name}  ·  Project loaded  ·  {manifest['frames']} frames")
        self.slider.reset_trim()
        self.clip_changed(self.slider.clip_in, self.slider.clip_out)
        self.refresh_results()
        self.seek(0)
        self.fps_label.setText(f"Source {fps:.2f} fps")
        if (project / "motion.bvh").exists():
            self.status.setText("Motion loaded. Play the animation or adjust the video clip range.")
        if (project / "keypoints.npz").exists() and not (project / "hmr4d_results.pt").exists():
            self.status.setText("2D observations loaded. Prepare the body model to reconstruct full-body motion.")
        self.sync_people(project,manifest)
        if workspace:
            trim=workspace.get('trim')
            if trim and 0<=trim[0]<trim[1]<=manifest['frames']:
                self.slider.clip_in,self.slider.clip_out=trim
                self.clip_changed(*trim)
            self.slider.setValue(max(0,min(int(workspace.get('frame',0)),manifest['frames']-1)))
            self.seek(self.slider.value())
        self.update_readiness()
        self.mamma_project_loaded(project,manifest)
        return True

    def seek(self, index):
        self.pear_seek(index)
        self.frame_number.blockSignals(True)
        self.frame_number.setValue(index + 1)
        self.frame_number.blockSignals(False)
        if self.timer.isActive() and not self.playback_tick:
            self.reset_play_clock()
        self.video_view.seek(index)
        fps = self.info["fps"] if self.info else 30
        total = self.info["duration"] if self.info else 0
        self.time_label.setText(f"{index/fps:05.2f} / {total:05.2f} s")

    def processing_range(self):
        """Map preview-local trim edges to the unchanged original source."""
        fps = self.info["fps"]
        first = self.source_offset + self.slider.clip_in / fps
        end = min(self.source_end, self.source_offset + self.slider.clip_out / fps)
        return first, end

    def clip_changed(self, first, end):
        if hasattr(self, "timer"):
            self.stop_playback()
        if self.info:
            start_seconds, end_seconds = self.processing_range()
            text = f"Clip: {first + 1}–{end}  |  {end_seconds-start_seconds:.2f} s"
            self.clip_label.setText(text)
            self.clip_label.setToolTip(tooltip_text(f"Video range: {start_seconds:.3f}–{end_seconds:.3f} s\nDrag the video track edges to trim playback and reconstruction. Existing motion files stay unchanged until you reconstruct again."))
        else:
            self.clip_label.setText("Clip: no video")

    def present_frame(self, index):
        if self._pear_active:
            self.pear_submit();return
        # 3D and overlays advance only when the corresponding decoded image is ready.
        self.motion_view.index = index
        self.motion_view.scene_time=self.source_offset+index/(self.info['fps'] if self.info else 30)
        self.motion_view.update()
        if self.timer.isActive():
            now = time.perf_counter()
            self.presented_times.append(now)
            while self.presented_times and now - self.presented_times[0] > 2:
                self.presented_times.popleft()
            if len(self.presented_times) > 2 and now - self.presented_times[0] >= 0.5:
                actual = (len(self.presented_times) - 1) / (now - self.presented_times[0])
                self.fps_label.setText(f"Source {self.info['fps']:.2f} | Play {actual:.1f} fps")

    def reset_play_clock(self, *_):
        self.play_anchor = self.slider.value()
        self.play_clock.start()
        self.presented_times.clear()
        if self.timer.isActive() and self.info:
            self.timer.start(max(1, math.ceil(1000 / (self.info['fps'] * self.play_speed.currentData()))))

    def stop_playback(self):
        self.timer.stop()
        self.timer_resolution.release()
        self.set_playback_indicator(False)
        self.pear_playback_stopped()

    def set_playback_indicator(self, playing):
        label = "Pause" if playing else "Play"
        self.play_button.setIcon(self.pause_icon if playing else self.play_icon)
        self.play_button.setToolTip(tooltip_text(label + " (Space)"))
        self.play_button.setAccessibleName(label)

    def step_frames(self, amount):
        if self._pear_active and self.pear_using_webcams():return
        self.stop_playback()
        self.slider.setValue(max(0, min(self.slider.maximum(), self.slider.value() + amount)))

    def toggle_play(self):
        if self._pear_active and self.pear_using_webcams():return
        if self.timer.isActive():
            self.stop_playback()
        elif self.video_path and self.info and self.video_readable:
            if not self.slider.clip_in <= self.slider.value() < self.slider.clip_out:
                self.slider.setValue(self.slider.clip_in)
            self.timer_resolution.acquire()
            self.reset_play_clock()
            self.timer.start(max(1, math.ceil(1000 / (self.info['fps'] * self.play_speed.currentData()))))
            self.set_playback_indicator(True)

    def next_frame(self):
        if not self.timer.isActive() or self.info is None or not self.video_readable:
            return
        elapsed = self.play_clock.nsecsElapsed() / 1e9
        rate = self.info["fps"] * self.play_speed.currentData()
        target = self.slider.clip_in + playback_frame(self.play_anchor - self.slider.clip_in,
                                elapsed, self.info["fps"], self.slider.clip_out - self.slider.clip_in,
                                self.play_speed.currentData())
        self.playback_tick = True
        try:
            self.slider.setValue(target)
        finally:
            self.playback_tick = False
        next_deadline = (math.floor(elapsed * rate + 1e-8) + 1) / rate
        remaining = next_deadline - self.play_clock.nsecsElapsed() / 1e9
        self.timer.start(max(1, math.ceil(remaining * 1000)))

    def shutdown_preview(self):
        if hasattr(self,'camera_tools'):self.camera_tools.close()
        self.shutdown_mamma()
        self.stop_playback()
        if hasattr(self,'character_loader'):
            self.character_loader.close();self.character_assets.clear()
        self.scene_loader.stop()
        self.scene_loader.wait()
        self.clear_people_scene()
        self.clear_preview()
        self.data_loader.stop()
        self.data_loader.wait()
        self.contact_writer.stop()
        self.contact_writer.wait()
        self.video_view.shutdown()
        self.motion_view.cleanup_gl()

    def changeEvent(self, event):
        if event.type() == QEvent.WindowStateChange and self.isMinimized() and hasattr(self, "timer"):
            self.stop_playback()
        super().changeEvent(event)

    def hideEvent(self, event):
        if hasattr(self, "timer"):
            self.stop_playback()
        super().hideEvent(event)

    def input_jobs(self, stages, result=None):
        if MOTION_TOOLS[self.model_choice.currentData()].input_setup_only:
            self.status.setText('MAMMA camera input setup is available; reconstruction is not integrated yet.')
            return
        if self.video_path is None:
            self.choose_video()
            return
        if self.info is None or not self.video_readable:
            self.status.setText("Wait for the video preview to finish opening.")
            return
        if 'people' not in stages and self._single_actor and not self._single_actor['checked']:
            self.status.setText('Check Person 1 in the People list to reconstruct it.')
            return
        if 'people' not in stages and self.input_person_jobs(stages,result):
            return
        unavailable = [model for stage in stages for model in ([] if self.model_choice.currentData()=='flowhmr' and stage=='flow_camera' and self.flow_sampling.currentData()==-1 else missing('flow_export' if self.model_choice.currentData()=='flowhmr' and stage in ('correct','export') else stage))]
        if self.model_choice.currentData()=='flowhmr' and self.flow_hamer.isChecked():
            from .flow_hamer import MODEL_KEYS
            from .models import BY_KEY
            unavailable += [BY_KEY[k] for k in MODEL_KEYS if not ready(BY_KEY[k])]
        if unavailable:
            names = ", ".join(dict.fromkeys(model.name for model in unavailable))
            self.status.setText(f"Prepare these models first: {names}")
            self.open_model_manager()
            return
        first, end = self.processing_range()
        prepare_args = ["--video", str(self.video_path), "--start", str(first), "--end", str(end)]
        if self.workspace_root is not None:
            prepare_args += ['--project',str(self.workspace_root)]
        elif self.project is not None:
            from .project_storage import storage_root
            if storage_root(self.project) is not None:
                prepare_args += ['--project', str(self.project)]
        if result or self.model_choice.currentData()!='gvhmr':
            prepare_args+=['--motion-tool',self.model_choice.currentData()]
        if result:
            prepare_args += ["--result", result]
            if result.startswith('corrected_') and not self.auto_floor.isChecked():
                prepare_args += ['--floor',str(self.floor_height.value())]
            if result.startswith('corrected_'):
                prepare_args += ['--correction-preset',self.correction_preset.currentData(),
                                 '--correction-mode',self.correction_mode.currentData(),
                                 '--foot-lock',str(int(self.foot_lock.isChecked())),'--hand-lock',str(int(self.hand_lock.isChecked())),
                                 '--pelvis-offset',str(self.pelvis_offset.value())]
        self.start_jobs([("prepare", prepare_args)]
                        + [(stage, ["--result", result] if stage in {"export", "flow_export", "footmr", "handflow", "correct"} and result else None) for stage in stages])

    def run_analysis(self):
        self.input_jobs(["track", "pva"])

    def run_full(self):
        tool = MOTION_TOOLS[self.model_choice.currentData()]
        if tool.key=='mamma':return self.run_mamma()
        if tool.key=='prompthmr':return self.run_prompthmr()
        if tool.key=='exmr' and self.people_root is not None:
            from .storage import atomic_json
            path=self.people_root/'project.json';state=json.loads(path.read_text(encoding='utf-8'))
            state['exmr_options']={'static_camera':self.exmr_camera.currentData(),'smoothing':'global' if self.exmr_smoothing.isChecked() else 'none'}
            atomic_json(path,state)
        if tool.key=='flowhmr' and self.people_root is not None:
            from .storage import atomic_json
            path=self.people_root/'project.json';state=json.loads(path.read_text(encoding='utf-8'))
            state['flow_options']={**state.get('flow_options',{}),**self.selected_flow_options()}
            atomic_json(path,state)
        refine = bool(tool.refinement) and self.htd_refine.isChecked()
        foot = tool.key=='gvhmr' and self.footmr_refine.isChecked()
        correction=self.motion_correction.isChecked()
        if correction:self.save_correction_options()
        hand=tool.key=='gvhmr' and self.handflow_refine.isChecked()
        self.input_jobs(tool.stages(refine, foot,correction,handflow=hand), result=result_kind(refine, foot,correction,handflow=hand))

    def save_correction_options(self):
        if self.project is not None:
            if self.people_root is not None:
                from .scene_floor import save_config
                save_config(self.people_root,self.selected_correction_options())
            from .storage import atomic_json
            path=self.project/'project.json';state=json.loads(path.read_text(encoding='utf-8'))
            state['correction']=self.selected_correction_options()
            atomic_json(path,state)

    def selected_correction_options(self):
        from .correction_settings import correction_options
        return correction_options({'auto_floor':self.auto_floor.isChecked(),'floor':self.floor_height.value(),
                                   'mode':self.correction_mode.currentData(),'foot_lock':self.foot_lock.isChecked(),'hand_lock':self.hand_lock.isChecked(),
                                   'preset':self.correction_preset.currentData(),'pelvis_offset':self.pelvis_offset.value()})

    def update_correction_description(self):
        descriptions = {
            'weak': 'Gentler correction, with some sliding allowed to preserve more of the original movement.',
            'normal': 'Balanced contact correction. Holds stable contacts while keeping transitions smooth.',
            'strong': 'Firmer holds, including less certain contacts. May resist intentional slides; review the result.',
        }
        self.preset_description.setText(descriptions[self.correction_preset.currentData()])
        if self.correction_mode.currentData()=='multi_level':
            self.floor_description.setText('Multi Level uses the inferred height of each foot contact. This floor value is unused; switch to Flat Ground to edit it.')
        else:
            self.floor_description.setText('Higher values raise the correction floor. Use the Correction Floor checkbox to preview it; Apply Correction updates the motion.')

    def restore_correction_options(self, options):
        from .correction_settings import correction_options
        options=correction_options(options)
        controls=(self.correction_mode,self.correction_preset,self.auto_floor,self.floor_height,self.pelvis_offset,self.foot_lock,self.hand_lock)
        for control in controls:control.blockSignals(True)
        try:
            self.correction_preset.setCurrentIndex(self.correction_preset.findData(options['preset']))
            self.correction_mode.setCurrentIndex(self.correction_mode.findData(options['mode']))
            self.foot_lock.setChecked(options['foot_lock']);self.hand_lock.setChecked(options['hand_lock'])
            self.auto_floor.setChecked(options['auto_floor']);self.floor_height.setValue(options['floor'])
            self.pelvis_offset.setValue(options['pelvis_offset'])
        finally:
            for control in controls:control.blockSignals(False)
        self.update_correction_description()

    def correction_settings_changed(self, *_):
        if self.sender() is self.floor_height and self.auto_floor.isChecked():return
        self.update_correction_description()
        self.save_correction_options()
        if self.sender() is self.correction_preset:
            self.request_people_scene(force=True)
            if self.preview_data is not None:self.apply_preview()
            self.update_readiness()
            return
        # Keep the last committed pose visible while editing the reference
        # plane. Only Apply Correction changes motion; stale export stays blocked.
        if self.preview_data is not None:
            self.correction_note.setText('Correction settings changed. Preview shows the last applied motion.\nApply Correction to update the pelvis, contacts and FBX.')
            self.correction_note.show()
        self.update_readiness()

    def apply_correction(self):
        if self.project is None or self.busy or self.contact_saving:return
        info=self.contact_tracks.get(base_result(self.active_result))
        if info and info.get('unsaved'):
            self.status.setText('Contact-lock timing was not saved. Edit or reset the tracks to retry saving.');return
        self.save_correction_options()
        kind=self.selected_result(True)
        self.set_active_result(kind)
        if self.people_root is not None:
            self.apply_scene_correction(kind);return
        self.start_jobs([('correct',['--result',kind]),('export',['--result',kind])], refresh_characters=True)

    def update_foot_contacts(self):
        base=base_result(self.active_result)
        info=self.contact_tracks.get(base)
        if info and self.info and (info['context']['frames']!=self.info['frames'] or info['context']['fps']!=self.info['fps']):info=None
        self.slider.set_contacts(info['ranges'] if info else None,info['automatic'] if info else None,
            editable=not self.busy and not self.contact_saving and self.motion_correction.isChecked(),
            identity=(str(self.project),base),token=id(info))
        from .foot_contacts import contact_levels
        self.motion_view.foot_lock_levels=lambda frame:contact_levels(info,frame)

    def foot_contacts_changed(self,ranges):
        self.save_foot_contacts(ranges)

    def reset_foot_contacts(self):
        self.save_foot_contacts(None)

    def save_foot_contacts(self,ranges):
        if self.project is None or self.busy or self.contact_saving:return
        base=base_result(self.active_result);info=self.contact_tracks.get(base)
        if info is None:return
        self.stop_playback();self.contact_saving=True
        try:self.contact_save_token=self.contact_writer.request(self.project,base,ranges,info['context'])
        except Exception as error:
            self.contact_saving=False;self.status.setText('Cannot save contact-lock timing: '+str(error));return
        # Keep immutable metadata while the writer commits off the GUI thread.
        self.contact_tracks[base]={**info,'ranges':ranges if ranges is not None else info['automatic'],'manual':ranges is not None,'unsaved':True}
        self.status.setText('Saving contact-lock timing...')
        self.update_readiness()

    def foot_contacts_saved(self,sequence,record):
        if self.project is None or str(self.project.resolve())!=record['project']:return
        info=self.contact_tracks.get(record['base'])
        if info is None:return
        self.contact_tracks[record['base']]={**info,'revision':record['state']['revision'],'unsaved':False}
        for key,scene in self.scene_cache.items():
            if base_result(key[1])==record['base']:
                take=scene.by_project.get(self.project)
                if take:
                    take.set_contacts(info['ranges'])
                    take.contact_revision=record['state']['revision']
        if sequence==self.contact_save_token:
            if self.preview_data is not None:self.apply_preview()
            self.status.setText('Contact-lock timing saved. Apply Correction to update the motion and FBX.')

    def foot_contacts_failed(self,sequence,message):
        if sequence==self.contact_save_token:self.status.setText('Cannot save contact-lock timing: '+message)

    def foot_contacts_settled(self,sequence):
        if sequence==self.contact_save_token:
            self.contact_saving=False;self.update_readiness()

    def prepare_mesh(self):
        if self.project is not None and not self.busy:
            self.prepare_target_preview()

    def resource_safety_changed(self,value):
        from .resource_safety import ENV,save,message
        try:save(value)
        except (OSError,ValueError) as error:
            self.resource_safety.blockSignals(True);self.resource_safety.setChecked(not value);self.resource_safety.blockSignals(False)
            self.log.append('Cannot save Resource Safety: '+str(error));return
        os.environ[ENV]='1' if value else '0'
        self.log.append(message(value))

    def start_jobs(self, jobs, refresh_characters=False):
        if self.busy:
            return
        jobs=list(jobs)
        from .resource_safety import message
        self._job_resource_safety=self.resource_safety.isChecked()
        self.log.append(message(self._job_resource_safety))
        self.character_refresh_queue=[]
        if refresh_characters:
            try:
                self.prepare_character_refresh(jobs)
            except (OSError, ValueError, KeyError) as exc:
                self.character_refresh_queue=[]
                self.status.setText(f'Cannot update linked characters: {exc}')
                self.update_readiness()
                return
        self._batch_started_at = time.monotonic()
        self._batch_has_motion = False
        self.task_state.reset(self._batch_started_at)
        self._task_manifests={}
        self.jobs, self.busy, self.cancelled, self.failed = list(jobs), True, False, False
        self._task_preparing=any(stage=='prepare' for stage,_ in self.jobs)
        self.update_readiness()
        self.next_job()

    def task_descriptor(self,stage,arguments):
        """Small metadata once per transition; no filesystem access on ticks."""
        arguments=arguments or []
        project=(Path(arguments[arguments.index('--project')+1]) if '--project' in arguments else
                 None if getattr(self,'_task_preparing',False) else self.project)
        tool=self.model_choice.currentData();state={};estimate=None
        if project is not None:
            project=Path(project)
            if project not in self._task_manifests:
                try:
                    file=project/'project.json'
                    state=json.loads(file.read_text(encoding='utf-8')) if file.stat().st_size<=1024**2 else {}
                except (OSError,ValueError):state={}
                if len(self._task_manifests)>=32:self._task_manifests.pop(next(iter(self._task_manifests)))
                self._task_manifests[project]=state
            state=self._task_manifests[project];tool=state.get('pipeline',{}).get('model',tool)
            estimate=state.get('stages',{}).get(stage,{}).get('elapsed')
            if stage.startswith('exmr'):
                try:
                    file=project/(stage+'_receipt.json')
                    if file.stat().st_size<=65536:estimate=json.loads(file.read_text(encoding='utf-8')).get('elapsed')
                except (OSError,ValueError):pass
        frames=state.get('frames',self.info.get('frames',0) if self.info else 0)
        if '--first-frame' in arguments and '--end-frame' in arguments:
            frames=int(arguments[arguments.index('--end-frame')+1])-int(arguments[arguments.index('--first-frame')+1])
        model=arguments[arguments.index('--model')+1] if '--model' in arguments else ''
        result=arguments[arguments.index('--result')+1] if '--result' in arguments else self.active_result
        key=(stage,tool,frames,model,result,self.compute_mode.currentData(),self.fast_sam.isChecked(),self.flow_batch4.isChecked(),self.flow_gpu_crop.isChecked(),self.flow_hand_stride4.isChecked(),self.flow_hamer.isChecked(),self.flow_sampling.currentData(),self.exmr_camera.currentData())
        if not isinstance(estimate,(int,float)) or not math.isfinite(estimate) or not 1<=estimate<=7*86400:estimate=None
        if stage in ('flow_batch','exmr_batch','gvhmr_batch'):
            from .queue_groups import argument
            items=json.loads(argument(arguments,'--batch-items'))
            key=(*key,argument(arguments,'--batch-phase'),tuple((item['project'],item['result']) for item in items),len(items))
            estimate=None
        estimate=self.task_state.history.get(key,estimate)
        return key,estimate

    def update_task_progress(self):
        if not self.busy or self.failed or self.cancelled or not self.task_state.active:return
        text,percent=self.task_state.status(time.monotonic(),self._queued_task_estimates)
        self.task_info.setText(text)
        # A real 0/N observation is loading, not a stationary empty bar.
        if percent is None:self.progress.setRange(0,0)
        else:
            total=self.task_state.total;current=self.task_state.current
            if total<=2147483647 and total.is_integer():
                self.progress.setRange(0,int(total));self.progress.setValue(int(current))
            else:
                self.progress.setRange(0,1000);self.progress.setValue(round(1000*current/total))

    def cleanup_batch_request(self):
        from .queue_groups import remove_request
        path=getattr(self,'_batch_request',None);self._batch_request=None
        try:remove_request(path)
        except OSError as error:self.log.append('Could not remove worker request: '+str(error))

    def next_job(self):
        self.cleanup_batch_request()
        self.guard_timer.stop()
        self.processing_guard = None
        if not self.jobs or self.failed or self.cancelled:
            self.task_timer.stop()
            elapsed = (max(0., time.monotonic() - self._batch_started_at)
                       if self._batch_started_at is not None else None)
            self._batch_started_at = None
            self.busy = False
            self.finish_single_tracking()
            self.task_state.active=False
            from .task_progress import duration
            self.task_info.setText(('Stopped' if self.failed or self.cancelled else 'Done')+
                                   (f' · Elapsed {duration(elapsed)}' if elapsed is not None else ''))
            self.progress.setRange(0, 100)
            self.progress.setValue(0 if self.failed or self.cancelled else 100)
            if self.current_stage in ('save_fbx','copy_bvh','save_blend','save_blend_scene'):
                # Saving a derived file does not replace scene/preview data.
                # Reloading the people scene here moves a following camera.
                if not self.failed and not self.cancelled:
                    paths=getattr(self,'_motion_export_targets',[])
                    self.status.setText('Motion export complete.'+(' '+paths[0] if len(paths)==1 else f' {len(paths)} files.' if paths else ''))
                    for path in paths:self.log.append('Exported: '+path)
                self.update_readiness()
                return
            if self.current_stage=='save_project':
                self.finish_project_save()
                return
            person_batch=self._people_batch
            completion_message=self.status.text()
            self.finish_people_jobs()
            if self.failed or self.cancelled:self.status.setText(completion_message)
            if not self.failed and not self.cancelled:
                if self.current_stage in ("export",'flow_export') and self.project is not None and not person_batch:
                    metadata = json.loads((self.project / "export.json").read_text(encoding="utf-8"))
                    self.set_active_result(metadata.get("result_kind", "refined"))
                if self.current_stage == "diagnose" and self.hardware_summary:
                    self.status.setText(self.hardware_summary)
                elif self.current_stage == "pva":
                    self.status.setText("2D analysis complete. View the estimated joints over the video.")
                elif self.current_stage == "download":
                    self.status.setText("Downloads complete. Check Model Manager for any required imports.")
                elif self.current_stage == "import_model":
                    self.status.setText("Model setup complete.")
                elif self.current_stage not in ('people','person'):
                    self.status.setText("Complete. Play the result or open the project folder.")
                if self._batch_has_motion and elapsed is not None:
                    seconds = round(elapsed)
                    minutes, seconds = divmod(seconds, 60)
                    hours, minutes = divmod(minutes, 60)
                    duration = (f'{hours}h {minutes:02d}m {seconds:02d}s' if hours else
                                f'{minutes}m {seconds:02d}s' if minutes else f'{elapsed:.1f} s')
                    self.status.setText(f'Complete. Total time: {duration}. Play the result or open the project folder.')
                    self.log.append(f'Total motion processing time: {elapsed:.2f} s')
            self.refresh_results()
            self.update_readiness()
            if self.failed or self.cancelled:
                self.character_refresh_queue=[]
            else:
                self.next_character_refresh()
            return
        from .queue_groups import join_single_pipeline
        self.jobs = join_single_pipeline(self.jobs, self.project)
        stage, arguments = self.jobs.pop(0)
        if stage in {'gvhmr', 'refine', 'footmr', 'handflow', 'correct',
                     'flow_camera', 'flow_features', 'flowhmr',
                     'exmr_preprocess', 'exmr_features', 'exmr'}:
            self._batch_has_motion = True
        if stage.startswith(('mamma_','prompthmr_')) or stage in ('flow_batch','exmr_batch','gvhmr_batch'):self._batch_has_motion=True
        self.current_stage = stage
        self._batch_inner_stage=stage
        if arguments is None:
            arguments = ["--project", str(self.project)]
        elif stage in {"export", "footmr", "handflow", "correct", 'flow_export'} and "--project" not in arguments:
            arguments = ["--project", str(self.project), *arguments]
        self.job_project = Path(arguments[arguments.index('--project')+1]) if '--project' in arguments else None
        if stage=='track':self.begin_single_tracking(self.job_project)
        if stage in {"download", "import_model"} and "--model" in arguments:
            from .models import BY_KEY
            model_key = arguments[arguments.index("--model") + 1]
            self.status.setText(("Downloading " if stage == "download" else "Importing ") + BY_KEY[model_key].name + "...")
        else:
            self.status.setText(STAGES[stage] + "...")
        key,estimate=self.task_descriptor(stage,arguments)
        self.task_state.begin(time.monotonic(),key,self.status.text(),estimate)
        actor=next((part for part in self.job_project.parts if part.startswith('person_')),'') if self.job_project else ''
        self._task_actor=actor.replace('_',' ').title().replace('000','')
        self._queued_task_estimates=[self.task_descriptor(s,a) for s,a in self.jobs]
        self.task_timer.start();self.update_task_progress()
        self.progress.setRange(0, 0)
        self.buffer = ""
        self._stdout_decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._stderr_decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._stderr_buffer = ""
        if stage in ('flow_batch','exmr_batch','gvhmr_batch'):
            from .queue_groups import write_request
            try:
                self._batch_request=write_request(stage,arguments)
            except (OSError,ValueError) as error:
                self.failed=True;self.log.append(str(error));self.status.setText('Cannot prepare people processing: '+str(error))
                self.next_job();return
            arguments=['--request',str(self._batch_request),'--project',str(self.job_project)]
        command = worker_command(stage, *arguments)
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONIOENCODING", "utf-8")
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("RGB_MOCAP_COMPUTE_MODE", self.compute_mode.currentData())
        environment.insert('RGB_MOCAP_RESOURCE_SAFETY','1' if self._job_resource_safety else '0')
        environment.insert('RGB_MOCAP_FLOW_INTERVAL',str(self.flow_sampling.currentData()))
        environment.insert('RGB_MOCAP_FLOW_FAST_SAM','1' if self.fast_sam.isChecked() else '0')
        environment.insert('RGB_MOCAP_FLOW_BATCH4','1' if self.flow_batch4.isChecked() else '0')
        environment.insert('RGB_MOCAP_FLOW_GPU_CROP','1' if self.flow_gpu_crop.isChecked() else '0')
        environment.insert('RGB_MOCAP_FLOW_HAND_STRIDE4','1' if self.flow_hand_stride4.isChecked() else '0')
        environment.insert('RGB_MOCAP_FLOW_HAMER','1' if self.flow_hamer.isChecked() else '0')
        environment.insert('RGB_MOCAP_EXMR_OPTIONS',json.dumps({'static_camera':self.exmr_camera.currentData(),'smoothing':'global' if self.exmr_smoothing.isChecked() else 'none'}))
        self.process.setProcessEnvironment(environment)
        self.process.readyReadStandardOutput.connect(self.read_stdout)
        self.process.readyReadStandardError.connect(self.read_stderr)
        self.process.finished.connect(self.process_finished)
        self.process.errorOccurred.connect(self.process_error)
        self.process.started.connect(self.start_processing_guard)
        self.process.start(command[0], command[1:])

    def start_processing_guard(self):
        from .processing_guard import ProcessingGuard
        try:
            self.processing_guard = ProcessingGuard(self.process.processId(),resource_safety=self._job_resource_safety)
            self.guard_timer.start()
        except Exception as error:
            self.log.append('Resource monitor unavailable: ' + str(error))

    def check_processing_resources(self):
        if not self.processing_guard or not self.process:return
        message = self.processing_guard.check()
        if message:
            self.failed = True
            self.cancel_job()
            self.status.setText(message)
            self.log.append(message)
            self.guard_timer.stop()
            if getattr(self,'job_project',None) is not None:
                from .storage import atomic_json
                path=self.job_project/'project.json'
                try:
                    state=json.loads(path.read_text(encoding='utf-8'))
                    state.setdefault('stages',{})[self._batch_inner_stage]={'status':'failed','error':message}
                    atomic_json(path,state)
                except (OSError,ValueError) as error:self.log.append('Could not save stopped stage: '+str(error))

    def read_stdout(self, final=False):
        if self.processing_guard:self.processing_guard.output()
        self.buffer += self._stdout_decoder.decode(bytes(self.process.readAllStandardOutput()), final=final)
        if final and self.buffer and not self.buffer.endswith("\n"):self.buffer += "\n"
        if len(self.buffer) > 65536 and "\n" not in self.buffer:
            self.buffer = self.buffer[-65536:]
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.startswith("@@MOCAP@@"):
                try:
                    self.handle_event(json.loads(line[len("@@MOCAP@@"):]))
                except (ValueError, KeyError) as error:
                    self.log.append(str(error))
            elif line.strip():
                self.log.append(line.rstrip())

    def read_stderr(self, final=False):
        if self.processing_guard:self.processing_guard.output()
        text = self._stderr_decoder.decode(bytes(self.process.readAllStandardError()), final=final)
        self._stderr_buffer = (self._stderr_buffer + text.replace("\r", "\n"))[-65536:]
        lines = self._stderr_buffer.split("\n");self._stderr_buffer = lines.pop()
        if final and self._stderr_buffer:lines.append(self._stderr_buffer);self._stderr_buffer = ""
        for line in lines:
            if line.strip():self.log.append(line[-8192:])

    def handle_event(self, event):
        kind = event["type"]
        self.single_tracking_event(event)
        if self.people_event(event):
            self._queued_task_estimates=[self.task_descriptor(s,a) for s,a in self.jobs]
            self.update_task_progress();return
        if kind=='batch_item_start':
            self.job_project=Path(event['project']);self._batch_inner_stage=event['stage']
            self._task_actor=f"Actor/take {event['index']}/{event['count']}"
            self.task_state.update(dict(current=0,total=0,phase=f"batch:{event['phase']}:{event['index']}",message='Loading actor data'),time.monotonic())
            self.status.setText(self._task_actor+' · '+STAGES.get(event['stage'],event['stage']))
            self.update_task_progress()
        elif kind in ('mamma_ready','prompthmr_ready'):
            self.workspace_root=Path(event['root'])
            self.load_project(Path(event['project']))
            self.set_active_result(event['result'])
        elif kind == "prepared":
            self._task_preparing=False
            self._task_manifests={}
            self.project = Path(event["project"])
            from .project_storage import storage_root
            saved_root=storage_root(self.project)
            from .project_workspace import is_workspace
            if saved_root is not None and not is_workspace(saved_root):
                from .storage import atomic_json
                atomic_json(saved_root/'workspace_state.json',dict(version=1,name=saved_root.name,
                    active=self.project.resolve().relative_to(saved_root).as_posix(),frame=0,trim=None))
            self.load_project(self.project)
        elif kind == 'project_saved':
            self._project_saved_root=Path(event['root'])
        elif kind == 'save_cancelled':
            self.cancelled=True
            self.status.setText(event['message'])
        elif kind in ('resource_mode', 'info'):
            self.log.append(event['message'])
        elif kind == "progress":
            message=event.get('message','Processing...')
            self.log.status_terminal(message)
            self.status.setText((self._task_actor+' · ' if getattr(self,'_task_actor','') else '')+message)
            self.task_state.update(event,time.monotonic())
            self.update_task_progress()
        elif kind == "diagnostics":
            data = event["result"]
            self.log.append(json.dumps(data, ensure_ascii=False, indent=2))
            self.hardware_summary = f"{data.get('gpu', 'GPU unavailable')} · RAM {data.get('ram_gib', '?')}GB · CUDA check {'Passed' if data.get('cuda_matmul_verified') else 'Unavailable'}"
            self.status.setText(self.hardware_summary)
        elif kind in {"failed", "blocked"}:
            self.failed = True
            self.task_timer.stop();self.task_info.setText('Failed')
            self.status.setText("Processing failed. " + event.get("message", "See the console for details."))
            self.log.append(self.status.text())
        elif kind == "complete":
            if not event.get('batch_inner'):self.task_state.cached|=bool(event.get('cached',False))
            self.log.append(f"{STAGES.get(event['stage'])}: {event.get('elapsed', 0):.2f} s · {event.get('output')}")
        self.log.ensureCursorVisible()

    def process_finished(self, code, _):
        self.read_stdout(final=True)
        self.read_stderr(final=True)
        if code != 0 and not self.cancelled:
            self.failed = True
            if "failed" not in self.status.text():
                self.status.setText("Processing stopped. See the console for details.")
        self.process.deleteLater()
        self.process = None
        self.task_state.finish(time.monotonic(),not self.failed and not self.cancelled and code==0)
        self.cleanup_fbx_export()
        self.next_job()

    def process_error(self, error):
        if error == QProcess.FailedToStart:
            self.character_refresh_queue=[]
            self.task_timer.stop();self.task_state.active=False;self.task_info.setText('Failed to start')
            self.guard_timer.stop()
            self.processing_guard = None
            if self.process is not None:
                self.process.deleteLater()
                self.process = None
            self.jobs.clear()
            self.cleanup_fbx_export()
            self.cleanup_batch_request()
            self.cleanup_project_save()
            self.failed = True
            self.busy = False
            self.finish_single_tracking()
            self.finish_people_jobs()
            self.status.setText("Cannot start the processing engine. Check the app installation.")
            self.update_readiness()

    def cancel_job(self):
        self.pear_leave()
        self.pear_stop_scan()
        if self.character_process is not None:
            self.cancel_character();return
        if self.current_stage=='save_project' and getattr(self,'_project_save_request',None):
            self._project_save_request.with_suffix('.cancel').touch()
            self.task_info.setText('Stopping...')
            self.status.setText('Cancelling project save...')
            self.cancel_button.setEnabled(False)
            return
        from shiboken6 import isValid
        pending=getattr(self,'cancel_process',None)
        if pending is not None and isValid(pending) and pending.state()!=QProcess.NotRunning:
            return
        self.task_timer.stop();self.task_state.active=False;self.task_info.setText('Stopping...')
        self.guard_timer.stop()
        self.cancelled = True
        self.jobs.clear()
        worker = self.process
        if worker and isValid(worker) and worker.processId():
            if os.name == "nt":
                # Stop FFmpeg/other descendants before losing their parent PID.
                helper = QProcess(self)
                self.cancel_process = helper
                def fallback(error):
                    if error == QProcess.FailedToStart and self.process is worker and isValid(worker):
                        worker.kill()
                    if error == QProcess.FailedToStart:
                        if self.cancel_process is helper:self.cancel_process=None
                        if isValid(helper):helper.deleteLater()
                helper.errorOccurred.connect(fallback)
                def cancelled(code, _):
                    if code and self.process is worker and isValid(worker):
                        worker.kill()
                    if self.cancel_process is helper:self.cancel_process=None
                    if isValid(helper):helper.deleteLater()
                helper.finished.connect(cancelled)
                helper.start("taskkill.exe", ["/PID", str(worker.processId()), "/T", "/F"])
            else:
                worker.kill()
        self.status.setText("Cancelled. Completed stages will be reused next time.")

    def refresh_results(self):
        if self.project:
            signature = project_signature(self.project)
            if signature != self.preview_signature:
                self.preview_signature = signature
                self.data_loader.request(self.project)
            elif self.preview_data is not None:
                self.apply_preview()

    def clear_preview(self):
        self.preview_help.setText('')
        self.pelvis_auto_note.setText('Automatic lowering varies by frame.')
        self.data_loader.cancel()
        self.preview_signature = None
        self.slider.has_motion = False
        self.slider.update()
        self.video_view.set_observations({})
        self.motion_view.set_motion(None)
        self.refine_preview_note.hide()
        self.correction_note.hide()
        self.contact_tracks.clear()
        self.slider.set_contacts(None,identity=None,token=None)
        if self.preview_data is not None:
            self.preview_data.close()
            self.preview_data = None

    @batch_readiness
    def preview_ready(self, generation, result):
        if generation != self.data_loader.generation:
            result.close()
            return
        if result.signature != project_signature(result.project):
            # The worker committed another generation during loading. Never
            # pair these arrays with the new on-disk receipts.
            result.close()
            self.preview_signature=None
            self.refresh_results()
            return
        self.video_view.set_observations({})
        previous=self.preview_data
        preserve_view=previous is not None and previous.project==result.project
        auto_fit=self.motion_view.auto_fit
        if preserve_view:self.motion_view.auto_fit=False
        self.preview_data = result
        self.contact_tracks=dict(getattr(result,'contact_tracks',{}))
        try:self.apply_preview()
        finally:
            self.motion_view.auto_fit=auto_fit
            self.restore_workspace_camera(consume=True)
            if previous is not None:previous.close()

    @batch_readiness
    @read_only_checks
    def apply_preview(self):
        data = self.preview_data
        signature=project_signature(data.project)
        if data.signature != signature:
            # Hide stale payloads without forgetting this project's camera
            # anchor while the next committed correction is loading.
            self.motion_view.set_motion(None,path=data.project/(self.active_result+'_joints.npz'))
            self.slider.has_motion=False;self.slider.update()
            if not self.busy and signature!=self.preview_signature:self.refresh_results()
            return
        self.video_view.set_observations(data.arrays)
        display_kind=self.active_result
        name = display_kind + "_joints"
        corrected=self.active_result.startswith('corrected_')
        from .result_artifacts import artifacts_current
        coherent=artifacts_current(self.project,display_kind)
        current=coherent and result_is_current(self.project,display_kind)
        pending=coherent and corrected and not current and result_is_current(self.project,display_kind,ignore_contacts=True)
        available = name in data.arrays and (current or pending)
        self.correction_note.hide()
        self.refine_preview_note.hide()
        if not available:
            result = {"initial":MOTION_TOOLS[self.model_choice.currentData()].label, "refined":"HTD", "footmr_initial":"FootMR", "footmr_refined":"HTD + FootMR"}[body_result(self.active_result)]
            if has_handflow(self.active_result):result += " + HandFlow"
            note=self.correction_note if corrected else self.refine_preview_note
            missing_hand=has_handflow(self.active_result) and not result_is_current(self.project,base_result(self.active_result))
            note.setText('No current HandFlow preview. Run Reconstruct 3D to infer finger motion.' if missing_hand else
                'Correction settings changed or result unavailable.\nUse Apply Correction to update the preview.' if corrected else f"No {result} preview yet. Run Reconstruct 3D to create it.")
            note.show()
        warning=' '.join(filter(None,(data.summaries.get(name,{}).get('correction_warning',''),data.summaries.get(name,{}).get('handflow_warning',''))))
        pelvis_summary=data.summaries.get(name,{})
        auto_drop=pelvis_summary.get('pelvis_auto_drop') if corrected else None
        self.pelvis_auto_note.setText('Automatic lowering varies by frame.' if auto_drop is None else
            f'Last applied: automatic lowering up to {auto_drop*100:.1f} cm; '
            f'automatic crouch lift up to {pelvis_summary.get("pelvis_auto_raise",0.)*100:.1f} cm; '
            f'user lift up to {pelvis_summary.get("pelvis_applied_lift",0.)*100:.1f} cm.')
        if self.people_root is not None:
            self.sync_shared_floor_controls()
        elif available and self.auto_floor.isChecked():
            self.floor_height.blockSignals(True)
            self.floor_height.setValue(float(data.summaries[name]['floor']))
            self.floor_height.blockSignals(False)
        if available and warning:
            self.correction_note.setText(warning);self.correction_note.show()
        if available and pending:
            self.correction_note.setText('Contact-lock timing changed. Preview shows the last applied correction.\nApply Correction to update the motion and FBX.');self.correction_note.show()
        self.slider.has_motion = available
        self.slider.update()
        overlay_name=self.active_result+'_joints'
        overlay_current=artifacts_current(self.project,self.active_result) and result_is_current(self.project,self.active_result,ignore_contacts=True)
        overlay_available=overlay_current and overlay_name in data.arrays
        from .bone_overlay import preview_current
        target_current=overlay_available and preview_current(self.project,self.active_result,data.target_info.get(self.active_result))
        camera_name=name if available else overlay_name
        rig_points=data.arrays.get(camera_name)
        # Source pose is the comparison reference. Never deform this surface
        # to make an imperfect export skeleton appear to fit it exactly.
        selected_mesh=None
        # Preserve a selected legacy baked preview if the export skin has not
        # been prepared yet. Never pair it with a different result's bones.
        self.motion_view.set_motion((rig_points if rig_points is not None else data.arrays.get(camera_name)) if available or overlay_available else None,
                                   data.summaries.get(camera_name),data.project/(camera_name+'.npz'),
                                   rotations=data.arrays.get(camera_name.replace('_joints','_rotations')) if available or overlay_available else None,
                                   mesh_frames=(selected_mesh if selected_mesh is not None else data.arrays.get(name.replace('_joints','_mesh'))) if available else None,mesh_asset=data.mesh_asset)
        self.motion_view.target_frames=data.arrays.get(self.active_result+'_rig') if target_current else None
        self.motion_view.target_asset=data.target_assets.get(self.active_result) if target_current else None
        self.motion_view.mesh_binding=self.motion_view.target_asset.get('binding') if selected_mesh is not None and self.motion_view.target_asset else None
        self.motion_view.raw_asset=None
        self.motion_view.native_vertices=None
        self.motion_view.native_required=False
        if self.model_choice.currentData()=='flowhmr':
            self.motion_view.raw_asset=self.motion_view.target_asset
            self.motion_view.native_vertices=data.arrays.get(self.active_result+'_native') if target_current else None
            self.motion_view.native_required=True
        info=data.target_info.get(self.active_result,{})
        warning=info.get('correction',{}).get('warning','')
        source_warning=data.summaries.get(name,{}).get('correction_warning','') if corrected else ''
        if source_warning and source_warning not in warning:warning=' '.join(filter(None,(warning,source_warning)))
        help_text='The skeleton is fitted to the selected motion mesh. Small differences may remain.'
        if info.get('fit_version',0)>=15:
            help_text='The skeleton follows the selected motion mesh, with steadier arm and leg directions. Errors in the AI motion may remain.'
        if corrected:
            short=int(pelvis_summary.get('short_contact_segments',0))
            if short:help_text+=f'\n{short} short foot-lock intervals never reach full strength because of boundary blending.'
        notices=[]
        if not available and corrected:notices.append('Selected motion unavailable. Use Apply Correction to update the preview.')
        if warning:notices.append(warning)
        selected_fit=info.get('selected_source_fit',{})
        if selected_fit:
            help_text+=f'\nSkeleton fit: mean {selected_fit["mean_m"]*100:.1f} cm; max {selected_fit["max_m"]*100:.1f} cm.'
            facing=selected_fit.get('anatomical_facing_deg',{})
            for label,joints in (('Knee',('LeftLeg','RightLeg')),('Elbow',('LeftForeArm','RightForeArm'))):
                values=[facing[j] for j in joints if j in facing]
                if values:help_text+=f'\n{label} anatomical axis vs AI source: mean {sum(v["mean"] for v in values)/len(values):.1f}°; max {max(v["max"] for v in values):.1f}°.'
        fitted=info.get('selected_source_fit',info.get('source_fit',{})).get('joints',{})
        elbow_offset=max((fitted.get(side+'ForeArm',fitted.get('mixamorig:'+side+'ForeArm',{})).get('max_m',0.) for side in ('Left','Right')),default=0.)
        if elbow_offset>.08:
            notices.append(f'Elbow pose differs from AI source (up to {elbow_offset*100:.1f} cm) after anatomical alignment.')
        if self.model_choice.currentData()=='flowhmr':
            from .flow_placement import needs_update
            if needs_update(self.project):notices.append('FlowHMR scene placement needs updating. Use Prepare Preview; no AI inference is required.')
            knee_offset=max((fitted.get(side+'Leg',{}).get('max_m',0.) for side in ('Left','Right')),default=0.)
            if knee_offset>.08:notices.append(f'Knee pose differs from AI source (up to {knee_offset*100:.1f} cm) after anatomical alignment.')
        self.bone_overlay.setToolTip(tooltip_text('Show the skeleton used for FBX export. Display only.\n'
            'In Blender, use Scale 1 and leave Automatic Bone Orientation off to keep its axes.\n'+help_text))
        if available and pending:notices.append('Contact-lock timing changed. Preview shows the last applied correction. Apply Correction to update the motion and FBX.')
        if corrected and available and not pending and not result_is_current(self.project,self.active_result):notices.append('The last applied correction is outdated or unavailable. Apply Correction before export.')
        self.preview_help.setText(help_text)
        self.correction_note.setText('\n'.join(dict.fromkeys(notices)));self.correction_note.setVisible(bool(notices))
        if available and self.motion_view.target_asset is None and not self.busy and os.environ.get('QT_QPA_PLATFORM')!='offscreen' and (self.project/RESULTS[base_result(self.active_result)][0]).is_file():
            token=(str(self.project),self.active_result,self.preview_signature)
            attempts=getattr(self,'_rig_attempts',deque(maxlen=16))
            if token not in attempts:
                attempts.append(token);self._rig_attempts=attempts
                QTimer.singleShot(0,self.prepare_target_preview)
        self.update_foot_contacts()
        self.update_readiness()

    def preview_failed(self, generation, message):
        if generation == self.data_loader.generation:
            changed=self.project is not None and self.preview_signature!=project_signature(self.project)
            self.preview_signature = None
            if changed:
                self.refresh_results()
                return
            self.status.setText("Cannot load the motion preview. " + message)

    def save_fbx(self):
        if self.selected_character() is not None:
            self.export_character(self.selected_character());return
        if self.busy or self.current_export() is None:
            self.status.setText('Selected motion is unavailable or has pending correction edits. Apply Correction / Prepare Preview before exporting.');return
        from .source_export_dialog import SourceExportDialog
        dialog=SourceExportDialog(self)
        if dialog.exec()!=QDialog.Accepted:return
        if self.current_export() is None:
            self.status.setText('Motion changed while choosing export options. Apply Correction / Prepare Preview first.');return
        self._source_export_options=dialog.options()
        if self._source_export_options['format']=='blend_scene':
            from .blend_export_ui import export_scene
            export_scene(self);return
        if self.export_checked_people():return
        source = self.current_export()
        if source:
            from .person_names import export_stem
            fmt=self._source_export_options['format']
            default=export_stem(self.export_person_name(self.project))+'.'+fmt
            name, _ = QFileDialog.getSaveFileName(self, "Export", default, fmt.upper()+" (*."+fmt+")",options=QFileDialog.DontConfirmOverwrite if fmt=='blend' else QFileDialog.Option(0))
            if name:
                if not Path(name).suffix:name+='.'+fmt
                self.submit_fbx_exports([(self.project,self.active_result,source,Path(name))])

    def submit_fbx_exports(self,files):
        if self.busy or not files or self.current_export() is None:return
        import uuid
        from .storage import atomic_json
        request=self.project/('.fbx-export-'+uuid.uuid4().hex+'.json')
        rows=[]
        if getattr(self,'_source_export_options',{}).get('format')=='blend':
            from .blend_export_ui import targets
            files=targets(files)
        try:
            for project,kind,source,target in files:
                row=dict(project=str(project),kind=kind,source=str(source),target=str(target))
                row['rig_name']=self.export_person_name(project)
                row['prepend_tpose']=getattr(self,'_source_export_options',{}).get('prepend_tpose',False)
                row['bvh_profile']=getattr(self,'_source_export_options',{}).get('bvh_profile','generic')
                for key,default in (('animate_camera',True),('background_refine',False),('camera_translation_scale',1.)):
                    row[key]=getattr(self,'_source_export_options',{}).get(key,default)
                if self.people_root is not None:
                    from .scene_export import context
                    row['scene']=context(self.people_root,project)
                rows.append(row)
        except (OSError,ValueError,KeyError) as error:
            self.status.setText('Cannot export shared scene: '+str(error));return
        atomic_json(request,rows)
        self._motion_export_targets=[str(target) for _,_,_,target in files]
        self._fbx_export_cleanup=(request,[target.with_name('.'+target.name+'.rgbmocap-'+request.stem+'.partial') for _,_,_,target in files])
        stage='copy_bvh' if getattr(self,'_source_export_options',{}).get('format','fbx')=='bvh' else 'save_fbx'
        if getattr(self,'_source_export_options',{}).get('format')=='blend':
            from .blend_support import partial
            stage='save_blend';self._fbx_export_cleanup=(request,[partial(target.parent,request) for _,_,_,target in files])
        self.start_jobs([(stage,['--project',str(self.project),'--output',str(request)])])

    def cleanup_fbx_export(self):
        pending=getattr(self,'_fbx_export_cleanup',None)
        if pending:
            request,paths=pending
            for path in [request,*paths]:
                try:
                    if path.is_dir():
                        from .blend_support import cleanup
                        cleanup(path,request)
                    else:path.unlink(missing_ok=True)
                except (OSError,ValueError):pass
            self._fbx_export_cleanup=None

    def open_folder(self):
        from .project_storage import storage_root
        target = self.workspace_root or ((storage_root(self.project) or self.project) if self.project else PROJECTS)
        target.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and not self.busy:
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls and not self.busy:
            if self._pear_active:self.load_pear_videos([Path(url.toLocalFile()) for url in urls])
            elif self.model_choice.currentData()=='mamma':self.load_mamma_videos([Path(url.toLocalFile()) for url in urls])
            else:self.load_video(Path(urls[0].toLocalFile()))

    def closeEvent(self, event):
        if getattr(self, '_preview_export', None) is not None:
            self._preview_export.cancel(close_window=True)
            event.ignore()
            return
        if self.busy and self.process:
            answer = QMessageBox.question(self, "Processing", "Cancel the current job and close the application?")
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.cancel_job()
            worker = self.process
            if worker:
                worker.waitForFinished(3000)
        self.pear_leave()
        self.pear_stop_scan()
        if self.character_process is not None:
            self.cancel_character();self.character_process.waitForFinished(5000)
        self.character_loader.close()
        self.task_timer.stop()
        if self.workspace_root is not None:
            try:self.remember_workspace_model()
            except (OSError,ValueError) as error:self.log.append('Could not save project view: '+str(error))
        self.shutdown_preview()
        event.accept()
