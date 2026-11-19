"""Transient live tool; offline projects and committed motion remain untouched."""
import time,subprocess,re,threading,json
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt,QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QWidget,QHBoxLayout,QVBoxLayout,QComboBox,QPushButton,QLabel,QCheckBox,QDoubleSpinBox,QMessageBox,QGroupBox,QSizePolicy

class CameraScan:
    def __init__(self):
        self.devices=None;self.cancelled=threading.Event();self.process=None
    def start(self):threading.Thread(target=self.run,daemon=True).start()
    def cancel(self):
        self.cancelled.set()
        if self.process and self.process.poll() is None:
            try:self.process.terminate()
            except OSError:pass
    def run(self):
        devices=[]
        try:
            from PySide6.QtMultimedia import QMediaDevices
            devices=[dict(name=d.description(),id=bytes(d.id()).hex(),mf_index=i) for i,d in enumerate(QMediaDevices.videoInputs())]
        except (ImportError,RuntimeError):pass
        try:
            from .paths import ROOT
            if self.cancelled.is_set():return
            self.process=subprocess.Popen([str(ROOT/'tools/bin/ffmpeg.exe'),'-hide_banner','-list_devices','true','-f','dshow','-i','dummy'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if self.cancelled.is_set():self.cancel()
            _,error=self.process.communicate(timeout=8)
            names=re.findall(r'"([^"\r\n]+)" \(video\)',error.decode('utf8',errors='replace'))
            for index,name in enumerate(names):
                matches=[d for d in devices if d['name']==name and 'mf_index' in d]
                if len(matches)==1 and names.count(name)==1:matches[0]['ds_index']=index
                elif not matches:devices.append(dict(name=name,id='dshow:'+name+':'+str(index),ds_index=index))
        except (OSError,subprocess.TimeoutExpired):
            if self.process:
                self.cancel();self.process.communicate()
        finally:self.devices=devices

from .pear_multi_ui import PearMultiWorkspace

class PearWorkspace(PearMultiWorkspace):
    def init_pear(self):
        self.init_pear_multi()
        self._pear_active=False;self._pear_failed=False;self.pear_session=None;self.pear_generation=0;self.pear_request=0;self._pear_last=None
        self._pear_scan_cache_at=0;self._pear_probe_at=0;self._pear_guard_at=0;self._pear_camera_state=None;self._pear_camera_scan=None;self._pear_closing=[]
        self._pear_presented_request=-1
        self._pear_ready=False
        self._pear_running=False
        self._pear_restart=0
        self._pear_source_signature=None
        self._pear_bone_reset=0;self._pear_reset_pose=None;self._pear_reset_pending=False
        self._pear_raw_pose=None;self._pear_world_anchor=None;self._pear_position_revision=0
        self._pear_start_align=False

    def build_pear_header(self,header):
        # Camera controls are built together below the video, never in its header.
        self.pear_scan_timer=QTimer(self);self.pear_scan_timer.setInterval(100);self.pear_scan_timer.timeout.connect(self.pear_finish_scan)

    def build_pear_settings(self):
        self.pear_page=QWidget();self.pear_page.setProperty("motion_tool","pear");outer=QVBoxLayout(self.pear_page);outer.setContentsMargins(0,0,0,0);outer.setAlignment(Qt.AlignTop)
        self.pear_page.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum)
        group=QGroupBox("PEAR Settings");outer.addWidget(group);layout=QVBoxLayout(group);layout.setSpacing(8);layout.setAlignment(Qt.AlignTop)
        self.pear_world_position=QCheckBox('World Position');self.pear_world_position.setChecked(True)
        self.pear_world_position.setToolTip('Move the pelvis using estimated position. Off fixes its current position, including height; rotation is unchanged.')
        layout.addWidget(self.pear_world_position);self.pear_world_position.toggled.connect(self.pear_position_changed)
        self.pear_foot_lock=QCheckBox('Foot Lock');self.pear_foot_lock.setChecked(True)
        self.pear_foot_lock.setToolTip('Stabilize planted feet with small, smooth leg corrections. Camera alignment is required.')
        layout.addWidget(self.pear_foot_lock);self.pear_foot_lock.toggled.connect(self.pear_filter_changed)
        self.pear_smoothing=QCheckBox('One Euro Filter');self.pear_smoothing.setChecked(True)
        self.pear_smoothing.setToolTip('Reduce motion jitter in video and webcam input.')
        layout.addWidget(self.pear_smoothing)
        components=QVBoxLayout();components.setContentsMargins(16,0,0,0);components.setSpacing(4)
        self.pear_smooth_rotations=QCheckBox('Bone Rotation');self.pear_smooth_rotations.setChecked(True)
        self.pear_smooth_rotations.setToolTip('Smooth body and finger rotations.')
        self.pear_smooth_position=QCheckBox('World Position');self.pear_smooth_position.setChecked(True)
        self.pear_smooth_position.setToolTip('Smooth estimated world position.')
        components.addWidget(self.pear_smooth_rotations);components.addWidget(self.pear_smooth_position)
        layout.addLayout(components)
        strength_row=QHBoxLayout();strength_row.addWidget(QLabel('Filter Strength'))
        self.pear_strength=QDoubleSpinBox();self.pear_strength.setRange(1,100);self.pear_strength.setDecimals(0);self.pear_strength.setValue(50);self.pear_strength.setSingleStep(5)
        self.pear_strength.setToolTip('Higher values reduce jitter but add more delay.')
        strength_row.addWidget(self.pear_strength);layout.addLayout(strength_row)
        for control in (self.pear_smoothing,self.pear_smooth_rotations,self.pear_smooth_position):
            control.toggled.connect(self.pear_filter_controls_changed)
        self.pear_strength.valueChanged.connect(self.pear_filter_changed)
        self.pear_resolution=QComboBox();self.pear_resolution.addItem('Webcam 1280 × 720',720);self.pear_resolution.addItem('Webcam 640 × 480',480)
        self.pear_resolution.setToolTip('Requested webcam capture size. Actual size depends on the driver. Lower resolution may reduce camera and detector cost.')
        layout.addWidget(self.pear_resolution)
        self.pear_status=QLabel('Click Start Live Tracking to begin.');self.pear_status.setWordWrap(True);layout.addWidget(self.pear_status)
        self.pear_status.setToolTip('Fixed-length body and finger bones with lightweight live foot locking. No per-frame body mesh, recording, offline correction, retargeting or export. Keep cameras fixed.')
        self.pear_alignment_warning=QLabel();self.pear_alignment_warning.setWordWrap(True);self.pear_alignment_warning.hide();layout.addWidget(self.pear_alignment_warning)
        self.pear_tracking_button=QPushButton('Start Live Tracking');layout.addWidget(self.pear_tracking_button);self.pear_tracking_button.clicked.connect(self.pear_toggle_tracking)
        self.pear_tracking_button.setToolTip('Try camera alignment, then start tracking. If alignment fails, track the selected camera without fusion. Stop releases the models and webcam.')
        self.pear_reset_button=QPushButton('Reset Bones');self.pear_reset_button.setEnabled(False);layout.addWidget(self.pear_reset_button)
        self.pear_reset_button.setToolTip('Recalculate bones from the displayed pose, then align cameras again. Video playback pauses.')
        self.pear_reset_button.clicked.connect(self.pear_reset_bones)
        self.pear_resolution.currentIndexChanged.connect(self.pear_source_changed)
        self.pear_align_button=QPushButton('Camera Align');layout.addWidget(self.pear_align_button)
        self.pear_align_button.setToolTip('Align one or more fixed views using VGGT and GeoCalib. A single view uses an estimated body-to-depth scale. Show the person and floor; multiple videos must be synchronized. Start Live Tracking aligns automatically.')
        self.pear_align_tooltip=self.pear_align_button.toolTip()
        self.pear_align_button.clicked.connect(self.pear_camera_align)
        self.model_pages.addWidget(self.pear_page)
        self.pear_poll_timer=QTimer(self);self.pear_poll_timer.setInterval(16);self.pear_poll_timer.timeout.connect(self.pear_poll)

    def pear_scan_cameras(self,force=False):
        if force:self.pear_capture_failed=None;self.pear_capture_suspended=None
        if not force and time.monotonic()-self._pear_scan_cache_at<5:return
        if self._pear_camera_scan is not None:return
        self._pear_camera_scan=CameraScan();self.pear_refresh.setEnabled(False)
        self._pear_camera_scan.start();self.pear_scan_timer.start()

    def pear_finish_scan(self):
        scan=self._pear_camera_scan
        if scan is None or scan.devices is None:return
        devices=scan.devices;self._pear_scan_cache_at=time.monotonic();self._pear_camera_scan=None;self.pear_scan_timer.stop()
        self.pear_fill_devices(devices);self.pear_refresh.setEnabled(True)

    def pear_stop_scan(self):
        self.pear_scan_timer.stop()
        if self._pear_camera_scan:self._pear_camera_scan.cancel()
        self._pear_camera_scan=None;self.pear_refresh.setEnabled(True)

    def pear_select_model(self):
        selected=self.model_choice.currentData()=='pear'
        if selected:
            if self.busy or self.contact_saving or self.mamma_pending or self._mamma_switching:
                self.model_choice.blockSignals(True);self.model_choice.setCurrentIndex(self.model_choice.findData(self._selected_tool));self.model_choice.blockSignals(False);return True
            if not self._pear_active:
                self.stop_playback();self.remember_workspace_model()
                self._pear_active=True;self._pear_running=False;self._pear_last=None
                self.suspend_stream()
                self._pear_media_restore=dict(info=self.info,video_path=self.video_path,source_offset=self.source_offset,source_end=self.source_end,
                    value=self.slider.value(),minimum=self.slider.minimum(),maximum=self.slider.maximum(),clip_in=self.slider.clip_in,clip_out=self.slider.clip_out,
                    filename=self.filename.text(),fps_label=self.fps_label.text(),track=self.slider.video_track_label,source_key=self.video_view._source_key,path=self.video_view._source_key[0] if self.video_view._source_key else None)
                m=self.motion_view
                self._pear_camera_state={k:np.copy(getattr(m,k)) if isinstance(getattr(m,k),np.ndarray) else getattr(m,k) for k in ('yaw','pitch','distance','pan','scene_center','body_height','scene_span','auto_fit')}
                self._pear_contact_edit=self.slider.contact_edit_enabled;self._pear_status_text=self.status.text();self._pear_bone_tooltip=self.bone_overlay.toolTip();self._pear_timeline_help=self.timeline_help_dialog.text()
                self._pear_observations=self.video_view.show_observations;self.video_view.show_observations=False
                m.live_active=True;m.live_data=None;m.live_asset=None;m.live_message='Open a video or select a webcam.';m.live_frame_first=True
                self.pear_scan_cameras();self.pear_poll_timer.start()
                self.pear_source_changed()
            self.pear_sync_controls();return True
        if self._pear_active:self.pear_leave()
        return False

    def pear_leave(self):
        if not self._pear_active:return
        self._pear_active=False;self.pear_poll_timer.stop();self.pear_close_session();self.pear_shutdown_multi()
        self.stream_tab_changed(self.people_tabs.currentIndex())
        if self.pear_inputs and self._pear_media_restore:
            saved=self._pear_media_restore;self.info=saved['info'];self.video_path=saved['video_path'];self.source_offset=saved['source_offset'];self.source_end=saved['source_end']
            self.slider.setRange(saved['minimum'],saved['maximum']);self.slider.clip_in=saved['clip_in'];self.slider.clip_out=saved['clip_out'];self.slider.video_track_label=saved['track'];self.slider.setValue(saved['value'])
            self.filename.setText(saved['filename']);self.fps_label.setText(saved['fps_label'])
            if self.info:self.slider.set_media(self.info['fps'])
            self.clip_changed(self.slider.clip_in,self.slider.clip_out)
            self.video_view.set_external_frame(None)
            if saved['path']:
                key=saved['source_key'];self._pear_restoring_media=True
                opened=self.video_view.open(Path(saved['path']),source_size=key[3],timeline_fps=key[4],timeline_start=key[5],reuse=True)
                if not opened:self.pear_restore_media_ready()
            else:
                self.video_readable=False;self.video_view.shutdown();self.video_view.update()
        self.pear_inputs=[];self.pear_calibration=None
        self.video_view.set_preview_rotation(0.)
        self.pear_header.hide();self.timeline_panel.setEnabled(True);self.video_view.set_external_frame(None)
        self.video_view.show_observations=self._pear_observations
        self.video_overlay.show();self.status.setText(self._pear_status_text)
        self.bone_overlay.setToolTip(self._pear_bone_tooltip)
        self.timeline_help_dialog.setText(self._pear_timeline_help)
        for check in self.viewport_checks.values():check.show()
        self.slider.contact_edit_enabled=self._pear_contact_edit
        m=self.motion_view;m.live_active=False;m.live_data=None;m.live_asset=None
        for k,v in self._pear_camera_state.items():setattr(m,k,v)
        m.release_live_renderer();m.update()
        for widget in (self.people_panel,self.find_person_button,self.correction_group,self.compute_mode.parentWidget(),self.run_button,self.mesh_button,self.cancel_button,self.viewport_footer):widget.show()
        self.import_character_action.setEnabled(True);self.model_pages.setVisible(True)
        self.slider.live_preview=False;self.slider._cache_key=None;self.slider.update()

    def pear_close_session(self):
        self._pear_running=False
        self._pear_reset_pending=False;self._pear_reset_pose=None
        self._pear_raw_pose=None;self._pear_world_anchor=None
        if self.pear_session:
            self._pear_closing.append(self.pear_session);self.pear_session.close();self.pear_session=None
        self._pear_last=None
        self._pear_ready=False
        if self._pear_active:
            self.motion_view.live_data=None;self.motion_view.live_asset=None;self.motion_view.release_live_renderer()
        self.pear_update_tracking_button()

    def pear_update_tracking_button(self):
        closing=any(s.process.poll() is None for s in self._pear_closing)
        active=self._pear_running or bool(self.pear_align_phase)
        self.pear_tracking_button.setText('Stop' if active else 'Stopping…' if closing else 'Start Live Tracking')
        self.pear_tracking_button.setEnabled(active or not closing)
        self.pear_reset_button.setEnabled(self._pear_running and not self.pear_align_phase and self._pear_ready and self.motion_view.live_data is not None and not self._pear_reset_pending)

    def pear_reset_bones(self):
        if not self.pear_reset_button.isEnabled():return
        if self.pear_calibration:
            self.pear_calibration=None
            self.pear_status.setText('Bone scale changed; run Camera Align again.')
        meta=self.pear_last_metrics
        if not self.pear_using_webcams():
            frame=int(meta['timeline_frame']);self.stop_playback();self.slider.setValue(frame);self.seek(frame)
        self._pear_bone_reset+=1;self._pear_reset_pose=meta['pose_id'];self._pear_reset_pending=True
        self._pear_last=None;self.pear_status.setText('Resetting bones from the displayed pose…')
        self.pear_update_tracking_button();self.pear_submit()

    def pear_position_changed(self,*_):
        self._pear_position_revision+=1
        current=self.motion_view.live_data
        self._pear_world_anchor=current['joints'][0].copy() if not self.pear_world_position.isChecked() and current is not None else None
        if self._pear_active and current is not None and self._pear_raw_pose is not None:
            self.pear_position_pose(self._pear_raw_pose)
            self.motion_view.live_key=('pear-position',self.pear_generation,self._pear_position_revision,self._pear_presented_request)
            self.motion_view.update()
        if self._pear_active:self._pear_last=None;self.pear_submit()

    def pear_position_pose(self,pose):
        self._pear_raw_pose=pose
        if pose is None:self.motion_view.live_data=None;return
        if self.pear_world_position.isChecked():self.motion_view.live_data=pose;return
        if self._pear_world_anchor is None:self._pear_world_anchor=pose['joints'][0].copy()
        from .pear.projection import lock_position
        self.motion_view.live_data=lock_position(pose,self._pear_world_anchor)

    def pear_toggle_tracking(self):
        if not self._pear_active:return
        if self._pear_running or self.pear_align_phase:
            try:self.pear_capture_suspended=json.dumps(self.pear_capture_request(),sort_keys=True)
            except ValueError:self.pear_capture_suspended=None
            self.pear_cancel_alignment("Tracking stopped.")
            self.pear_close_capture()
            self._pear_running=False;self.pear_generation+=1
            self.pear_alignment_warning.clear();self.pear_alignment_warning.hide()
            self.stop_playback();self.pear_close_session()
            self.video_view.set_external_frame(None)
            self.pear_status.setText('Tracking stopped.')
            self.motion_view.live_message='Click Start Live Tracking to begin.';self.motion_view.update()
            if self.info:self.fps_label.setText(f"Source {self.info['fps']:.2f} fps")
            return
        if any(s.process.poll() is None for s in self._pear_closing):return
        try:
            if not self.pear_sources():raise ValueError('Open a video or assign a webcam before starting.')
        except (ValueError,OSError) as e:self.pear_status.setText(str(e));return
        from .pear_runtime import status
        from .models import missing
        ok,message=status()
        if not ok or missing('pear_assets'):
            self.pear_status.setText(message if not ok else 'Prepare missing PEAR models in Model Manager.');return
        self.pear_capture_suspended=None;self.pear_capture_failed=None
        self._pear_running=True;self._pear_failed=False;self._pear_probe_at=0
        self._pear_start_align=True
        self._pear_restart+=1;self.pear_update_tracking_button();self.pear_source_changed()

    def pear_sync_controls(self):
        from .models import ready,BY_KEY,STAGE_REQUIREMENTS
        from .pear_models import KEYS
        from .pear_runtime import status
        ok,message=status()
        missing_models=[BY_KEY[key] for key in KEYS if not ready(BY_KEY[key])]
        unavailable=[model for model in missing_models if model.key in STAGE_REQUIREMENTS['pear_assets']]
        self.viewport_footer.show()
        for key,check in self.viewport_checks.items():check.setVisible(key=='show_grid')
        self.bone_overlay.setToolTip('Show fixed-length PEAR body and finger bones. Use Reset Bones to recalibrate their lengths from the displayed pose. Display only; no animation export in this mode.')
        self.mamma_thumbnails.hide()
        self.pear_input_panel.show();self.pear_slot_widget.show()
        self.timeline_help_dialog.setText("PEAR live video preview\n\nOpen a video or select a webcam, then click Start Live Tracking. Stop releases the models and webcam.\n\nFor video, click or drag the frame ruler to infer the current frame. During playback, the processed video frame and its pose appear together with a small delay. Old requests are skipped. Pause or seek to inspect an exact frame.\n\nSelecting a webcam disables video timeline controls. No motion is recorded or exported in this mode.")
        self.video_overlay.hide();self.slider.contact_edit_enabled=False
        self.status.setText("PEAR experimental preview · Tracking and pose estimates may be unstable.")
        self.pear_header.show();self.model_pages.setCurrentWidget(self.pear_page);self.model_pages.show()
        self.people_tracker.hide()
        for widget in (self.people_panel,self.find_person_button,self.correction_group,self.compute_mode.parentWidget(),self.run_button,self.mesh_button,self.cancel_button,self.htd_refine,self.refine_description,self.footmr_refine,self.handflow_refine,self.footmr_description,self.refine_preview_note,self.analyze_button):widget.hide()
        self.models_group.setTitle('PEAR Models');self.readiness.setText(f'PEAR: {len(KEYS)-len(missing_models)}/{len(KEYS)} models ready')
        self.readiness.setStyleSheet('color:#ffb45c; font-weight:600;' if missing_models else '')
        self.readiness.setToolTip(message+('\nMissing models:\n'+'\n'.join(model.name for model in missing_models) if missing_models else ''))
        self.export_action.setEnabled(False);self.save_project_action.setEnabled(False);self.import_character_action.setEnabled(False)
        self.slider.contacts_supported=False;self.slider.live_preview=True;self.slider._cache_key=None;self.slider.update()
        self.play_button.setEnabled(self.info is not None and self.video_readable and not self.pear_using_webcams())
        if not ok or unavailable:
            self.pear_status.setText(message if not ok else 'Prepare missing PEAR models in Model Manager.')
            self.motion_view.live_message=self.pear_status.text();self.motion_view.update()
        self.model_choice.setEnabled(not self.busy and not self.contact_saving)
        self.models_button.setEnabled(not self.busy)
        if self.busy:self.pear_close_session()
        self.pear_update_tracking_button()

    def pear_source_changed(self,*_):
        if not self._pear_active:return
        self._pear_failed=False
        camera=self.pear_camera.currentData() if self.pear_using_webcams() else None
        if self.pear_using_webcams() and camera is None:camera=True
        path=self.video_view._source_key[0] if self.video_view._source_key and self.video_readable and camera is None else None
        self.pear_multi_invalidate()
        if camera:
            self.pear_make_thumbnails([(f'Camera {i+1}: '+(c.currentData()['name'] if c.currentData() else 'None'),None) for i,c in enumerate(self.pear_cameras)])
        self.timeline_panel.setEnabled(camera is None);self.pear_resolution.setEnabled(camera is not None)
        try:
            sources=self.pear_sources();signature=(self.pear_input_signature(),self._pear_restart)
        except (ValueError,OSError) as e:
            self.pear_status.setText(str(e));self.pear_generation+=1;self.pear_close_session();return
        keep_pair=signature==self._pear_source_signature and self.motion_view.live_data is not None
        self._pear_source_signature=signature
        self.pear_generation+=1;self._pear_last=None
        if not keep_pair:
            self._pear_reset_pending=False;self._pear_reset_pose=None
            self._pear_raw_pose=None;self._pear_world_anchor=None
            self.motion_view.live_data=None;self.motion_view.live_frame_first=True
            self.motion_view.live_message='Waiting for live tracking…';self.motion_view.update()
        self.timeline_panel.setEnabled(camera is None)
        self.pear_sync_rotations();self.pear_smoothing.setEnabled(True);self.pear_resolution.setEnabled(camera is not None)
        if not self._pear_running:
            self.video_view.set_external_frame(None)
            self.pear_status.setText('Click Start Live Tracking to begin.')
            self.motion_view.live_message=self.pear_status.text();self.motion_view.update()
            self.pear_sync_controls();return
        if not keep_pair:
            if camera:
                self.stop_playback()
                if not self.video_view.external_mode:self.video_view.set_external_frame(QImage())
            elif self.video_readable:self.video_view.set_external_frame(None)
            else:self.video_view.set_external_frame(None)
        self.pear_sync_controls()
        if self._pear_start_align or (not self.pear_calibration and not self.pear_align_phase):
            self._pear_start_align=False;self.pear_begin_alignment()
        else:self.pear_submit()

    def pear_filter_changed(self,*_):
        if self._pear_active:self._pear_last=None;self.pear_submit()

    def pear_filter_controls_changed(self,*_):
        enabled=self.pear_smoothing.isChecked()
        self.pear_smooth_rotations.setEnabled(enabled);self.pear_smooth_position.setEnabled(enabled)
        self.pear_strength.setEnabled(enabled and (self.pear_smooth_rotations.isChecked() or self.pear_smooth_position.isChecked()))
        self.pear_filter_changed()

    def pear_submit(self):
        if not self._pear_active or not self._pear_running or self._pear_failed or self.busy:return
        if self.pear_input_pending or self.pear_switching or self.pear_align_phase in ('closing','solve'):return
        try:sources=self.pear_sources()
        except (ValueError,OSError):return
        camera=(sources[0] if sources else None) if self.pear_using_webcams() else None;view=self.video_view
        path=view._source_key[0] if view._source_key and self.video_readable else None
        playing=self.timer.isActive() and camera is None
        value=dict(generation=self.pear_generation,mode='camera' if camera else 'video' if path else 'idle',camera=camera,path=None if camera else path,frame=view.requested_index,
            timeline_frame=view.timeline_requested_index,pair_video=playing,restart=self._pear_restart,
            rotation=self.pear_rotation.value(),smoothing=self.pear_smoothing.isChecked(),smoothing_strength=self.pear_strength.value(),resolution=self.pear_resolution.currentData())
        value.update(smooth_rotations=self.pear_smooth_rotations.isChecked(),smooth_position=self.pear_smooth_position.isChecked())
        value.update(bone_reset=self._pear_bone_reset,reset_pose=self._pear_reset_pose)
        value.update(foot_lock=self.pear_foot_lock.isChecked(),world_position=self.pear_world_position.isChecked(),
            position_anchor=self._pear_world_anchor.tolist() if self._pear_world_anchor is not None else None)
        try:self.pear_multi_request(value)
        except (ValueError,OSError) as exc:
            if not self.pear_align_phase:self.pear_status.setText(str(exc))
            return
        if value==self._pear_last:return

        if not self.pear_session:
            if any(s.process.poll() is None for s in self._pear_closing):
                self.pear_status.setText('Stopping previous PEAR session…');return
            if time.monotonic()-self._pear_probe_at<1:return
            self._pear_probe_at=time.monotonic()
            from .pear_runtime import status
            from .models import missing
            if not status()[0] or missing('pear_assets'):return
            from .pear.session import Session
            try:self.pear_session=Session()
            except Exception as e:
                self._pear_running=False;self.pear_update_tracking_button();self.pear_status.setText(str(e));return
            self.pear_started=time.monotonic();self._pear_failed=False;self._pear_ready=False
        self._pear_last=value.copy();self.pear_request+=1;value['request']=self.pear_request;value['requested_at']=time.monotonic()
        self.pear_session.submit(value)
        if not camera and not playing:
            if self.motion_view.live_data is None:self.motion_view.live_message='Tracking current video frame…' if path else 'Open a video or select a webcam.'
            else:self.pear_status.setText('Updating current video frame…')
            self.motion_view.update()

    def pear_seek(self,index):
        if not self._pear_active or not self._pear_running or self.pear_using_webcams():return
        # Ordinary forward playback accepts completed frames with bounded delay.
        # A scrub or loop starts a new epoch: old results must not cross it.
        if index!=self.video_view.timeline_requested_index and (not self.playback_tick or index<self.video_view.timeline_requested_index):
            self.pear_generation+=1;self._pear_last=None
            if not (self.timer.isActive() and self.video_view.external_mode):
                if self.motion_view.live_data is None:self.motion_view.live_message='Tracking current video frame…'
                else:self.pear_status.setText('Updating current video frame…')
                self.motion_view.update()

    def pear_playback_stopped(self):
        if not self._pear_active or not self._pear_running or self.pear_using_webcams():return
        if self.info:self.fps_label.setText(f"Source {self.info['fps']:.2f} fps")
        if self.video_view.external_mode:
            self.pear_generation+=1;self._pear_last=None
            if self.motion_view.live_data is None:self.motion_view.live_message='Tracking current video frame…'
            else:self.pear_status.setText('Updating current video frame…')
            # Keep the last correctly rotated image/pose pair until the exact
            # paused request arrives. Do not briefly restore unrotated pixels.
            self.motion_view.update()

    def pear_poll(self):
        self._pear_closing=[s for s in self._pear_closing if s.process.poll() is None]
        if not self._pear_active:return
        self.pear_capture_poll()
        if self.pear_alignment_poll():return
        self.pear_update_tracking_button()
        if not self._pear_running:return
        self.pear_submit()
        if not self.pear_session:return
        if time.monotonic()-self._pear_guard_at>.5:
            self._pear_guard_at=time.monotonic()
            idle=self._pear_ready and (self._pear_failed or (not self.pear_using_webcams() and not self.timer.isActive() and self._pear_presented_request==self.pear_request))
            if idle:self.pear_session.guard.output()
            warning=self.pear_session.guard.check()
            if warning:
                if self.pear_align_phase=='samples':self.pear_alignment_failed('Camera Align failed: '+warning);return
                self._pear_raw_pose=None
                self.pear_status.setText(warning);self.motion_view.live_data=None;self.motion_view.live_message=warning
                self.pear_close_session();self._pear_failed=True;self.log.append('PEAR: '+warning);self.motion_view.update();return
        top,packet=self.pear_session.poll()
        if top:
            self._pear_ready=True
            meta,a=top
            self.motion_view.live_asset=dict(key='pear-live',render_mode='bones') if meta.get('render_mode')=='bones' else dict(key='pear-live',topology_key='pear-live',mode='raw',vertex_count=meta['vertices'],**{k:a[k] for k in ('faces','offsets','neighbors')})
            self.motion_view.live_parents=a['parents']
        if packet:
            meta,a=packet;kind=meta['type']
            if self.pear_multi_packet(meta,a):return
            if 'preview' in meta and meta['preview']!=self.pear_preview:return
            if kind=='error':
                if meta.get('generation') is not None and meta['generation']!=self.pear_generation:
                    if meta.get('recoverable'):return
                    self.pear_close_session();return
                if self.pear_align_phase=='samples':self.pear_alignment_failed(meta['text']);return
                self._pear_raw_pose=None
                self.pear_status.setText(meta['text']);self.motion_view.live_data=None;self.motion_view.live_message=meta['text'];self.motion_view.update()
                if meta.get('recoverable'):
                    self._pear_failed=True;self.log.append('PEAR: '+meta['text']);return
                self.log.append('PEAR: '+meta['text']);self.pear_close_session();self._pear_failed=True;return
            if kind=='status':
                if meta.get('generation',self.pear_generation)!=self.pear_generation:return
                if meta.get('idle'):self._pear_presented_request=meta['request']
                self.pear_status.setText(meta['text']);self.motion_view.live_message=meta['text'];self.motion_view.update();return
            if meta.get('generation')!=self.pear_generation:return
            if meta.get('mode')=='video':
                playing=self.timer.isActive() and not self.pear_using_webcams()
                if bool(meta.get('pair_video'))!=playing:return
                if playing:
                    if meta.get('request',-1)<=self._pear_presented_request or ('image' not in a and kind!='missing'):return
                elif meta.get('request')!=self.pear_request:return
            self._pear_presented_request=meta.get('request',-1)
            if 'image' in a:
                im=a['image'];h,w,_=im.shape;self.video_view.set_external_frame(QImage(im.data,w,h,im.strides[0],QImage.Format_RGB888).copy())
                if meta.get('mode')=='video':self.video_view.index=meta['timeline_frame']
            if kind=='frame':a['rig_revision']=meta.get('bone_revision',0)
            self.pear_position_pose(a if kind=='frame' else None)
            self.motion_view.live_key=(self.pear_generation,meta.get('frame'),meta.get('request'),self._pear_position_revision)
            self.motion_view.live_message=meta.get('text','')
            if kind=='frame':
                if self._pear_reset_pending and meta.get('bone_reset_handled',-1)>=self._pear_bone_reset:
                    self._pear_reset_pending=False;self.pear_last_metrics=meta
                    self.pear_begin_alignment();return
                self.pear_last_metrics=meta
                ms=meta['ms'];extra=(f" · {meta['available_views']}/{meta['views']} views" if meta.get('views') else '')
                if meta.get('views') and not meta.get('aligned'):extra+=' · Camera Align required to combine views'
                if max(meta.get('foot_lock',{}).get('residual_mm',[0.]))>30:extra+=' · Foot lock limited by reach'
                self.pear_status.setText(meta.get('bone_reset_warning') or f'Live tracking · {ms:.0f} ms/frame · {1000/max(ms,1):.1f} inference fps'+extra)
                if meta.get('pair_video'):
                    delay=max(0,(time.monotonic()-meta['requested_at'])*1000)
                    self.pear_last_metrics['display_delay_ms']=delay
                    self.fps_label.setText(f"Preview {meta['timeline_frame']+1} · {delay:.0f} ms delay")
                if self.motion_view.live_frame_first:self.motion_view.live_frame_first=False;self.motion_view.frame_all()
            else:self.pear_status.setText(meta.get('text','No person visible.'))
            self.pear_update_tracking_button()
            self.motion_view.update()
        elif self.pear_session and not self._pear_ready:
            self.pear_status.setText(f'Loading PEAR · {time.monotonic()-self.pear_started:.0f} s')
