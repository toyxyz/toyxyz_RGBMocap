"""Shared workspace controls and queue setup for ViDiHand2."""
from pathlib import Path
import json

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFileDialog

from .motion import ARM_LENGTHS, TAKE_NAME, TILT_TAKE_NAME


class FirstPersonWorkspace:
    def choose_first_person_camera(self):
        selected,_=QFileDialog.getOpenFileName(self,"Open ViDiHand2 Camera JSON",filter="Camera JSON (*.json)")
        if selected:
            self.first_person_camera_path=Path(selected)
            self.first_person_camera.setCurrentIndex(self.first_person_camera.findData("manual"))
            self.first_person_camera_file.setToolTip(selected)
            self.status.setText("Calibration selected: "+self.first_person_camera_path.name)

    def clear_first_person_calibration(self):
        self.first_person_camera_path = None
        self.first_person_camera_file.setToolTip("")

    def sync_first_person_controls(self):
        selected=self.model_choice.currentData()=="vidihand2"
        self.track_face.setVisible(not selected)
        self.rife_interpolation.setVisible(not selected)
        self.mesh_button.setVisible(not selected)
        self.face_overlay.setVisible(not selected)
        self.mano_overlay.setEnabled(selected and self.motion_view.first_person_mano is not None)
        for name in ("show_correction_floor","show_trajectory","show_foot_locks","show_names"):
            self.viewport_checks[name].setVisible(not selected)
        if not selected:
            self.cancel_first_person_camera_edit()
            if hasattr(self,"_camera_before_first_person"):
                checked=self._camera_before_first_person
                del self._camera_before_first_person
                self.viewport_checks["show_cameras"].setChecked(checked)
            return
        self.find_person_button.hide()
        self.correction_group.hide()
        self.slider.contacts_supported=False
        self.slider.update()
        self.first_person_camera.setEnabled(not self.busy)
        self.first_person_camera_file.setEnabled(not self.busy)
        self.first_person_auto_arms.setEnabled(not self.busy)
        self.first_person_auto_lengths.setEnabled(not self.busy)
        self.update_arm_length_controls()
        self.update_first_person_camera_editor()
        self.run_button.setToolTip("Track hands and fingers and derive head motion from the camera.")
        self.people_help_dialog.setText("The camera wearer is the actor. ViDiHand2 tracks both hands and fingers.")

    def update_first_person_camera_editor(self):
        view = self.motion_view
        available = self.first_person_take_ready()
        if not available:
            self.cancel_first_person_camera_edit()
        editing = view.first_person_camera_edit is not None
        self.first_person_adjust_camera.setEnabled(available and not editing)
        self.first_person_edit_controls.setVisible(editing)
        self.first_person_fit_arms.setEnabled(available and not editing and (self.project/"input.mp4").is_file())

    def first_person_take_ready(self):
        view = self.motion_view
        return (self.model_choice.currentData()=="vidihand2" and not self.busy and not self.contact_saving
                and self.project is not None and view.first_person_head is not None
                and view.path == self.project/TAKE_NAME)

    def apply_first_person_arms(self):
        if self.first_person_fit_arms.isEnabled():
            self.stop_playback()
            self.start_jobs([("vidihand_arms",["--project",str(self.project),*self.arm_length_arguments()])])

    def arm_length_arguments(self):
        if self.first_person_auto_lengths.isChecked():
            return ["--auto-arm-lengths"]
        return ["--arm-lengths",*(str(field.value()/100.) for field in self.first_person_arm_lengths)]

    def update_arm_length_controls(self):
        for field in self.first_person_arm_lengths:
            field.setEnabled(not self.busy and not self.first_person_auto_lengths.isChecked())

    def restore_arm_lengths(self,lengths=ARM_LENGTHS,automatic=False):
        for field,length in zip(self.first_person_arm_lengths,lengths):
            field.setValue(length*100.)
        self.first_person_auto_lengths.setChecked(automatic)

    def start_first_person_camera_edit(self):
        if not self.first_person_adjust_camera.isEnabled():
            return
        from .camera_alignment import read_alignment
        from .camera_gizmo import CameraGizmo
        from .motion import Y_UP, load
        from ..result_artifacts import file_stat

        path = self.project/TAKE_NAME
        try:
            stamp = file_stat(path)
            data = load(path)
            alignment = read_alignment(data)
            center = Y_UP @ data["camera_position"][0]
        except (OSError, ValueError, KeyError) as error:
            self.status.setText(str(error))
            return
        self.stop_playback()
        self.slider.setValue(0)
        self.seek(0)
        self.view_choice.setCurrentText("Perspective")
        self.viewport_checks["show_cameras"].setChecked(True)
        view = self.motion_view
        view.pan = center - view.scene_center
        view.auto_fit = False
        view.first_person_camera_edit = CameraGizmo(path, stamp, center, alignment)
        self.update_first_person_camera_editor()
        view.update()

    def cancel_first_person_camera_edit(self):
        view = self.motion_view
        if view.first_person_camera_edit is not None:
            view.first_person_camera_edit = None
            view.update()
        if hasattr(self, "first_person_edit_controls"):
            self.first_person_edit_controls.hide()
            available = self.first_person_take_ready()
            self.first_person_adjust_camera.setEnabled(available)
            self.first_person_fit_arms.setEnabled(available and (self.project/"input.mp4").is_file())

    def reset_first_person_camera_edit(self):
        edit = self.motion_view.first_person_camera_edit
        if edit is not None:
            center = edit.center
            edit.reset()
            self.motion_view.pan += edit.center - center
            self.motion_view.update()

    def apply_first_person_camera_edit(self):
        edit = self.motion_view.first_person_camera_edit
        if edit is None or self.busy or self.project is None or edit.path != self.project/TAKE_NAME:
            return
        from .camera_alignment import save_alignment

        try:
            save_alignment(edit.path, edit.transform, edit.stamp)
        except (OSError, ValueError, KeyError) as error:
            self.status.setText(str(error))
            return
        self.cancel_first_person_camera_edit()
        self.refresh_results()
        self.status.setText("First-frame camera adjustment applied to the take.")

    def run_first_person(self):
        from .runtime import load_settings
        from ..models import missing
        from ..workflow_jobs import video_jobs

        if self.busy or self.contact_saving:
            return
        if self.video_path is None:
            self.choose_video()
            return
        if not self.info or not self.video_readable:
            self.status.setText("Wait for the video preview to finish opening.")
            return
        try:
            load_settings()
        except FileNotFoundError as error:
            self.status.setText(str(error))
            self.log.append(str(error))
            return
        except (OSError,ValueError) as error:
            self.status.setText("Check ViDiHand2 setup. See docs/FIRST_PERSON_ARMS.md.")
            self.log.append(str(error))
            return
        mode=self.first_person_camera.currentData()
        if mode=="manual" and not (self.first_person_camera_path and self.first_person_camera_path.is_file()):
            self.choose_first_person_camera()
            if not (self.first_person_camera_path and self.first_person_camera_path.is_file()):
                return
        if mode=="automatic" and missing("flow_camera"):
            self.status.setText("Prepare VGGT-Omega in Model Manager.")
            self.open_model_manager()
            return
        first,end=self.processing_range()
        jobs=video_jobs(self.video_path,first,end,"vidihand2",("vidihand_track","vidihand_publish"),result="initial",
                        workspace=self.workspace_root,interpolation=False)
        if mode=="automatic":
            jobs.insert(1,("flow_vidihand_camera",None))
        if self.first_person_auto_arms.isChecked():
            jobs.append(("vidihand_arms",self.arm_length_arguments()))
        self._first_person_job_mode=mode
        self._first_person_job_camera=self.first_person_camera_path if mode=="manual" else None
        self.start_jobs(jobs)

    def first_person_command(self,project,*,refit=False):
        from .runtime import command,load_settings

        state=json.loads((project/"project.json").read_text(encoding="utf-8"))
        mode=self._first_person_job_mode
        camera=self._first_person_job_camera
        offset=0.
        if mode=="automatic":
            camera=project/"flow_camera.npz"
        elif mode=="manual":
            from ..project_workspace import read,root_for
            root=root_for(project,state)
            offset=max(0.,state.get("start",0.)-(float(read(root).get("time_offset",0.)) if root else 0.))
        output=project/(TILT_TAKE_NAME if refit else TAKE_NAME)
        if refit:
            output.unlink(missing_ok=True)
        self._first_person_cancel=project/"vidihand.cancel"
        self._first_person_cancel.unlink(missing_ok=True)
        return command(load_settings(),project/"input.mp4",camera,output,self._first_person_cancel,
                       camera_start=offset,refit_from=project/TAKE_NAME if refit else None)

    def cancel_first_person(self):
        self._first_person_cancel.touch()
        self.cancelled=True
        self.jobs.clear()
        worker=self.process
        self.cancel_button.setEnabled(False)
        self.status.setText("Cancelling ViDiHand2...")
        QTimer.singleShot(4500,lambda: worker.kill() if self.process is worker else None)

    def apply_first_person_preview(self, data):
        anchor=data.arrays.get("first_person_anchor")
        summary=data.summaries.get("first_person_anchor")
        if getattr(self,"_restore_arm_lengths_for",None)==data.project:
            self._restore_arm_lengths_for=None
            self.restore_arm_lengths(summary["arm_lengths"] if summary else ARM_LENGTHS,
                                     summary["arm_auto_lengths"] if summary else False)
        elif summary and self.first_person_auto_lengths.isChecked() and summary["arm_auto_lengths"]:
            self.restore_arm_lengths(summary["arm_lengths"],automatic=True)
        self.slider.has_motion=anchor is not None
        self.slider.update()
        self.motion_view.set_motion(anchor,summary,
                                    data.project/TAKE_NAME,
                                    first_person_hands=data.arrays.get("first_person_hands"),
                                    first_person_bones=data.arrays.get("first_person_bones"),
                                    first_person_head=data.arrays.get("first_person_head"),
                                    first_person_mano=data.target_assets.get("first_person_mano"),
                                    first_person_arms=data.arrays.get("first_person_arms"))
        if anchor is not None and not hasattr(self,"_camera_before_first_person"):
            camera=self.viewport_checks["show_cameras"]
            self._camera_before_first_person=camera.isChecked()
            camera.setChecked(True)
        self.motion_view.native_required=False
        self.correction_note.hide()
        self.refine_preview_note.hide()
        if self.motion_view.first_person_arms is not None:
            self.preview_help.setText(f"Arm IK solved {summary['arm_count']}/{summary['hand_count']} tracked hand poses.")
        else:
            self.preview_help.setText("ViDiHand2 hands and camera-driven head." if anchor is not None else "")
        self.update_readiness()
