"""Link the optional face worker to the ordinary offline processing queue."""
from PySide6.QtCore import QSignalBlocker
from .face_data import FaceArchive, FILE, root_for


class FaceWorkspace:
    def face_project(self):
        if self.project is None or self._pear_active:
            return None
        root = self.people_root if self.people_root is not None else root_for(self.project)
        return root if (root/'input.mp4').is_file() else None

    def face_job(self):
        root = self.face_project()
        if root is None:
            return None
        arguments = ['--project',str(root)]
        if self.people_state is not None:
            selected = self.checked_people()
            if not selected:
                return None
            for person in selected:
                arguments += ['--person',str(person['id'])]
        return 'face_track',arguments

    def update_face_overlay(self, refresh=False):
        enabled = self.track_face.isChecked()
        try:
            root = self.face_project() if enabled else None
        except (OSError,ValueError,KeyError):
            root = None
        key = None
        if root is not None:
            try:
                stamp = (root/FILE).stat()
                key = (root,stamp.st_size,stamp.st_mtime_ns)
            except OSError:
                pass
        if refresh or key != getattr(self,'_face_overlay_key',None):
            self._face_overlay_key = key
            archive=None
            if key is not None:
                try:
                    archive=FaceArchive(root)
                except (OSError,ValueError,KeyError) as error:
                    self.log.append('Face overlay unavailable: '+str(error))
            self.motion_view.face_archive=archive
            self.motion_view.face_surfaces.clear()
            self.motion_view.face_manifests.clear()
            self.motion_view.update()

    def restore_track_face(self, enabled):
        with QSignalBlocker(self.track_face):
            self.track_face.setChecked(enabled is True)
        self.update_face_overlay()

    def track_face_toggled(self, enabled):
        self.update_face_overlay(refresh=True)
        if enabled and not self.busy and not self.contact_saving:
            try:
                job=self.face_job()
                if job is not None:
                    self.start_jobs([job])
            except (OSError,ValueError) as error:
                self.status.setText('Face tracking unavailable: '+str(error))

    def queue_face_after_reconstruction(self):
        if (self.jobs or self.failed or self.cancelled or not self.track_face.isChecked()
                or not self._batch_has_motion or self.current_stage not in
                ('export','flow_export','gemx_reconstruct','mamma_publish','prompthmr_publish')):
            return
        try:
            job=self.face_job()
            if job is not None:
                self.jobs.append(job)
        except (OSError,ValueError) as error:
            self.log.append('Face tracking unavailable: '+str(error))
