"""Link the optional face worker to the ordinary offline processing queue."""
from PySide6.QtCore import QSignalBlocker, Qt
from .face_data import root_for


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

    def face_playback_archive(self, root):
        from .face_playback import FacePlayback

        if not hasattr(self, '_face_playback'):
            self._face_playback = FacePlayback(self)
            self._face_playback.changed.connect(self.face_playback_ready)
        return self._face_playback.request(root)

    def face_playback_ready(self):
        cache = self._face_playback
        if cache.closed:
            return
        if cache.error:
            self.log.append('Face playback unavailable: ' + cache.error)
        self._stream_frame = None
        self.update_face_overlay(refresh=True)

    def close_face_playback(self):
        if hasattr(self, '_face_playback'):
            self._face_playback.close()

    def neutral_face_person(self):
        item = self.people_list.currentItem()
        if item is not None and isinstance(item.data(Qt.UserRole), str):
            raise ValueError('Select a person in the People list, rather than a character.')
        person = self.current_person()
        if person is None or not isinstance(person.get('id'), int):
            raise ValueError('Select a person in the People list first.')
        return person

    def update_neutral_face_controls(self, archive):
        if not hasattr(self, 'set_neutral_face'):
            return
        ready = archive is not None and not self.busy and not self.contact_saving and not self._pear_active
        try:
            person = self.neutral_face_person()
            ident = person['id']
            ready = ready and ident in archive.index
        except (ValueError, AttributeError):
            person, ident, ready = None, None, False
        frame = archive.neutral_frames.get(ident) if archive is not None else None
        name = person.get('name', 'Person ' + str(ident)) if person else 'Face'
        self.set_neutral_face.setEnabled(bool(ready))
        self.reset_neutral_face.setEnabled(bool(ready and frame is not None))
        self.neutral_face_note.setText(name + (' · Neutral frame ' + str(frame + 1)
                                               if frame is not None else ' · Neutral: Default'))

    def set_current_neutral_face(self):
        if self.busy or self.contact_saving:
            return
        try:
            from .face_neutral import set_neutral
            root = self.face_project()
            archive = self.face_playback_archive(root)
            if archive is None:
                raise ValueError('Wait for the saved face tracking data to load, or run Track Face first.')
            person = self.neutral_face_person()
            project, frame = self.project, self.motion_view.index
            if self.people_state:
                if self.scene_data is None:
                    raise ValueError('Wait for the people scene to load.')
                current = self.scene_data.take_at(person['id'], self.motion_view.scene_time, self.project)
                if current is None:
                    raise ValueError('This person has no take at the current timeline position.')
                take, frame = current
                project = take.project
            slot, sample, source = archive.resolve(project, frame, person=person['id'])
            if sample < 0 or sample != source:
                raise ValueError('Choose a frame where this person’s face was tracked successfully, rather than a held expression.')
            self.stop_playback()
            set_neutral(archive, person['id'], int(source))
            self._stream_frame = None
            self.update_face_overlay(refresh=True)
            self.status.setText(person.get('name', 'Person') + ' neutral face set at capture frame ' + str(source + 1) + '.')
        except (OSError, ValueError, KeyError, IndexError) as error:
            self.status.setText('Cannot set neutral face: ' + str(error))

    def reset_current_neutral_face(self):
        if self.busy or self.contact_saving:
            return
        try:
            from .face_neutral import reset_neutral
            root = self.face_project()
            if root is None:
                return
            person = self.neutral_face_person()
            reset_neutral(root, person['id'])
            self._stream_frame = None
            self.update_face_overlay(refresh=True)
            self.status.setText(person.get('name', 'Person') + ' neutral face reset.')
        except (OSError, ValueError, KeyError) as error:
            self.status.setText('Cannot reset neutral face: ' + str(error))

    def update_face_overlay(self, refresh=False):
        enabled = self.track_face.isChecked()
        archive = None
        try:
            # Also let active streaming reuse this preparation when overlay is off.
            root = self.face_project()
            archive = self.face_playback_archive(root)
        except (OSError, ValueError, KeyError):
            root = None
        self.update_neutral_face_controls(archive)
        visible_archive = archive if enabled else None
        key = (root, id(visible_archive)) if enabled and root is not None else None
        if refresh or key != getattr(self, '_face_overlay_key', None):
            self._face_overlay_key = key
            self.motion_view.face_archive = visible_archive
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
