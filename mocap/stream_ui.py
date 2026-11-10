"""Stream the prepared offline motion without changing People visibility."""
from hashlib import sha256
import json
from queue import Empty

from PySide6.QtCore import QSignalBlocker, QTimer, Qt
from PySide6.QtWidgets import QComboBox, QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from .bone_stream import BoneStream, MAX_ACTORS, MAX_BAKE_FRAMES, PORT


def stream_rig(asset):
    return asset.get('stream_rig') if asset else None


class StreamWorkspace:
    def build_stream_page(self):
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(5, 5, 5, 5)
        column.addWidget(QLabel('Target'))
        self.stream_target = QComboBox()
        self.stream_target.addItems(['None', 'Blender'])
        self.stream_target.setAccessibleName('Stream target')
        column.addWidget(self.stream_target)
        self.stream_people = QListWidget()
        self.stream_people.setObjectName('streamPeopleList')
        self.stream_people.setAccessibleName('People to stream')
        column.addWidget(self.stream_people, 1)
        self.stream_status = QLabel()
        self.stream_status.setWordWrap(True)
        self.stream_status.setObjectName('muted')
        column.addWidget(self.stream_status)

        self._stream_sender = None
        self._stream_scope = None
        self._stream_checked = set()
        self._stream_rows = None
        self._stream_frame = None
        self._stream_bake_actor = None
        self._stream_timer = QTimer(page)
        self._stream_timer.setTimerType(Qt.PreciseTimer)
        self._stream_timer.timeout.connect(self.update_stream)
        self.stream_target.currentIndexChanged.connect(self.change_stream_target)
        self.stream_people.itemChanged.connect(self.stream_person_changed)
        return page

    def stream_tab_changed(self, index):
        if not self._pear_active and (index == 1 or self._stream_sender is not None):
            self.update_stream()
            self._stream_timer.start(33 if self._stream_sender else 250)
        else:
            self._stream_timer.stop()

    def change_stream_target(self, index):
        self.close_stream()
        if index == 1 and not self._pear_active:
            try:
                self._stream_sender = BoneStream()
            except OSError as error:
                with QSignalBlocker(self.stream_target):
                    self.stream_target.setCurrentIndex(0)
                self.stream_status.setText(f'Cannot start Blender stream on port {PORT}.')
                self.log.append(f'Blender stream could not start: {error}')
                return
        self.stream_tab_changed(self.people_tabs.currentIndex())

    def close_stream(self):
        if self._stream_sender is not None:
            self._stream_sender.close()
            self._stream_sender = None
        self._stream_frame = None
        self._stream_bake_actor = None

    def suspend_stream(self):
        self.stream_target.setCurrentIndex(0)
        self._stream_timer.stop()

    def stream_person_changed(self, item):
        ident = item.data(Qt.UserRole)
        if item.checkState() == Qt.Checked:
            if len(self._stream_checked) >= MAX_ACTORS:
                with QSignalBlocker(self.stream_people):
                    item.setCheckState(Qt.Unchecked)
                return
            self._stream_checked.add(ident)
        else:
            self._stream_checked.discard(ident)
        self.update_stream()

    def stream_people_with_motion(self):
        if self.people_state:
            if self.scene_data is None:
                return ()
            return tuple((person['id'], person['name']) for person in self.people_state['people']
                         if any(stream_rig(take.target_asset) is not None and 'rig' in take.arrays
                                for take in self.scene_data.takes.get(person['id'], ())))
        if self.motion_view.target_frames is not None and stream_rig(self.motion_view.target_asset):
            actor = self._single_actor
            return ((1, actor['name'] if actor else 'Person 1'),)
        return ()

    def stream_sources_at_playhead(self, rows):
        view = self.motion_view
        for ident, name in rows:
            if ident not in self._stream_checked:
                continue
            if self.people_state:
                current = self.scene_data.take_at(ident, view.scene_time, self.project)
                if current is None:
                    continue
                take, frame = current
                rig, frames, transform = stream_rig(take.target_asset), take.arrays.get('rig'), take.transform
            else:
                rig, frames = stream_rig(view.target_asset), view.target_frames
                frame, transform = view.index, None
            if rig is not None and frames is not None and 0 <= frame < len(frames):
                yield ident, name, rig, frames, frame, transform

    def stream_bake_plan(self, ident, actor, key):
        fps = float(self.info['fps'])
        origin, count = self.source_offset, int(self.info['frames'])
        preferred = None

        def segment(first, end, start, rate, frames, rig, transform):
            if rig is None or frames is None:
                raise ValueError('Prepare the motion for all takes before baking.')
            return first, end, start, rate, getattr(frames, 'path', frames), rig, transform

        if self.people_state:
            scene = self.scene_data
            if scene is None:
                raise ValueError('Wait for the people scene to load.')
            manifest = json.loads((self.people_root/'project.json').read_text(encoding='utf-8'))
            fps, origin, count = float(manifest['fps']), float(manifest['start']), int(manifest['frames'])

            def take_segment(first, end, take):
                return segment(first, end, take.start, take.fps, take.arrays.get('rig'),
                               stream_rig(take.target_asset), take.transform)

            spans = [take_segment(first, end, take) for first, end, take in scene.spans.get(ident, ())]
            take = scene.by_project.get(self.project)
            if take is not None and take.ident == ident:
                preferred = take_segment(take.start, take.end, take)
        else:
            view = self.motion_view
            if view.target_frames is not None:
                count = min(count, len(view.target_frames))
            spans = [segment(origin, origin + count/fps, origin, fps, view.target_frames,
                             stream_rig(view.target_asset), None)]
        rigs = [span[5] for span in spans + ([preferred] if preferred else [])]
        active = next((rig for rig in rigs if rig.key == key), None)
        if active is None:
            raise ValueError('The streaming skeleton changed. Wait for the updated rig.')
        topology = [bone[:2] for bone in active.bones]
        if any([bone[:2] for bone in rig.bones] != topology for rig in rigs):
            raise ValueError('The takes use different bone hierarchies.')
        if not 0 < count <= MAX_BAKE_FRAMES:
            raise ValueError(f'Bake supports up to {MAX_BAKE_FRAMES:,} timeline frames.')
        return actor, key, fps, origin, count, spans, [span[0] for span in spans], preferred

    def stream_actor_id(self, ident):
        return sha256(f'{self._stream_scope!r}/{ident}'.encode('utf-8')).hexdigest()

    def stream_bake_commands(self, sender):
        for _ in range(8):
            try:
                connection, kind, actor, key = sender.commands.get_nowait()
            except Empty:
                return
            if not sender.connected or connection != sender.connection:
                continue
            ident = next((i for i in self._stream_checked if self.stream_actor_id(i) == actor), None)
            if kind == ord('A'):
                if self._stream_bake_actor == (connection, actor):
                    if ident is not None:
                        self._stream_checked.discard(ident)
                        self._stream_rows = None
                    self._stream_bake_actor = None
                continue
            try:
                if ident is None or not self.info:
                    raise ValueError('Check the person in RGB MoCap before baking.')
                reply = self.stream_bake_plan(ident, actor, key)
                self._stream_bake_actor = connection, actor
            except (OSError, ValueError, KeyError, TypeError) as error:
                reply = str(error)
            sender.bake_reply = connection, reply
            sender.changed.set()

    def update_stream(self):
        if self._pear_active:
            return

        scope = (self.people_root or self.project, self.model_choice.currentData())
        if scope != self._stream_scope:
            self._stream_scope = scope
            self._stream_checked.clear()
            self._stream_rows = None
            self._stream_frame = None
        sender = self._stream_sender
        if sender is not None:
            self.stream_bake_commands(sender)
        rows = self.stream_people_with_motion()
        if rows != self._stream_rows:
            with QSignalBlocker(self.stream_people):
                self.stream_people.clear()
                for ident, name in rows:
                    item = QListWidgetItem(name, self.stream_people)
                    item.setData(Qt.UserRole, ident)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Checked if ident in self._stream_checked else Qt.Unchecked)
            self._stream_rows = rows

        if sender is None:
            status = 'Select a target to stream.' if rows else 'Prepare motion to list people.'
        else:
            error = sender.error()
            if error:
                self.log.append(f'Blender stream: {error}')
            connected = sender.connected
            sources = tuple(self.stream_sources_at_playhead(rows)) if connected else ()
            checked = tuple(sorted(self._stream_checked)) if connected else None
            key = (checked, tuple((ident, name, rig.key, id(frames), frame, id(transform))
                                  for ident, name, rig, frames, frame, transform in sources))
            if key != self._stream_frame:
                selected = {
                    ident: self.stream_actor_id(ident)
                    for ident in checked or ()
                }
                try:
                    actors = tuple(
                        (selected[ident], name,
                         rig, frame, frames[frame], transform)
                        for ident, name, rig, frames, frame, transform in sources)
                except (OSError, ValueError, IndexError) as error:
                    self.log.append(f'Blender stream could not read motion: {error}')
                    actors = ()
                sender.publish(actors, tuple(selected.values()) if checked is not None else None)
                self._stream_frame = key
            if not connected:
                status = f'Waiting for Blender\n127.0.0.1:{PORT}'
            elif not self._stream_checked:
                status = 'Connected. Check people to stream.'
            else:
                status = f'Connected · {len(sources)} people at this frame'
        if self.stream_status.text() != status:
            self.stream_status.setText(status)
