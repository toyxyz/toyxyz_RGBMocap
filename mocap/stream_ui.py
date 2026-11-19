"""Stream the prepared offline motion without changing People visibility."""
from hashlib import sha256
import json
from queue import Empty

from PySide6.QtCore import QSignalBlocker, QTimer, Qt
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from .bone_stream import BoneStream, MAX_ACTORS, MAX_BAKE_FRAMES, PORT
from .paths import ROOT
from .unreal_stream import UnrealStream, subject_name


def stream_rig(asset):
    return asset.get('stream_rig') if asset else None


class StreamWorkspace:
    def build_stream_page(self):
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(5, 5, 5, 5)
        column.addWidget(QLabel('Target'))
        self.stream_target = QComboBox()
        self.stream_target.addItems(['None', 'Blender', 'Unreal Live Link'])
        self.stream_target.setAccessibleName('Stream target')
        column.addWidget(self.stream_target)
        self.stream_port_controls = QWidget()
        port_controls = QVBoxLayout(self.stream_port_controls)
        port_controls.setContentsMargins(0, 0, 0, 0)
        port_controls.setSpacing(4)
        port_label = QLabel('Port')
        port_controls.addWidget(port_label)
        port_row = QHBoxLayout()
        self.stream_port = QSpinBox()
        self.stream_port.setRange(1024, 65535)
        self.stream_port.setValue(PORT)
        self.stream_port.setKeyboardTracking(False)
        self.stream_port.setAccessibleName('Blender stream port')
        self.stream_port.setToolTip('Match this port in Blender. Press Enter or leave the field to apply.')
        port_label.setBuddy(self.stream_port)
        port_row.addWidget(self.stream_port, 1)
        reset_port = QPushButton('Reset')
        reset_port.setAccessibleName('Reset Blender stream port')
        reset_port.setToolTip(f'Restore the default Blender stream port ({PORT}).')
        reset_port.clicked.connect(self.reset_stream_port)
        port_row.addWidget(reset_port)
        port_controls.addLayout(port_row)
        column.addWidget(self.stream_port_controls)
        self.stream_port_controls.hide()
        self.stream_unreal_controls = QWidget()
        unreal_controls = QVBoxLayout(self.stream_unreal_controls)
        unreal_controls.setContentsMargins(0, 0, 0, 0)
        note = QLabel('Body bones · offline motion\nIn Unreal: Live Link > Add Source > Message Bus Source > RGB MoCap.')
        note.setWordWrap(True)
        unreal_controls.addWidget(note)
        retry = QPushButton('Retry connection')
        retry.setToolTip('Restart the optional Live Link sender after setup or a connection error.')
        retry.clicked.connect(self.restart_unreal_stream)
        unreal_controls.addWidget(retry)
        self.stream_reference_button = QPushButton('Export Reference FBX...')
        self.stream_reference_button.setToolTip('Export the exact reference rig for the highlighted person at the current frame. Uses the Blender runtime.')
        self.stream_reference_button.clicked.connect(self.export_stream_reference)
        unreal_controls.addWidget(self.stream_reference_button)
        column.addWidget(self.stream_unreal_controls)
        self.stream_unreal_controls.hide()
        self.stream_people = QListWidget()
        self.stream_people.setObjectName('streamPeopleList')
        self.stream_people.setAccessibleName('People to stream')
        column.addWidget(self.stream_people, 1)
        self.stream_status = QLabel()
        self.stream_status.setWordWrap(True)
        self.stream_status.setObjectName('muted')
        column.addWidget(self.stream_status)

        self._stream_sender = None
        self._stream_port_error = None
        self._stream_scope = None
        self._stream_checked = set()
        self._stream_rows = None
        self._stream_frame = None
        self._stream_bake_actor = None
        self._stream_timer = QTimer(page)
        self._stream_timer.setTimerType(Qt.PreciseTimer)
        self._stream_timer.timeout.connect(self.update_stream)
        self.stream_target.currentIndexChanged.connect(self.change_stream_target)
        self.stream_port.editingFinished.connect(self.apply_stream_port)
        self.stream_people.itemChanged.connect(self.stream_person_changed)
        return page

    def stream_tab_changed(self, index):
        if not self._pear_active and (index == 1 or self._stream_sender is not None):
            self.update_stream()
            self._stream_timer.start(33 if self._stream_sender else 250)
        else:
            self._stream_timer.stop()

    def change_stream_target(self, index):
        blender = index == 1 and not self._pear_active
        unreal = index == 2 and not self._pear_active
        self.stream_port_controls.setVisible(blender)
        self.stream_unreal_controls.setVisible(unreal)
        self._stream_port_error = None
        if blender:
            if isinstance(self._stream_sender, UnrealStream):
                self.close_stream()
            self.apply_stream_port()
        elif unreal:
            self.restart_unreal_stream()
        else:
            self.close_stream()
            self.stream_tab_changed(self.people_tabs.currentIndex())

    def restart_unreal_stream(self):
        if self.stream_target.currentIndex() != 2 or self._pear_active:
            return
        self.close_stream()
        self._stream_port_error = None
        try:
            self._stream_sender = UnrealStream(ROOT)
        except (OSError, ValueError) as error:
            self._stream_port_error = str(error)
            self.log.append('Unreal Live Link: ' + str(error))
        self.stream_tab_changed(self.people_tabs.currentIndex())

    def reset_stream_port(self):
        self.stream_port.setValue(PORT)
        self.apply_stream_port()

    def apply_stream_port(self):
        if self.stream_target.currentIndex() != 1 or self._pear_active:
            return
        port = self.stream_port.value()
        active = self._stream_sender
        self._stream_port_error = None
        if active is None or active.port != port:
            try:
                sender = BoneStream(port)
            except OSError as error:
                if active is None:
                    self._stream_port_error = f'Cannot open port {port}. Choose another port or press Enter to retry.'
                else:
                    self.stream_port.setValue(active.port)
                    self._stream_port_error = f'Cannot open port {port}. Still listening on 127.0.0.1:{active.port}.'
                self.log.append(f'Blender stream could not open port {port}: {error}')
            else:
                self.close_stream()
                self._stream_sender = sender
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

    def stream_source_at_playhead(self, ident, name):
        view = self.motion_view
        if self.people_state:
            if self.scene_data is None:
                return None
            current = self.scene_data.take_at(ident, view.scene_time, self.project)
            if current is None:
                return None
            take, frame = current
            rig, frames, transform = stream_rig(take.target_asset), take.arrays.get('rig'), take.transform
        else:
            rig, frames = stream_rig(view.target_asset), view.target_frames
            frame, transform = view.index, None
        if rig is not None and frames is not None and 0 <= frame < len(frames):
            return ident, name, rig, frames, frame, transform
        return None

    def stream_sources_at_playhead(self, rows):
        for ident, name in rows:
            if ident in self._stream_checked:
                source = self.stream_source_at_playhead(ident, name)
                if source is not None:
                    yield source

    def stream_reference_problem(self, message):
        self.status.setText(message)
        self.log.append(message)
        QMessageBox.warning(self, 'Export Live Link Reference Rig', message)

    def export_stream_reference(self):
        if self._pear_active:
            self.stream_reference_problem('Reference export requires prepared offline motion. Leave PEAR and select an offline model.')
            return
        if self.busy:
            self.stream_reference_problem('Wait for the current job to finish before exporting a reference rig.')
            return
        from pathlib import Path
        from uuid import uuid4
        from .unreal_stream import rig_schema
        from .storage import atomic_json
        from .blend_support import partial
        item = self.stream_people.currentItem()
        if item is None and len(self._stream_rows or ()) == 1:
            self.stream_people.setCurrentRow(0)
            item = self.stream_people.currentItem()
        if item is None:
            self.stream_reference_problem('Select a person by clicking their name in the Stream list, then export the reference rig. Checking a box only enables streaming.')
            return
        ident = item.data(Qt.UserRole)
        row = next(((i, n) for i, n in self._stream_rows or () if i == ident), None)
        source = self.stream_source_at_playhead(*row) if row else None
        if source is None:
            self.stream_reference_problem('No prepared motion for this person at the current frame. Select a ready take and seek inside its frame range. If correction is pending, apply it first.')
            return
        try:
            # Small metadata only; FBX writing runs in a separate CPU worker.
            schema = rig_schema(source[2])
            subject = subject_name(self.stream_actor_id(ident), row[1])
            filename = subject + '_rig_' + schema['rig'][:8] + '.fbx'
            path, _ = QFileDialog.getSaveFileName(self, 'Export Live Link Reference Rig',
                                                 str(Path.home() / 'Documents' / filename), 'FBX files (*.fbx)')
            if not path:
                return
            if self.busy:
                raise ValueError('Wait for the current job before exporting a reference rig.')
            current = self.stream_source_at_playhead(*row)
            if current is None or current[2].key != schema['rig']:
                raise ValueError('The current reference rig changed. Select the take and export again.')
            target = Path(path).resolve()
            if target.suffix.lower() != '.fbx':
                target = target.with_suffix('.fbx')
                if target.exists():
                    raise ValueError('Choose the complete .fbx filename to confirm replacing an existing file.')
            project_root = self.people_root or self.project
            if project_root and target.is_relative_to(Path(project_root).resolve()):
                raise ValueError('Choose an export destination outside the source project folder.')
            request = ROOT / 'cache/unreal_stream' / ('reference-' + uuid4().hex + '.json')
            request.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(request, dict(target=str(target), schema=schema))
            self._fbx_export_cleanup = request, [partial(target, request)]
            self._motion_export_targets = [str(target)]
            self.start_jobs([('stream_reference', ['--output', str(request)])])
        except (OSError, ValueError, KeyError) as error:
            self.stream_reference_problem('Cannot export reference rig: ' + str(error))

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
        if isinstance(sender, BoneStream):
            self.stream_bake_commands(sender)
        rows = self.stream_people_with_motion()
        if rows != self._stream_rows:
            with QSignalBlocker(self.stream_people):
                self.stream_people.clear()
                for ident, name in rows:
                    item = QListWidgetItem(name, self.stream_people)
                    item.setData(Qt.UserRole, ident)
                    item.setToolTip('Unreal subject: ' + subject_name(self.stream_actor_id(ident), name))
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Checked if ident in self._stream_checked else Qt.Unchecked)
            self._stream_rows = rows

        self.stream_reference_button.setEnabled(bool(rows) and not getattr(self, 'busy', False))
        if sender is None:
            status = 'Select a target to stream.' if rows else 'Prepare motion to list people.'
        else:
            error = sender.error()
            if error:
                target = 'Unreal Live Link' if isinstance(sender, UnrealStream) else 'Blender stream'
                self.log.append(f'{target}: {error}')
                if isinstance(sender, UnrealStream):
                    self._stream_port_error = str(error)
            unreal = isinstance(sender, UnrealStream)
            connected = sender.connected
            sources = tuple(self.stream_sources_at_playhead(rows)) if connected or unreal else ()
            checked = tuple(sorted(self._stream_checked)) if connected or unreal else None
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
                    self.log.append(f'Bone stream could not read motion: {error}')
                    actors = ()
                sender.publish(actors, tuple(selected.values()) if checked is not None else None)
                self._stream_frame = key
            if unreal and not sender.ready:
                status = 'Starting Unreal Live Link sender…'
            elif unreal and not connected:
                status = 'Waiting for Unreal. Add Message Bus Source > RGB MoCap.'
            elif not connected:
                status = f'Waiting for Blender\n127.0.0.1:{sender.port}'
            elif not self._stream_checked:
                status = 'Connected. Check people to stream.'
            else:
                status = f'Connected · {len(sources)} people at this frame'
        status = self._stream_port_error or status
        if self.stream_status.text() != status:
            self.stream_status.setText(status)
