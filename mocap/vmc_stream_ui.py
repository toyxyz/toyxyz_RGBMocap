"""Shared Stream-tab VMC control for accepted live and prepared offline poses."""
import time
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import QWidget,QVBoxLayout,QListWidgetItem
from .pear_vmc_ui import PearVmcWorkspace


class VmcStreamWorkspace(PearVmcWorkspace):
    def build_vmc_stream(self, column):
        self.stream_vmc_controls = QWidget()
        layout = QVBoxLayout(self.stream_vmc_controls); layout.setContentsMargins(0,0,0,0)
        self.build_pear_vmc(layout)
        column.addWidget(self.stream_vmc_controls); self.stream_vmc_controls.hide()
        self._vmc_adapter = None; self._vmc_frame_key = None; self._vmc_sent_at = 0.

    def pear_vmc_toggle(self, enabled):
        super().pear_vmc_toggle(enabled)
        self._vmc_frame_key = None
        if hasattr(self,'people_tabs'):
            self.stream_tab_changed(self.people_tabs.currentIndex())

    def pear_vmc_publish(self, meta=None):
        # The PEAR hook only submits already-accepted preview poses; no inference.
        if not self._pear_active or self.stream_target.currentText() != 'VMC':
            return
        super().pear_vmc_publish(meta)

    def vmc_person_changed(self, item):
        ident = item.data(Qt.UserRole)
        if item.checkState() == Qt.Checked:
            self._stream_checked = {ident}
            self._stream_rows = None
        else:
            self._stream_checked.discard(ident)
        self._vmc_frame_key = None
        self.update_stream()

    def update_vmc_stream(self):
        self.pear_vmc_update_status()
        live = self._pear_active
        self.stream_people.setVisible(not live)
        scope = ('pear-live',) if live else (self.people_root or self.project,self.model_choice.currentData())
        if scope != self._stream_scope:
            self._stream_scope = scope; self._stream_checked.clear(); self._stream_rows = None
            self._vmc_frame_key = None; self._vmc_adapter = None
            if self.pear_vmc_sender is not None:
                self.pear_vmc_sender.submit(None)
        if live:
            self.stream_status.setText('Live PEAR: send the current tracked person. Start Live Tracking separately.')
            return

        rows = self.stream_people_with_motion()
        # VMC has no actor ID. Never blend multiple people on a single port.
        if len(self._stream_checked)>1:
            self._stream_checked = {next(i for i,_ in rows if i in self._stream_checked)} if any(i in self._stream_checked for i,_ in rows) else set()
            self._stream_rows = None
        if rows != self._stream_rows:
            self._stream_checked.intersection_update(i for i,_ in rows)
            with QSignalBlocker(self.stream_people):
                self.stream_people.clear()
                for ident,name in rows:
                    item=QListWidgetItem(name,self.stream_people); item.setData(Qt.UserRole,ident)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Checked if ident in self._stream_checked else Qt.Unchecked)
            self._stream_rows = rows
        sender = self.pear_vmc_sender
        if sender is None:
            self.stream_status.setText('Select one person and start VMC Output.' if rows else 'Prepare motion to list people.')
            return
        sources=tuple(self.stream_sources_at_playhead(rows))
        key=(scope,tuple(self._stream_checked),tuple((i,rig.key,id(frames),frame,id(transform))
                for i,name,rig,frames,frame,transform in sources))
        now=time.monotonic()
        if not sources:
            if key!=self._vmc_frame_key:
                sender.submit(None); self._vmc_frame_key=key
            self.stream_status.setText('Select one person to stream.' if not self._stream_checked else 'No prepared motion for this person at the current frame.')
            return
        ident,name,rig,frames,frame,transform=sources[0]
        if key!=self._vmc_frame_key or now-self._vmc_sent_at>=.25:
            try:
                adapter_key=(scope,rig.key)
                if self._vmc_adapter is None or self._vmc_adapter[0]!=adapter_key:
                    from .vmc_rig import OfflineRig
                    self._vmc_adapter=(adapter_key,OfflineRig(rig))
                sender.submit_offline(self._vmc_adapter[1],frames[frame],transform)
            except (OSError,ValueError,KeyError,IndexError,TypeError) as exc:
                sender.submit(None); self.stream_status.setText('VMC: '+str(exc))
                self._vmc_frame_key=None; self._vmc_sent_at=now
                return
            self._vmc_frame_key=key; self._vmc_sent_at=now
        self.stream_status.setText(f'{name}: timeline frame {frame+1}. Play or scrub to stream motion.')
