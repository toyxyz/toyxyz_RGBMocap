"""Opt-in VMC controls shared by live and offline motion streams."""
from PySide6.QtWidgets import QPushButton, QGroupBox, QLabel, QLineEdit, QSpinBox, QFormLayout


class PearVmcWorkspace:
    def build_pear_vmc(self, layout):
        self.pear_vmc_sender = None
        self.pear_vmc_closing = []
        group = QGroupBox('VMC Output (Experimental)'); form = QFormLayout(group)
        self.pear_vmc_host = QLineEdit('127.0.0.1')
        self.pear_vmc_port = QSpinBox(); self.pear_vmc_port.setRange(1, 65535)
        self.pear_vmc_port.setValue(39539)
        self.pear_vmc_enabled = QPushButton('Start VMC Output')
        self.pear_vmc_enabled.setCheckable(True)
        self.pear_vmc_enabled.setToolTip('Start or stop sending motion to the destination IP and port.')
        self.pear_vmc_status = QLabel('Output off.'); self.pear_vmc_status.setWordWrap(True)
        form.addRow('Destination IP', self.pear_vmc_host)
        form.addRow('UDP Port', self.pear_vmc_port)
        form.addRow(self.pear_vmc_enabled); form.addRow(self.pear_vmc_status)
        group.setToolTip('Send source-local body/finger bones to Operator or a VMC receiver. '
                         'Metres, Unity axes. Avatar bind-pose retargeting may be required. '
                         'Does not start tracking or record motion.')
        self.pear_vmc_enabled.toggled.connect(self.pear_vmc_toggle)
        layout.addWidget(group)

    def pear_vmc_toggle(self, enabled):
        self.pear_vmc_enabled.setText('Stop VMC Output' if enabled else 'Start VMC Output')
        if not enabled:
            sender = self.pear_vmc_sender; self.pear_vmc_sender = None
            if sender is not None:
                sender.close()
                self.pear_vmc_closing.append(sender)
                self.pear_vmc_enabled.setEnabled(False)
                self.pear_vmc_enabled.setText('Stopping VMC Output...')
            self.pear_vmc_host.setEnabled(True); self.pear_vmc_port.setEnabled(True)
            self.pear_vmc_status.setText('Output off.'); return
        try:
            from .pear_vmc import Sender
            self.pear_vmc_sender = Sender(self.pear_vmc_host.text(), self.pear_vmc_port.value())
        except (OSError, ValueError) as exc:
            self.pear_vmc_enabled.blockSignals(True); self.pear_vmc_enabled.setChecked(False)
            self.pear_vmc_enabled.blockSignals(False)
            self.pear_vmc_enabled.setText('Start VMC Output')
            self.pear_vmc_status.setText('VMC: '+str(exc)); return
        self.pear_vmc_host.setEnabled(False); self.pear_vmc_port.setEnabled(False)
        self.pear_vmc_status.setText('Waiting for poses. Select a person, or start PEAR live tracking.')

    def pear_vmc_stop(self):
        if hasattr(self, 'pear_vmc_enabled'):
            self.pear_vmc_enabled.setChecked(False)

    def pear_vmc_publish(self, meta=None):
        sender = getattr(self, 'pear_vmc_sender', None)
        if sender is None:
            return
        pose = self.motion_view.live_data
        sender.submit(pose if meta and meta.get('type') == 'frame' else None,
                      getattr(self.motion_view, 'live_parents', None),
                      bool(meta and meta.get('aligned')))

    def pear_vmc_update_status(self):
        # An old socket must finish its final OK=0 before a replacement starts.
        self.pear_vmc_closing = [s for s in self.pear_vmc_closing if s.thread.is_alive()]
        self.pear_vmc_enabled.setEnabled(not self.pear_vmc_closing)
        self.pear_vmc_enabled.setText('Stopping VMC Output...' if self.pear_vmc_closing else
            'Stop VMC Output' if self.pear_vmc_enabled.isChecked() else 'Start VMC Output')
        sender = getattr(self, 'pear_vmc_sender', None)
        if sender is not None:
            self.pear_vmc_status.setText('VMC: '+sender.error if sender.error else
                f'Sent {sender.frames} poses - '+('tracking' if sender.tracking else 'waiting / tracking unavailable'))
