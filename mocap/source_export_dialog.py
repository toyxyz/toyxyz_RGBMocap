"""Options for exporting the displayed source skeleton."""
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QComboBox,
                              QCheckBox, QLabel, QDialogButtonBox, QDoubleSpinBox)


class SourceExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Export Motion')
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)
        self.format = QComboBox()
        self.format.addItem('FBX (.fbx)', 'fbx')
        self.format.addItem('Blender project + video (.blend)', 'blend')
        self.format.addItem('Blender MAMMA scene: meshes + all cameras (.blend)', 'blend_scene')
        self.format.addItem('BVH — 3ds Max Biped (.bvh)', 'bvh_biped')
        self.format.addItem('BVH — Generic / Blender (.bvh)', 'bvh')
        form.addRow('Format', self.format)
        self.tpose = QCheckBox('Insert T-pose at frame 0')
        self.tpose.setStyleSheet("""
            QCheckBox { background:#35465a; color:#ffffff; border:1px solid #7399bf;
                        border-radius:6px; padding:12px; spacing:12px; font-weight:600; }
            QCheckBox::indicator { width:22px; height:22px; border:2px solid #d6e6f5;
                                   border-radius:3px; background:#18222e; }
            QCheckBox::indicator:checked { background:#38bdf8; border:2px solid #ffffff; }
            QCheckBox:focus { border:2px solid #38bdf8; }
        """)
        self.tpose.toggled.connect(lambda checked: self.tpose.setText(
            'Insert T-pose at frame 0 — ON' if checked else 'Insert T-pose at frame 0 — OFF'))
        layout.addWidget(self.tpose)
        note = QLabel('Adds one reference frame before the complete motion. '
                      'Original frame 0 becomes frame 1. Importers may number '
                      'the first frame as 1. BVH units: metres; up axis: Y.')
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText('Save…')
        destination_note = QLabel('Next: choose the output filename and folder. '
                                  'For multiple takes, choose an output folder.')
        destination_note.setWordWrap(True)
        layout.addWidget(destination_note)
        self.blend_note=QLabel('Blender: creates a new subfolder with the project and video.mp4. Video is a camera-view background, not a render background. Uses saved ExMR/FlowHMR camera rotation when enabled; no new tracking or metric camera translation is calculated.')
        self.blend_note.setWordWrap(True)
        self.blend_note.setVisible(False)
        layout.addWidget(self.blend_note)
        self.format.currentIndexChanged.connect(lambda: self.blend_note.setVisible(self.format.currentData()=='blend'))
        self.animate_camera=QCheckBox('Export saved camera tracking (ExMR / FlowHMR)')
        self.animate_camera.setChecked(True)
        self.animate_camera.setVisible(False)
        layout.addWidget(self.animate_camera)
        self.format.currentIndexChanged.connect(lambda: self.animate_camera.setVisible(self.format.currentData()=='blend'))
        self.background_refine=QCheckBox('Refine camera rotation from background (experimental)')
        self.background_refine.setToolTip('Tracks visible background and accepts only validated rotation improvements. Does not estimate metric camera translation or repair body motion.')
        self.background_refine.setVisible(False);layout.addWidget(self.background_refine)
        self.format.currentIndexChanged.connect(lambda: self.background_refine.setVisible(self.format.currentData()=='blend'))
        self.background_refine.toggled.connect(lambda enabled:self.animate_camera.setChecked(True) if enabled else None)
        self.animate_camera.toggled.connect(lambda enabled:self.background_refine.setChecked(False) if not enabled else None)
        self.camera_translation=QCheckBox('Experimental camera translation — FlowHMR only')
        self.camera_translation.setVisible(False)
        layout.addWidget(self.camera_translation)
        self.translation_scale=QDoubleSpinBox();self.translation_scale.setRange(0,100);self.translation_scale.setDecimals(3);self.translation_scale.setValue(1);self.translation_scale.setSingleStep(.1)
        self.translation_scale.setPrefix('Translation scale: ');self.translation_scale.setEnabled(False);self.translation_scale.setVisible(False)
        self.translation_scale.setToolTip('0 keeps the camera position fixed. 1 uses native VO units; adjust to match the video. Existing bone animation is unchanged.')
        layout.addWidget(self.translation_scale)
        self.camera_translation.toggled.connect(self.translation_scale.setEnabled)
        self.camera_translation.toggled.connect(lambda enabled: self.animate_camera.setChecked(True) if enabled else None)
        self.animate_camera.toggled.connect(lambda enabled: self.camera_translation.setChecked(False) if not enabled else None)
        self.format.currentIndexChanged.connect(lambda: self.camera_translation.setVisible(self.format.currentData()=='blend'))
        self.format.currentIndexChanged.connect(lambda: self.translation_scale.setVisible(self.format.currentData()=='blend'))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.scene_note=QLabel('MAMMA scene: checked people, exact animated body meshes, reference skeletons and all calibrated cameras with videos in one folder. Keeps the original timeline; no inserted T-pose. Mesh caches can be large.')
        self.scene_note.setWordWrap(True);layout.insertWidget(layout.count()-1,self.scene_note)
        def scene_mode():
            enabled=self.format.currentData()=='blend_scene'
            self.scene_note.setVisible(enabled);self.tpose.setEnabled(not enabled)
        self.format.currentIndexChanged.connect(scene_mode);scene_mode()
        self.setMinimumWidth(420)

    def options(self):
        return dict(format='bvh' if self.format.currentData()=='bvh_biped' else self.format.currentData(),
                    bvh_profile='biped' if self.format.currentData()=='bvh_biped' else 'generic',
                    background_refine=self.background_refine.isChecked(),prepend_tpose=self.tpose.isChecked(),animate_camera=self.animate_camera.isChecked(),camera_translation=self.camera_translation.isChecked(),camera_translation_scale=self.translation_scale.value())
