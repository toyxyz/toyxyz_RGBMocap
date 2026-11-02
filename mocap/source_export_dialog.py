"""Source-motion export options; existing character exports remain separate."""
from PySide6.QtWidgets import QDialog,QVBoxLayout,QFormLayout,QComboBox,QCheckBox,QLabel,QDialogButtonBox,QDoubleSpinBox,QLayout

class SourceExportDialog(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent);self.setWindowTitle('Export');self.setMinimumWidth(480)
        self.model=parent.model_choice.currentData() if parent is not None and hasattr(parent,'model_choice') else None
        layout=QVBoxLayout(self);layout.setSizeConstraint(QLayout.SetMinimumSize);form=QFormLayout();layout.addLayout(form)
        self.format=QComboBox();self.format.addItem('FBX (.fbx)','fbx');self.format.addItem('BVH (.bvh)','bvh');self.format.addItem('Blender project + video (.blend)','blend')
        form.addRow('Format',self.format)
        self.profile=QComboBox();self.profile.addItem('Generic / Blender','generic');self.profile.addItem('3ds Max Biped','biped')
        self.profile.setToolTip('Generic / Blender keeps anatomical names and original rotation orders. 3ds Max Biped uses Character Studio names and uniform ZXY rotations. Both keep metre units and Y up.')
        self.profile_label=QLabel('BVH profile');form.addRow(self.profile_label,self.profile)
        self.whole_scene=QCheckBox('MAMMA whole scene: body meshes and all cameras');layout.addWidget(self.whole_scene)
        self.whole_scene.setToolTip('Export checked people, exact animated surfaces, reference skeletons and every calibrated camera/video. Keeps the original timeline. Mesh caches can be large.')
        self.tpose=QCheckBox('Add T-pose before motion');self.tpose.setChecked(False)
        self.tpose.setToolTip('Insert a T-pose at the first motion frame root position. Original frames shift by one; their poses and timing intervals stay unchanged.')
        layout.addWidget(self.tpose)
        self.animate_camera=QCheckBox('Export saved camera tracking');self.animate_camera.setChecked(True);layout.addWidget(self.animate_camera)
        self.background_refine=QCheckBox('Refine Camera Rotation');layout.addWidget(self.background_refine)
        from .camera_ui import BACKGROUND_HELP
        self.background_refine.setToolTip(BACKGROUND_HELP)
        if parent is not None and hasattr(parent,'camera_tools'):self.background_refine.setChecked(parent.camera_tools.background_options().get(self.model,False))
        self.translation_scale=QDoubleSpinBox();self.translation_scale.setRange(0,100);self.translation_scale.setDecimals(3);self.translation_scale.setValue(1);self.translation_scale.setSingleStep(.1)
        if parent is not None and hasattr(parent,'camera_tools'):self.translation_scale.setValue(parent.camera_tools.scale.value())
        self.translation_scale.setToolTip('Approximate saved camera movement. 0 fixes the camera center; 1 uses the estimate. This does not change the source animation.')
        self.scale_label=QLabel('Camera movement scale');form.addRow(self.scale_label,self.translation_scale)
        self.note=QLabel();self.note.setWordWrap(True);layout.addWidget(self.note)
        buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText('Continue...');buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
        for signal in (self.format.currentIndexChanged,self.whole_scene.toggled,self.animate_camera.toggled):signal.connect(self.update_format)
        self.update_format()
    def update_format(self):
        fmt=self.format.currentData();blend=fmt=='blend';whole=blend and self.model=='mamma' and self.whole_scene.isChecked()
        self.profile_label.setVisible(fmt=='bvh');self.profile.setVisible(fmt=='bvh')
        self.whole_scene.setVisible(blend and self.model=='mamma');self.tpose.setEnabled(not whole);self.tpose.setVisible(not whole)
        self.animate_camera.setVisible(blend and not whole)
        camera=blend and not whole and self.model in ('flowhmr','exmr')
        self.background_refine.setVisible(camera);self.background_refine.setEnabled(self.animate_camera.isChecked())
        self.scale_label.setVisible(camera);self.translation_scale.setVisible(camera);self.translation_scale.setEnabled(self.animate_camera.isChecked())
        text=('Creates a new folder containing the .blend, videos and mesh caches. Keep the folder together. Bodies are baked PC2 surfaces; skeletons are separate editable references. No T-pose is inserted.' if whole else
              'Creates a new folder containing the .blend and synchronized video. Keep both together. The video is a camera-view background, not a render background. Saved camera movement is approximate.' if blend else
              'Exports the selected source skeleton. BVH uses metres and Y up. The T-pose is an extra animation frame; it does not replace the character bind pose.')
        self.note.setText(text)
        self.layout().activate();self.adjustSize()
    def options(self):
        fmt=self.format.currentData();whole=fmt=='blend' and self.model=='mamma' and self.whole_scene.isChecked()
        value=dict(format='blend_scene' if whole else fmt,bvh_profile=self.profile.currentData() if fmt=='bvh' else 'generic',prepend_tpose=self.tpose.isChecked() and not whole)
        if fmt=='blend':value.update(animate_camera=self.animate_camera.isChecked(),background_refine=self.background_refine.isChecked() and self.animate_camera.isChecked() and self.model in ('flowhmr','exmr'),camera_translation_scale=self.translation_scale.value())
        return value
