"""Export choices for an already baked character."""
from PySide6.QtWidgets import QDialog,QVBoxLayout,QFormLayout,QComboBox,QCheckBox,QLabel,QDialogButtonBox

from .face_export_ui import FaceExportControls
from .local_choices import apply_choices, control_values, remember, restore

CHOICES = 'character_export'


class CharacterExportDialog(QDialog):
    def __init__(self,kind,parent=None):
        super().__init__(parent)
        self.setWindowTitle('Export Character')
        self.setMinimumWidth(max(440,self.fontMetrics().horizontalAdvance('M')*32))
        layout=QVBoxLayout(self);form=QFormLayout();layout.addLayout(form)
        self.format=QComboBox();self.format.addItem('FBX (.fbx)','fbx')
        if kind=='pmx':self.format.addItem('VMD (.vmd)','vmd')
        form.addRow('Format',self.format)
        self.position_label = QLabel('Position')
        self.vmd_placement = QComboBox()
        self.vmd_placement.addItem('Keep original position', 'original')
        self.vmd_placement.addItem('Place at origin', 'origin')
        self.vmd_placement.addItem('In Place', 'in-place')
        self.position_label.setBuddy(self.vmd_placement)
        self.vmd_placement.setToolTip('Place at origin centers the first frame and keeps travel. In Place removes horizontal root travel. Both keep height and rotation.')
        form.addRow(self.position_label, self.vmd_placement)
        self.root_motion_label = QLabel('Root Motion Bone')
        self.vmd_root_bone = QComboBox()
        self.vmd_root_bone.addItem('Current root', 'root')
        self.vmd_root_bone.addItem('Center (センター)', 'center')
        self.root_motion_label.setBuddy(self.vmd_root_bone)
        self.vmd_root_bone.setToolTip(
            'Center puts the root translation and rotation on センター, leaving the master root free for placement in MMD. '
            'The PMX model must have a compatible Center hierarchy.'
        )
        form.addRow(self.root_motion_label, self.vmd_root_bone)
        self.mesh=QCheckBox('Include character mesh');self.mesh.setChecked(True)
        self.mesh.setToolTip('Checked: save the character mesh and skeleton animation. Unchecked: save only the skeleton and animation.')
        layout.addWidget(self.mesh)
        self.face_export = FaceExportControls()
        layout.addWidget(self.face_export)
        self.note=QLabel();self.note.setWordWrap(True);layout.addWidget(self.note)
        buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
        self.defaults_button = buttons.addButton('Defaults', QDialogButtonBox.ResetRole)
        self.defaults_button.clicked.connect(self.reset_defaults)
        self.initial_choices = control_values(self.remembered_controls())
        self.kept_choices=restore(self.remembered_controls(),CHOICES)
        self.format.currentIndexChanged.connect(self.update_format);self.update_format()

    def accept(self):
        try:remember(self.remembered_controls(),CHOICES,self.kept_choices)
        except (OSError,ValueError) as error:
            log=getattr(self.parent(),'log',None)
            if log is not None:log.append('Cannot save export options: '+str(error))
        super().accept()

    def reset_defaults(self):
        apply_choices(self.remembered_controls(), self.initial_choices)
        self.kept_choices.clear()

    def remembered_controls(self):
        return dict(format=self.format, include_mesh=self.mesh, vmd_placement=self.vmd_placement,
                    vmd_root_bone=self.vmd_root_bone, **self.face_export.controls())

    def update_format(self):
        vmd=self.format.currentData()=='vmd';self.mesh.setVisible(not vmd)
        self.position_label.setVisible(vmd)
        self.vmd_placement.setVisible(vmd)
        self.root_motion_label.setVisible(vmd)
        self.vmd_root_bone.setVisible(vmd)
        self.note.setText('Motion only, sampled at 30 fps for the same PMX model. Body and finger animation is saved; facial animation and physics are not generated.' if vmd else 'Save the retargeted motion. Each take is saved as a separate file.')
        self.layout().activate()
        self.fit_note()
        self.adjustSize()

    def fit_note(self):
        self.note.setMinimumHeight(self.note.heightForWidth(max(1,self.note.width())))

    def resizeEvent(self,event):
        super().resizeEvent(event)
        self.fit_note()

    def options(self):
        fmt = self.format.currentData()
        options = dict(format=fmt, include_mesh=fmt=='fbx' and self.mesh.isChecked(), **self.face_export.options())
        if fmt == 'vmd':
            options['vmd_placement'] = self.vmd_placement.currentData()
            options['vmd_root_bone'] = self.vmd_root_bone.currentData()
        return options
