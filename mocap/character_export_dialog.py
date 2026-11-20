"""Export choices for an already baked character."""
from PySide6.QtWidgets import QDialog,QVBoxLayout,QFormLayout,QComboBox,QCheckBox,QLabel,QDialogButtonBox

from .face_export_ui import FaceExportControls
from .local_choices import remember, restore

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
        self.mesh=QCheckBox('Include character mesh');self.mesh.setChecked(True)
        self.mesh.setToolTip('Checked: save the character mesh and skeleton animation. Unchecked: save only the skeleton and animation.')
        layout.addWidget(self.mesh)
        self.face_export = FaceExportControls()
        layout.addWidget(self.face_export)
        self.note=QLabel();self.note.setWordWrap(True);layout.addWidget(self.note)
        buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
        self.kept_choices=restore(self.remembered_controls(),CHOICES)
        self.format.currentIndexChanged.connect(self.update_format);self.update_format()

    def accept(self):
        try:remember(self.remembered_controls(),CHOICES,self.kept_choices)
        except (OSError,ValueError) as error:
            log=getattr(self.parent(),'log',None)
            if log is not None:log.append('Cannot save export options: '+str(error))
        super().accept()

    def remembered_controls(self):
        return dict(format=self.format,include_mesh=self.mesh,**self.face_export.controls())

    def update_format(self):
        vmd=self.format.currentData()=='vmd';self.mesh.setVisible(not vmd)
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
        return dict(format=self.format.currentData(),include_mesh=self.format.currentData()=='fbx' and self.mesh.isChecked(),
                    **self.face_export.options())
