"""Source-motion export options; existing character exports remain separate."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox,
    QCheckBox, QLabel, QDialogButtonBox, QDoubleSpinBox, QWidget, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QPlainTextEdit, QAbstractItemView)

from .export_options import FBX_A_POSE_DEGREES
from .tooltips import tooltip_text


class SourceExportDialog(QDialog):
    def __init__(self,parent=None,*,show_take_scope=False,default_all_takes=False,validate=None,take_choices=None):
        super().__init__(parent);self.setWindowTitle('Export');self.setFixedSize(660,485)
        self.show_take_scope=show_take_scope
        self.validate=validate
        self.checked_exports=None
        self.take_choices=take_choices
        self.selected_take_paths=None
        self.model=parent.model_choice.currentData() if parent is not None and hasattr(parent,'model_choice') else None
        layout=QVBoxLayout(self);layout.setContentsMargins(18,18,18,18);layout.setSpacing(12)
        form=QFormLayout();form.setVerticalSpacing(12);layout.addLayout(form)
        self.format=QComboBox()
        self.format.addItem('FBX (.fbx)','fbx');self.format.addItem('BVH (.bvh)','bvh')
        self.format.addItem('Blender project + video (.blend)','blend')
        form.addRow('Format',self.format)
        self.take_scope=QComboBox()
        self.take_scope.addItem('Current take', 'current')
        self.take_scope.addItem('All takes for checked people', 'all')
        self.take_scope.addItem('Selected takes', 'selected')
        self.take_scope.setCurrentIndex(int(default_all_takes and show_take_scope))
        self.take_scope_label=QLabel('Export scope')
        self.take_controls=QWidget()
        range_layout=QHBoxLayout(self.take_controls)
        range_layout.setContentsMargins(0,0,0,0)
        range_layout.addWidget(self.take_scope,1)
        self.choose_takes_button=QPushButton('Choose takes…')
        self.choose_takes_button.clicked.connect(self.choose_takes)
        range_layout.addWidget(self.choose_takes_button)
        form.addRow(self.take_scope_label,self.take_controls)
        self.blend_layout = QComboBox()
        self.blend_layout.addItem('Separate Blender files', 'separate')
        self.blend_layout.addItem('One Blender file', 'combined')
        self.blend_layout.setToolTip('Combine the selected people and takes on one Blender timeline, or export one bundle per take.')
        self.blend_layout_label = QLabel('Blender output')
        form.addRow(self.blend_layout_label, self.blend_layout)
        self.profile=QComboBox();self.profile.addItem('Generic / Blender','generic')
        if self.model!='gemx':self.profile.addItem('3ds Max Biped','biped')
        self.profile.setToolTip('Generic / Blender keeps anatomical names and original rotation orders. 3ds Max Biped uses Character Studio names and uniform ZXY rotations. Both keep metre units and Y up.')
        self.profile_label=QLabel('BVH profile');form.addRow(self.profile_label,self.profile)
        self.fbx_rest_pose = QComboBox()
        self.fbx_rest_pose.addItem('T-pose', 't')
        self.fbx_rest_pose.addItem('A-pose', 'a')
        self.fbx_rest_pose.setToolTip(
            f'Choose the exported FBX reference pose. A-pose lowers the upper arms to '
            f'{FBX_A_POSE_DEGREES}° below horizontal. Motion keys keep their original poses.')
        self.fbx_rest_label = QLabel('Rest pose')
        form.addRow(self.fbx_rest_label, self.fbx_rest_pose)
        self.fbx_compatibility=QCheckBox('FBX Compatibility (Maya)')
        self.fbx_compatibility.setToolTip('Use the included Blender runtime to write an FBX for Maya and Blender. Unchecked uses the original FBX writer.')
        layout.addWidget(self.fbx_compatibility)
        self.whole_scene=QCheckBox('MAMMA whole scene: body meshes and all cameras');layout.addWidget(self.whole_scene)
        self.whole_scene.setToolTip('Export checked people, exact animated surfaces, reference skeletons and every calibrated camera/video. Keeps the original timeline. Mesh caches can be large.')
        self.tpose=QCheckBox('Add T-pose before motion');self.tpose.setChecked(False)
        layout.addWidget(self.tpose)
        self.place_at_origin=QCheckBox('Place at origin');layout.addWidget(self.place_at_origin)
        self.place_at_origin.setToolTip("Center each person's first exported frame on world X/Z. Height and travel are preserved. People are centered independently; their relative placement changes.")
        self.symmetric_rest=QCheckBox('Symmetrical rest pose');layout.addWidget(self.symmetric_rest)
        self.symmetric_rest.setToolTip('Use mirrored left/right rest bones with averaged paired lengths and a straight vertical spine. Animation rotations and timing are retained; changed proportions can move hands and feet. Exported files only.')
        self.animate_camera=QCheckBox('Export saved camera tracking');self.animate_camera.setChecked(True);layout.addWidget(self.animate_camera)
        self.background_refine=QCheckBox('Refine Camera Rotation');layout.addWidget(self.background_refine)
        from .camera_ui import BACKGROUND_HELP
        self.background_refine.setToolTip(BACKGROUND_HELP)
        if parent is not None and hasattr(parent,'camera_tools'):self.background_refine.setChecked(parent.camera_tools.background_options().get(self.model,False))
        self.translation_scale=QDoubleSpinBox();self.translation_scale.setRange(0,100);self.translation_scale.setDecimals(3);self.translation_scale.setValue(1);self.translation_scale.setSingleStep(.1)
        if parent is not None and hasattr(parent,'camera_tools'):self.translation_scale.setValue(parent.camera_tools.scale.value())
        self.translation_scale.setToolTip('Approximate saved camera movement. 0 fixes the camera center; 1 uses the estimate. This does not change the source animation.')
        self.scale_label=QLabel('Camera movement scale');form.addRow(self.scale_label,self.translation_scale)
        self.export_face=QCheckBox('Export Face (.txt)')
        self.export_face.setToolTip('Save frame numbers, source and export times, and ARKit-style expression values for each tracked person.')
        layout.addWidget(self.export_face)
        self.note=QLabel();self.note.setWordWrap(True);layout.addWidget(self.note)
        self.validation_error=QPlainTextEdit()
        self.validation_error.setReadOnly(True)
        self.validation_error.setFixedHeight(90)
        self.validation_error.setAccessibleName('Export blockers and fixes')
        self.validation_error.setStyleSheet('color: #f2b899;')
        self.validation_error.hide()
        layout.addWidget(self.validation_error)
        self.take_scope.currentIndexChanged.connect(self.update_format)
        layout.addStretch(1)
        buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText('Continue...');buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
        for signal in (self.format.currentIndexChanged,self.fbx_rest_pose.currentIndexChanged,self.blend_layout.currentIndexChanged,self.whole_scene.toggled,self.animate_camera.toggled):signal.connect(self.update_format)
        self.update_format()
    def update_format(self):
        fmt=self.format.currentData();blend=fmt=='blend';whole=blend and self.model=='mamma' and self.whole_scene.isChecked()
        self.take_scope_label.setVisible(self.show_take_scope)
        self.take_controls.setVisible(self.show_take_scope)
        self.choose_takes_button.setVisible(self.take_scope.currentData()=='selected')
        self.validation_error.hide()
        self.profile_label.setVisible(fmt=='bvh');self.profile.setVisible(fmt=='bvh')
        self.fbx_compatibility.setVisible(fmt=='fbx')
        self.blend_layout_label.setVisible(blend and not whole)
        self.blend_layout.setVisible(blend and not whole)
        self.fbx_rest_label.setVisible(fmt == 'fbx')
        self.fbx_rest_pose.setVisible(fmt == 'fbx')
        leading = self.fbx_rest_pose.currentText() if fmt == 'fbx' else 'T-pose'
        self.tpose.setText(f'Add {leading} before motion')
        self.tpose.setToolTip(f'Insert the {leading} at the first motion frame root position. The original motion starts one frame later.')
        self.whole_scene.setVisible(blend and self.model=='mamma');self.tpose.setEnabled(not whole);self.tpose.setVisible(not whole)
        for control in (self.place_at_origin,self.symmetric_rest):control.setVisible(not blend);control.setEnabled(not blend)
        self.animate_camera.setVisible(blend and not whole)
        camera=blend and not whole and self.model in ('flowhmr','exmr')
        self.background_refine.setVisible(camera);self.background_refine.setEnabled(self.animate_camera.isChecked())
        self.scale_label.setVisible(camera);self.translation_scale.setVisible(camera);self.translation_scale.setEnabled(self.animate_camera.isChecked())
        if self.model=='gemx':
            self.symmetric_rest.hide();self.symmetric_rest.setEnabled(False)
            if blend:self.tpose.hide();self.animate_camera.hide()
            self.note.setText('Exports the native SOMA skeleton, exact animated surface and synchronized video in a new Blender bundle folder. Keep the files together.' if blend else
                              'Exports the selected SOMA Native77 skeleton and motion at 30 fps. The optional first frame uses the selected FBX rest pose.' if fmt == 'fbx' else
                              'Exports the selected SOMA Native77 skeleton and motion at 30 fps. BVH uses metres and Y up. The optional first frame uses the native rest pose.')
            self.layout().activate()
            return
        text=('Creates a new folder containing the .blend, videos and mesh caches. Keep the folder together. Bodies are baked PC2 surfaces; skeletons are separate editable references. No T-pose is inserted.' if whole else
              'Puts the selected people and takes in one .blend on a shared timeline. Keep the synchronized videos beside it.' if blend and self.blend_layout.currentData() == 'combined' else
              'Creates a new folder containing the .blend and synchronized video. Keep both together. The video is a camera-view background, not a render background. Saved camera movement is approximate.' if blend else
              'Exports the selected source skeleton with the chosen FBX rest pose. The matching first animation frame is optional.' if fmt == 'fbx' else
              'Exports the selected source skeleton. BVH uses metres and Y up. The T-pose is an optional first animation frame.')
        self.note.setText(text)
        self.layout().activate()
    def choose_takes(self):
        if self.take_choices is None:
            return
        try:
            choices=self.take_choices()
        except (OSError, ValueError, KeyError) as error:
            self.validation_error.setPlainText(str(error))
            self.validation_error.show()
            return
        initial=self.selected_take_paths
        if initial is None:
            initial={getattr(self.parent(), 'project', None)}
        dialog=TakeSelectionDialog(choices, initial, self)
        if dialog.exec()==QDialog.Accepted:
            self.selected_take_paths=dialog.selected_paths()
            count=len(self.selected_take_paths)
            self.choose_takes_button.setText(f'Choose takes… ({count})')
            self.validation_error.hide()

    def accept(self):
        if self.validate is not None:
            try:
                self.checked_exports=self.validate(self.options())
            except (OSError, ValueError, KeyError) as error:
                from .storage import path_error_message
                self.validation_error.setPlainText(path_error_message(error))
                self.validation_error.show()
                return
        super().accept()

    def options(self):
        fmt=self.format.currentData();whole=fmt=='blend' and self.model=='mamma' and self.whole_scene.isChecked()
        scope=self.take_scope.currentData() if self.show_take_scope else ('all' if whole else 'current')
        selection=dict(take_scope=scope)
        if scope=='selected':
            selection['selected_takes']=self.selected_take_paths or ()
        if self.model=='gemx' and fmt=='blend':
            return dict(format='blend',animate_camera=False,blend_layout=self.blend_layout.currentData(),
                        export_face=self.export_face.isChecked(),**selection)
        value=dict(format='blend_scene' if whole else fmt,bvh_profile=self.profile.currentData() if fmt=='bvh' else 'generic',prepend_tpose=self.tpose.isChecked() and not whole,**selection)
        if fmt == 'fbx':
            value['fbx_compatibility'] = self.fbx_compatibility.isChecked()
            value['fbx_rest_pose'] = self.fbx_rest_pose.currentData()
        value.update(place_at_origin=self.place_at_origin.isChecked() and fmt in ('fbx','bvh'),symmetric_rest=self.model!='gemx' and self.symmetric_rest.isChecked() and fmt in ('fbx','bvh'))
        if fmt=='blend':value.update(blend_layout=self.blend_layout.currentData(),animate_camera=self.animate_camera.isChecked(),background_refine=self.background_refine.isChecked() and self.animate_camera.isChecked() and self.model in ('flowhmr','exmr'),camera_translation_scale=self.translation_scale.value())
        value['export_face']=self.export_face.isChecked()
        return value


class TakeSelectionDialog(QDialog):
    def __init__(self, choices, initial, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Choose takes')
        self.resize(620,380)
        self.choices=choices
        self.paths=[choice.project for choice in choices]
        layout=QVBoxLayout(self)
        self.table=QTableWidget(len(choices),3,self)
        self.table.setHorizontalHeaderLabels(['Person / Take','Frame range','Status'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1,QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2,QHeaderView.ResizeToContents)
        layout.addWidget(self.table)
        for row,choice in enumerate(choices):
            name=QTableWidgetItem(f'{choice.person_name} · Take {choice.number}')
            flags=Qt.ItemIsEnabled | Qt.ItemIsSelectable
            name.setFlags(flags | Qt.ItemIsUserCheckable if choice.source is not None else flags)
            if choice.source is not None:
                name.setCheckState(Qt.Checked if choice.project in initial else Qt.Unchecked)
            self.table.setItem(row,0,name)
            for column,text in ((1,choice.frames),(2,choice.status)):
                item=QTableWidgetItem(text)
                item.setFlags(flags)
                self.table.setItem(row,column,item)
            if choice.issue is not None:
                detail=choice.issue.report(choice.label)
                for column in range(self.table.columnCount()):
                    self.table.item(row,column).setToolTip(tooltip_text(detail))
        self.details=QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setFixedHeight(85)
        self.details.setAccessibleName('Selected take details')
        layout.addWidget(self.details)
        self.table.currentCellChanged.connect(self.update_details)
        if choices:
            first_problem=next((row for row,choice in enumerate(choices) if choice.issue is not None),0)
            self.table.setCurrentCell(first_problem,2)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        accept=buttons.button(QDialogButtonBox.Ok)
        accept.setEnabled(bool(self.selected_paths()))
        self.table.itemChanged.connect(lambda _item: accept.setEnabled(bool(self.selected_paths())))

    def update_details(self, row, *_):
        if row<0:
            self.details.clear()
            return
        choice=self.choices[row]
        self.details.setPlainText(choice.issue.report(choice.label) if choice.issue else
                                  f'{choice.label}\nReady to export.')

    def selected_paths(self):
        return {path for row,path in enumerate(self.paths)
                if self.table.item(row,0).checkState()==Qt.Checked}
