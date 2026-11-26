"""Switch cached model results inside one project without running inference."""
import json
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QHeaderView,
    QLabel, QTreeWidget, QTreeWidgetItem, QVBoxLayout)
from . import project_workspace as workspace
from .exmr_options import DEFAULTS as EXMR_DEFAULTS
from .local_choices import offered
from .theme import set_appearance_style, ui_px
from .project_storage import manifest
from .storage import atomic_json
from .video import select_preview
from .vggt_camera_opt import DEFAULT_ENABLED, enabled as vggt_opt_enabled

DEFAULTS = 'default_settings'
# Stored keys map to the editor controls. Checkboxes supply their own labels.
DEFAULT_SETTINGS = {
    'gvhmr_options': ('GVHMR', {
        'htd': ('htd_refine', None),
        'footmr': ('footmr_refine', None),
        'handflow': ('handflow_refine', None),
    }),
    'flow_options': ('FlowHMR', {
        'camera_interval': ('flow_sampling', 'Camera / Sampling'),
        'vggt_opt': ('flow_vggt_opt', None),
        'fast_sam': ('fast_sam', None),
        'sam_batch4': ('flow_batch4', None),
        'sam_gpu_crop': ('flow_gpu_crop', None),
        'sam_hand_stride4': ('flow_hand_stride4', None),
        'hamer': ('flow_hamer', None),
        'refine_camera_rotation': ('flowhmr_background', None),
    }),
    'exmr_options': ('ExMR', {
        'static_camera': ('exmr_camera', 'Camera'),
        'smoothing': ('exmr_smoothing', None),
        'refine_camera_rotation': ('exmr_background', None),
    }),
    'prompthmr_options': ('PromptHMR', {
        'camera': ('prompthmr_camera', 'Camera'),
        'interaction': ('prompthmr_interaction', None),
    }),
    'mamma_options': ('MAMMA', {
        'vggt_opt': ('mamma_vggt_opt', None),
    }),
    'correction': ('Motion Correction', {
        'mode': ('correction_mode', 'Correction Mode'),
        'preset': ('correction_preset', 'Lock Preset'),
        'foot_lock': ('foot_lock', None),
        'hand_lock': ('hand_lock', None),
    }),
    'processing_options': ('Processing Resources', {
        'resource_mode': ('compute_mode', 'Resource Mode'),
    }),
}
PROJECT_DEFAULT_GROUPS = ('flow_options', 'exmr_options', 'prompthmr_options', 'mamma_options', 'correction')
DEFAULT_RESOURCE_MODE = 'auto'
GVHMR_DEFAULTS = dict(htd=True, footmr=True, handflow=True)
FLOW_DEFAULTS = dict(camera_interval=0, vggt_opt=DEFAULT_ENABLED, fast_sam=True, sam_batch4=True, sam_gpu_crop=True,
                     sam_hand_stride4=True, hamer=False)


class DefaultSettingsDialog(QDialog):
    def __init__(self, parent, groups, *, reset):
        super().__init__(parent)
        self.setWindowTitle('Reset Default Settings' if reset else 'Save Settings as Default')
        prompt = ('Restore the built-in starting settings?' if reset else
                  'Save these options as your starting settings?')
        scope = ('These values apply the next time you open a video or launch the app. '
                 'The current video keeps its settings.')
        layout = QVBoxLayout(self)
        for text in (prompt, scope):
            label = QLabel(text)
            label.setWordWrap(True)
            layout.addWidget(label)

        values = QTreeWidget()
        values.setHeaderLabels(('Setting', 'Built-in Value' if reset else 'Value to Save'))
        values.setAccessibleName('Starting settings and values')
        values.setSelectionMode(QTreeWidget.SelectionMode.NoSelection)
        values.setRootIsDecorated(False)
        values.setItemsExpandable(False)
        set_appearance_style(values, 'QTreeWidget::item { padding: 0 8px; }')
        values.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        values.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        for title, rows in groups:
            group = QTreeWidgetItem(values, [title])
            font = group.font(0)
            font.setBold(True)
            group.setFont(0, font)
            for label, value in rows:
                QTreeWidgetItem(group, [label, value])
        values.expandAll()
        layout.addWidget(values, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        cancel = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        cancel.setDefault(True)
        cancel.setFocus()
        area = parent.screen().availableGeometry()
        self.resize(min(ui_px(600), area.width() - 32), min(ui_px(560), area.height() - 64))


class ModelWorkspace:
    def selected_flow_options(self):
        return dict(camera_interval=self.flow_sampling.currentData(), vggt_opt=self.flow_vggt_opt.isChecked(),
                    fast_sam=self.fast_sam.isChecked(),
                    sam_batch4=self.flow_batch4.isChecked(), sam_gpu_crop=self.flow_gpu_crop.isChecked(),
                    sam_hand_stride4=self.flow_hand_stride4.isChecked(),hamer=self.flow_hamer.isChecked())

    def workspace_settings(self):
        return dict(result=self.active_result,correction=self.selected_correction_options(),
            mamma_options=self.mamma_options(),
            prompthmr_options=self.prompthmr_options(),
            flow_options=self.selected_flow_options(),
            exmr_options={'static_camera':self.exmr_camera.currentData(),'smoothing':'global' if self.exmr_smoothing.isChecked() else 'none'})

    def restore_workspace_settings(self,settings):
        from .motion_tools import result_options,has_handflow,result_kind
        gvhmr=self.model_choice.currentData()=='gvhmr'
        kind=settings.get('result',result_kind(gvhmr,gvhmr,True,gvhmr))
        for widget,value in ((self.htd_refine,result_options(kind)[0]),(self.footmr_refine,result_options(kind)[1]),
                             (self.handflow_refine,has_handflow(kind)),(self.motion_correction,kind.startswith('corrected_'))):
            widget.blockSignals(True);widget.setChecked(value);widget.blockSignals(False)
        self.restore_correction_options(settings.get('correction',{}))
        self.restore_workflow_options(settings)
        self.active_result=self.selected_result()
        self.restore_mamma_options(settings.get('mamma_options',{}))
        self.restore_prompthmr_options(settings.get('prompthmr_options',{}))

    def restore_workflow_options(self, settings):
        flow = {**FLOW_DEFAULTS, **settings.get('flow_options', {})}
        self.flow_hamer.setChecked(False)
        self.flow_sampling.setCurrentIndex(max(0, self.flow_sampling.findData(flow['camera_interval'])))
        self.flow_vggt_opt.setChecked(vggt_opt_enabled(flow))
        self.fast_sam.setChecked(flow['fast_sam'])
        self.flow_batch4.setChecked(flow['sam_batch4'])
        self.flow_gpu_crop.setChecked(flow['sam_gpu_crop'])
        self.flow_hand_stride4.setChecked(flow['sam_hand_stride4'])
        self.flow_hamer.setChecked(flow['hamer'])
        exmr = {**EXMR_DEFAULTS, **settings.get('exmr_options', {})}
        self.exmr_camera.setCurrentIndex(self.exmr_camera.findData(exmr['static_camera']))
        self.exmr_smoothing.setChecked(exmr['smoothing'] == 'global')

    def default_option_text(self, control, key, value):
        if isinstance(control, QComboBox):
            index = offered(control, value)
            return control.itemText(index) if index >= 0 else None
        if key == 'smoothing':
            if type(value) is not str or value not in ('global', 'none'):
                return None
            value = value == 'global'
        if type(value) is bool:
            return 'On' if value else 'Off'
        return None

    def confirm_default_settings(self, settings, *, reset):
        groups = []
        for group, (title, fields) in DEFAULT_SETTINGS.items():
            rows = []
            for key, (name, label) in fields.items():
                control = getattr(self, name)
                value = self.default_option_text(control, key, settings[group][key])
                rows.append((label if label is not None else control.text(), value or 'Unavailable'))
            groups.append((title, rows))
        groups.append(('Face Tracking', [(self.track_face.text(), 'On' if settings['track_face'] else 'Off')]))
        return DefaultSettingsDialog(self, groups, reset=reset).exec() == QDialog.DialogCode.Accepted

    def save_default_settings(self):
        from .resource_safety import update

        settings = self.workspace_settings()
        settings['prompthmr_options']['camera'] = self.prompthmr_camera.currentData()
        if self._selected_tool == 'gvhmr':
            htd, footmr, handflow = (control.isChecked() for control in
                                     (self.htd_refine, self.footmr_refine, self.handflow_refine))
        else:
            htd, footmr, _, handflow = self._tool_refinement.get('gvhmr', (True, True, True, True))
        settings['gvhmr_options'] = dict(htd=htd, footmr=footmr, handflow=handflow)
        camera_background = self.camera_tools.background_options()
        settings['flow_options']['refine_camera_rotation'] = camera_background['flowhmr']
        settings['exmr_options']['refine_camera_rotation'] = camera_background['exmr']
        settings['processing_options'] = {'resource_mode': self.compute_mode.currentData()}
        value = {group: {key: settings[group][key] for key in fields}
                 for group, (_, fields) in DEFAULT_SETTINGS.items()}
        value['track_face'] = self.track_face.isChecked()
        if not self.confirm_default_settings(value, reset=False):
            return
        try:
            update({DEFAULTS: value})
        except (OSError, ValueError) as error:
            self.status.setText('Could not save default settings. See Console.')
            self.log.append('Cannot save default settings: ' + str(error))
            return
        self.status.setText('Default settings saved for new videos.')
        self.log.append('Default settings saved. New videos, the next start and models a project has no settings for use them.')

    def reset_default_settings(self):
        from .correction_settings import correction_options
        from .prompthmr_pipeline import options as prompthmr_options
        from .resource_safety import update

        built_in = dict(gvhmr_options=GVHMR_DEFAULTS,
                        flow_options={**FLOW_DEFAULTS, 'refine_camera_rotation': False},
                        exmr_options={**EXMR_DEFAULTS, 'refine_camera_rotation': False},
                        mamma_options={'vggt_opt': DEFAULT_ENABLED},
                        prompthmr_options={'camera': 'static', **prompthmr_options({})},
                        correction=correction_options(), track_face=False,
                        processing_options={'resource_mode': DEFAULT_RESOURCE_MODE})
        if not self.confirm_default_settings(built_in, reset=True):
            return
        try:
            update({DEFAULTS: None})
        except (OSError, ValueError) as error:
            self.status.setText('Could not reset default settings. See Console.')
            self.log.append('Cannot reset default settings: ' + str(error))
            return
        self._reset_starting_settings = True
        self.status.setText('Default settings reset for new videos.')
        self.log.append('Default settings reset. New videos and the next start use the built-in settings.')

    def default_groups(self):
        """Saved defaults by group, only values that pass their check, plus Track Face; None when nothing is saved."""
        from .local_choices import saved
        start = saved(DEFAULTS)
        if not isinstance(start, dict):
            return None
        groups = {}
        for group, (_, fields) in DEFAULT_SETTINGS.items():
            values = start.get(group)
            groups[group] = {}
            if isinstance(values, dict):
                for key, (name, _) in fields.items():
                    if key in values and self.default_option_text(getattr(self, name), key, values[key]) is not None:
                        groups[group][key] = values[key]
        return dict(groups,track_face=start.get('track_face') is True)

    def with_default_groups(self, settings, model):
        defaults = self.default_groups()
        if defaults is None:
            return settings
        # Saved project groups and preview selections take precedence.
        settings = {**settings, **{group: defaults[group] for group in PROJECT_DEFAULT_GROUPS
                                   if not isinstance(settings.get(group), dict)}}
        if model == 'gvhmr' and 'result' not in settings and defaults['gvhmr_options']:
            from .motion_tools import result_kind
            options = {**GVHMR_DEFAULTS, **defaults['gvhmr_options']}
            settings['result'] = result_kind(options['htd'], options['footmr'], True, options['handflow'])
        return settings

    def saved_track_face(self,view):
        # A new project's view is empty until the app first saves it.
        if 'track_face' in view:return view['track_face']
        defaults=self.default_groups()
        return defaults is not None and defaults['track_face']

    def apply_default_settings(self):
        # New work. A missing or unusable value keeps restore_workspace_settings' built-in value for that setting.
        defaults=self.default_groups()
        reset = self._reset_starting_settings
        self._reset_starting_settings = False
        if defaults is None:
            if reset:
                # These settings otherwise carry over between clips in the same session.
                self.compute_mode.setCurrentIndex(self.compute_mode.findData(DEFAULT_RESOURCE_MODE))
                self.camera_tools.restore_background({})
                self._tool_refinement['gvhmr'] = (GVHMR_DEFAULTS['htd'], GVHMR_DEFAULTS['footmr'],
                                                  True, GVHMR_DEFAULTS['handflow'])
                self.restore_prompthmr_options({'people': self.prompthmr_people.value()})
                self.mamma_vggt_opt.setChecked(DEFAULT_ENABLED)
                self.restore_track_face(False)
            return
        from .motion_tools import result_kind

        mode = defaults['processing_options'].get('resource_mode', DEFAULT_RESOURCE_MODE)
        self.compute_mode.setCurrentIndex(self.compute_mode.findData(mode))
        self.camera_tools.restore_background({
            'flowhmr': defaults['flow_options'].get('refine_camera_rotation', False),
            'exmr': defaults['exmr_options'].get('refine_camera_rotation', False),
        })
        options = {**GVHMR_DEFAULTS, **defaults['gvhmr_options']}
        self._tool_refinement['gvhmr'] = (options['htd'], options['footmr'], True, options['handflow'])
        settings = self.workspace_settings()
        if self._selected_tool == 'gvhmr':
            settings['result'] = result_kind(options['htd'], options['footmr'],
                                             self.motion_correction.isChecked(), options['handflow'])
        for group in PROJECT_DEFAULT_GROUPS:
            fields = DEFAULT_SETTINGS[group][1]
            settings[group] = {**{key: value for key, value in settings[group].items() if key not in fields},
                               **defaults[group]}
        try:self.restore_workspace_settings(settings)
        except (OSError,ValueError,KeyError,TypeError) as error:self.log.append('Default settings not applied: '+str(error))
        self.restore_track_face(defaults['track_face'])

    def workspace_view(self):
        fps=self.info['fps'] if self.info else 30.
        return dict(track_face=self.track_face.isChecked(),
            camera_background=self.camera_tools.background_options(),source_time=self.source_offset+self.slider.value()/fps,
            trim_time=[self.source_offset+self.slider.clip_in/fps,self.source_offset+self.slider.clip_out/fps],
            camera={**{key:np.asarray(getattr(self.motion_view,key)).tolist()
                    for key in ('yaw','pitch','distance','pan','scene_center')},'target':self.motion_view.target().tolist()})

    def remember_workspace_model(self):
        root=self.workspace_root
        if root is None:return
        if self.project is not None:
            state=manifest(self.project)
            state.setdefault('pipeline',{})['result']=self.active_result
            settings=self.workspace_settings()
            for key in ('correction','flow_options','exmr_options','prompthmr_options'):state[key]=settings[key]
            atomic_json(self.project/'project.json',state)
            saved=workspace.set_active(root,state['pipeline']['model'],self.project,self.workspace_view())
            model=state['pipeline']['model']
        else:
            model=self._selected_tool
            saved=workspace.set_active(root,model,view=self.workspace_view())
        saved['models'][model]['settings']=self.workspace_settings()
        if self.mamma_inputs:
            saved['mamma_inputs']=self.mamma_inputs
            saved['mamma_active_camera']=self.mamma_active_camera
        atomic_json(root/'project.json',saved)

    def restore_workspace_view(self,view):
        self.camera_tools.restore_background(view.get('camera_background',{}))
        if self.info:
            fps=self.info['fps']
            times=view.get('trim_time')
            if times:
                first=max(0,round((times[0]-self.source_offset)*fps))
                end=min(self.slider.maximum()+1,round((times[1]-self.source_offset)*fps))
                if first<end:self.slider.set_clip(first,end)
            frame=round((view.get('source_time',self.source_offset)-self.source_offset)*fps)
            self.slider.setValue(max(0,min(self.slider.maximum(),frame)))
            self.seek(self.slider.value())
        if view.get('camera'):
            self._workspace_camera=view['camera']
            self.restore_workspace_camera()

    def restore_workspace_camera(self,consume=False):
        camera=getattr(self,'_workspace_camera',None)
        if not camera:return
        for key,value in camera.items():
            if key=='target':continue
            setattr(self.motion_view,key,np.asarray(value,float) if key in ('pan','scene_center') else float(value))
        if 'target' in camera:self.motion_view.pan+=np.asarray(camera['target'])-self.motion_view.target()
        self.motion_view.auto_fit=False
        self.motion_view.update()
        if consume and (self.people_root is None or self.scene_data is not None):self._workspace_camera=None

    def open_workspace(self,root):
        try:
            root=Path(root).resolve();state=workspace.read(root)
            workspace.model_project(root,state['active_model'],state)
            if self.workspace_root is not None and self.workspace_root!=root and not self.busy:
                self.remember_workspace_model()
            if self.workspace_root != root:
                self.clear_first_person_calibration()
            self.clear_mamma_inputs()
            self.workspace_root=root
            self._workspace_camera=None
            if not self.switch_workspace_model(state['active_model'],opening=True):
                return False
            self.restore_track_face(self.saved_track_face(state.get('view',{})))
            return True
        except (OSError,ValueError,KeyError) as error:
            self.status.setText('Cannot open project: '+str(error));return False

    def switch_workspace_model(self,model,opening=False):
        root=self.workspace_root
        try:
            state=workspace.read(root)
            if not opening:
                view=self.workspace_view()
                self.remember_workspace_model()
                state=workspace.read(root)
            else:view=state.get('models',{}).get(model,{}).get('view',state.get('view',{}))
            target=workspace.model_project(root,model,state)
            self.stop_playback()
            self._workspace_loading=True
            try:
                if (target/'project.json').is_file():
                    self.load_project(target)
                else:
                    self.clear_preview();self.reset_people();self.project=None
                    self.video_view.set_observations({})
                    self.model_choice.blockSignals(True)
                    self.model_choice.setCurrentIndex(self.model_choice.findData(model))
                    self.model_choice.blockSignals(False)
                    self.select_model()
                    settings = state.get('models', {}).get(model, {}).get('settings', {})
                    self.restore_workspace_settings(self.with_default_groups(settings, model))
                    if opening or (model=='mamma' and state.get('mamma_inputs') and not self.mamma_inputs):self.open_workspace_video(state)
                    self.status.setText('MAMMA inputs loaded. Prepare its runtime and models in Model Manager, then Reconstruct 3D.' if model=='mamma' else
                        f'{self.model_choice.currentText()} has no saved motion in this project. Use Reconstruct 3D.')
            finally:self._workspace_loading=False
            workspace.set_active(root,model,target if (target/'project.json').is_file() else None,view)
            self._workspace_pending_view=view
            self.restore_workspace_view(view)
            self.setWindowTitle('RGB MoCap | '+root.name)
            self.update_readiness()
            return True
        except (OSError,ValueError,KeyError) as error:
            self.model_choice.blockSignals(True)
            self.model_choice.setCurrentIndex(self.model_choice.findData(self._selected_tool))
            self.model_choice.blockSignals(False)
            self.status.setText('Cannot switch model: '+str(error));return False

    def open_workspace_video(self,state):
        self._rife_source_fps = float(state.get('source', {}).get('fps') or 0)
        self.rife_interpolation.setChecked(False)
        self.video_path=workspace.media_path(self.workspace_root,state)
        self.video_readable=False
        if self.model_choice.currentData()=='mamma' and state.get('mamma_inputs'):
            view=state.get('models',{}).get('mamma',{}).get('view',state.get('view',{}))
            self.restore_mamma_project_inputs(state,self.workspace_root,view)
            return
        relative=state.get('reference')
        reference=workspace.child(self.workspace_root,relative) if relative else None
        self._workspace_video_reference=bool(reference and (reference/'project.json').is_file() and (reference/'input.mp4').is_file())
        if self._workspace_video_reference:
            data=manifest(reference);fps=data['fps']
            self.info={'fps':fps,'frames':data['frames'],'duration':data['frames']/fps}
            self.source_offset,self.source_end=data['start'],data['end']
            self.slider.set_media(fps);self.slider.setRange(0,data['frames']-1);self.slider.reset_trim()
            self.video_view.open(select_preview(reference),(data['width'],data['height']))
            self.filename.setText(self.workspace_root.name+' · No saved motion for the selected model')
            self.fps_label.setText(f'Source {fps:.2f} fps')
        else:
            self.info=None;self.source_offset=float(state.get('time_offset',0.))
            self.source_end=self.source_offset+state['source']['duration']
            self.video_view.open(self.video_path)

    def workspace_project_loaded(self,project,state):
        root=workspace.root_for(project,state)
        self.workspace_root=root
        self._workspace_video_reference=root is not None
        if root is not None:
            workspace.set_active(root,state['pipeline']['model'],project)
