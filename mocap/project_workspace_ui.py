"""Switch cached model results inside one project without running inference."""
import json
from pathlib import Path
import numpy as np
from . import project_workspace as workspace
from .project_storage import manifest
from .storage import atomic_json
from .video import select_preview


class ModelWorkspace:
    def selected_flow_options(self):
        return dict(camera_interval=self.flow_sampling.currentData(), fast_sam=self.fast_sam.isChecked(),
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
        flow=settings.get('flow_options',{})
        self.flow_hamer.setChecked(False)
        self.flow_sampling.setCurrentIndex(max(0,self.flow_sampling.findData(flow.get('camera_interval',0))))
        self.fast_sam.setChecked(flow.get('fast_sam',True))
        self.flow_batch4.setChecked(flow.get('sam_batch4',True))
        self.flow_gpu_crop.setChecked(flow.get('sam_gpu_crop',True))
        self.flow_hand_stride4.setChecked(flow.get('sam_hand_stride4',True))
        self.flow_hamer.setChecked(flow.get('hamer',False))
        exmr=settings.get('exmr_options',{})
        self.exmr_camera.setCurrentIndex(self.exmr_camera.findData(exmr.get('static_camera',True)))
        self.exmr_smoothing.setChecked(exmr.get('smoothing','global')=='global')
        self.active_result=self.selected_result()
        self.restore_mamma_options(settings.get('mamma_options',{}))
        self.restore_prompthmr_options(settings.get('prompthmr_options',{}))

    def workspace_view(self):
        fps=self.info['fps'] if self.info else 30.
        return dict(camera_background=self.camera_tools.background_options(),source_time=self.source_offset+self.slider.value()/fps,
            trim_time=[self.source_offset+self.slider.clip_in/fps,self.source_offset+self.slider.clip_out/fps],
            camera={**{key:np.asarray(getattr(self.motion_view,key)).tolist()
                    for key in ('yaw','pitch','distance','pan','scene_center')},'target':self.motion_view.target().tolist()})

    def remember_workspace_model(self):
        if getattr(self,"_pear_active",False):return
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
            self.clear_mamma_inputs()
            self.workspace_root=root
            self._workspace_camera=None
            return self.switch_workspace_model(state['active_model'],opening=True)
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
                    self.restore_workspace_settings(state.get('models',{}).get(model,{}).get('settings',{}))
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
