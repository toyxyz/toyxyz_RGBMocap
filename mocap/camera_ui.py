"""Display-only saved cameras; one bounded loader and the existing video frame."""
from threading import Condition
import time
import numpy as np
from PySide6.QtCore import QThread,Signal,Qt,QPointF
from PySide6.QtGui import QPainter,QPen,QColor
from PySide6.QtWidgets import QDoubleSpinBox
from .viewer import VideoView
from .camera_tracks import at,load,centers

BACKGROUND_HELP='Refine saved camera rotation using background features. Only validated corrections up to 2 degrees are applied. Unreliable samples retain the original camera. CPU processing; no motion re-inference or camera-position correction. Off keeps the original tracking.'

class CameraLoader(QThread):
    ready=Signal(int,object)
    failed=Signal(int,str)
    activity=Signal(int,str)
    def __init__(self,parent):
        super().__init__(parent);self.condition=Condition();self.generation=0;self.pending=None;self.stopping=False;self.references={};self.backgrounds={}
    def request(self,project,group,background=False):
        with self.condition:
            self.generation+=1;self.pending=(self.generation,project,group,background);self.condition.notify();return self.generation
    def cancel(self):
        with self.condition:self.generation+=1;self.pending=None
    def close(self):
        with self.condition:self.stopping=True;self.condition.notify()
        self.wait();self.references.clear();self.backgrounds.clear()
    def run(self):
        while True:
            with self.condition:
                while self.pending is None and not self.stopping:self.condition.wait()
                if self.stopping:return
                token,project,group,background=self.pending;self.pending=None
            cancelled=lambda:self.stopping or token!=self.generation
            try:
                result=load(project,group,cancelled,self.references)
                if background and result is not None and result['model'] in ('exmr','flowhmr') and not cancelled():
                    from .camera_background_worker import refined
                    self.ready.emit(token,dict(result,background_pending=True));started=time.monotonic()
                    def activity(current,total):
                        elapsed=time.monotonic()-started
                        done=f' {current}/{total}' if total else ''
                        remaining=f' | about {elapsed*(total-current)/current:.0f}s remaining' if current and total>current else ''
                        if not cancelled():self.activity.emit(token,f'Refining camera background rotation{done} | {elapsed:.0f}s{remaining}')
                    try:result=refined(result,project,cancelled,activity,self.backgrounds)
                    except InterruptedError:
                        if not cancelled():raise
                    except (OSError,ValueError,RuntimeError,KeyError) as error:
                        result=dict(result,warning=(result.get('warning','')+' Background camera refinement failed: '+str(error)+'. Original camera retained.').strip(),background_report=dict(failed=str(error)))
                if background and result is not None and not cancelled():
                    from .camera_background_worker import current
                    if not current(list(result.get('camera_sources',{}).items())):raise ValueError('Saved camera changed during background refinement. Reload the camera preview.')
                if not cancelled():self.ready.emit(token,result)
            except FileNotFoundError:
                if not cancelled():self.failed.emit(token,'Saved camera data is incomplete. Prepare the motion preview before displaying cameras.')
            except Exception as error:
                if not cancelled():self.failed.emit(token,str(error))

def actors(view):
    def skeleton(frames,asset):
        frames=np.asarray(frames);points=np.array(frames[:,:3,3],float);parents=np.array(asset['parents'],int)
        # Terminal tips use the same outgoing local tails as the 3D overlay.
        bones=asset.get('bones',{})
        for bone,child,_,_ in bones.get('endpoint_vertices',[]):
            if child==-1:
                tip=frames[bone,:3,:3]@bones['segments'][bone][2]+points[bone]
                points=np.vstack((points,tip));parents=np.r_[parents,bone]
        return points,parents
    if view.people_mode:
        for take,index in view.scene_actors():
            if 'rig' in take.arrays and take.target_asset is not None and 'parents' in take.target_asset:
                p,parents=skeleton(take.arrays['rig'][index],take.target_asset)
                yield p@take.transform[:3,:3].T+take.transform[:3,3],parents,view.scene_colors.get(take.ident,'#ffcf4a')
    elif view.single_actor_visible and view.target_frames is not None and view.target_asset is not None and 'parents' in view.target_asset:
        if 0<=view.index<len(view.target_frames):
            p,parents=skeleton(view.target_frames[view.index],view.target_asset)
            yield p,parents,'#ffcf4a'

class CameraCanvas(VideoView):
    def __init__(self,tools):
        super().__init__();self.tools=tools;self.show_observations=False;self.pair=None
    def mirror(self,*args):
        tool=self.tools;w=tool.w;source=w.video_view;view=w.motion_view
        if not tool.data or source.frame is None:
            self.frame=None;self.pair=None;self.update();return
        # The source view emits only after its matching decoded image is committed.
        # Freeze this complete pair; later source/rig selections cannot mix it.
        ready=(w.project is None or (view.path is not None and view.path.parent==w.project))
        ci=getattr(w,'mamma_active_camera',0) if tool.data['model']=='mamma' else 0
        cameras=tool.data['cameras'];camera=cameras[min(max(0,ci),len(cameras)-1)]
        # Approximate displacement is a camera-gizmo aid, not a replacement for
        # the saved projection used to inspect the source/correction overlay.
        matrix=at(camera,view.scene_time,0)
        self.pair=(camera,matrix,list(actors(view)) if ready else [])
        self.frame=source.frame;self.source_size=source.source_size;self.index=source.index;self.update();source.update()
    def paintEvent(self,event):
        super().paintEvent(event)
        painter=QPainter(self);painter.setRenderHint(QPainter.Antialiasing)
        self.tools.paint_overlay(painter,self.image_rect(),self.tools.w.motion_view.show_bones,labels=True)

class CameraTools:
    def __init__(self,w,stack,header):
        self.w=w;self.stack=stack;self.data=None;self.message='';self.key=None;self.token=0;self.scale_values={};self.background_log_key=None
        self.canvas=CameraCanvas(self);stack.addWidget(w.motion_view);stack.addWidget(self.canvas)
        self.scale=QDoubleSpinBox();self.scale.setRange(0,100);self.scale.setDecimals(3);self.scale.setValue(1)
        self.scale.setPrefix('Motion x ');self.scale.setSpecialValueText('Rotation only');self.scale.setMaximumWidth(150)
        self.scale.setToolTip('Estimated camera movement: 1 uses the stored estimate, 0 fixes the camera center. Adjust the approximate distance scale. Camera gizmos only; video overlays and motion are unchanged.')
        header.addWidget(self.scale);self.scale.hide();self.scale.valueChanged.connect(self.scale_changed)
        self.loader=CameraLoader(w);self.loader.ready.connect(self.loaded);self.loader.failed.connect(self.failed);self.loader.activity.connect(self.activity);self.loader.start()
        w.video_view.frame_presented.connect(self.canvas.mirror)
        w.video_view.projected_overlay=self.paint_video_overlay
    def background_options(self):
        return {model:bool(getattr(self.w,model+'_background',None) and getattr(self.w,model+'_background').isChecked()) for model in ('flowhmr','exmr')}
    def restore_background(self,values):
        values=values if isinstance(values,dict) else {}
        for model in ('flowhmr','exmr'):
            control=getattr(self.w,model+'_background',None)
            if control is not None:
                control.blockSignals(True);control.setChecked(values.get(model) is True);control.blockSignals(False);control.setToolTip(BACKGROUND_HELP)
        self.key=None
    def background_changed(self,*args):
        self.key=None;self.sync()
    def activity(self,token,message):
        if token!=self.token:return
        self.message=message;self.w.motion_view.camera_message=message
        self.canvas.update();self.w.video_view.update();self.w.motion_view.update()
    def preview(self):return self.w.view_choice.currentText()=='Camera Preview'
    def set_view(self,name):
        self.stack.setCurrentIndex(int(name=='Camera Preview'))
        if name!='Camera Preview':self.w.motion_view.set_view(name)
        else:self.canvas.mirror()
        self.sync()
    def sync(self):
        w=self.w;view=w.motion_view;pear=getattr(w,'_pear_active',False)
        self.scale.setVisible(bool(self.data and not pear and view.show_cameras and not self.preview() and any(c['displacement'] is not None for c in self.data['cameras'])))
        item=w.view_choice.model().item(w.view_choice.findText('Camera Preview'))
        if item is not None:item.setEnabled(not pear)
        if pear and self.preview():
            w.view_choice.setCurrentText(view.view_name)
        left=w.video_view.show_observations and w.project is not None and w.preview_data is not None
        wanted=(view.show_cameras or self.preview() or left) and not pear
        background=self.background_options().get(w.model_choice.currentData(),False)
        key=(str(w.project),str(w.people_root),id(w.preview_data),bool(w.busy),pear,wanted,background)
        if key==self.key:
            self.canvas.mirror();return
        self.key=key;self.loader.cancel();self.token=self.loader.generation;self.data=None;self.scale.hide()
        view.camera_records=[];view.camera_message='';self.canvas.frame=None;self.canvas.pair=None
        self.message=''
        if wanted:
            self.message='Camera data is paused during processing.' if w.busy else 'Loading saved cameras...'
            if w.project is None:self.message='Open a reconstructed project to show saved cameras.'
            elif not w.busy:self.token=self.loader.request(w.project,w.people_root,background)
            view.camera_message=self.message
        self.canvas.update();w.video_view.update();view.update()
    def loaded(self,token,data):
        if token!=self.token:return
        self.data=data;self.message='Refining camera background rotation...' if data.get('background_pending') else data.get('warning','');self.w.motion_view.camera_message=self.message
        report=data.get('background_report')
        if report is not None:
            summary=report.get('skipped') or report.get('failed') or f"{report.get('accepted',0)}/{len(report.get('samples',[]))} samples accepted; "+('cached; no new processing' if report.get('cache_hit') else f"{report.get('elapsed_seconds',0):.2f}s; CPU")
            if 'accepted' in report and not report['accepted']:summary+='; original camera retained'
            control=getattr(self.w,data['model']+'_background',None)
            if control is not None:control.setToolTip(BACKGROUND_HELP+' Last result: '+summary)
            log_key=(str(self.w.project),data['model'],summary)
            if log_key!=self.background_log_key:
                self.background_log_key=log_key;self.w.log.append('Camera background refinement ('+data['model']+'): '+summary)
        self.scale.blockSignals(True);self.scale.setValue(self.scale_values.get(data['model'],1));self.scale.blockSignals(False)
        self.scale.setVisible(self.w.motion_view.show_cameras and not self.preview() and any(c['displacement'] is not None for c in data['cameras']))
        self.w.viewport_checks['show_cameras'].setToolTip('Show saved cameras. Display only. '+data['note'])
        self.canvas.setToolTip(data['note']);self.scale_changed()
    def failed(self,token,message):
        if token!=self.token:return
        self.data=None;self.message=message;self.w.motion_view.camera_message=message
        self.w.motion_view.update();self.canvas.update();self.w.video_view.update()
    def scale_changed(self,*args):
        view=self.w.motion_view
        if self.data:self.scale_values[self.data['model']]=self.scale.value()
        view.camera_records=[] if not self.data else [(c,centers(c,self.scale.value())) for c in self.data['cameras']]
        view.camera_translation_scale=self.scale.value();view.update();self.canvas.mirror();self.w.video_view.update()
    def paint_video_overlay(self,painter,rect):
        source=self.w.video_view
        if getattr(self.w,'_pear_active',False) or source.external_mode:return
        if source.frame is not None and self.data is None:
            self.paint_overlay(painter,rect,False);return
        if (source.frame is None or self.canvas.frame is None or self.canvas.index!=source.index
                or source.frame.cacheKey()!=self.canvas.frame.cacheKey()):return
        self.paint_overlay(painter,rect,True)
    def paint_overlay(self,painter,rect,bones,labels=False):
        message=self.message
        pair=self.canvas.pair
        painter.save()
        try:
            if self.canvas.frame is not None and pair:
                camera,matrix,paired_actors=pair
                if matrix is None:message='Outside saved camera timing.'
                else:
                    if labels:message=camera['name']+' | saved camera / displayed source frame'
                    if bones:
                        painter.setClipRect(rect);k=camera['K'];width,height=camera['size']
                        for points,parents,color in paired_actors:
                            cam=points@matrix[:3,:3].T+matrix[:3,3]
                            painter.setPen(QPen(QColor(color),2,Qt.SolidLine,Qt.RoundCap))
                            for j,parent in enumerate(parents):
                                if parent<0:continue
                                a,b=cam[parent].copy(),cam[j].copy()
                                if not np.isfinite([a,b]).all() or max(a[2],b[2])<=.03:continue
                                if a[2]<.03:a+=(b-a)*((.03-a[2])/(b[2]-a[2]))
                                if b[2]<.03:b+=(a-b)*((.03-b[2])/(a[2]-b[2]))
                                q=np.array([a,b])@k.T;q=q[:,:2]/q[:,2:]
                                q[:,0]=rect.left()+q[:,0]*rect.width()/width;q[:,1]=rect.top()+q[:,1]*rect.height()/height
                                painter.drawLine(QPointF(*q[0]),QPointF(*q[1]))
                        painter.setClipping(False)
                    if labels and not paired_actors:message+=' | no source bones at this time'
                    if labels and self.message:message+='\n'+self.message
            if message:
                painter.setPen(QColor('#e8c38d' if self.message else '#dddddd'))
                painter.drawText(rect.adjusted(12,10,-12,-12),Qt.AlignTop|Qt.TextWordWrap,message)
        finally:painter.restore()
    def close(self):
        self.loader.close();self.w.video_view.projected_overlay=None;self.data=None;self.w.motion_view.camera_records=[];self.canvas.shutdown();self.canvas.frame=None
