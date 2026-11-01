"""Synchronized multi-view inputs and MAMMA settings."""
from pathlib import Path
from threading import Condition
import math
import cv2
from PySide6.QtCore import Qt, QSize, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QIcon
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QGridLayout,QScrollArea,QToolButton,
    QComboBox,QLabel,QPushButton,QFileDialog,QSizePolicy,QSpinBox,QDoubleSpinBox)
from .mamma_inputs import video_paths, common_media, read_calibration, match_calibration, resolved_entries
from .ui_refresh import batch_readiness


class CameraThumbnail(QToolButton):
    def minimumSizeHint(self):return QSize(48,88)
    def sizeHint(self):return QSize(100,88)
    def resizeEvent(self,event):
        self.setIconSize(QSize(max(16,min(100,self.width()-10)),48))
        super().resizeEvent(event)


class InputInspector(QThread):
    finished_input = Signal(str,int,object)
    failed_input = Signal(str,int,str)

    def __init__(self,parent):
        super().__init__(parent)
        self.condition=Condition();self.pending={};self.tokens={};self.stopping=False

    def request(self,kind,value):
        with self.condition:
            token=self.tokens.get(kind,0)+1;self.tokens[kind]=token
            self.pending[kind]=(token,value);self.condition.notify()
        return token

    def obsolete(self,kind,token):
        with self.condition:return self.stopping or self.tokens.get(kind)!=token

    def stop(self):
        with self.condition:self.stopping=True;self.pending.clear();self.condition.notify()

    def run(self):
        while True:
            with self.condition:
                while not self.pending and not self.stopping:self.condition.wait()
                if self.stopping:return
                kind=next(iter(self.pending));token,value=self.pending.pop(kind)
            try:
                if kind=='calibration':result=read_calibration(value)
                else:
                    result=[]
                    for path in video_paths(value):
                        if self.obsolete(kind,token):break
                        cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,
                            [cv2.CAP_PROP_N_THREADS,1,cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,1500,cv2.CAP_PROP_READ_TIMEOUT_MSEC,1500])
                        try:
                            fps=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            w,h=int(cap.get(3)),int(cap.get(4));ok,im=cap.read()
                            if not cap.isOpened() or not ok or not math.isfinite(fps) or min(fps,frames,w,h)<=0:
                                raise ValueError('Cannot read camera video: '+path.name)
                            # A bounded early sample is more useful than a black
                            # first frame in recordings that begin with lights off.
                            cap.set(cv2.CAP_PROP_POS_FRAMES,min(frames-1,round(fps)))
                            sample_ok,sample=cap.read()
                            if sample_ok:im=sample
                            scale=min(144/w,81/h)
                            im=cv2.resize(im,(max(1,round(w*scale)),max(1,round(h*scale))))
                            im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
                            image=QImage(im.data,im.shape[1],im.shape[0],im.strides[0],QImage.Format_RGB888).copy()
                            result.append((dict(id=path.stem,path=str(path),**({'file_stamp':[path.stat().st_size,path.stat().st_mtime_ns]} if kind=='pear_videos' else {}),metadata=dict(width=w,height=h,fps=fps,frames=frames,duration=frames/fps)),image))
                        finally:cap.release()
                    if not self.obsolete(kind,token):common_media([e for e,image in result])
                if not self.obsolete(kind,token):self.finished_input.emit(kind,token,result)
            except Exception as error:
                message=('Missing camera calibration field: '+str(error) if kind=='calibration' and isinstance(error,KeyError) else str(error))
                if not self.obsolete(kind,token):self.failed_input.emit(kind,token,message)


class MammaWorkspace:
    def init_mamma(self):
        self.mamma_inputs=[];self.mamma_active_camera=0;self.mamma_calibration=None
        self.mamma_inspector=None;self.mamma_tokens={};self.mamma_pending=False
        self._mamma_restore=None;self._mamma_switching=False;self._mamma_resume=False
        self._mamma_primary_opening=False

    def build_mamma_thumbnails(self,parent_layout):
        self.mamma_thumbnails=QScrollArea()
        self.mamma_thumbnails.setWidgetResizable(True)
        self.mamma_thumbnails.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.mamma_thumbnails.setMaximumHeight(220)
        self.mamma_thumbnails.setAccessibleName('MAMMA camera thumbnails')
        inner=QWidget();self.mamma_thumbnail_grid=QGridLayout(inner)
        self.mamma_thumbnail_grid.setContentsMargins(4,4,4,4);self.mamma_thumbnail_grid.setSpacing(4)
        for c in range(4):self.mamma_thumbnail_grid.setColumnStretch(c,1)
        self.mamma_thumbnails.setWidget(inner);self.mamma_thumbnails.hide()
        parent_layout.addWidget(self.mamma_thumbnails)
        self.mamma_thumbnail_buttons=[]

    def build_mamma_settings(self):
        page=QWidget();page.setProperty('motion_tool','mamma')
        box=QVBoxLayout(page);box.setContentsMargins(0,0,0,0)
        box.setSpacing(10);box.setAlignment(Qt.AlignTop)
        box.addWidget(QLabel('Camera Calibration'))
        self.mamma_camera_mode=QComboBox()
        self.mamma_camera_mode.addItem('VGGT-Omega + GeoCalib (Automatic)','vggt')
        self.mamma_camera_mode.addItem('Manual Camera Data','manual')
        self.mamma_camera_mode.setToolTip('VGGT-Omega estimates fixed-camera placement. GeoCalib estimates background vertical direction before MAMMA 3D fitting. Manual uses your calibration without automatic gravity changes.')
        self.mamma_camera_mode.currentIndexChanged.connect(self.mamma_settings_changed)
        box.addWidget(self.mamma_camera_mode)
        self.mamma_camera_import=QPushButton('Import Camera Data...')
        self.mamma_camera_import.setToolTip('Load JSON or MAMMA YAML calibration for these videos. Camera names must match filenames without extensions; use original image sizes and metres. See docs/MAMMA.md for the format and docs/samples/mamma-camera-example.json for a synthetic example.')
        self.mamma_camera_import.clicked.connect(self.choose_mamma_calibration);box.addWidget(self.mamma_camera_import)
        self.mamma_camera_note=QLabel('VGGT-Omega + GeoCalib (Automatic)');self.mamma_camera_note.setWordWrap(True)
        self.mamma_camera_note.setMaximumHeight(72)
        self.mamma_camera_note.setTextFormat(Qt.PlainText)
        box.addWidget(self.mamma_camera_note)
        self.mamma_people=QSpinBox();self.mamma_people.setRange(0,8);self.mamma_people.setValue(0)
        people_label=QLabel('People');people_label.setBuddy(self.mamma_people)
        self.mamma_people.setAccessibleName('Number of people')
        self.mamma_people.setToolTip('0 = automatically detect the number of people. Set 1-8 to specify how many people to track if someone is missed or a background person is included. Count each person once across all cameras. Find person is not needed.')
        people_label.setToolTip(self.mamma_people.toolTip());box.addWidget(people_label)
        self.mamma_people.valueChanged.connect(self.mamma_settings_changed);box.addWidget(self.mamma_people)
        self.mamma_height=QDoubleSpinBox();self.mamma_height.setRange(.5,2.5);self.mamma_height.setDecimals(2);self.mamma_height.setValue(1.7);self.mamma_height.setSingleStep(.01)
        self.mamma_height.setAccessibleName('Reference height in metres')
        height_label=QLabel('Reference Height (m)');height_label.setBuddy(self.mamma_height)
        self.mamma_height.setToolTip('VGGT has no absolute metric scale. Approximate standing height sets the scene scale. This estimate assumes visible, initially standing people; it does not force every actor to this height. Supplied metric calibration does not use this value.')
        height_label.setToolTip(self.mamma_height.toolTip());box.addWidget(height_label)
        self.mamma_height.valueChanged.connect(self.mamma_settings_changed);box.addWidget(self.mamma_height)
        self.mamma_setup_notice=QLabel('MAMMA runtime required. Open Model Manager > Runtime Setup.')
        self.mamma_setup_notice.setWordWrap(True);self.mamma_setup_notice.setStyleSheet('color:#ffb45c')
        box.addWidget(self.mamma_setup_notice)
        page.setToolTip('Open synchronized, fixed-camera videos of the same performance. All videos share one trim range, starting at the same time. Select a thumbnail to change the preview camera.')
        self.model_pages.addWidget(page)

    def mamma_worker(self):
        if self.mamma_inspector is None:
            self.mamma_inspector=InputInspector(self)
            self.mamma_inspector.finished_input.connect(self.mamma_input_ready)
            self.mamma_inspector.failed_input.connect(self.mamma_input_failed)
            self.mamma_inspector.start()
        return self.mamma_inspector

    def clear_mamma_inputs(self):
        self.mamma_inputs=[];self.mamma_active_camera=0;self.mamma_pending=False;self._mamma_restore=None
        self._mamma_switching=False;self._mamma_resume=False
        self._mamma_primary_opening=False
        if self.mamma_inspector:
            with self.mamma_inspector.condition:
                for kind in ('videos','calibration'):
                    token=self.mamma_inspector.tokens.get(kind,0)+1
                    self.mamma_inspector.tokens[kind]=self.mamma_tokens[kind]=token
                    self.mamma_inspector.pending.pop(kind,None)
        self.mamma_thumbnails.hide()
        self.slider.video_track_label='Source clip';self.slider._cache_key=None

    def load_mamma_videos(self,paths,restore=None):
        if self.busy or self.contact_saving:return False
        try:
            paths=video_paths(paths)
            self.stop_playback();self.mamma_pending=True;self._mamma_restore=restore
            self.mamma_tokens['videos']=self.mamma_worker().request('videos',paths)
            self.status.setText('Checking camera videos and preparing thumbnails...')
            self.update_readiness();return True
        except (OSError,ValueError) as error:
            self.status.setText('Cannot open camera videos: '+str(error));return False

    def mamma_input_ready(self,kind,token,result):
        if token!=self.mamma_tokens.get(kind):return
        if kind=='calibration':
            try:
                if self.mamma_inputs:match_calibration(result,self.mamma_inputs)
                self.mamma_calibration=result
                self.mamma_camera_mode.setCurrentIndex(self.mamma_camera_mode.findData('manual'))
                self.mamma_settings_changed()
                self.status.setText('Camera calibration loaded: '+result['filename'])
            except (ValueError,KeyError,TypeError) as error:self.mamma_input_failed(kind,token,str(error))
            return
        self.mamma_pending=False
        if self.model_choice.currentData()!='mamma':self.update_readiness();return
        entries=[e for e,image in result];info=common_media(entries)
        restore=self._mamma_restore;self._mamma_restore=None
        if restore:
            for entry,saved in zip(entries,restore.get('entries',[])):
                if saved.get('media'):entry['media']=saved['media']
        if not restore:
            if self.workspace_root is not None:self.remember_workspace_model()
            self.workspace_root=None;self._workspace_video_reference=False;self._workspace_camera=None
            self.clear_preview();self.reset_people();self.project=None
            self.motion_view.auto_fit=True
            self.video_path=Path(entries[0]['path'])
            self.setWindowTitle('RGB MoCap | MAMMA Multi-View Inputs')
            for control in (self.htd_refine,self.footmr_refine,self.motion_correction):
                control.blockSignals(True);control.setChecked(True);control.blockSignals(False)
            self.restore_correction_options({});self.active_result=self.selected_result()
            self.mamma_calibration=None;self.mamma_camera_mode.blockSignals(True)
            self.mamma_camera_mode.setCurrentIndex(0);self.mamma_camera_mode.blockSignals(False)
        self.mamma_inputs=entries;self.info=info;self.source_offset=0.;self.source_end=info['duration']
        if restore and restore.get('result_project'):
            import json
            state=json.loads((Path(restore['result_project'])/'project.json').read_text(encoding='utf-8'))
            info=dict(fps=state['fps'],frames=state['frames'],duration=state['frames']/state['fps'])
            self.info=info;self.source_offset=state['start'];self.source_end=state['end']
        self.slider.set_media(info['fps']);self.slider.setRange(0,info['frames']-1);self.slider.reset_trim()
        self.slider.video_track_label=f'{len(entries)} cameras · common trim';self.slider._cache_key=None
        while self.mamma_thumbnail_grid.count():
            widget=self.mamma_thumbnail_grid.takeAt(0).widget()
            if widget:widget.deleteLater()
        self.mamma_thumbnail_buttons=[]
        for i,(entry,image) in enumerate(result):
            button=CameraThumbnail();button.setCheckable(True);button.setAutoExclusive(True)
            button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon);button.setIcon(QIcon(QPixmap.fromImage(image)))
            button.setIconSize(QSize(100,48));button.setText(str(i+1));button.setMinimumWidth(0)
            button.setFixedHeight(88);button.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
            button.setToolTip(entry['id']+'\n'+entry['path']);button.setAccessibleName('Preview camera '+entry['id'])
            button.setStyleSheet('QToolButton{background:#303030;border:1px solid #555;padding:3px;color:#ddd}QToolButton:checked{border:2px solid #8ac1ff;background:#385571}')
            button.clicked.connect(lambda checked=False,i=i:self.select_mamma_camera(i))
            self.mamma_thumbnail_grid.addWidget(button,i//4,i%4);self.mamma_thumbnail_buttons.append(button)
        self.mamma_thumbnails.setFixedHeight(min(220,8+math.ceil(len(entries)/4)*92))
        self.mamma_thumbnails.setVisible(len(entries)>1)
        if restore:
            self.restore_mamma_options(restore.get('options',{}))
            self.restore_workspace_view(restore.get('view',{}))
        self.mamma_active_camera=-1
        self.select_mamma_camera(min(int((restore or {}).get('active_camera',0)),len(entries)-1))
        self.clip_changed(self.slider.clip_in,self.slider.clip_out)
        self.status.setText(f'{len(entries)} camera videos loaded. Common trim applies to every camera; video starts must already be synchronized. Use synchronized views of the same performance, then Reconstruct 3D.')
        self.mamma_settings_changed()

    def mamma_input_failed(self,kind,token,error):
        if token!=self.mamma_tokens.get(kind):return
        if kind=='videos':self.mamma_pending=False;self._mamma_restore=None
        self.status.setText(('Cannot open camera videos: ' if kind=='videos' else 'Cannot import camera calibration: ')+error)
        self.update_readiness()

    @batch_readiness
    def select_mamma_camera(self,index,*,refresh=False):
        if not 0<=index<len(self.mamma_inputs) or (index==self.mamma_active_camera and not refresh):return
        self._mamma_resume=self.timer.isActive();self.stop_playback();self.mamma_active_camera=index
        self.mamma_thumbnail_buttons[index].setChecked(True)
        self._mamma_switching=True;self.video_readable=False
        entry=self.mamma_inputs[index]
        path=Path(entry['path']);first=0
        if self.project is not None:
            import json
            state=json.loads((self.project/'project.json').read_text(encoding='utf-8'))
            if state.get('pipeline',{}).get('model')=='mamma':
                saved=next((v for v in state.get('mamma_preview_videos',[]) if v['id']==entry['id']),None)
                if saved:
                    path=(self.project/saved['path']).resolve();first=saved['first']
                    if self.workspace_root is None or not path.is_relative_to(self.workspace_root.resolve()):
                        raise ValueError('Invalid saved MAMMA preview camera path.')
        opened=self.video_view.open(path,timeline_fps=self.info['fps'],timeline_start=first,reuse=True)
        self.filename.setText(f"{len(self.mamma_inputs)} camera videos · {entry['id']} · common range {self.info['duration']:.2f} s")
        if not opened:self.mamma_video_ready(dict(fps=self.video_view.native_fps))
        self.update_readiness()

    def mamma_video_ready(self,info):
        if self._mamma_primary_opening:
            self._mamma_primary_opening=False;self.video_readable=True
            self.seek(self.slider.value());self.update_readiness();return True
        if self.model_choice.currentData()!='mamma' or not self.mamma_inputs:return False
        self.video_readable=True;self._mamma_switching=False
        self.fps_label.setText(f"Camera {self.mamma_active_camera+1} · Source {info['fps']:.2f} | Timeline {self.info['fps']:.0f} fps")
        self.seek(self.slider.value())
        if self._mamma_resume:self._mamma_resume=False;self.toggle_play()
        self.update_readiness();return True

    @batch_readiness
    def mamma_project_loaded(self,project,state):
        if state.get('pipeline',{}).get('model')!='mamma':return
        import json
        root=self.workspace_root
        if root is None:return
        saved=json.loads((root/'project.json').read_text(encoding='utf-8'))
        entries=resolved_entries(root,saved.get('mamma_inputs',[]))
        if not entries:return
        active=min(saved.get('mamma_active_camera',0),len(entries)-1)
        options=state.get('mamma_options',{})
        import copy
        self._mamma_committed_options=copy.deepcopy(options)
        if [(e['id'],str(Path(e['path']).resolve())) for e in self.mamma_inputs]==[(e['id'],e['path']) for e in entries] and self.mamma_thumbnail_buttons:
            self.restore_mamma_options(options)
            # The selected camera is independent of the selected person.
            self.select_mamma_camera(min(self.mamma_active_camera,len(entries)-1),refresh=True)
        else:
            # Loading an actor resets the viewport before thumbnails arrive.
            # That temporary default is not the user's saved scene camera.
            view=saved.get('models',{}).get('mamma',{}).get('view') or {}
            if not view:
                view=dict(source_time=state['start'],trim_time=[state['start'],state['end']])
            self.load_mamma_videos([e['path'] for e in entries],restore=dict(options=options,
                active_camera=active,view=view,entries=entries,result_project=str(project)))

    def mamma_mode_changed(self,previous,model):
        if previous=='mamma' and model!='mamma' and self.mamma_inputs and self.project is None:
            # A single-view model must never show camera 3 while processing
            # the primary camera 1. This is display switching, not inference.
            self.stop_playback();self._mamma_primary_opening=True;self.video_readable=False
            self.video_view.open(Path(self.mamma_inputs[0]['path']),timeline_fps=self.info['fps'])
        elif model=='mamma' and previous!='mamma' and self.mamma_inputs and self.project is None:
            oldfps=self.info['fps'];start,end=self.processing_range();frame=self.source_offset+self.slider.value()/oldfps
            self.info=common_media(self.mamma_inputs);self.source_offset=0.;self.source_end=self.info['duration']
            self.slider.set_media(30.);self.slider.setRange(0,self.info['frames']-1)
            self.slider.set_clip(round(start*30),round(end*30));self.slider.setValue(round(frame*30))
            self.slider.video_track_label=f'{len(self.mamma_inputs)} cameras · common trim'
            self.mamma_active_camera=-1;self.select_mamma_camera(0)

    def choose_mamma_calibration(self):
        name,_=QFileDialog.getOpenFileName(self,'Import Camera Calibration','','Camera calibration (*.json *.yaml *.yml)')
        if name:
            self.mamma_tokens['calibration']=self.mamma_worker().request('calibration',name)
            self.status.setText('Checking camera calibration...')

    def mamma_options(self):
        return dict(camera_mode=self.mamma_camera_mode.currentData(),calibration=self.mamma_calibration,people=self.mamma_people.value(),height_m=self.mamma_height.value())

    def restore_mamma_options(self,options):
        if not isinstance(options,dict) or options.get('camera_mode','vggt') not in {'vggt','manual'}:
            raise ValueError('Invalid saved MAMMA camera settings.')
        people=options.get('people',0);height=options.get('height_m',1.7)
        if isinstance(people,bool) or not isinstance(people,int) or not 0<=people<=8:
            raise ValueError('Invalid saved MAMMA person count.')
        if isinstance(height,bool) or not isinstance(height,(int,float)) or not math.isfinite(height) or not .5<=height<=2.5:
            raise ValueError('Invalid saved MAMMA reference height.')
        self.mamma_people.blockSignals(True);self.mamma_height.blockSignals(True)
        self.mamma_people.setValue(int(options.get('people',0)));self.mamma_height.setValue(float(options.get('height_m',1.7)))
        self.mamma_people.blockSignals(False);self.mamma_height.blockSignals(False)
        self.mamma_calibration=options.get('calibration')
        if self.mamma_calibration is not None and (not isinstance(self.mamma_calibration,dict) or
            not isinstance(self.mamma_calibration.get('filename'),str) or not isinstance(self.mamma_calibration.get('cameras'),dict)):
            self.mamma_calibration=None
            raise ValueError('Invalid saved MAMMA calibration data.')
        self.mamma_camera_mode.blockSignals(True)
        self.mamma_camera_mode.setCurrentIndex(max(0,self.mamma_camera_mode.findData(options.get('camera_mode','vggt'))))
        self.mamma_camera_mode.blockSignals(False)
        self.mamma_settings_changed()

    def mamma_settings_changed(self,*_):
        manual=self.mamma_camera_mode.currentData()=='manual';message=''
        if manual:
            if self.mamma_calibration:
                message=self.mamma_calibration['filename']
                try:
                    if self.mamma_inputs:match_calibration(self.mamma_calibration,self.mamma_inputs)
                except ValueError as error:message='Warning: '+str(error)
            else:message='Import camera calibration to use Manual Camera Data.'
        self.mamma_camera_note.setText(message[:240]+('…' if len(message)>240 else ''))
        self.mamma_camera_note.setToolTip(message)
        self.mamma_camera_note.setVisible(manual)
        self.mamma_camera_import.setEnabled(not self.busy)
        self.mamma_camera_mode.setEnabled(not self.busy)
        self.mamma_height.setEnabled(not self.busy and not manual)
        self.mamma_people.setEnabled(not self.busy)
        if hasattr(self,'readiness'):self.update_readiness()

    def run_mamma(self):
        import hashlib,json,uuid
        from .paths import HOME,PROJECTS
        from .storage import atomic_json
        from .mamma_pipeline import STAGES
        from .mamma_runtime import configuration
        from .models import missing
        if self.busy or self.mamma_pending or self.contact_saving:return
        try:
            if len(self.mamma_inputs)<2:
                raise ValueError('Use at least two synchronized fixed-camera videos for MAMMA reconstruction.')
            configuration()
            options=self.mamma_options()
            if options['camera_mode']=='manual':match_calibration(options['calibration'],self.mamma_inputs)
            else:
                from .mamma_gravity import support_identity
                try:support_identity()
                except (OSError,ValueError,KeyError,TypeError) as error:
                    raise ValueError('Automatic MAMMA cameras need background gravity support. Close RGB MoCap and run install.cmd, then prepare Scene Gravity / GeoCalib in Model Manager.') from error
            absent=[m for m in missing('mamma_assets') if options['camera_mode']!='manual' or m.key not in {'flow_vggt','pear_geocalib'}]
            if absent:
                self.status.setText('Prepare the missing MAMMA models first.');self.open_model_manager();return
            first,end=self.processing_range()
            root=self.workspace_root
            if root is None:
                token=hashlib.sha256('|'.join(e['path'] for e in self.mamma_inputs).encode()).hexdigest()[:14]
                root=PROJECTS/(Path(self.mamma_inputs[0]['path']).stem[:80]+'_'+token+'_project')
            folder=HOME/'cache/mamma-jobs';folder.mkdir(parents=True,exist_ok=True)
            request=folder/(uuid.uuid4().hex+'.json')
            atomic_json(request,dict(version=1,root=str(root),inputs=self.mamma_inputs,start=first,end=end,
                options=options,active_camera=self.mamma_active_camera,
                correction_enabled=self.motion_correction.isChecked(),correction=self.selected_correction_options()))
            self.start_jobs([(stage,['--request',str(request)]) for stage in STAGES])
        except (OSError,ValueError,RuntimeError,KeyError) as error:
            self.status.setText('Cannot start MAMMA: '+str(error))

    def sync_mamma_controls(self):
        mamma=self.model_choice.currentData()=='mamma'
        self.find_person_button.setVisible(not mamma)
        from .people_ui import PEOPLE_HELP
        self.people_help_dialog.setText(
            'MAMMA detects and matches people across the camera videos during Reconstruct 3D. Find person is not needed. '
            'Set People to 0 for automatic detection, or 1-8 for the expected number of people. The first camera should show the intended people clearly.\n\n'
            'After reconstruction, check rows to show people and select a row to edit its contact timings. '
            'Right-click to rename; click the color square to change its preview color. Source motion is preserved.' if mamma else PEOPLE_HELP)
        self.mamma_thumbnails.setVisible(mamma and len(self.mamma_inputs)>1)
        if mamma:
            self.run_button.setEnabled(not self.busy and self.video_readable and len(self.mamma_inputs)>=2)
            self.find_person_button.setEnabled(False)
            self.models_button.setEnabled(not self.busy)
            self.run_button.setToolTip('Reconstruct the same performance from two or more synchronized fixed-camera videos. No fixed camera-count maximum; available RAM and VRAM determine practical capacity. Uses SAM 2.1 masks, MAMMA FP32 landmarks, VGGT or supplied cameras and the full MAMMA body fit. No automatic video synchronization is performed.')
            try:
                from .mamma_runtime import configuration
                configuration()
                message=''
                if self.mamma_inputs and len(self.mamma_inputs)<2:
                    message='Reconstruction requires at least two synchronized camera videos.'
                elif self.project is not None and getattr(self,'_mamma_committed_options',None)!=self.mamma_options():
                    message='Camera/person settings changed. Reconstruct 3D to apply; the previous motion is still displayed.'
                self.mamma_setup_notice.setText(message);self.mamma_setup_notice.setVisible(bool(message))
            except (OSError,ValueError,RuntimeError) as error:
                self.mamma_setup_notice.setText(str(error));self.mamma_setup_notice.show()
                self.run_button.setEnabled(False)
            self.mamma_camera_import.setEnabled(not self.busy)
            self.mamma_camera_mode.setEnabled(not self.busy)
        else:
            self.run_button.setToolTip('Create 3D motion for checked people in the trimmed video range using the selected model.')
        if self.mamma_pending:
            self.save_project_action.setEnabled(False);self.slider.trim_enabled=False;self.play_button.setEnabled(False)
            self.run_button.setEnabled(False);self.find_person_button.setEnabled(False)
        for button in self.mamma_thumbnail_buttons:button.setEnabled(not self.busy and not self.mamma_pending)

    def restore_mamma_project_inputs(self,state,root,view):
        entries=resolved_entries(root,state.get('mamma_inputs',[]))
        if not entries:return False
        active=state.get('mamma_active_camera',0)
        if not isinstance(active,int) or not 0<=active<len(entries):
            raise ValueError('Invalid saved preview camera.')
        options=state.get('models',{}).get('mamma',{}).get('settings',{}).get('mamma_options',{})
        self.restore_mamma_options(options)
        return self.load_mamma_videos([e['path'] for e in entries],restore=dict(
            options=options,active_camera=active,view=view,entries=entries))

    def shutdown_mamma(self):
        if self.mamma_inspector:
            for kind in ('videos','calibration'):self.mamma_tokens[kind]=self.mamma_tokens.get(kind,0)+1
            self.mamma_inspector.stop();self.mamma_inspector.wait();self.mamma_inspector.deleteLater();self.mamma_inspector=None
