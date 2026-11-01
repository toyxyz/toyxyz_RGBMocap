"""PEAR multiview input controls; inference/calibration never run on the GUI thread."""
from pathlib import Path
import json,hashlib,time,math
from PySide6.QtCore import Qt,QSize
from PySide6.QtGui import QImage,QIcon,QPixmap
from PySide6.QtWidgets import QWidget,QGridLayout,QHBoxLayout,QScrollArea,QComboBox,QSpinBox,QDoubleSpinBox,QLabel,QPushButton,QSizePolicy
from .mamma_ui import InputInspector,CameraThumbnail

class PearMultiWorkspace:
    def init_pear_multi(self):
        self.pear_inputs=[];self.pear_preview=0;self.pear_cameras=[];self.pear_devices=[]
        self.pear_rotations=[];self.pear_slot_rows=[];self.pear_device_rotations={};self.pear_video_rotations={}
        self.pear_capture=None;self.pear_capture_key=None;self.pear_capture_command=None;self.pear_capture_generation=0;self.pear_capture_failed=None;self.pear_capture_suspended=None;self.pear_capture_guard_at=0
        self.pear_capture_streams={};self.pear_align_frame=0;self.pear_align_generation=0
        self.pear_inspector=None;self.pear_input_token=0;self.pear_input_pending=False;self.pear_switching=False;self.pear_resume=False
        self.pear_calibration=None;self.pear_alignment=None;self.pear_align_phase=None;self.pear_align_folder=None;self.pear_align_signature=None
        self.pear_align_started=0;self.pear_align_text="Camera Align";self.pear_align_guard_at=0;self.pear_thumb_at=0;self.pear_thumbnails=None;self._pear_media_restore=None;self._pear_restoring_media=False
    def build_pear_multi_header(self,row):
        row.addWidget(QLabel('Webcams'));row.addSpacing(8);row.addWidget(QLabel('Count'))
        self.pear_camera_count=QSpinBox();self.pear_camera_count.setRange(1,32);self.pear_camera_count.setValue(1);self.pear_camera_count.setFixedWidth(48)
        self.pear_camera_count.setToolTip('Number of webcam slots. Assign a different device to every slot. Video files determine their own camera count.')
        row.addWidget(self.pear_camera_count)
        self.pear_refresh=QPushButton('↻');self.pear_refresh.setFixedWidth(28);self.pear_refresh.setToolTip('Refresh connected webcams.')
        row.addWidget(self.pear_refresh);row.addStretch();self.pear_refresh.clicked.connect(lambda:self.pear_scan_cameras(force=True))
    def build_pear_thumbnails(self,parent):
        self.pear_input_panel=QWidget();outer=QGridLayout(self.pear_input_panel);outer.setContentsMargins(0,0,0,0)
        outer.setVerticalSpacing(4)
        self.pear_header=QWidget();self.pear_header.setObjectName('editorHeader')
        controls=QHBoxLayout(self.pear_header);controls.setContentsMargins(7,3,7,3);controls.setSpacing(6)
        self.build_pear_multi_header(controls);outer.addWidget(self.pear_header,0,0)
        self.pear_slot_widget=QScrollArea();self.pear_slot_widget.setWidgetResizable(True);self.pear_slot_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        slots=QWidget();self.pear_slot_grid=QGridLayout(slots);self.pear_slot_grid.setContentsMargins(7,4,7,4);self.pear_slot_grid.setHorizontalSpacing(8);self.pear_slot_grid.setVerticalSpacing(4);self.pear_slot_grid.setAlignment(Qt.AlignTop);self.pear_slot_widget.setWidget(slots)
        for col,title in enumerate(('Camera','Device','Rotation')):
            label=QLabel(title);label.setObjectName('muted');self.pear_slot_grid.addWidget(label,0,col)
        self.pear_slot_grid.setColumnStretch(1,1);outer.addWidget(self.pear_slot_widget,1,0)
        self.pear_resize_slots(1);self.pear_camera_count.valueChanged.connect(self.pear_resize_slots)
        self.pear_thumbnails=QScrollArea();self.pear_thumbnails.setWidgetResizable(True);self.pear_thumbnails.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.pear_thumbnails.setAccessibleName('PEAR camera thumbnails');inner=QWidget();self.pear_thumbnail_grid=QGridLayout(inner)
        self.pear_thumbnail_grid.setContentsMargins(4,4,4,4);self.pear_thumbnail_grid.setSpacing(4)
        for c in range(4):self.pear_thumbnail_grid.setColumnStretch(c,1)
        self.pear_thumbnails.setWidget(inner);outer.addWidget(self.pear_thumbnails,2,0);parent.addWidget(self.pear_input_panel)
        self.pear_thumbnail_buttons=[];self.pear_input_panel.hide();self.pear_thumbnails.hide();self.pear_slot_widget.hide()
    def pear_resize_slots(self,count):
        if not hasattr(self,'pear_slot_grid'):return
        while len(self.pear_cameras)>count:
            self.pear_cameras.pop();self.pear_rotations.pop()
            for widget in self.pear_slot_rows.pop():self.pear_slot_grid.removeWidget(widget);widget.deleteLater()
        while len(self.pear_cameras)<count:
            combo=QComboBox();combo.setMinimumWidth(100);combo.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon);combo.setMinimumContentsLength(10)
            combo.addItem('None — use video' if not self.pear_cameras else 'None',None)
            for d in self.pear_devices:combo.addItem(d['name'],d)
            i=len(self.pear_cameras);combo.setToolTip(f'Webcam slot {i+1}');combo.currentIndexChanged.connect(lambda _,i=i:self.pear_camera_slot_changed(i))
            label=QLabel(str(i+1));rotation=self.pear_make_rotation(i)
            for col,widget in enumerate((label,combo,rotation)):self.pear_slot_grid.addWidget(widget,i+1,col)
            self.pear_slot_rows.append((label,combo,rotation));self.pear_cameras.append(combo);self.pear_rotations.append(rotation)
        self.pear_camera=self.pear_cameras[0];self.pear_rotation=self.pear_rotations[0]
        row_height=max(self.pear_camera.sizeHint().height(),self.pear_rotation.sizeHint().height())
        self.pear_slot_widget.setFixedHeight(min(220,12+self.pear_slot_grid.itemAtPosition(0,0).widget().sizeHint().height()+count*(row_height+4)))
        self.pear_slot_widget.show();self.pear_preview=min(self.pear_preview,count-1)
        self.pear_sync_rotations()
        self.pear_source_changed()
    def pear_make_rotation(self,index):
        control=QDoubleSpinBox();control.setRange(-360,360);control.setDecimals(1);control.setSingleStep(90);control.setSuffix('°');control.setFixedWidth(88)
        control.setAccessibleName(f'Camera {index+1} rotation');control.setToolTip('Rotate this input clockwise before display and tracking. 90 turns right; −90 turns left.')
        control.valueChanged.connect(lambda _,i=index:self.pear_rotation_changed(i));return control
    def pear_video_rotation_key(self):
        if self.pear_inputs:return self.pear_inputs[min(self.pear_preview,len(self.pear_inputs)-1)]['path']
        view=self.video_view
        return view._source_key[0] if view._source_key and self.video_readable else None
    def pear_sync_rotations(self):
        webcam=self.pear_using_webcams()
        if not webcam and self.pear_inputs:self.pear_preview=min(self.pear_preview,len(self.pear_inputs)-1)
        for i,(combo,control) in enumerate(zip(self.pear_cameras,self.pear_rotations)):
            device=combo.currentData()
            if webcam:
                value=self.pear_device_rotations.get(device['id'],0.) if device else 0.
                enabled=device is not None
            else:
                key=self.pear_video_rotation_key() if i==0 else None
                value=self.pear_video_rotations.get(key,0.);enabled=key is not None
            control.blockSignals(True);control.setValue(value);control.blockSignals(False);control.setEnabled(enabled)
            control.setAccessibleName(f'Webcam {i+1} rotation' if webcam else 'Selected video rotation')
        self.video_view.set_preview_rotation(self.pear_rotation.value() if self._pear_active and not webcam else 0.)
    def pear_camera_slot_changed(self,index):
        self.pear_sync_rotations();self.pear_source_changed()
    def pear_rotation_changed(self,index):
        if self.pear_using_webcams():
            device=self.pear_cameras[index].currentData()
            if device:self.pear_device_rotations[device['id']]=self.pear_rotations[index].value()
        elif index==0:
            key=self.pear_video_rotation_key()
            if key:self.pear_video_rotations[key]=self.pear_rotation.value()
        self.pear_source_changed()
    def pear_input_rotations(self):
        if self.pear_using_webcams():return [self.pear_device_rotations.get(c.currentData()['id'],0.) if c.currentData() else 0. for c in self.pear_cameras]
        return [self.pear_video_rotations.get(s['path'],0.) for s in self.pear_sources()]
    def pear_using_webcams(self):return any(c.currentData() is not None for c in self.pear_cameras)
    def pear_fill_devices(self,devices):
        changed=False;self.pear_devices=devices
        for combo in self.pear_cameras:
            current=combo.currentData();combo.blockSignals(True);combo.clear();combo.addItem('None — use video' if combo is self.pear_camera else 'None',None)
            for d in devices:combo.addItem(d['name'],d)
            index=next((i for i in range(1,combo.count()) if current and combo.itemData(i)['id']==current['id']),0)
            combo.setCurrentIndex(index);combo.blockSignals(False);changed|=current!=combo.currentData()
        self.pear_sync_rotations()
        if changed:self.pear_source_changed()
    def pear_sources(self):
        if self.pear_using_webcams():
            devices=[c.currentData() for c in self.pear_cameras]
            if any(d is None for d in devices):raise ValueError('Assign a webcam to every slot, or set all slots to None to use videos.')
            if len({d['id'] for d in devices})!=len(devices):raise ValueError('Each webcam slot must use a different device.')
            return devices
        if self.pear_inputs:return self.pear_inputs
        view=self.video_view
        return [dict(path=view._source_key[0])] if view._source_key and self.video_readable else []
    def pear_input_signature(self):
        sources=self.pear_sources();stamps=[]
        for s in sources:
            if 'path' in s:
                stamps.append([s['path'],s.get('file_stamp')])
            else:stamps.append(s)
        value=[stamps,self.pear_input_rotations(),self.pear_resolution.currentData() if self.pear_using_webcams() else None]
        return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode('utf8')).hexdigest()
    def pear_multi_invalidate(self):
        try:signature=self.pear_input_signature()
        except (ValueError,OSError):signature=None
        if self.pear_calibration and self.pear_calibration[0]!=signature:self.pear_calibration=None
        if self.pear_align_phase and signature!=self.pear_align_signature:self.pear_cancel_alignment('Inputs changed; camera alignment cancelled.')
    def pear_make_thumbnails(self,entries):
        if self.pear_thumbnails is None:return
        while self.pear_thumbnail_grid.count():
            w=self.pear_thumbnail_grid.takeAt(0).widget()
            if w:w.deleteLater()
        self.pear_thumbnail_buttons=[]
        for i,(title,image) in enumerate(entries):
            button=CameraThumbnail();button.setCheckable(True);button.setAutoExclusive(True);button.setText(str(i+1))
            button.setFixedHeight(88);button.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed);button.setMinimumWidth(0)
            button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon);button.setToolTip(title)
            if image is not None:button.setIcon(QIcon(QPixmap.fromImage(image)));button.setIconSize(QSize(100,48))
            button.setStyleSheet('QToolButton{background:#303030;border:1px solid #555;color:#ddd}QToolButton:checked{border:2px solid #8ac1ff;background:#385571}')
            button.clicked.connect(lambda checked=False,i=i:self.pear_select_preview(i))
            self.pear_thumbnail_grid.addWidget(button,i//4,i%4);self.pear_thumbnail_buttons.append(button)
        self.pear_preview=min(self.pear_preview,max(0,len(entries)-1))
        if entries:self.pear_thumbnail_buttons[self.pear_preview].setChecked(True)
        self.pear_thumbnails.setFixedHeight(min(220,8+math.ceil(len(entries)/4)*92));self.pear_thumbnails.setVisible(len(entries)>1)
    def pear_select_preview(self,index):
        if index==self.pear_preview:return
        self.pear_preview=index;self._pear_last=None;self.pear_sync_rotations()
        if self.pear_using_webcams():self.pear_submit();return
        if not self.pear_inputs:return
        self.pear_resume=self.timer.isActive();self.stop_playback();self.pear_switching=True;self.video_readable=False
        self.video_view.open(Path(self.pear_inputs[index]['path']),timeline_fps=self.info['fps'],reuse=True)
    def load_pear_videos(self,paths):
        if not 1<=len(paths)<=32:self.pear_status.setText('Choose 1–32 PEAR camera videos.');return
        self.stop_playback();self.pear_cancel_alignment('Camera inputs replaced.');self.pear_calibration=None
        for combo in self.pear_cameras:combo.blockSignals(True);combo.setCurrentIndex(0);combo.blockSignals(False)
        if self.pear_inspector is None:
            self.pear_inspector=InputInspector(self);self.pear_inspector.finished_input.connect(self.pear_inputs_ready);self.pear_inspector.failed_input.connect(self.pear_inputs_failed);self.pear_inspector.start()
        self.pear_input_pending=True;self.pear_input_token=self.pear_inspector.request('pear_videos',list(map(str,paths)))
        self.pear_status.setText('Opening PEAR camera videos…')
    def pear_inputs_ready(self,kind,token,result):
        if not self._pear_active or token!=self.pear_input_token:return
        from .mamma_inputs import common_media
        self.pear_input_pending=False;self.pear_inputs=[e for e,image in result];self.info=common_media(self.pear_inputs)
        self.video_path=Path(self.pear_inputs[0]['path'])
        self.source_offset=0.;self.source_end=self.info['duration'];self.slider.set_media(self.info['fps']);self.slider.setRange(0,self.info['frames']-1);self.slider.reset_trim()
        self.slider.video_track_label=f'{len(result)} PEAR cameras · common trim';self.slider._cache_key=None
        self.pear_preview=0;self.pear_sync_rotations();self.pear_make_thumbnails([(e['id'],im) for e,im in result]);self.pear_switching=True;self.video_readable=False
        self.video_view.open(Path(self.pear_inputs[0]['path']),timeline_fps=self.info['fps'],reuse=True)
        self.filename.setText(f'{len(result)} PEAR camera videos · synchronized starts required · common range {self.info["duration"]:.2f} s')
        self.pear_source_changed()
    def pear_inputs_failed(self,kind,token,error):
        if token==self.pear_input_token:self.pear_input_pending=False;self.pear_status.setText(error)
    def pear_restore_media_ready(self):
        if not self._pear_restoring_media:return False
        self._pear_restoring_media=False;self.video_readable=True
        if self._pear_media_restore['info'] is None:return False
        self.seek(self._pear_media_restore['value']);return True
    def pear_video_ready(self,info):
        if not self._pear_active or not self.pear_inputs:return False
        self.pear_switching=False;self.video_readable=True
        self.fps_label.setText(f'Camera {self.pear_preview+1} · Source {info["fps"]:.2f} | Timeline {self.info["fps"]:.0f} fps')
        self.seek(self.slider.value());self.pear_source_changed()
        if self.pear_resume:self.pear_resume=False;self.toggle_play()
        return True
    def pear_camera_align(self):
        if self.pear_align_phase:self.pear_cancel_alignment('Camera alignment cancelled.');return
        if not self._pear_running:self.pear_toggle_tracking();return
        self.pear_begin_alignment()
    def pear_check_alignment(self):
        from .models import missing
        from .pear.alignment_support import configuration
        if not self.pear_sources():raise ValueError('Open a video or assign a webcam before Camera Align.')
        configuration()
        if missing('pear_align'):raise ValueError('Prepare VGGT and GeoCalib in PEAR Model Manager first.')
    def pear_begin_alignment(self):
        self.pear_alignment_warning.clear();self.pear_alignment_warning.hide()
        try:
            self.pear_check_alignment()
            self._pear_failed=False;self.pear_calibration=None
            if not self.pear_using_webcams() and not self.slider.clip_in<=self.slider.value()<self.slider.clip_out:
                self.slider.setValue(self.slider.clip_in);self.seek(self.slider.clip_in)
            self.pear_align_frame=self.slider.value();self.pear_align_generation=self.pear_generation
            if not self.pear_using_webcams():self.video_view.set_external_frame(None)
            self.pear_align_signature=self.pear_input_signature();self.pear_align_phase='samples'
            self.pear_align_started=time.monotonic();self.pear_align_text='Camera Align · Preparing background samples';self._pear_last=None;self.pear_status.setText('Camera Align · Preparing background and body-scale samples…');self.pear_submit()
        except (ValueError,OSError,KeyError) as e:
            self.pear_alignment_failed('Camera Align failed: '+str(e))
    def pear_cancel_alignment(self,message):
        if self.pear_alignment:self._pear_closing.append(self.pear_alignment);self.pear_alignment.close();self.pear_alignment=None
        if self.pear_align_phase=='samples' and self.pear_session:self.pear_close_session()
        self.pear_align_phase=None;self.pear_align_folder=None
        if hasattr(self,'pear_align_button'):
            self.pear_align_button.setText('Camera Align');self.pear_align_button.setToolTip(self.pear_align_tooltip)
        if hasattr(self,'pear_status'):self.pear_status.setText(message)
    def pear_alignment_failed(self,message):
        self.pear_cancel_alignment(message);self.pear_calibration=None
        self._pear_running=True;self._pear_failed=False;self._pear_start_align=False
        self.pear_generation+=1;self._pear_last=None;self._pear_ready=False;self._pear_probe_at=0
        warning=message+' Tracking continues in the selected camera view without alignment; aligned foot lock is unavailable.'
        self.pear_alignment_warning.setText(warning);self.pear_alignment_warning.show()
        self.pear_status.setText('Starting tracking without camera alignment…')
        self.motion_view.live_message=self.pear_status.text();self.motion_view.update()
        self.pear_update_tracking_button()
        self.log.append(warning)
    def pear_alignment_poll(self):
        if not self.pear_align_phase:return False
        self.pear_align_button.setText('Cancel Camera Align')
        self.pear_status.setText(self.pear_align_text+f' · {time.monotonic()-self.pear_align_started:.0f} s')
        if self.pear_align_phase=='closing':
            if any(s.process.poll() is None for s in self._pear_closing):return True
            from .pear.session import Session
            try:
                self.pear_alignment=Session('flow_pear_align');self.pear_alignment.submit(dict(folder=self.pear_align_folder));self.pear_align_phase='solve'
            except (OSError,RuntimeError,ValueError) as exc:
                self.pear_alignment_failed('Camera Align failed: '+str(exc));return True
        if self.pear_align_phase=='solve':
            if time.monotonic()-self.pear_align_guard_at>.5:
                self.pear_align_guard_at=time.monotonic();warning=self.pear_alignment.guard.check()
                if warning:self.pear_alignment_failed(warning);return True
            _,packet=self.pear_alignment.poll()
            if packet:
                meta,_=packet
                if meta['type']=='error':self.pear_alignment_failed(meta['text']);return True
                if meta['type']=='status':self.pear_align_text=meta['text']
                if meta['type']=='alignment' and meta.get('phase')=='complete':
                    if self.pear_input_signature()!=self.pear_align_signature:self.pear_cancel_alignment('Inputs changed; alignment discarded.');return True
                    self.pear_calibration=(self.pear_align_signature,self.pear_align_folder)
                    self.log.append(f'PEAR Camera Align: {time.monotonic()-self.pear_align_started:.1f} s · '+self.pear_align_folder)
                    self.pear_cancel_alignment('Camera alignment complete. Restarting tracking…');self.pear_generation+=1;self._pear_running=True;self._pear_failed=False;self._pear_last=None;self._pear_probe_at=0
            elif self.pear_alignment:self.pear_align_button.setToolTip(f'Aligning fixed cameras · {time.monotonic()-self.pear_align_started:.0f} s elapsed. Click to cancel.')
            return True
        return self.pear_align_phase=='closing'
    def pear_multi_request(self,value):
        sources=self.pear_sources();rotations=self.pear_input_rotations();value.update(sources=sources,preview=self.pear_preview,signature=self.pear_input_signature(),rotations=rotations,rotation=rotations[0] if rotations else 0.,
            fps=self.info['fps'] if self.info else 30.,duration=self.slider.clip_out/self.info['fps'] if self.info else 0.,
            calibration=self.pear_calibration[1] if self.pear_calibration else None,align_job=self.pear_align_phase=='samples')
        if len(sources)>1:value['frame']=value['timeline_frame']
        if self.pear_using_webcams():
            capture=self.pear_capture_request()
            if not self.pear_capture or self.pear_capture.process.poll() is not None or self.pear_capture_key!=self.pear_capture_session_key(capture):raise ValueError('Waiting for webcam capture streams…')
            streams=[self.pear_capture_streams.get(i) for i in range(len(sources))]
            if any(s is None or s.get('device_id')!=d['id'] for s,d in zip(streams,sources)):
                raise ValueError('Waiting for webcam capture streams…')
            value['camera_streams']=streams
        if self.pear_align_phase=='samples':
            value['generation']=self.pear_align_generation
            if not self.pear_using_webcams():value['frame']=self.pear_align_frame;value['timeline_frame']=self.pear_align_frame
        return value
    def pear_multi_packet(self,meta,arrays):
        if meta['type']=='alignment' and meta.get('phase')=='prepared':
            if self.pear_align_phase!='samples' or meta['generation']!=self.pear_align_generation:return True
            self.pear_align_folder=meta['folder'];self.pear_close_session();self.pear_align_phase='closing';return True
        if self.pear_align_phase=='samples' and meta['type'] in ('frame','missing'):return True
        if 'thumbnails' in arrays and time.monotonic()-self.pear_thumb_at>.25:
            self.pear_thumb_at=time.monotonic()
            for b,im in zip(self.pear_thumbnail_buttons,arrays['thumbnails']):
                h,w,_=im.shape;b.setIcon(QIcon(QPixmap.fromImage(QImage(im.data,w,h,im.strides[0],QImage.Format_RGB888).copy())))
        return False
    def pear_shutdown_multi(self):
        self.pear_close_capture()
        self.pear_cancel_alignment('Tracking stopped.')
        if self.pear_inspector:self.pear_inspector.stop();self.pear_inspector.wait(4000);self.pear_inspector=None
        if self.pear_thumbnails:self.pear_input_panel.hide()
    def pear_capture_request(self):
        sources=[];slots=[];angles=[]
        for i,(combo,angle) in enumerate(zip(self.pear_cameras,self.pear_input_rotations())):
            device=combo.currentData()
            if device:sources.append(device);slots.append(i);angles.append(angle)
        if not sources:return None
        if len({s['id'] for s in sources})!=len(sources):raise ValueError('Each webcam slot must use a different device.')
        return dict(sources=sources,slots=slots,slot_count=len(self.pear_cameras),rotations=angles,resolution=self.pear_resolution.currentData())
    def pear_close_capture(self):
        if self.pear_capture:self._pear_closing.append(self.pear_capture);self.pear_capture.close();self.pear_capture=None
        self.pear_capture_key=None;self.pear_capture_command=None;self.pear_capture_streams={}
    def pear_capture_session_key(self,request):
        return json.dumps([request['sources'],request['slots'],request['slot_count'],request['resolution']],sort_keys=True)
    def pear_capture_error(self,message,key):
        self.pear_cancel_alignment(message);self.pear_close_session();self.pear_close_capture()
        self.pear_capture_failed=key;self._pear_failed=True;self.video_view.set_external_frame(QImage())
        self.pear_status.setText(message);self.motion_view.live_message=message;self.motion_view.update();self.log.append(message)
    def pear_capture_poll(self):
        try:request=self.pear_capture_request()
        except ValueError as exc:
            self.pear_close_capture()
            if not self._pear_running:self.video_view.set_external_frame(QImage());self.pear_status.setText(str(exc))
            return
        key=json.dumps(request,sort_keys=True) if request else None
        if request is None:
            self.pear_close_capture();return
        if key in (self.pear_capture_failed,self.pear_capture_suspended):return
        session_key=self.pear_capture_session_key(request)
        if self.pear_capture_key!=session_key:
            self.pear_close_capture()
            if any(s.process.poll() is None for s in self._pear_closing):return
            from .pear.session import Session
            try:self.pear_capture=Session('pear_camera_preview')
            except Exception as exc:self.pear_capture_failed=key;self.pear_status.setText(str(exc));return
            self.pear_capture_key=session_key
            self.video_view.set_external_frame(QImage())
            if not self._pear_running and not self.pear_align_phase and not self._pear_failed:self.pear_status.setText('Opening webcam preview…')
        command=dict(request,preview=self.pear_preview)
        if command!=self.pear_capture_command:
            self.pear_capture_generation+=1;self.pear_capture.submit(dict(command,generation=self.pear_capture_generation));self.pear_capture_command=command
        if time.monotonic()-self.pear_capture_guard_at>.5:
            self.pear_capture_guard_at=time.monotonic();warning=self.pear_capture.guard.check()
            if warning:self.pear_capture_error(warning,key);return
        _,packet=self.pear_capture.poll()
        if not packet:return
        meta,arrays=packet
        if meta['type']=='error':
            self.pear_capture_error(meta['text'],key);return
        if meta.get('generation')!=self.pear_capture_generation:return
        if meta['type']=='camera_preview':
            self.pear_capture_streams={s['slot']:s for s in meta.get('streams',[])}
            self.pear_multi_packet(meta,arrays)
            if meta['preview']!=self.pear_preview:return
            if self._pear_running and self._pear_ready and not self.pear_align_phase and not self._pear_failed:return
            im=arrays.get('image')
            if im is not None:
                h,w,_=im.shape;self.video_view.set_external_frame(QImage(im.data,w,h,im.strides[0],QImage.Format_RGB888).copy())
                if not self._pear_running and not self.pear_align_phase and not self._pear_failed:self.pear_status.setText('Webcam preview · '+('Assign every slot before starting tracking.' if len(request['sources'])<request['slot_count'] else 'Click Start Live Tracking for pose tracking.'))
                self.fps_label.setText('Webcam preview')
            else:
                self.video_view.set_external_frame(QImage())
                if not self._pear_running and not self.pear_align_phase and not self._pear_failed:self.pear_status.setText('Waiting for webcam frames…' if self.pear_preview in request['slots'] else 'Choose an assigned webcam thumbnail to preview it.')
        elif meta['type']=='status' and not self._pear_running and not self.pear_align_phase and not self._pear_failed:self.pear_status.setText(meta['text'])
