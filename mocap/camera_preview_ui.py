"""Video and export-skeleton overlay in the viewport's camera mode."""
import json,tempfile
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt,QProcess,QTimer,QPointF
from PySide6.QtGui import QPainter,QPen,QColor
from PySide6.QtWidgets import QWidget,QHBoxLayout,QStackedWidget,QComboBox,QCheckBox,QDoubleSpinBox,QPushButton
from .viewer import VideoView

class CameraView(VideoView):
    def __init__(self):
        super().__init__();self.points=None;self.delta=None;self.meta=None;self.scale=0.;self.message='Select a reconstructed person/take.';self.outside=False
        self.show_observations=False
    def release(self):
        self.shutdown()
        for name in ('points','delta'):
            data=getattr(self,name)
            if data is not None:data._mmap.close()
            setattr(self,name,None)
        self.meta=None
    def paintEvent(self,event):
        super().paintEvent(event)
        painter=QPainter(self);painter.setRenderHint(QPainter.Antialiasing)
        if self.frame is not None and self.points is not None and not self.outside and 0<=self.index<len(self.points):
            points=np.array(self.points[self.index],float)+self.scale*self.delta[self.index]
            k=np.asarray(self.meta['K']);r=self.image_rect()
            painter.setClipRect(r);painter.setPen(QPen(QColor('#ffcf4a'),2,Qt.SolidLine,Qt.RoundCap))
            def project(p):
                q=k@p;return QPointF(r.left()+q[0]/q[2]*r.width()/self.meta['width'],r.top()+q[1]/q[2]*r.height()/self.meta['height'])
            for j,parent in enumerate(self.meta['parents']):
                if parent<0:continue
                a=points[parent].copy();b=points[j].copy()
                if a[2]<=.03 and b[2]<=.03:continue
                if a[2]<.03:a+=(b-a)*((.03-a[2])/(b[2]-a[2]))
                if b[2]<.03:b+=(a-b)*((.03-b[2])/(a[2]-b[2]))
                painter.drawLine(project(a),project(b))
            painter.setClipping(False)
        painter.setPen(QColor('white'))
        label='Outside selected take' if self.outside else self.message
        painter.drawText(self.rect().adjusted(12,12,-12,-12),Qt.AlignTop|Qt.TextWordWrap,label)

class CameraPreview:
    def __init__(self,w,header,stack):
        self.w=w;self.view=CameraView();self.stack=stack;self.stack.addWidget(w.motion_view);self.stack.addWidget(self.view)
        self.mode=QComboBox();self.mode.addItems(['Free View','Camera Preview']);header.addWidget(self.mode)
        self.translation=QCheckBox('Camera move (experimental)');self.scale=QDoubleSpinBox();self.scale.setRange(0,100);self.scale.setDecimals(3);self.scale.setSingleStep(.1);self.scale.setValue(1);self.scale.setPrefix('Scale ')
        self.background=QCheckBox('Refine background rotation');self.background.setToolTip('Experimental: validate camera rotation against background features. May take a few seconds; no motion inference.')
        self.reload=QPushButton('Refresh camera')
        self.controls=QWidget();controls_layout=QHBoxLayout(self.controls);controls_layout.setContentsMargins(6,2,6,2)
        for item in (self.background,self.translation,self.scale,self.reload):controls_layout.addWidget(item)
        controls_layout.addStretch();self.controls.hide()
        self.process=None;self.temp=None;self.signature=None;self.log='';self.closed=False
        self.debounce=QTimer(w);self.debounce.setSingleShot(True);self.debounce.setInterval(150);self.debounce.timeout.connect(self.prepare)
        self.mode.currentIndexChanged.connect(self.toggle);self.translation.toggled.connect(self.change_scale);self.scale.valueChanged.connect(self.change_scale);self.reload.clicked.connect(self.refresh)
        self.background.toggled.connect(self.refresh)
        self.view.metadata_ready.connect(lambda _:self.seek());self.view.decode_error.connect(self.failed)
    def toggle(self,index):
        self.stack.setCurrentIndex(index)
        self.controls.setVisible(bool(index))
        self.w.view_choice.setEnabled(not index);self.w.bone_overlay.setEnabled(not index)
        if index:self.refresh()
        else:self.stop();self.signature=None
    def change_scale(self,*args):
        self.view.scale=self.scale.value() if self.translation.isChecked() else 0
        self.scale.setEnabled(self.translation.isChecked());self.view.update()
    def refresh(self,*args):
        self.signature=None;self.check()
    def check(self):
        if self.closed or not self.mode.currentIndex():return
        w=self.w;project=getattr(w,'project',None);kind=getattr(w,'active_result','initial')
        meta=project/(kind+'_mixamo.json') if project else None
        signature=(str(project),kind,meta.stat().st_mtime_ns if meta and meta.exists() else None, bool(w.busy),str(w.selected_character()),w.current_export() is not None if project else False)
        if signature==self.signature:return
        self.signature=signature;self.stop();self.view.message='Preparing camera preview...';self.view.update()
        if w.selected_character() is not None:self.failed('Select a source person for Camera Preview.');return
        if w.busy:self.failed('Camera preview paused during processing.');return
        if not signature[-1]:self.failed('Select a ready take; Apply Correction or Prepare Preview if needed.');return
        self.debounce.start()
    def stop(self):
        self.debounce.stop()
        if self.process is not None:
            process=self.process;self.process=None;process.blockSignals(True);process.kill();process.waitForFinished(3000);process.deleteLater()
        self.view.release()
        if self.temp is not None:self.temp.cleanup();self.temp=None
    def prepare(self):
        if not self.mode.currentIndex() or self.closed:return
        self.temp=tempfile.TemporaryDirectory(prefix='rgbmocap-camera-');self.log=''
        from .paths import worker_command
        command=worker_command('camera_preview','--project',str(self.w.project),'--result',self.w.active_result,'--output',self.temp.name)
        if self.background.isChecked():command.append('--background-refine')
        p=QProcess(self.w);self.process=p;p.setProcessChannelMode(QProcess.MergedChannels)
        p.readyReadStandardOutput.connect(self.read);p.finished.connect(self.finished)
        p.errorOccurred.connect(lambda _:self.failed('Camera preview worker could not start.') if p.error()==QProcess.FailedToStart else None)
        p.start(command[0],command[1:])
    def read(self):
        if self.process:self.log=(self.log+bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace'))[-4000:]
    def finished(self,code,*args):
        self.read()
        if code:self.failed('Camera preview failed: '+self.log[-500:]);return
        try:
            folder=Path(self.temp.name);meta=json.loads((folder/'ready.json').read_text())
            self.view.meta=meta;self.view.points=np.load(folder/'points.npy',mmap_mode='r');self.view.delta=np.load(folder/'translation.npy',mmap_mode='r')
            self.translation.setEnabled(meta['model']=='flowhmr')
            if meta['model']!='flowhmr':self.translation.setChecked(False)
            report=meta.get('background_refinement');self.view.setToolTip(json.dumps(report,indent=2) if report else '')
            self.change_scale();self.view.message=('Camera Preview - '+report['reason']) if report else 'Camera Preview - selected take / export skeleton';self.view.open(Path(meta['video']),(meta['width'],meta['height']));self.seek()
        except Exception as error:self.failed(str(error))
    def failed(self,message):
        self.view.message=message;self.view.update()
    def seek(self,*args):
        if not self.mode.currentIndex() or self.view.meta is None:return
        meta=self.view.meta;index=round((self.w.motion_view.scene_time-meta['start'])*meta['fps'])
        self.view.outside=not 0<=index<meta['frames']
        self.view.seek(max(0,min(meta['frames']-1,index)))
    def apply_export(self,dialog):
        if self.mode.currentIndex():
            dialog.background_refine.setChecked(self.background.isChecked())
            dialog.animate_camera.setChecked(True);dialog.camera_translation.setChecked(self.translation.isChecked());dialog.translation_scale.setValue(self.scale.value())
    def close(self):
        self.closed=True;self.stop()
