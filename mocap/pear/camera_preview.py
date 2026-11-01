"""Camera-only preview: bounded latest frames, no PEAR/Torch/model loading."""
import sys,json,threading,time,traceback

class CameraPreview:
    def __init__(self,cv):
        self.cv=cv;self.cameras=[];self.sources=None;self.latest=[]
        self.publishers=[];self.publish_lock=threading.Lock();self.closing=False
    def close(self):
        self.closing=True;cameras=self.cameras;self.cameras=[];self.sources=None;self.latest=[]
        errors=[]
        for camera in cameras:
            try:camera.close()
            except Exception as exc:errors.append(exc)
        if errors:raise errors[0]
        for publisher in self.publishers:
            if publisher:publisher.close()
        self.publishers=[]
    def publish(self,index,image,timestamp):
        from .camera_stream import FrameWriter,MAX_TOTAL
        with self.publish_lock:
            if self.closing:return
            if self.publishers[index] is None:
                if sum(p.size for p in self.publishers if p)+image.nbytes>MAX_TOTAL:raise RuntimeError('Shared webcam frames exceed 512 MiB. Choose a lower webcam resolution.')
                self.publishers[index]=FrameWriter(image.shape)
            publisher=self.publishers[index]
        publisher.publish(image,timestamp)
    def update(self,request,force=False):
        import numpy as np
        from .worker import LatestCamera
        from .input_transform import rotate_frame,input_rotations
        sources=request['sources'];slots=request['slots'];count=int(request['slot_count'])
        if not 1<=len(sources)<=count<=32 or len(slots)!=len(sources) or len(set(slots))!=len(slots) or any(not 0<=s<count for s in slots):raise ValueError('Invalid webcam preview slots')
        if len({s['id'] for s in sources})!=len(sources):raise ValueError('Each webcam slot must use a different device.')
        angles=input_rotations(request,len(sources))
        key=json.dumps([sources,request['resolution']],sort_keys=True)
        if key!=self.sources:
            self.close()
            self.closing=False;self.publishers=[None]*len(sources)
            try:
                for i,device in enumerate(sources):self.cameras.append(LatestCamera(device,request['resolution'],self.cv,on_frame=lambda im,t,i=i:self.publish(i,im,t)))
            except Exception:self.close();raise
            self.sources=key;self.latest=[None]*len(sources);force=True
        fresh=False
        for i,camera in enumerate(self.cameras):
            try:item=camera.take(timeout=0.)
            except Exception as exc:raise RuntimeError(sources[i]['name']+': '+str(exc)) from exc
            if item is not None:self.latest[i]=item[0];fresh=True
        if not fresh and not force:return None
        thumbs=np.zeros((count,72,128,3),np.uint8);arrays={};selected=int(request['preview'])
        for slot,raw,angle in zip(slots,self.latest,angles):
            if raw is None:continue
            image=rotate_frame(raw,angle);h,w=image.shape[:2];scale=min(128/w,72/h)
            small=self.cv.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))))
            thumbs[slot,:small.shape[0],:small.shape[1]]=small[...,::-1]
            if slot==selected:
                scale=min(1,1280/w,720/h)
                if scale<1:image=self.cv.resize(image,(round(w*scale),round(h*scale)))
                arrays['image']=np.ascontiguousarray(image[...,::-1])
        if count>1:arrays['thumbnails']=thumbs
        with self.publish_lock:
            streams=[dict(p.descriptor(),slot=slot,device_id=device['id']) for slot,device,p in zip(slots,sources,self.publishers) if p is not None]
        return dict(type='camera_preview',generation=request['generation'],preview=selected,ready='image' in arrays,streams=streams),arrays

def main():
    pipe=sys.stdout.buffer;sys.stdout=sys.stderr
    from .wire import send
    condition=threading.Condition();state={'request':None,'serial':0,'stop':False}
    def controls():
        try:
            while True:
                line=sys.stdin.buffer.readline(65537)
                if not line or len(line)>65536:break
                request=json.loads(line)
                with condition:
                    if request.get('stop'):break
                    state['request']=request;state['serial']+=1;condition.notify_all()
        finally:
            with condition:state['stop']=True;condition.notify_all()
    threading.Thread(target=controls,daemon=True).start()
    preview=None
    try:
        import cv2
        cv2.setNumThreads(2);preview=CameraPreview(cv2);seen=-1;notice=0;last_frame=time.monotonic()
        while True:
            started=time.monotonic()
            with condition:
                while not state['stop'] and state['request'] is None:condition.wait(.1)
                if state['stop']:break
                request=dict(state['request']);force=seen!=state['serial'];seen=state['serial']
            packet=preview.update(request,force)
            if packet:
                meta,arrays=packet;send(pipe,meta,**arrays)
                if meta['ready']:last_frame=time.monotonic()
            elif time.monotonic()-last_frame>2 and time.monotonic()-notice>1:
                notice=time.monotonic();send(pipe,dict(type='status',generation=request['generation'],text='Waiting for webcam frames…'))
            with condition:condition.wait(max(.001,1/30-(time.monotonic()-started)))
    except Exception as exc:
        traceback.print_exc();send(pipe,dict(type='error',text='Webcam preview: '+str(exc)))
    finally:
        if preview:preview.close()
