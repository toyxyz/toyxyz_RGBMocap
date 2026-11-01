"""Latest-frame PEAR preview. No inference archives, recordings or project writes."""
import sys,threading,json,time,traceback

def main():
    import os
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(key,'4')
    pipe=sys.stdout.buffer
    sys.stdout=sys.stderr
    import faulthandler
    faulthandler.enable(file=sys.stderr)
    faulthandler.dump_traceback_later(40,repeat=True,file=sys.stderr)
    from .wire import send
    condition=threading.Condition();state={'value':None,'serial':0,'stop':False}
    def controls():
        try:
            while True:
                line=sys.stdin.buffer.readline(65537)
                if not line or len(line)>65536:break
                value=json.loads(line)
                with condition:
                    if value.get('stop'):break
                    state['value']=value;state['serial']+=1;condition.notify_all()
        finally:
            with condition:state['stop']=True;condition.notify_all()
    cap=None;camera=None;multi=None
    def release_inputs():
        nonlocal cap,camera
        current=camera;camera=None
        try:
            if current:current.close()
        finally:
            if cap:cap.release();cap=None
    try:
        send(pipe,{'type':'status','text':'Loading PEAR models…'})
        from .core import Pear,cv2,np,torch
        cv2.setNumThreads(2)
        from .projection import Placement
        from .temporal_filter import ParameterFilter
        from .input_transform import rotate_frame,input_rotations
        if not torch.cuda.is_available():raise RuntimeError('PEAR requires a compatible NVIDIA GPU.')
        from ..compute_policy import gpu_reserve
        free,total=torch.cuda.mem_get_info()
        budget=max(1024**2,free-gpu_reserve(total))
        torch.cuda.set_per_process_memory_fraction(min(1.,budget/total))
        model=Pear(mesh=False,fixed_rig=True,profile='fast',progress=lambda text:send(pipe,{'type':'status','text':text}))
        send(pipe,{'type':'status','text':'Warming up live inference…'})
        model.run(np.zeros((480,640,3),dtype=np.uint8),box_override=np.array([160,10,480,470],dtype=np.float32))
        # Detector imports its backend lazily on its first call.
        model.detector.predict(np.zeros((480,640,3),dtype=np.uint8),device=0,classes=[0],conf=.5,imgsz=640,verbose=False)
        model.previous=None;model.frame_counter=0
        faulthandler.cancel_dump_traceback_later()
        # Windows Fortran/CRT DLL initialization can contend with a blocking
        # stdin reader. Start the control reader only after native imports/load.
        threading.Thread(target=controls,daemon=True).start()
        placement=Placement();smooth=ParameterFilter()
        from .fixed_rig import CalibrationFrames
        references=CalibrationFrames();rig=model.rig;pose_id=0;reset_handled=0;reset_warning='';calibration_pose=None
        send(pipe,{'type':'topology','render_mode':'bones','load_seconds':model.load_seconds},parents=np.asarray(model.parents,dtype='int32'))
        seen=-1;source=None;last_index=None;last_options=None;last_epoch=None;last_playing=False;frame_number=0
        while True:
            with condition:
                while not state['stop'] and (state['value'] is None or (state['serial']==seen and camera is None and not (multi and multi.inputs and multi.inputs.mode=='camera'))):
                    condition.wait(.25)
                if state['stop']:break
                request=dict(state['value']);seen=state['serial']
            mode=request.get('mode','video');generation=int(request['generation'])
            if len(request.get('sources',[]))>1 or request.get('align_job') or request.get('calibration'):
                if multi is None:
                    release_inputs()
                    from .multi_worker import MultiPreview
                    send(pipe,{'type':'status','text':'Preparing bounded multi-camera inference…','generation':generation})
                    multi=MultiPreview(model,cv2,lambda meta,**a:send(pipe,meta,**a))
                try:
                    prepared=multi.process(request)=='prepared'
                except (ValueError,OSError) as exc:
                    send(pipe,{'type':'error','recoverable':True,'text':str(exc),'generation':generation,'request':request['request']})
                    prepared=False
                if prepared:
                    # Preserve the prepared event until the GUI closes this worker.
                    with condition:
                        while not state['stop']:condition.wait(.2)
                    break
                continue
            if multi is not None:
                multi.close();multi=None
                # The batched path owns the full network graph; use its eager
                # original after switching back, rather than an obsolete encoder graph.
                model.model.config=dict(model.model.config,graph=False)
                source=None
            angles=input_rotations(request,0 if mode=='idle' else 1)
            smoothing=bool(request.get('smoothing',True))
            options=(angles[0] if angles else 0.,smoothing and bool(request.get('smooth_rotations',True)),
                smoothing and bool(request.get('smooth_position',True)))
            strength=float(request.get('smoothing_strength',50.))
            if strength!=smooth.strength:smooth.set_strength(strength)
            desired=(mode,request.get('camera'),request.get('path'),request.get('resolution'),request.get('restart',0),request.get('camera_streams'))
            if desired!=source:
                release_inputs()
                model.previous=None;placement.reset();smooth.reset();rig.clear();references.clear();reset_warning='';calibration_pose=None;last_index=None;last_epoch=None;source=desired;reset_handled=int(request.get('bone_reset',0))
                try:
                    if mode=='camera':
                        send(pipe,{'type':'status','text':'Opening camera…','generation':generation})
                        from .camera_stream import open_camera
                        streams=request.get('camera_streams')
                        camera=open_camera(request['camera'],request.get('resolution',720),cv2,streams[0] if streams else None)
                    elif mode=='video' and request.get('path'):
                        cap=cv2.VideoCapture(request['path'])
                        if not cap.isOpened():raise RuntimeError('Cannot open the selected video.')
                except Exception as exc:
                    release_inputs();source=None
                    send(pipe,{'type':'error','recoverable':True,'text':str(exc),'generation':generation,'request':request['request']});continue
            if mode=='idle':
                send(pipe,{'type':'status','idle':True,'text':'PEAR ready · Open a video or select a webcam.','generation':generation,'request':request['request']});continue
            if options!=last_options:
                if last_options is None or options[0]!=last_options[0]:
                    model.previous=None;placement.reset();references.clear();smooth.reset()
                else:
                    if options[1]!=last_options[1]:smooth.reset()
                    if options[2]!=last_options[2]:placement.reset_filter()
                last_options=options
            reset_id=int(request.get('bone_reset',0))
            if reset_id>reset_handled:
                reset_handled=reset_id;reset_warning=''
                reference=references.get(request.get('reset_pose'))
                if request.get('reset_pose') is not None:
                    if reference is None:reset_warning='Reset Bones: displayed pose expired. Pause the video or try again.'
                    else:
                        send(pipe,{'type':'status','text':'Resetting bones from the displayed pose…','generation':generation})
                        try:rig.calibrate(*reference);calibration_pose=request['reset_pose']
                        except (ValueError,RuntimeError) as exc:reset_warning='Reset Bones: '+str(exc)
            if mode=='camera':
                try:item=camera.take()
                except Exception as exc:
                    release_inputs();source=None
                    send(pipe,{'type':'error','recoverable':True,'text':str(exc),'generation':generation,'request':request['request']});continue
                if item is None:continue
                image,timestamp=item;index=frame_number;frame_number+=1
                image=rotate_frame(image,options[0])
                started=time.perf_counter()
            elif cap:
                started=time.perf_counter();index=int(request.get('frame',0));fps=max(float(cap.get(cv2.CAP_PROP_FPS)),1.);timestamp=index/fps
                playing=bool(request.get('pair_video'))
                forward=playing and last_playing and last_index is not None and 0<(index-last_index)/fps<=.25
                continuous=forward and last_epoch==generation
                if not continuous:
                    smooth.reset();placement.reset_filter()
                    if not forward and index!=last_index:model.previous=None
                # One requested frame, one prediction. A seek seeds the filter
                # from that pose; it never reconstructs hidden history frames.
                if last_index is None or not last_index<index<=last_index+8:cap.set(cv2.CAP_PROP_POS_FRAMES,index)
                else:
                    for _ in range(index-last_index-1):cap.grab()
                ok,image=cap.read();last_index=index;last_epoch=generation;last_playing=playing
                if not ok:
                    smooth.reset();model.previous=None
                    send(pipe,{'type':'missing','generation':generation,'request':request['request'],'pair_video':bool(request.get('pair_video')),'mode':'video','frame':index,'timeline_frame':request.get('timeline_frame',index),'text':'Cannot decode this video frame.'});continue
                image=rotate_frame(image,options[0])
            else:continue
            result=model.run(image,parameter_filter=smooth if options[1] else None,timestamp=timestamp)
            meta={'type':'frame' if result else 'missing','generation':generation,'request':request['request'],'frame':index,'mode':mode,
                'pair_video':bool(request.get('pair_video')),'timeline_frame':request.get('timeline_frame',index),'requested_at':request.get('requested_at',time.monotonic()),'timestamp':timestamp,
                'smoothing':smoothing,'smooth_rotations':options[1],'smooth_position':options[2],
                'smoothing_strength':strength,'rotation':options[0],'context_frames':0,'inference_frames':1}
            arrays={}
            if result:
                pose_id+=1
                if rig.rest is None:
                    send(pipe,{'type':'status','text':'Calibrating fixed bone lengths…','generation':generation})
                    rig.calibrate(result['body'],result['flame']);calibration_pose=pose_id
                v,j,rot,facing=rig.pose(result['body'],result['camera'])
                v,j=placement.points(v,j,result['camera'],result['crop_bbox'],result['image_size'],smoothing=options[2],timestamp=timestamp,strength=strength)
                arrays.update(bounds=np.ascontiguousarray(v[:,[0,2,1]]*[1,1,-1],dtype='float32'),joints=np.ascontiguousarray(j[:,[0,2,1]]*[1,1,-1],dtype='float32'),rotations=rot,bone_facing=facing)
                references.remember(pose_id,result['body'],result['flame'])
                meta.update(render_mode='bones',pose_id=pose_id,bone_revision=rig.revision,calibration_pose=calibration_pose,
                    bone_reset_handled=reset_handled,bone_reset_warning=reset_warning)
                meta['ms']=round((time.perf_counter()-started)*1000,2)
                meta['peak_vram_mib']=round(torch.cuda.max_memory_allocated()/1024**2,1)
            else:smooth.reset();model.previous=None;meta['text']='No person visible.'
            if mode in ('camera','video'):
                h,w=image.shape[:2];scale=min(1,1280/w,720/h)
                if scale<1:image=cv2.resize(image,(round(w*scale),round(h*scale)))
                arrays['image']=np.ascontiguousarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            send(pipe,meta,**arrays)
    except (BrokenPipeError,EOFError):pass
    except Exception as exc:
        traceback.print_exc()
        try:send(pipe,{'type':'error','text':str(exc),'generation':locals().get('generation')})
        except (OSError,EOFError):pass
    finally:
        release_inputs()
        if multi is not None:multi.close()

class LatestCamera:
    def __init__(self,device,height,cv,on_frame=None):
        self.cv=cv;self.condition=threading.Condition();self.latest=None;self.error=None;self.stop=False;self.cap=None
        self.on_frame=on_frame
        self.thread=threading.Thread(target=self.run,args=(device,height),daemon=True);self.thread.start()
    def run(self,device,height):
        try:
            for backend,index in ((self.cv.CAP_MSMF,device.get('mf_index')),(self.cv.CAP_DSHOW,device.get('ds_index'))):
                if index is None:continue
                cap=None
                try:
                    cap=self.cv.VideoCapture(int(index),backend)
                    if cap.isOpened():
                        cap.set(self.cv.CAP_PROP_FRAME_WIDTH,1280 if height==720 else 640);cap.set(self.cv.CAP_PROP_FRAME_HEIGHT,height)
                        cap.set(self.cv.CAP_PROP_FPS,30)
                        ok,im=cap.read()
                        if ok:self.cap=cap;break
                except Exception:pass
                if cap:cap.release()
            if self.cap is None:raise RuntimeError('Cannot open this webcam. Close other camera applications and try again.')
            while not self.stop:
                timestamp=time.monotonic()
                if self.on_frame:self.on_frame(im,timestamp)
                with self.condition:self.latest=(im,timestamp);self.condition.notify_all()
                ok,im=self.cap.read()
                if not ok:raise RuntimeError('Webcam disconnected or stopped delivering frames.')
        except Exception as e:
            with self.condition:self.error=str(e);self.condition.notify_all()
        finally:
            if self.cap:self.cap.release()
    def take(self,timeout=.2):
        with self.condition:
            if self.latest is None and not self.error:self.condition.wait(timeout)
            if self.error:raise RuntimeError(self.error)
            item=self.latest;self.latest=None;return item
    def close(self):
        self.stop=True;self.thread.join(2)
        if self.thread.is_alive():raise RuntimeError('Camera did not stop; restart PEAR to release the device.')
