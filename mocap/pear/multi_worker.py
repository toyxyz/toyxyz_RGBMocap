"""Resident multiview inputs; bounded capture buffers, parallel CPU decode/crops."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,time,uuid,re,shutil,zipfile
import numpy as np
from .input_transform import rotate_frame,input_rotations
from .multiview import Fusion,observations,local_matrices,fk,CV
from .projection import Placement
from .fixed_rig import CalibrationFrames
from .temporal_filter import ParameterFilter
from ..paths import ROOT
from ..storage import atomic_json,sha256

def owned_folder(path):
    p=Path(path).resolve();root=(ROOT/'cache/pear/alignment').resolve()
    if not p.is_relative_to(root) or p==root:raise ValueError('Invalid PEAR alignment cache path')
    return p

def write_sample(path,image,cv):
    ok,encoded=cv.imencode('.png',image)
    if not ok:raise OSError('Cannot encode camera alignment sample')
    Path(path).write_bytes(encoded.tobytes())

def read_parameters(path):
    if path.stat().st_size>2*1024**2:raise ValueError('Alignment parameter file is too large')
    with zipfile.ZipFile(path) as z:
        entries=z.infolist()
        if len(entries)>64 or sum(p.file_size for p in entries)>2*1024**2:raise ValueError('Invalid alignment parameter archive')
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def prune_samples(keep=None):
    """At most four private jobs; never follow links or delete non-job paths."""
    root=ROOT/'cache/pear/alignment';root.mkdir(parents=True,exist_ok=True)
    jobs=[p for p in root.iterdir() if p.is_dir() and re.fullmatch(r'[0-9a-f]{32}',p.name) and not p.is_symlink() and not (getattr(p.lstat(),'st_file_attributes',0)&0x400)]
    ordered=sorted(jobs,key=lambda p:p.stat().st_mtime,reverse=True)
    retained=ordered[:3]
    protected=next((p for p in ordered if keep and p.resolve()==Path(keep).resolve()),None)
    if protected is not None and protected not in retained:retained=retained[:2]+[protected]
    for p in ordered:
        if p not in retained:owned_folder(p);shutil.rmtree(p)

class Inputs:
    def __init__(self,sources,mode,height,pool,cv,streams=None):
        self.sources=sources;self.mode=mode;self.pool=pool;self.cv=cv;self.caps=[];self.last=[None]*len(sources)
        try:
            if streams is not None and len(streams)!=len(sources):raise ValueError('Shared camera count changed')
            for i,s in enumerate(sources):
                if mode=='camera':
                    from .camera_stream import open_camera
                    cap=open_camera(s,height,cv,streams[i] if streams is not None else None)
                else:
                    cap=cv.VideoCapture(s['path'],cv.CAP_FFMPEG,[cv.CAP_PROP_N_THREADS,1,cv.CAP_PROP_OPEN_TIMEOUT_MSEC,2000,cv.CAP_PROP_READ_TIMEOUT_MSEC,2000])
                    if not cap.isOpened():cap.release();raise ValueError('Cannot open camera video: '+Path(s['path']).name)
                self.caps.append(cap)
        except Exception:self.close();raise
    def read(self,timestamp,rotation):
        angles=input_rotations({'rotations':rotation},len(self.caps)) if isinstance(rotation,(list,tuple)) else input_rotations({'rotation':rotation},len(self.caps))
        def one(i):
            cap=self.caps[i]
            if self.mode=='camera':
                item=cap.take()
                return None if item is None else (rotate_frame(item[0],angles[i]),item[1])
            fps=float(cap.get(self.cv.CAP_PROP_FPS))
            if not np.isfinite(fps) or fps<=0:raise ValueError('Camera video has invalid timing')
            index=round(timestamp*fps);last=self.last[i]
            if last is None or not last<index<=last+8:cap.set(self.cv.CAP_PROP_POS_FRAMES,index)
            else:
                for _ in range(index-last-1):cap.grab()
            ok,im=cap.read();self.last[i]=index
            if not ok:raise ValueError('Cannot decode camera '+str(i+1)+' at this time.')
            return rotate_frame(im,angles[i]),index/max(fps,1.)
        items=list(self.pool.map(one,range(len(self.caps))))
        if any(v is None for v in items):return None
        if self.mode=='camera':
            # Arrival-time synchronization is approximate: no claim of hardware genlock.
            times=[v[1] for v in items]
            if max(times)-min(times)>.08:return None
        return [v[0] for v in items],float(np.mean([v[1] for v in items]))
    def close(self):
        errors=[]
        for cap in self.caps:
            try:cap.close() if self.mode=='camera' else cap.release()
            except Exception as e:errors.append(e)
        self.caps=[]
        if errors:raise errors[0]

class MultiPreview:
    def __init__(self,model,cv,send):
        from .batch import BatchRunner
        self.pool=ThreadPoolExecutor(4,thread_name_prefix='pear-cameras');self.batch=BatchRunner(model,self.pool)
        self.model=model;self.cv=cv;self.send=send;self.inputs=None;self.key=None;self.fusion=None;self.align_key=None;self.calibration_sizes=None
        from .foot_lock import LiveFootLock
        self.foot_lock=LiveFootLock();self.lock_options=None;self.position_anchor=None
        self.placement=Placement();self.filter=ParameterFilter();self.last_time=None;self.last_generation=None;self.pose_id=0;self.reference=None;self.reset_handled=0;self.references=CalibrationFrames()
    def close(self):
        try:
            if self.inputs:self.inputs.close();self.inputs=None
        finally:self.pool.shutdown(wait=True,cancel_futures=True)
    def configure(self,request):
        sources=request['sources'];mode=request['mode']
        if not 1<=len(sources)<=32:raise ValueError('PEAR needs 1–32 input slots.')
        if mode=='video':
            for entry in sources:
                st=Path(entry['path']).stat()
                if entry.get('file_stamp') and entry['file_stamp']!=[st.st_size,st.st_mtime_ns]:raise ValueError('A camera video changed. Open the videos again before tracking.')
        self.rotations=input_rotations(request,len(sources))
        key=json.dumps([mode,sources,request['resolution'],self.rotations,request.get('camera_streams')],sort_keys=True)
        if key!=self.key:
            if self.inputs:self.inputs.close();self.inputs=None
            self.inputs=Inputs(sources,mode,request['resolution'],self.pool,self.cv,request.get('camera_streams'));self.key=key
            self.batch.reset(len(sources));self.model.rig.clear();self.fusion=None;self.align_key=None
            self.filter.reset();self.placement.reset();self.last_time=None;self.reference=None;self.references.clear();self.reset_handled=int(request.get('bone_reset',0))
        calibration=request.get('calibration')
        if calibration!=self.align_key:
            self.fusion=None;self.align_key=None;self.calibration_sizes=None
            if calibration:
                folder=owned_folder(calibration);data=json.loads((folder/'cameras.json').read_text('utf8'))
                if data['signature']!=request['signature'] or len(data['intrinsics'])!=len(sources):raise ValueError('Camera alignment no longer matches these inputs.')
                if data.get('rig_sha256')!=sha256(folder/'rig.npz'):raise ValueError('Camera alignment rig changed. Run Camera Align again.')
                z=read_parameters(folder/'rig.npz')
                body={k[5:]:v for k,v in z.items() if k.startswith('body_')}
                flame={k[6:]:v for k,v in z.items() if k.startswith('flame_')}
                self.send({'type':'status','text':'Restoring aligned fixed-rig calibration…','generation':request['generation']})
                self.model.rig.calibrate(body,flame);self.reference=(body,flame)
                self.fusion=Fusion(self.model.rig,data);self.align_key=calibration;self.calibration_sizes=data['sizes']
    def thumbnails(self,images):
        out=[]
        for im in images:
            h,w=im.shape[:2];s=min(128/w,72/h);thumb=np.zeros((72,128,3),np.uint8)
            small=self.cv.resize(im,(max(1,round(w*s)),max(1,round(h*s))))
            thumb[:small.shape[0],:small.shape[1]]=small[...,::-1];out.append(thumb)
        return np.stack(out)
    def prepare_alignment(self,request,images,packets,timestamp):
        prune_samples(request.get('calibration'))
        folder=ROOT/'cache/pear/alignment'/uuid.uuid4().hex;folder.mkdir(parents=True)
        n=len(images);samples=[];uv=[];boxes=[];sizes=[];paths=[];camera_relative=[]
        for sample in range(2):
            if sample:
                if request['mode']=='camera':
                    until=time.monotonic()+1.;fresh=None
                    while time.monotonic()<until:
                        item=self.inputs.read(timestamp,self.rotations)
                        if item:fresh=item
                    if fresh is None:raise ValueError('Webcams did not deliver synchronized fresh samples. Check connections.')
                    images,_=fresh;packets=self.batch.run(images)
                else:
                    end=float(request.get('duration',timestamp+2))-1/float(request.get('fps',30))
                    next_time=min(timestamp+2,end)
                    if next_time<=timestamp:raise ValueError('Leave at least two video frames after the playhead for Camera Align.')
                    images,_=self.inputs.read(next_time,self.rotations);packets=self.batch.run(images)
            if sum(p is not None for p in packets)<min(2,n):raise ValueError('Camera Align needs the same person visible in the selected views. Choose another time.')
            obs=np.full((n,145,2),np.nan);bb=np.full((n,4),np.nan);relative=np.full((n,55,3),np.nan)
            for c,(im,p) in enumerate(zip(images,packets)):
                path=folder/f'sample-{sample}-{c}.png'
                # Limit calibration source pixels; intrinsics are returned in original dimensions.
                h,w=im.shape[:2];scale=min(1.,1920/w,1080/h);small=self.cv.resize(im,(round(w*scale),round(h*scale))) if scale<1 else im
                write_sample(path,small,self.cv)
                paths.append(path.name)
                if p is not None:
                    obs[c]=observations(p,self.model.rig);bb[c]=p['bbox']
                    local,_=fk(local_matrices(p['body']),self.model.rig)
                    relative[c]=local[:55]@(CV@p['camera'][:3,:3]).T
                if sample==0:sizes.append([w,h])
                elif sizes[c]!=[w,h]:raise ValueError('Camera dimensions changed during alignment.')
            uv.append(obs);boxes.append(bb);camera_relative.append(relative)
        body,flame=self.reference
        np.savez(folder/'rig.npz',**{'body_'+k:v for k,v in body.items()},**{'flame_'+k:v for k,v in flame.items()})
        np.savez(folder/'observations.npz',uv=np.asarray(uv),boxes=np.asarray(boxes),sizes=np.asarray(sizes),rest=self.model.rig.rest,camera_relative=np.asarray(camera_relative))
        atomic_json(folder/'job.json',dict(version=1,signature=request['signature'],images=paths,count=n,
            rig_sha256=sha256(folder/'rig.npz'),observations_sha256=sha256(folder/'observations.npz')))
        self.send({'type':'alignment','phase':'prepared','folder':str(folder),'generation':request['generation']})
    def process(self,request):
        started=time.perf_counter();self.configure(request)
        timestamp=float(request.get('timeline_frame',0))/float(request.get('fps',30))
        item=self.inputs.read(timestamp,self.rotations)
        if item is None:
            if self.inputs.mode=='camera':self.send({'type':'status','text':'Waiting for synchronized camera frames (arrival skew ≤80 ms)…','generation':request['generation']})
            return
        images,source_time=item
        if self.fusion and [list(im.shape[1::-1]) for im in images]!=self.calibration_sizes:raise ValueError('Camera image dimensions changed. Run Camera Align again.')
        if request['mode']=='camera':timestamp=source_time
        continuous=self.last_time is not None and 0<timestamp-self.last_time<=.25 and self.last_generation==request['generation']
        if not continuous:
            self.filter.reset();self.placement.reset_filter();self.foot_lock.reset();self.position_anchor=None
            if self.fusion:self.fusion.reset()
            if self.last_time!=timestamp:self.batch.reset(len(images))
        self.last_time=timestamp;self.last_generation=request['generation']
        packets=self.batch.run(images);rig=self.model.rig
        available=[p for p in packets if p is not None]
        if rig.rest is None and available:
            rig.calibrate(available[0]['body'],available[0]['flame']);self.reference=(available[0]['body'],available[0]['flame'])
        if request.get('align_job'):
            if not available:raise ValueError('No person visible for Camera Align.')
            self.prepare_alignment(request,images,packets,timestamp);return 'prepared'
        reset=int(request.get('bone_reset',0))
        if reset>self.reset_handled:
            # UI invalidates camera calibration before allowing a new body scale.
            self.reset_handled=reset
            reference=self.references.get(request.get('reset_pose'))
            if reference is None:raise ValueError('Reset Bones: displayed pose expired. Pause and try again.')
            rig.calibrate(*reference);self.reference=reference;self.filter.reset();self.placement.reset_filter()
        selected=int(request.get('preview',0));selected=max(0,min(selected,len(images)-1));arrays={}
        smooth_rotations=bool(request['smoothing'] and request.get('smooth_rotations',True))
        smooth_position=bool(request['smoothing'] and request.get('smooth_position',True))
        if self.fusion:arrays=self.fusion.run(packets,timestamp,request['smoothing'],request['smoothing_strength'],
            smooth_rotations=request.get('smooth_rotations',True),smooth_position=request.get('smooth_position',True)) or {}
        elif packets[selected] is not None:
            p=packets[selected];body=p['body'];self.filter.set_strength(request['smoothing_strength'])
            if smooth_rotations:body,_=self.filter.update(body,p['flame'],timestamp)
            else:self.filter.reset()
            v,j,r,f=rig.pose(body,p['camera']);v,j=self.placement.points(v,j,p['camera'],p['crop_bbox'],p['image_size'],smooth_position,timestamp,strength=request['smoothing_strength'])
            arrays=dict(bounds=np.ascontiguousarray(v[:,[0,2,1]]*[1,1,-1],dtype='float32'),joints=np.ascontiguousarray(j[:,[0,2,1]]*[1,1,-1],dtype='float32'),rotations=r,bone_facing=f)
        foot_metrics={}
        options=(request.get('foot_lock',True),request.get('world_position',True),rig.revision,self.align_key)
        if options!=self.lock_options:self.foot_lock.reset();self.position_anchor=None;self.lock_options=options
        if arrays:
            if not request.get('world_position',True):
                from .projection import lock_position
                anchor=request.get('position_anchor')
                if anchor is not None:self.position_anchor=np.asarray(anchor,dtype='float32')
                if self.position_anchor is None:self.position_anchor=arrays['joints'][0].copy()
                arrays=lock_position(arrays,self.position_anchor)
            arrays,foot_metrics=self.foot_lock.apply(arrays,rig,timestamp,
                enabled=bool(request.get('foot_lock',True) and self.fusion is not None),move_root=request.get('world_position',True))
        else:self.foot_lock.reset()
        valid=bool(arrays);self.pose_id+=1
        reference_packet=packets[selected] or (available[0] if available else None)
        if reference_packet:self.references.remember(self.pose_id,reference_packet['body'],reference_packet['flame'])
        im=images[selected];h,w=im.shape[:2];s=min(1,1280/w,720/h)
        if s<1:im=self.cv.resize(im,(round(w*s),round(h*s)))
        arrays['image']=np.ascontiguousarray(im[...,::-1]);arrays['thumbnails']=self.thumbnails(images)
        self.send(dict(type='frame' if valid else 'missing',generation=request['generation'],request=request['request'],mode=request['mode'],
            preview=selected,frame=self.pose_id if request['mode']=='camera' else request.get('timeline_frame',0),timeline_frame=request.get('timeline_frame',0),pair_video=request.get('pair_video',False),
            requested_at=request.get('requested_at',time.monotonic()),render_mode='bones',pose_id=self.pose_id,bone_revision=rig.revision,
            bone_reset_handled=self.reset_handled,ms=(time.perf_counter()-started)*1000,views=len(images),available_views=len(available),
            aligned=self.fusion is not None,smooth_rotations=smooth_rotations,smooth_position=smooth_position,
            foot_lock=foot_metrics,text='' if valid else 'No person visible in the selected inputs.'),**arrays)
