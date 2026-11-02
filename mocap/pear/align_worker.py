"""One-shot fixed-camera background alignment in a separate Flow-profile process."""
import sys,json,time,traceback,struct

def main():
    import os
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(key,'4')
    pipe=sys.stdout.buffer;sys.stdout=sys.stderr
    from .wire import send
    def progress(text):send(pipe,dict(type='status',text=text))
    try:
        request=json.loads(sys.stdin.buffer.readline(65537));folder=request['folder']
        align(folder,progress)
        send(pipe,dict(type='alignment',phase='complete',folder=folder))
    except Exception as exc:
        traceback.print_exc();send(pipe,dict(type='error',text='Camera Align failed: '+str(exc)))

def align(path,progress):
    from .multi_worker import owned_folder,read_parameters,write_sample
    from .alignment_support import activate
    activate()
    from ..paths import ROOT
    from ..storage import atomic_json,sha256
    from ..models import BY_KEY
    import numpy as np,torch,cv2,gc
    from scipy.spatial.transform import Rotation
    sys.path.insert(0,str(ROOT/'third_party/FlowHMR/third_party/vggt-omega'))
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera
    from ..gpu_offload import load_vggt,release
    from geocalib import GeoCalib
    from ..compute_policy import gpu_memory_fraction
    torch.set_num_threads(4);cv2.setNumThreads(2)
    folder=owned_folder(path);job=json.loads((folder/'job.json').read_text('utf8'));n=int(job['count'])
    if not 1<=n<=32 or len(job['images'])!=2*n:raise ValueError('Invalid camera alignment job')
    for key in ('flow_vggt','pear_geocalib'):
        m=BY_KEY[key]
        if sha256(m.path)!=m.expected_sha256:raise ValueError('Prepare '+m.name+' in Model Manager.')
    if job.get('rig_sha256')!=sha256(folder/'rig.npz') or job.get('observations_sha256')!=sha256(folder/'observations.npz'):raise ValueError('Alignment samples changed; run Camera Align again')
    z=read_parameters(folder/'observations.npz')
    uv=z['uv'];boxes=z['boxes'];sizes=z['sizes'];rest=z['rest']
    if uv.shape!=(2,n,145,2) or boxes.shape!=(2,n,4) or sizes.shape!=(n,2) or rest.shape!=(55,3):raise ValueError('Invalid alignment observations')
    images=[];masks=[];paths=[]
    for i,name in enumerate(job['images']):
        if name!=f'sample-{i//n}-{i%n}.png':raise ValueError('Invalid calibration image path')
        image_path=folder/name
        if image_path.stat().st_size>32*1024**2:raise ValueError('Calibration image too large')
        encoded=image_path.read_bytes()
        if len(encoded)<24 or encoded[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('Invalid calibration PNG')
        width,height=struct.unpack('>II',encoded[16:24])
        if not 0<width<=1920 or not 0<height<=1080:raise ValueError('Calibration image dimensions exceed the bounded sample size')
        im=cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_COLOR)
        if im is None:raise ValueError('Missing calibration sample')
        h,w=im.shape[:2];mask=np.ones((h,w),np.uint8);box=boxes[i//n,i%n].copy()
        if np.isfinite(box).all():
            box*=np.tile(np.array([w,h])/sizes[i%n],2);dx=(box[2]-box[0])*.12;dy=(box[3]-box[1])*.08
            x0,y0,x1,y1=np.rint(box+[-dx,-dy,dx,dy]).astype(int);mask[max(0,y0):min(h,y1),max(0,x0):min(w,x1)]=0
        bg=im.copy();bg[mask==0]=0;p=folder/f'background-{i}.png'
        write_sample(p,bg,cv2)
        # One view has no triangulation baseline. Keep actual person depth for
        # its rig-to-depth scale fit; floor/gravity still exclude the person.
        paths.append(image_path if n==1 else p);images.append(im);masks.append(mask)
    start=time.perf_counter();free,total=torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(max(.001,gpu_memory_fraction(free,total)))
    stages=[];progress('Camera Align · Loading VGGT camera/depth model…')
    torch.cuda.reset_peak_memory_stats();t=time.perf_counter();model=load_vggt(BY_KEY['flow_vggt'].path,'cuda',camera_only=False)
    progress('Camera Align · VGGT background cameras and depth…')
    inputs=load_and_preprocess_images([str(p) for p in paths],mode='balanced',image_resolution=512).cuda()
    with torch.inference_mode():pred=model(inputs);E,K=encoding_to_camera(pred['pose_enc'],inputs.shape[-2:])
    torch.cuda.synchronize();E=E[0].cpu().numpy();small=K[0].cpu().numpy();H,W=inputs.shape[-2:];K=small.copy()
    for i in range(2*n):K[i,0,:]*=sizes[i%n,0]/W;K[i,1,:]*=sizes[i%n,1]/H
    depth=pred['depth'][0].cpu().numpy().squeeze(-1);confidence=pred['depth_conf'][0].cpu().numpy()
    pixels=inputs.cpu().numpy();stages.append(dict(stage='VGGT',seconds=time.perf_counter()-t,peak_vram_gib=torch.cuda.max_memory_allocated()/2**30))
    del pred,inputs;release(model);del model;gc.collect();torch.cuda.empty_cache()
    RR=[];tt=[];KK=[]
    for c in range(n):
        es=E[[c,c+n]];r=Rotation.from_matrix(es[:,:,:3]).mean().as_matrix()
        centres=-np.einsum('nij,nj->ni',es[:,:,:3].transpose(0,2,1),es[:,:,3]);RR.append(r);tt.append(-r@np.median(centres,axis=0));KK.append(np.median(K[[c,c+n]],axis=0))
    RR,tt,KK=np.array(RR),np.array(tt),np.array(KK);P=KK@np.concatenate([RR,tt[...,None]],axis=-1)
    progress('Camera Align · GeoCalib background gravity…');t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    model=GeoCalib(str(BY_KEY['pear_geocalib'].path)).cuda().eval();gravity=[];active_mask=None
    def exclude_actor(module,args,out):
        for key in ('up_confidence','latitude_confidence'):
            mask=torch.nn.functional.interpolate(active_mask,size=out[key].shape[-2:],mode='nearest')
            out[key]=out[key]*(mask[:,0] if out[key].ndim==3 else mask)
        return out
    hook=model.model.perspective_decoder.register_forward_hook(exclude_actor)
    with torch.inference_mode():
        for i,(im,mask) in enumerate(zip(images,masks)):
            progress(f'Camera Align · GeoCalib {i+1}/{2*n}')
            tensor=torch.from_numpy(im[...,::-1].copy()).permute(2,0,1).float().cuda()/255
            active_mask=model.image_processor(torch.from_numpy(mask.astype('float32')).cuda()[None,None])['image']
            # The focal prior is in sample pixels, not the original 4K dimensions.
            focal=KK[i%n,1,1]*im.shape[0]/sizes[i%n,1]
            result=model.calibrate(tensor,priors={'focal':tensor.new_tensor(float(focal))})
            g=result['gravity'].vec3d[0].cpu().numpy();gravity.append(E[i,:,:3].T@g)
    stages.append(dict(stage='GeoCalib',seconds=time.perf_counter()-t,peak_vram_gib=torch.cuda.max_memory_allocated()/2**30))
    hook.remove();del model,result,tensor;gc.collect();torch.cuda.empty_cache()
    progress('Camera Align · Fixed-rig scale and background floor…')
    ratios=[]
    for sample in range(2):
        views=[c for c in range(n) if np.isfinite(uv[sample,c,:55]).all()]
        if len(views)<2:continue
        X=[]
        for k in range(55):
            a=np.stack([v for c in views for v in (uv[sample,c,k,0]*P[c,2]-P[c,0],uv[sample,c,k,1]*P[c,2]-P[c,1])])
            _,_,vh=np.linalg.svd(a);X.append(vh[-1,:3]/vh[-1,3])
        X=np.array(X)
        for a,b in ((1,4),(4,7),(2,5),(5,8),(16,18),(18,20),(17,19),(19,21)):
            d=np.linalg.norm(X[a]-X[b])
            if np.isfinite(d) and d>1e-5:ratios.append(np.linalg.norm(rest[a]-rest[b])/d)
    if n==1:
        from .single_alignment import metric_scale
        ratios=metric_scale(z,KK[0],depth,confidence)
    if not ratios:raise ValueError('Insufficient person observations for estimated metric scale')
    scale=float(np.median(ratios))
    if not .01<scale<100:raise ValueError('Unstable camera scale; use clearer, overlapping fixed views')
    up=np.mean(gravity,axis=0);up/=np.linalg.norm(up)
    scatter=float(np.degrees(np.arccos(np.clip(np.asarray(gravity)@up,-1,1))).max())
    if not np.isfinite(up).all() or scatter>30:raise ValueError('Background gravity estimates disagree. Try another calibration time.')
    forward=RR[0].T@np.array([0.,0.,1.]);forward-=up*np.dot(forward,up)
    if np.linalg.norm(forward)<.05:raise ValueError('Reference camera is too close to vertical')
    forward/=np.linalg.norm(forward);right=np.cross(forward,up);Q=np.stack([right,forward,up]);origin=-scale*Q@(-RR[0].T@tt[0]);heights=[]
    for i in range(2*n):
        h,w=depth[i].shape;yy,xx=np.mgrid[:h,:w];rays=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(small[i]).T
        X=scale*((rays*depth[i][...,None]-E[i,:,3])@E[i,:,:3]@Q.T)+origin
        normal=np.cross(X[1:-1,2:]-X[1:-1,:-2],X[2:,1:-1]-X[:-2,1:-1]);normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-9)
        dot=np.zeros((h,w));dot[1:-1,1:-1]=abs(normal[...,2])
        background=cv2.resize(masks[i],(w,h),interpolation=cv2.INTER_NEAREST)>0
        valid=(yy>h*.65)&background&(dot>.95)&(confidence[i]>np.percentile(confidence[i],35))&(pixels[i].max(0)>.04)&np.isfinite(X).all(-1)
        heights.extend(X[valid,2])
    heights=np.array(heights)
    if len(heights)<100:raise ValueError('Not enough visible horizontal background floor. Choose a frame showing the floor.')
    # Fixed bin budget avoids pathological predicted depth ranges allocating huge histograms.
    low,high=np.percentile(heights,[1,99]);bins=min(8192,max(2,int((high-low)/.025)+1))
    hist,edges=np.histogram(heights,bins=bins,range=(low,high));mode=(edges[hist.argmax()]+edges[hist.argmax()+1])/2
    selected=heights[abs(heights-mode)<.06]
    if len(selected)<100:raise ValueError('Background floor estimate is unstable')
    floor=float(np.median(selected));origin[2]-=floor
    extrinsics=np.array([np.c_[r@Q.T,scale*t-(r@Q.T)@origin] for r,t in zip(RR,tt)])
    if not np.isfinite(extrinsics).all():raise ValueError('Nonfinite camera calibration')
    record=dict(version=1,signature=job['signature'],rig_sha256=job['rig_sha256'],intrinsics=KK.tolist(),extrinsics=extrinsics.tolist(),sizes=sizes.tolist(),
        seconds=time.perf_counter()-start,stages=stages,gravity_scatter_deg=scatter,scale=scale,
        floor_residual_p95_m=float(np.percentile(abs(selected-floor),95)),scale_is_estimated=True,
        scale_method='single-view inferred metric rig / predicted person depth' if n==1 else 'multi-view body triangulation')
    atomic_json(folder/'cameras.json',record)
    # Retain only tiny calibration/rig data after success, not captured personal images.
    for p in folder.glob('*.png'):p.unlink()
    progress(f'Camera Align complete · {record["seconds"]:.1f} s')
