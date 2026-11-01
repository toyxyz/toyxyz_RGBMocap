"""VGGT fixed-camera estimation and explicit original-pixel calibration.

Metric scale and gravity are priors when calibration is absent; they are not
measured scene geometry. No per-frame camera or body-pose smoothing is added.
"""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from .storage import atomic_json
from .image_io import read_image, write_image


def infer(work,job,emit):
    import cv2,torch,gc
    from .models import BY_KEY
    from .paths import ROOT
    sys.path.insert(0,str(ROOT/'third_party/FlowHMR/third_party/vggt-omega'))
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images,_balanced_target_shape
    from vggt_omega.utils.pose_enc import encoding_to_camera
    entries=json.loads((work/'inputs.json').read_text(encoding='utf-8'))
    frames=job['frames'];first=0
    def frame(entry,index):
        cap=cv2.VideoCapture(entry['path'])
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,im=cap.read()
            if not ok:raise ValueError('Cannot read a VGGT camera frame.')
            return im
        finally:cap.release()
    for f in sorted(set(min(frames-1,x) for x in (0,10,20,30))):
        if all(float(frame(e,f).mean())>20 for e in entries):first=f;break
    else:raise ValueError('The initial camera images are too dark for VGGT. Adjust the common trim.')
    selected=sorted(set((first,min(max(first+30,60),frames-1),min(max(first+60,90),frames-1))))
    images_dir=work/'camera-images';images_dir.mkdir(exist_ok=True)
    paths=[];sizes=[];camera_ids=[]
    for f in selected:
        for c,entry in enumerate(entries):
            im=frame(entry,f);h,w=im.shape[:2]
            if not .5<=h/w<=2:raise ValueError('VGGT camera calibration currently requires image aspect ratios between 1:2 and 2:1.')
            for mask in (work/'clean-masks/motion'/entry['id']/'masks').glob(f'mask_{f:04}_*.png'):
                m=read_image(str(mask),0)
                if m is None:raise ValueError('Unreadable camera exclusion mask.')
                y,x=np.where(m>127)
                if len(x):im[max(0,y.min()-18):min(h,y.max()+19),max(0,x.min()-18):min(w,x.max()+19)]=0
            path=images_dir/f'{f:06}-{c}.png'
            if not write_image(str(path),im):raise OSError('Cannot prepare a VGGT input image.')
            paths.append(str(path));sizes.append((w,h,_balanced_target_shape(h/w,512,16)));camera_ids.append(c)
    emit('progress',current=0,total=0,message=f'VGGT: estimate {len(entries)} fixed cameras from {len(paths)} background images')
    from .memory_mode import run as memory_run
    from .gpu_offload import load_vggt,release
    def predict(low):
        model=load_vggt(BY_KEY['flow_vggt'].path,'cuda',low_memory=low)
        try:
            images=load_and_preprocess_images(paths,mode='balanced',image_resolution=512).cuda()
            with torch.inference_mode():
                prediction=model(images);E,K=encoding_to_camera(prediction['pose_enc'],images.shape[-2:])
            return E[0].cpu().numpy(),K[0].cpu().numpy(),*images.shape[-2:]
        finally:release(model)
    E,K,H,W=memory_run('MAMMA VGGT camera',predict,6.+.1*len(paths),emit)
    for i,(w,h,(hh,ww)) in enumerate(sizes):
        K[i,0,2]-=(W-ww)//2;K[i,1,2]-=(H-hh)//2
        K[i,0,:]*=w/ww;K[i,1,:]*=h/hh
    if not np.isfinite(E).all() or not np.isfinite(K).all():raise ValueError('VGGT returned invalid cameras.')
    gc.collect();torch.cuda.empty_cache()
    ci=np.array(camera_ids);RR=[];tt=[];KK=[]
    for c in range(len(entries)):
        es=E[ci==c];r=Rotation.from_matrix(es[:,:,:3]).mean().as_matrix()
        centres=-np.einsum('nij,nj->ni',es[:,:,:3].transpose(0,2,1),es[:,:,3])
        RR.append(r);tt.append(-r@np.median(centres,0));KK.append(np.median(K[ci==c],0))
    RR,tt,KK=np.array(RR),np.array(tt),np.array(KK)
    from .mamma_gravity import infer as background_gravity
    up=background_gravity(work,entries,selected,RR,KK,emit)
    uv=[];vis=[]
    for entry in entries:
        with np.load(work/'landmarks/motion'/(entry['id']+'.npz'),allow_pickle=False) as z:
            uv.append(z['landmarks'][...,:2]);vis.append(z['visibilities'])
    uv=np.stack(uv,axis=3);vis=np.stack(vis,axis=3);U=uv.reshape(-1,len(entries),2);V=vis.reshape(-1,len(entries))
    P=KK@np.concatenate((RR,tt[...,None]),axis=-1)
    X=np.empty((len(U),3),np.float64)
    for first in range(0,len(U),4096):
        u=U[first:first+4096];weights=np.clip(V[first:first+4096],.05,1)
        A=np.stack([v for c in range(len(entries)) for v in (
            (u[:,c,0,None]*P[c,2]-P[c,0])*weights[:,c,None],
            (u[:,c,1,None]*P[c,2]-P[c,1])*weights[:,c,None])],axis=1)
        _,_,vh=np.linalg.svd(A,full_matrices=False)
        X[first:first+len(u)]=vh[:,-1,:3]/vh[:,-1,3,None]
    X=X.reshape(*uv.shape[:3],3)
    if not np.isfinite(X).all():raise ValueError('VGGT ray geometry is degenerate. Use manual camera calibration or better-separated views.')
    import pickle
    with BY_KEY['mamma_vertices'].path.open('rb') as f:D=pickle.load(f).cpu().numpy()
    with np.load(BY_KEY['mamma_body'].path,allow_pickle=False) as z:neutral=D@z['v_template']
    head=neutral[:,1]>np.quantile(neutral[:,1],.94);sole=neutral[:,1]<np.quantile(neutral[:,1],.025)
    train=np.arange(min(10,frames-1),min(60,frames),5);cal=X[train]
    feet=np.median(cal[:,:,sole],axis=2);heads=np.median(cal[:,:,head],axis=2);vectors=heads-feet
    lens=np.linalg.norm(vectors,axis=-1)
    valid=(lens>max(1e-6,np.percentile(lens,10)*.5))&(lens<np.percentile(lens,90)*2)
    if not valid.any():raise ValueError('Cannot estimate camera scale from visible standing subjects.')
    from .mamma_gravity import world_basis
    Q=world_basis(up,RR[0])
    # Reference size remains a human-height assumption, independent of gravity.
    # Do not use body lean to rotate the camera world or shrink projected height.
    height=float(np.median(lens[valid]))
    ratio=(np.median(neutral[head,1])-np.median(neutral[sole,1]))/np.ptp(neutral[:,1])
    scale=float(job['options'].get('height_m',1.7))*ratio/height
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid estimated metric camera scale.')
    origin=np.array([0.,0.,scale*np.median((cal[:,:,sole]@Q.T)[...,2])])
    cameras={}
    for c,entry in enumerate(entries):
        r=RR[c]@Q.T;t=scale*tt[c]+r@origin
        cameras[entry['id']]=dict(intrinsic_matrix=KK[c].tolist(),extrinsics_matrix=np.column_stack((r,t)).tolist(),image_size=entry['size'],distortion_coeffs=[0.]*4)
    atomic_json(work/'camera-estimation.json',dict(assumed_height_m=job['options'].get('height_m',1.7),scale=scale,
        initial_frames=train.tolist(),gravity_source='GeoCalib background consensus',gravity_applied_before_3d_fit=True,
        floor_is_measured=False,warning='VGGT-Omega camera geometry and GeoCalib background gravity are estimates. Metric scale assumes the reference person height; the initial sole datum is not measured ground.'))
    return cameras


def run(engine,work,job,emit):
    import yaml
    entries=json.loads((work/'inputs.json').read_text(encoding='utf-8'))
    if job['options']['camera_mode']=='manual':
        from .mamma_inputs import match_calibration
        match_calibration(job['options']['calibration'],job['inputs'])
        cameras={}
        for entry in entries:
            cam=dict(job['options']['calibration']['cameras'][entry['id']])
            K=np.asarray(cam['intrinsic_matrix'],float);w,h=cam['image_size'];rw,rh=entry['size']
            K[0]*=rw/w;K[1]*=rh/h
            cameras[entry['id']]={**cam,'intrinsic_matrix':K.tolist(),'image_size':entry['size']}
    else:cameras=infer(work,job,emit)
    output={'cameras':{}}
    for name,cam in cameras.items():
        E=np.array(cam['extrinsics_matrix']);R=E[:,:3];K=np.array(cam['intrinsic_matrix'])
        q=Rotation.from_matrix(R.T).as_quat()
        output['cameras'][name]=dict(camera_model='pinhole',distortion_model='radtan',
            intrinsics=[float(K[0,0]),float(K[1,1]),float(K[0,2]),float(K[1,2])],resolution=cam['image_size'],
            translation=(-R.T@E[:,3]).tolist(),rotation_quaternion=q[[3,0,1,2]].tolist(),distortion_coeffs=cam.get('distortion_coeffs',[0.]*4))
    (work/'calibration.yaml').write_text(yaml.safe_dump(output,sort_keys=False),encoding='utf-8')
    atomic_json(work/'camera-world.json',cameras)
    from .mamma_pipeline import run_script
    run_script(engine,'capture','run_ma_cap.py',['--videos_dir',work/'videos','--calibration',work/'calibration.yaml',
        '--cam_names',*[e['id'] for e in entries],'--out',work/'capture','--seq_name','motion','--start',0,'--end',job['frames']])
