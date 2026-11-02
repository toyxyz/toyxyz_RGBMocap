"""Optional finger-only HaMeR pass, isolated from FlowHMR's Python runtime."""
from pathlib import Path
import json,os,subprocess,uuid,zipfile
from .paths import ROOT,UPSTREAM

MODEL_KEYS=('hand_hamer','hand_mean','mano_right')


def enabled(project):
    from .flow_sam import saved_options
    return saved_options(project,json.loads((Path(project)/'project.json').read_text(encoding='utf-8')))['hamer']


def prediction_name(project,generated=False):
    if generated:use_hamer=enabled(project)
    else:
        try:use_hamer=bool(json.loads((Path(project)/'flowhmr_receipt.json').read_text(encoding='utf-8'))['identity']['code'].get('hamer'))
        except (OSError,ValueError,KeyError,TypeError):use_hamer=False
    return 'flow_hamer_prediction.pt' if use_hamer else 'flow_prediction.pt'


def source_identity():
    from .resource_cache import code_hash
    from .models import BY_KEY
    from .result_artifacts import file_stat
    from .runtime_registry import PROFILES
    files=[ROOT/'mocap/flow_hamer.py',ROOT/'mocap/flow_hamer_entry.py',
           ROOT/'mocap/exmr_backend.py',UPSTREAM/'ExMR/configs/hamer_model_config.yaml',
           UPSTREAM/'ExMR/exmr/preprocessing/hand_features.py',UPSTREAM/'ExMR/exmr/geometry/smoothing.py']
    files+=sorted((UPSTREAM/'ExMR/third_party/hamer/hamer').rglob('*.py'))
    return dict(version=1,code={str(p.relative_to(ROOT)):code_hash(p) for p in files},
                models={k:file_stat(BY_KEY[k].path) for k in MODEL_KEYS},runtime=PROFILES['exmr'],
                hands='every frame, ExMR crops, FP16 network, Gaussian rotation sigma=1 frame')


def hand_identity(project):
    from .result_artifacts import file_stat
    return dict(source=source_identity(),inputs={n:file_stat(Path(project)/n) for n in ('input.mp4','flow_hand_boxes.npy')})


def load_hands(project,n):
    import numpy as np
    with np.load(Path(project)/'flow_hamer_hands.npz',allow_pickle=False) as z:
        r=z['rotations'].copy();valid=z['valid'].copy()
    if r.shape!=(n,2,15,3,3) or valid.shape!=(n,2) or valid.dtype!=bool or not np.isfinite(r).all():
        raise ValueError('Invalid HaMeR finger cache')
    if np.max(np.abs(r.swapaxes(-1,-2)@r-np.eye(3)))>2e-4 or np.min(np.linalg.det(r))<.999:
        raise ValueError('Invalid HaMeR finger rotations')
    return r,valid


def refine(project,output,emit):
    import torch,numpy as np
    from .model_session import current
    from .result_artifacts import file_stat
    project=Path(project);n=output['rot6d'].shape[1];identity=hand_identity(project)
    receipt=project/'flow_hamer_receipt.json';payload=project/'flow_hamer_hands.npz';cached=False
    try:
        value=json.loads(receipt.read_text(encoding='utf-8'))
        cached=value['identity']==identity and value['output']==file_stat(payload)
        if cached:load_hands(project,n)
    except (OSError,ValueError,KeyError,TypeError,EOFError,zipfile.BadZipFile):cached=False
    if not cached:
        # Do not hold FlowHMR weights while the independent ExMR worker runs.
        session=current()
        if session is not None:session.discard()
        import gc
        gc.collect();torch.cuda.empty_cache()
        python=ROOT/'runtime_exmr/python.exe'
        if not python.is_file():python=ROOT/'.venv-exmr/Scripts/python.exe'
        if not python.is_file():raise RuntimeError('HaMeR needs the base ExMR runtime. Run install.cmd to repair it.')
        emit('info',message='FlowHMR HaMeR: every-frame fingers; ExMR crops and Gaussian smoothing. Body, wrists and trajectory are retained.')
        command=[str(python),'-I','-X','utf8',str(ROOT/'mocap/flow_hamer_entry.py'),str(project)]
        from collections import deque
        tail=deque(maxlen=20)
        with subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding='utf-8',errors='replace',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)) as child:
            try:
                for line in child.stdout:
                    tail.append(line[-1000:])
                    if line.startswith('@@FLOW_HAND@@'):
                        event=json.loads(line[len('@@FLOW_HAND@@'):]);kind=event.pop('type');emit(kind,**event)
                if child.wait():raise RuntimeError('FlowHMR HaMeR failed:\n'+''.join(tail)[-4000:])
            except BaseException:
                child.kill();child.wait();raise
        if hand_identity(project)!=identity:raise RuntimeError('HaMeR inputs changed during inference. Reconstruct 3D again.')
    r,valid=load_hands(project,n)
    result=dict(output);result['rot6d']=output['rot6d'].clone()
    for side,offset in enumerate((22,37)):
        pose=torch.from_numpy(r[:,side,:,:,:2].reshape(n,15,6).astype(np.float32))
        result['rot6d'][0,valid[:,side],offset:offset+15]=pose[valid[:,side]]
    result['body_rot6d']=result['rot6d'][:,:,1:]
    result.pop('local_joints_positions',None)
    emit('info',message=f'FlowHMR HaMeR: {int(valid[:,0].sum())}/{n} left and {int(valid[:,1].sum())}/{n} right hand frames; unavailable crops retain FlowHMR fingers.'+(' Verified hand cache reused.' if cached else ''))
    return result


def run(project,emit):
    """Called only by the separate pinned ExMR runtime."""
    import cv2,numpy as np,torch,gc,time,psutil
    started=time.monotonic()
    from scipy.spatial.transform import Rotation
    from .storage import atomic_json
    from .result_artifacts import file_stat
    from .memory_mode import run as memory_run
    from .model_session import set_memory_budget
    from .compute_policy import configure_worker_cpu
    project=Path(project).resolve();state=json.loads((project/'project.json').read_text(encoding='utf-8'));n=state['frames']
    if state.get('pipeline',{}).get('model')!='flowhmr' or not enabled(project) or not 1<=n<=5000:
        raise ValueError('Invalid FlowHMR HaMeR project')
    identity=hand_identity(project)
    boxes=np.load(project/'flow_hand_boxes.npy',allow_pickle=False)
    if boxes.shape!=(n,2,4):raise ValueError('SAM hand boxes do not match the motion')
    torch.set_num_threads(configure_worker_cpu('exmr_features'))
    if not torch.cuda.is_available():raise RuntimeError('HaMeR requires a compatible NVIDIA GPU')
    free,total=torch.cuda.mem_get_info();set_memory_budget(free,total)
    from .exmr_backend import activate,prepare_mano
    root,weights=activate();prepare_mano(root,weights)
    from exmr.preprocessing.hand_features import HamerHandFeatures,_expand_to_aspect_ratio,_crop_hand
    from exmr.geometry.smoothing import gaussian_smooth_rotations
    from .batch_prefetch import batch_source
    W,H=state['width'],state['height'];size=boxes[:,:,2:]-boxes[:,:,:2]
    clipped=np.maximum(0,np.minimum(boxes[:,:,2:],[W,H])-np.maximum(boxes[:,:,:2],0))
    valid=np.isfinite(boxes).all(-1)&(size.min(-1)>2)&(clipped.min(-1)>=4)&(clipped.prod(-1)/np.maximum(size.prod(-1),1)>=.1)
    def infer(low):
        emit('progress',current=0,total=n,phase='hamer',message='Load HaMeR finger model')
        model=HamerHandFeatures(weights/'hamer',fp16=True,batch_size=4,num_workers=0,low_memory=low)
        rotations=np.tile(np.eye(3,dtype=np.float32),(n,2,15,1,1));batch_size=1 if low else 8
        emit('info',message=f'HaMeR fingers: batch {batch_size} hand crop(s); FP16 network; ExMR crops; rotation Gaussian sigma=1 frame.')
        try:
            cfg=model.cfg;mean=255*np.asarray(cfg.MODEL.IMAGE_MEAN);std=255*np.asarray(cfg.MODEL.IMAGE_STD)
            def batches():
                cap=cv2.VideoCapture(str(project/'input.mp4'));pending=[]
                try:
                    for i in range(n):
                        ok,img=cap.read()
                        if not ok:raise ValueError(f'Cannot decode HaMeR frame {i+1}')
                        for side in np.flatnonzero(valid[i]):
                            b=boxes[i,side];center=(b[2:]+b[:2])/2
                            scale=_expand_to_aspect_ratio(2*(b[2:]-b[:2]),cfg.MODEL.BBOX_SHAPE).max()
                            crop=_crop_hand(img.copy(),*center,scale,cfg.MODEL.IMAGE_SIZE,flip=side==0)
                            crop=crop[:,:,::-1].transpose(2,0,1).astype(np.float32)
                            pending.append((i,int(side),((crop-mean[:,None,None])/std[:,None,None]).astype(np.float32)))
                            if len(pending)==batch_size:yield pending;pending=[]
                    if cap.read()[0]:raise ValueError('HaMeR video frame count differs from the project')
                    if pending:yield pending
                finally:cap.release()
            with batch_source(batches(),batch_bytes=batch_size*3*256*256*4) as items,torch.inference_mode():
                for batch in items:
                    image=torch.from_numpy(np.stack([x[2] for x in batch])).cuda()
                    with torch.autocast('cuda',dtype=torch.float16):
                        features=model.model.backbone(image[:,:,:,32:-32]);params,_,_=model.model.mano_head(features)
                    hand=params['hand_pose'].float().cpu().numpy()
                    for k,(i,side,_) in enumerate(batch):
                        r=hand[k]
                        if side==0:
                            mirror=np.diag([-1.,1.,1.]);r=mirror@r@mirror
                        rotations[i,side]=Rotation.from_matrix(r).as_matrix()
                    model._tokens=None
                    emit('progress',current=batch[-1][0]+1,total=n,phase='hamer',message='HaMeR: infer every-frame fingers')
        finally:
            model.close();del model;gc.collect();torch.cuda.empty_cache()
        return rotations
    rotations=memory_run('FlowHMR HaMeR fingers',infer,5.5,emit) if valid.any() else np.tile(np.eye(3,dtype=np.float32),(n,2,15,1,1))
    for side in range(2):
        edges=np.diff(np.r_[False,valid[:,side],False].astype(np.int8));starts=np.flatnonzero(edges==1);stops=np.flatnonzero(edges==-1)
        for a,b in zip(starts,stops):
            aa=Rotation.from_matrix(rotations[a:b,side].reshape(-1,3,3)).as_rotvec().reshape(1,b-a,45)
            with torch.inference_mode():smoothed=gaussian_smooth_rotations(torch.from_numpy(aa.astype(np.float32)),sigma=1.,dim=-2)
            rotations[a:b,side]=Rotation.from_rotvec(smoothed.numpy().reshape(-1,3)).as_matrix().reshape(b-a,15,3,3)
    if hand_identity(project)!=identity:raise RuntimeError('HaMeR inputs changed before commit')
    target=project/'flow_hamer_hands.npz';temp=target.with_name(target.name+'.'+uuid.uuid4().hex+'.partial')
    try:
        with temp.open('wb') as f:np.savez(f,rotations=rotations,valid=valid)
        os.replace(temp,target)
    finally:temp.unlink(missing_ok=True)
    load_hands(project,n)
    atomic_json(project/'flow_hamer_receipt.json',dict(identity=identity,output=file_stat(target)))
    emit('progress',current=n,total=n,phase='hamer',message='HaMeR finger smoothing complete')
    emit('info',message=f'HaMeR finger pass: {time.monotonic()-started:.2f} s; peak allocated VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB; peak worker RAM {psutil.Process().memory_info().peak_wset/2**30:.2f} GiB.')
