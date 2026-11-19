"""FP32 HandFlow inference, bounded images/windows, no body or wrist replacement."""
import contextlib,gc,io,json,os,sys,time
from pathlib import Path


def assign_hands(boxes,confidence,classes,wrists,actor):
    import numpy as np
    result=np.zeros((2,4),np.float32);valid=np.zeros(2,bool);scores=np.zeros(2,np.float32)
    span=max(float(actor[3]-actor[1]),40);pairs=[]
    for side in range(2):
        if wrists[side,2]<.1:continue
        for j,box in enumerate(boxes):
            distance=np.linalg.norm((box[:2]+box[2:])/2-wrists[side,:2])/span
            if distance<.22:pairs.append((distance+.035*(classes[j]!=side)-.015*confidence[j],side,j))
    used=set()
    for _,side,j in sorted(pairs):
        if valid[side] or j in used:continue
        result[side]=boxes[j];valid[side]=True;scores[side]=confidence[j];used.add(j)
    for side in range(2):
        if not valid[side]:
            radius=max(span*.055,12);result[side]=[* (wrists[side,:2]-radius),*(wrists[side,:2]+radius)]
    return result,valid,scores


def infer(project,emit,n):
    import cv2,numpy as np,torch
    from .engines import activate
    from .paths import UPSTREAM,COMPAT,MODELS
    from .models import BY_KEY
    from .compute_policy import retry_cuda
    os.environ['PYOPENGL_PLATFORM']='win32' if os.name=='nt' else 'egl'
    os.environ['WANDB_MODE']='disabled'
    os.environ['YOLO_CONFIG_DIR']=str(MODELS/'handflow/settings')
    # Only pinned/checksummed official weights enter this worker. Upstream
    # Ultralytics 8.2 uses torch.load without the post-2.6 explicit argument.
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    torch.manual_seed(0);np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=activate('HandFlow');sys.path.insert(0,str(UPSTREAM/'HaMeR'));sys.path.insert(0,str(COMPAT))
    started=time.monotonic()
    with np.load(project/'tracking.npz') as tracking:actors=tracking['boxes']
    wrists=torch.load(project/'gvhmr_keypoints.pt',map_location='cpu',weights_only=False).numpy()[:,[9,10]]
    K=torch.load(project/'camera.pt',map_location='cpu',weights_only=False)['K'].numpy()
    if K.ndim==3:K=K[0]
    if actors.shape!=(n,4) or wrists.shape!=(n,2,3):raise ValueError('HandFlow observations have the wrong frame count.')
    allboxes=np.zeros((n,2,4),np.float32);valid=np.zeros((n,2),bool);scores=np.zeros((n,2),np.float32)
    # Match the validated independent pipeline: finish detection before loading
    # the large hand networks. Only small boxes/scores survive this phase.
    from ultralytics import YOLO
    detector=YOLO(str(BY_KEY['hand_detector'].path)).to('cuda')
    capture=cv2.VideoCapture(str(project/'input.mp4'))
    try:
        for i in range(n):
            ok,image=capture.read()
            if not ok:raise ValueError('Cannot read HandFlow video frame '+str(i))
            prediction=detector.predict(image,conf=.3,verbose=False,imgsz=640,device=0,half=False)[0].boxes
            allboxes[i],valid[i],scores[i]=assign_hands(prediction.xyxy.cpu().numpy(),prediction.conf.cpu().numpy(),prediction.cls.cpu().numpy(),wrists[i],actors[i])
            if i%30==0:emit('progress',stage='handflow',current=i,total=n,message='HandFlow: detecting this actor\'s hands')
    finally:capture.release()
    del detector,prediction,image;gc.collect();torch.cuda.empty_cache()
    from omegaconf import OmegaConf
    from yacs.config import CfgNode
    from model.feature_extractors.image_refiner import FrameCompressor
    cfg=OmegaConf.merge(OmegaConf.load(root/'configs/model.yaml'),OmegaConf.load(root/'configs/inference.yaml'))
    os.environ['HANDFLOW_NORMALIZATION_STATS']=str(BY_KEY['handflow_stats'].path)
    os.environ['HAMER_CKPT']=str(BY_KEY['hand_hamer'].path)
    # Compress each frame immediately; never cache gigabytes of patch tokens.
    compressor=FrameCompressor(512,input_dim=1280,num_heads=16,num_layers=8).eval()
    state=torch.load(BY_KEY['handflow'].path,map_location='cpu',weights_only=False)
    state=state.get('module',state);state={k.removeprefix('module.'):v for k,v in state.items()}
    compressor.load_state_dict({k.removeprefix('frame_compressor.'):v for k,v in state.items() if k.startswith('frame_compressor.')},strict=True)
    del state;gc.collect();compressor.cuda()
    from hamer.models import HAMER
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    config=CfgNode.load_cfg(BY_KEY['hand_config'].path.read_text(encoding='utf-8'))
    config.MANO.MODEL_PATH=str(BY_KEY['mano_right'].path.parent);config.MANO.MEAN_PARAMS=str(BY_KEY['hand_mean'].path)
    config.MODEL.BBOX_SHAPE=[192,256];config.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS',None)
    hamer=HAMER(config,init_renderer=False).eval()
    checkpoint=torch.load(BY_KEY['hand_hamer'].path,map_location='cpu',weights_only=False)
    hamer.load_state_dict(checkpoint['state_dict'],strict=True);del checkpoint;gc.collect();hamer.cuda()
    tokens=np.empty((n,2,512),np.float32);landmarks=np.zeros((n,2,21,2),np.float32)
    intrinsics=np.zeros((n,2,4),np.float32)
    with torch.inference_mode():empty=compressor(torch.zeros(1,1,192,1280,device='cuda')).cpu().numpy()[0,0]
    tokens[:]=empty
    captured=[];hook=hamer.backbone.register_forward_hook(lambda m,i,o:captured.append(o))
    free,_=torch.cuda.mem_get_info();batch_limit=8 if free>3*1024**3 else 1
    pending=[]
    def execute(entries):
        images=torch.from_numpy(np.stack([e[4]['img'] for e in entries])).cuda()
        def predict(size):
            outputs=[]
            for first in range(0,len(entries),size):
                captured.clear();prediction=hamer({'img':images[first:first+size]})
                feature=captured[-1].flatten(2).transpose(1,2)
                compressed=compressor(feature[None]).squeeze(0).cpu().numpy()
                outputs.extend([(compressed[k],prediction['pred_cam'][k].cpu().numpy(),prediction['pred_keypoints_3d'][k].cpu().numpy()) for k in range(len(compressed))])
                captured.clear();del prediction
            return outputs
        predictions=retry_cuda(predict,min(batch_limit,len(entries)),lambda size:None)
        for (i,side,H,W,item),(token,pc,points) in zip(entries,predictions):
            tokens[i,side]=token;center=item['box_center'];size=float(item['box_size']);scale=256/size
            cx=(float(K[0,2])-center[0])*scale+128;cy=(float(K[1,2])-center[1])*scale+128
            if side==0:cx=256-cx
            intrinsics[i,side]=[K[0,0]*scale,K[1,1]*scale,cx,cy]
            bc=center.copy();sourceK=K.copy()
            if side==0:bc[0]=W-1-bc[0];sourceK[0,2]=W-1-sourceK[0,2]
            focal=config.EXTRA.FOCAL_LENGTH/config.MODEL.IMAGE_SIZE*max(W,H);bs=size*pc[0]+1e-9
            camera=np.array([2*(bc[0]-W/2)/bs+pc[1],2*(bc[1]-H/2)/bs+pc[2],2*focal/bs]);points=points+camera
            uv=points[:,:2]/points[:,2:]*focal+np.array([W/2,H/2]);uv=np.clip(uv/[W,H],0,1)*[W,H]
            landmarks[i,side]=(uv-sourceK[:2,2])/np.array([sourceK[0,0],sourceK[1,1]])*intrinsics[i,side,:2]/256+intrinsics[i,side,2:]/256
    capture=cv2.VideoCapture(str(project/'input.mp4'))
    try:
        with torch.inference_mode():
            for i in range(n):
                ok,image=capture.read()
                if not ok:raise ValueError('Cannot read HandFlow video frame '+str(i))
                boxes=allboxes[i]
                sides=np.flatnonzero(valid[i])
                if len(sides):
                    dataset=ViTDetDataset(config,image,boxes[sides],sides.astype(np.float32),rescale_factor=2.)
                    with contextlib.redirect_stdout(io.StringIO()):
                        pending.extend((i,int(side),image.shape[0],image.shape[1],dataset[k]) for k,side in enumerate(sides))
                if len(pending)>=batch_limit:execute(pending);pending=[]
                if i%30==0:emit('progress',stage='handflow',current=i,total=n,message='HandFlow: detecting and encoding this actor\'s hands')
            if pending:execute(pending)
    finally:capture.release();hook.remove();captured.clear()
    del hamer,compressor,pending;gc.collect();torch.cuda.empty_cache()
    # Cached image tokens bypass the frozen image encoder. Do not allocate a
    # second unused HaMeR copy while loading the official temporal denoiser.
    import model.flow_matching.denoiser as architecture
    architecture.HaMeRBackbone=lambda **kwargs:torch.nn.Identity()
    model=architecture.HandPoseDenoiser(cfg).eval()
    state=torch.load(BY_KEY['handflow'].path,map_location='cpu',weights_only=False);state=state.get('module',state)
    state={k.removeprefix('module.'):v for k,v in state.items() if not k.removeprefix('module.').startswith('hamer_backbone.')}
    model.load_state_dict(state,strict=True);del state;gc.collect();model.cuda()
    from utils.inference_utils import split_sequence_into_windows,run_fm_inference_with_overlap
    class WindowBatches(torch.nn.Module):
        def __init__(self):super().__init__();self.network=model;self.limit=16
        def forward(self,x,t,batch):
            def predict(size):
                return torch.cat([self.network(x[i:i+size],t[i:i+size],{k:(v[i:i+size] if torch.is_tensor(v) and v.shape[0]==len(x) else v) for k,v in batch.items()}) for i in range(0,len(x),size)])
            return retry_cuda(predict,min(self.limit,len(x)),lambda size:setattr(self,'limit',size))
    wrapper=WindowBatches();rotations=np.zeros((n,2,15,3,3),np.float32)
    from scipy.spatial.transform import Rotation
    with torch.inference_mode():
        for side in range(2):
            torch.manual_seed(0)
            batch=dict(mano_params=torch.zeros(1,n,48),mano_trans=torch.zeros(1,n,3),mano_betas=torch.zeros(1,10),
                padding_mask=torch.zeros(1,n,dtype=torch.bool),images=torch.zeros(1,n,3,1,1,dtype=torch.uint8),
                hamer_landmarks=torch.from_numpy(landmarks[:,side])[None],crop_intrinsics=torch.from_numpy(intrinsics[:,side])[None],
                hamer_confidence=torch.from_numpy(scores[:,side])[None],image_tokens=torch.from_numpy(tokens[:,side])[None],side=['right'],source=['custom'])
            windows,_=split_sequence_into_windows(batch,16,14,torch.device('cuda'))
            windows={k:v.cuda() if torch.is_tensor(v) else v for k,v in windows.items()}
            pose,_,_=run_fm_inference_with_overlap(wrapper,model,windows,16,2,3,torch.device('cuda'),overlap_method='vblend')
            r=Rotation.from_rotvec(pose[:n,3:].cpu().numpy().reshape(-1,3)).as_matrix().reshape(n,15,3,3)
            if side==0:
                mirror=np.diag([-1.,1.,1.]);r=mirror@r@mirror
            rotations[:,side]=r
            emit('progress',stage='handflow',current=side+1,total=2,message='HandFlow: refining finger motion over time')
    if not np.isfinite(rotations).all():raise ValueError('HandFlow produced invalid finger rotations.')
    return rotations,valid,dict(fp32=True,seed=0,window=16,overlap=2,ode_steps=3,window_batch=wrapper.limit,
        crop_batch=batch_limit,seconds=time.monotonic()-started,body_and_wrists_unchanged=True,
        missing='Temporal prediction; unavailable detections remain flagged',image_cache='Per-frame 512-D compressed tokens; no full-video image or patch cache')
