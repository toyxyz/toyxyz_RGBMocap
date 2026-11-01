"""Official ExMR in its own Python 3.12 worker; shared SMPL-X postprocessing.

Native predictions are immutable inputs. Only the hand-mean representation is
converted for the shared evaluator; this does not rotate or refit the source.
"""
from pathlib import Path
import gc, hashlib, json, math, os, sys, time
import numpy as np
from .paths import ROOT, HOME, UPSTREAM
from .storage import atomic_json, sha256
from .resource_cache import code_hash
from .model_session import set_memory_budget
from .compute_policy import gpu_memory_fraction
from .result_artifacts import file_stat
from .exmr_options import saved

STAGES=('exmr_preprocess','exmr_features','exmr')
VERSION=2
LINKS={'exmr':'exmr/exmr.safetensors','smplx':'body_models/smplx/SMPLX_NEUTRAL.npz',
       'exmr_sapiens':'sapiens2/sapiens2_1b_pose.safetensors',
       'flow_sam':'sam-3d-body-dinov3/model.ckpt','flow_sam_config':'sam-3d-body-dinov3/model_config.yaml',
       'flow_mhr':'sam-3d-body-dinov3/assets/mhr_model.pt',
       'hand_hamer':'hamer/hamer_ckpts/checkpoints/hamer.ckpt',
       'hand_mean':'hamer/data/mano_mean_params.npz'}


def load(path):
    import torch
    return torch.load(path,map_location='cpu',weights_only=False,mmap=True)


def scene_root(project):
    state=json.loads((project/'project.json').read_text(encoding='utf-8'))
    if not state.get('people_root'):return project
    root=(project/state['people_root']).resolve()
    if not project.resolve().is_relative_to(root) or not (root/'people.json').is_file():
        raise ValueError('Invalid ExMR scene path')
    return root



def activate():
    import torch,cv2
    from .models import BY_KEY,ready
    root=UPSTREAM/'ExMR'
    for name in ('exmr','third_party/sapiens2/sapiens','third_party/hamer/hamer','third_party/dinov3'):
        if not (root/name).is_dir():raise RuntimeError('ExMR runtime is missing. Repair the application installation.')
    for path in (root,root/'third_party/sapiens2',root/'third_party/hamer'):
        if str(path) not in sys.path:sys.path.insert(0,str(path))
    os.environ.update(PYOPENGL_PLATFORM='win32',WANDB_MODE='disabled',PYTHONNOUSERSITE='1')
    weights=HOME/'cache/exmr/assets';weights.mkdir(parents=True,exist_ok=True)
    for key,name in LINKS.items():
        source=BY_KEY[key].path
        if not ready(BY_KEY[key]):continue
        target=weights/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.samefile(source):continue
        if target.exists():target.unlink()
        try:os.link(source,target)
        except OSError:
            try:target.symlink_to(source)
            except OSError as exc:raise RuntimeError('Keep ExMR models and the app on the same drive, or enable Windows symbolic links.') from exc
    original=torch.hub.load
    def local_hub(repository,*args,**kwargs):
        if repository=='facebookresearch/dinov3':
            repository=str(root/'third_party/dinov3');kwargs['source']='local'
        return original(repository,*args,**kwargs)
    if not getattr(original, '_rgb_exmr', False):
        local_hub._rgb_exmr=True
        torch.hub.load=local_hub
    config=root/'configs/hamer_model_config.yaml'
    if sha256(config)!='0e5eeb82752e47dfd01db8e13ccc4c5eba9bf83f53da8285523b8d3e87247aa3':
        raise RuntimeError('ExMR HaMeR configuration is damaged. Repair the application.')
    import shutil
    shutil.copy2(config,weights/'hamer/hamer_ckpts/model_config.yaml')
    from exmr.preprocessing import video as V
    from exmr.preprocessing.image_io import read_bgr_frame
    def read_frame(path):
        image=read_bgr_frame(path)
        return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    V.read_frame=read_frame
    return root,weights


def prepare_mano(root,weights):
    """Official restricted-array conversion; source registered model untouched."""
    from .models import BY_KEY
    import runpy,pickle
    source=BY_KEY['mano_right'].path;target=weights/'hamer/data/mano/MANO_RIGHT.pkl'
    receipt=target.with_suffix('.receipt.json');stamp=file_stat(source)
    if target.exists() and receipt.exists():
        prior=json.loads(receipt.read_text(encoding='utf-8'))
        if prior.get('source')==stamp and prior.get('output')==file_stat(target):return
    helper=runpy.run_path(str(root/'tools/download_models.py'))
    arrays=helper['_load_mano'](source);helper['_check_mano'](arrays,source)
    target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_suffix('.partial')
    with temporary.open('wb') as f:pickle.dump(arrays,f)
    os.replace(temporary,target);atomic_json(receipt,dict(source=stamp,output=file_stat(target)))


def camera(project,emit):
    """One rotation estimate for the whole scene, sliced for visible takes."""
    import torch
    root=scene_root(project);state=json.loads((root/'project.json').read_text(encoding='utf-8'))
    config=saved(root,state);context=dict(input=file_stat(root/'input.mp4'),frames=state['frames'],static=config['static_camera'],version=3,odometry=sha256(UPSTREAM/'ExMR/exmr/preprocessing/visual_odometry.py'),reader=sha256(Path(__file__).with_name('exmr_camera.py')))
    receipt=root/'exmr_camera.json';path=root/'exmr_camera.npy'
    valid=False
    try:valid=json.loads(receipt.read_text(encoding='utf-8'))['identity']==context and json.loads(receipt.read_text(encoding='utf-8'))['output']==file_stat(path)
    except (OSError,ValueError,KeyError):pass
    if valid:
        rotations=np.load(path,allow_pickle=False)
        valid=rotations.shape==(state['frames'],3,3) and np.isfinite(rotations).all()
    if not valid:
        if config['static_camera']:rotations=np.broadcast_to(np.eye(3,dtype=np.float32),(state['frames'],3,3)).copy()
        else:
            emit('progress',current=0,total=0,message='ExMR: estimate one scene camera rotation (SimpleVO)')
            from .exmr_camera import rotations as estimate_rotations
            rotations=np.asarray(estimate_rotations(root/'input.mp4',state['frames']),np.float32)
        if rotations.shape!=(state['frames'],3,3) or not np.isfinite(rotations).all():raise ValueError('ExMR camera has incomplete frames')
        temporary=root/'exmr_camera.partial.npy';np.save(temporary,rotations);os.replace(temporary,path)
        atomic_json(receipt,dict(identity=context,output=file_stat(path)))
    if not np.allclose(rotations.swapaxes(-1,-2)@rotations,np.eye(3),atol=2e-4):raise ValueError('Invalid ExMR camera rotations')
    local=json.loads((project/'project.json').read_text(encoding='utf-8'))
    first,end=local.get('person_frames',(0,state['frames']))
    # Root camera coordinates are retained in this receipt for shared placement.
    atomic_json(project/'exmr_camera_sources.json',dict(root=os.path.relpath(root,project),root_stat=file_stat(path),
        first=first,reference_rotation=rotations[0].tolist(),first_rotation=rotations[first].tolist()))
    return torch.from_numpy(rotations[first:end].copy())


def identity(project,stage):
    from .models import BY_KEY,STAGE_REQUIREMENTS
    state=json.loads((project/'project.json').read_text(encoding='utf-8'))
    files={'exmr_preprocess':['input.mp4','tracking.npz'],
           'exmr_features':['input.mp4','tracking.npz','exmr_keypoints.pt'],
           'exmr':['tracking.npz','exmr_keypoints.pt','exmr_body_features.pt','exmr_hand_features.pt']}[stage]
    root=scene_root(project);root_state=json.loads((root/'project.json').read_text(encoding='utf-8'))
    return dict(version=VERSION,stage=stage,inputs={name:file_stat(project/name) for name in files},
        memory_policy={name:sha256(ROOT/'mocap'/name) for name in ('memory_mode.py','gpu_offload.py','model_session.py','batch_prefetch.py')},
        root_input=file_stat(root/'input.mp4'),take=state.get('person_frames'),frames=state['frames'],fps=state['fps'],
        options=saved(root,root_state) if stage=='exmr' else {'pose':'Sapiens2-1B 448x336'},
        models={key:file_stat(BY_KEY[key].path) for key in STAGE_REQUIREMENTS[stage]},
        image_readers={name:sha256(UPSTREAM/'ExMR/exmr/preprocessing'/name) for name in {
            'exmr_preprocess':('image_io.py','keypoints/sapiens2.py'),
            'exmr_features':('image_io.py','body_features.py','hand_features.py'),
            'exmr':(),
        }[stage]},
        feature_decoder={name:sha256(UPSTREAM/'ExMR/third_party/sam-3d-body/sam_3d_body/models'/name)
            for name in ('decoders/promptable_decoder.py','meta_arch/sam3d_body.py')} if stage=='exmr_features' else {},
        upstream_api=sha256(UPSTREAM/'ExMR/exmr/pipeline.py'),
        motion_solver={name:sha256(UPSTREAM/'ExMR/exmr'/name) for name in
            ('model/exmr.py','model/global_traj.py','model/ik.py','geometry/camera.py')} if stage=='exmr' else {},
        code=code_hash(Path(__file__)),upstream=sha256(UPSTREAM/'ExMR/RGB_MOCAP_PROVENANCE.json'))


def frame_dir(project,n):
    import cv2
    from .video import executable
    import subprocess
    directory=project/'exmr_frames';directory.mkdir(exist_ok=True)
    identity=dict(input=file_stat(project/'input.mp4'),frames=n,version=1)
    try:
        if json.loads((directory/'receipt.json').read_text(encoding='utf-8'))==identity and len(list(directory.glob('*.jpg')))==n:return directory
    except (OSError,ValueError):pass
    for path in directory.glob('*.jpg'):path.unlink()
    subprocess.run([executable('ffmpeg'),'-v','error','-i',str(project/'input.mp4'),'-vsync','0','-q:v','2','-start_number','0','-threads','2','-y',str(directory/'%06d.jpg')],check=True,creationflags=0x08000000)
    if len(list(directory.glob('*.jpg')))!=n:raise ValueError('ExMR frame count differs from project')
    atomic_json(directory/'receipt.json',identity)
    return directory


def canonical_params(params,body):
    converted={k:v.detach().cpu().clone() for k,v in params.items()}
    # SMPL-X adds pose_mean to axis angles. ExMR's flat-hand AA is converted
    # exactly to that existing convention, preserving every finger rotation.
    for side in ('left','right'):
        converted[side+'_hand_pose']-=getattr(body,side+'_hand_mean').detach().cpu()
    return converted


def validate_native(native,n):
    import torch
    shapes={'body_pose':(n,63),'global_orient':(n,3),'transl':(n,3),'betas':(n,10),'left_hand_pose':(n,45),'right_hand_pose':(n,45)}
    for space in ('smplx_global','smplx_incam'):
        for key,shape in shapes.items():
            value=native[space][key]
            if tuple(value.shape)!=shape or not torch.isfinite(value).all():raise ValueError('Invalid ExMR '+space+'/'+key)
        beta=native[space]['betas']
        if not torch.allclose(beta,beta[:1].expand_as(beta),atol=1e-5,rtol=0):raise ValueError('ExMR returned changing actor shape')
    if tuple(native['static_conf_logits'].shape)!=(n,6) or not torch.isfinite(native['static_conf_logits']).all():raise ValueError('Invalid ExMR contact predictions')


def complete_stage(project,stage,output,elapsed,cached,emit):
    state=json.loads((project/'project.json').read_text(encoding='utf-8'))
    state.setdefault('stages',{})[stage]=dict(status='complete',output=str(output),elapsed=elapsed)
    atomic_json(project/'project.json',state)
    emit('complete',stage=stage,cached=cached,elapsed=0 if cached else elapsed,output=str(output))


def run_stage(project,stage,result,emit,force=False,feature_phase=None):
    import torch,cv2,psutil
    from .models import missing,BY_KEY
    from .engines import save_tensor
    project=Path(project);state=json.loads((project/'project.json').read_text(encoding='utf-8'));n=state['frames']
    if stage not in STAGES or result not in ('auto','initial','corrected_initial'):raise ValueError('Unsupported ExMR stage/result')
    if feature_phase not in (None,'body','hand') or (feature_phase and stage!='exmr_features'):raise ValueError('Invalid ExMR feature phase')
    if n<1:raise ValueError('ExMR clip is empty')
    absent=missing(stage)
    if absent:raise ValueError('Open ExMR Model Manager and prepare: '+', '.join(m.name for m in absent))
    signature=identity(project,stage);receipt=project/(stage+'_receipt.json');pending=project/(stage+'_pending.json')
    outputs={'exmr_preprocess':['exmr_keypoints.pt'],'exmr_features':['exmr_body_features.pt','exmr_hand_features.pt'],
             'exmr':['exmr_results.pt','hmr4d_results.pt','initial_sources.json','exmr_camera_sources.json']}[stage]
    cached=False
    try:
        old=json.loads(receipt.read_text(encoding='utf-8'))
        cached=not force and not pending.exists() and old['identity']==signature and all(file_stat(project/k)==v for k,v in old['outputs'].items())
        if cached:
            for name in outputs:
                if name.endswith('.pt'):
                    value=load(project/name)
                    if stage=='exmr' and name=='exmr_results.pt':validate_native(value,n)
                    elif stage!='exmr' and (len(value)!=n or not torch.isfinite(value).all()):cached=False
    except (OSError,ValueError,KeyError,RuntimeError,EOFError):cached=False
    if cached:
        complete_stage(project,stage,project/outputs[0],old.get('elapsed',0),True,emit);return
    if not torch.cuda.is_available():raise RuntimeError('ExMR requires a supported NVIDIA CUDA GPU. Check System in Help for details.')
    started=time.monotonic()
    free,total=torch.cuda.mem_get_info();set_memory_budget(free,total)
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(0);np.random.seed(0)
    torch.cuda.reset_peak_memory_stats();atomic_json(pending,dict(identity=signature))
    # Invalidate the old source generation as soon as new observations begin.
    atomic_json(project/'exmr_pending.json',dict(stage=stage,identity=signature))
    root,weights=activate()
    from .model_session import acquire,model_key,release
    from exmr import pipeline as P
    with np.load(project/'tracking.npz',allow_pickle=False) as tracking:
        boxes=torch.from_numpy(tracking['boxes'].copy()).float();k=torch.from_numpy(tracking['K'].copy()).float()
    if boxes.shape!=(n,4) or not torch.isfinite(boxes).all():raise ValueError('ExMR actor tracking has the wrong frame count')
    if k.ndim==2:k=k[None].repeat(n,1,1)
    if tuple(k.shape)!=(n,3,3) or not torch.isfinite(k).all():raise ValueError('Invalid original camera intrinsics')
    if stage=='exmr_preprocess':
        from exmr.preprocessing.keypoints.sapiens2 import Sapiens2Keypoints
        emit('progress',current=0,total=n,message='ExMR: body, hand and foot keypoints (Balanced)')
        from .memory_mode import run as memory_run
        directory=frame_dir(project,n)
        def keypoints(low):
            model=acquire(model_key('sapiens',BY_KEY['exmr_sapiens'].path,low_memory=low),
                lambda:Sapiens2Keypoints(BY_KEY['exmr_sapiens'].path,resolution='448x336',num_workers=0,low_memory=low))
            try:return model(directory,boxes)
            finally:release(model)
        values=memory_run('ExMR Sapiens',keypoints,6.,emit)
        save_tensor(values,project/outputs[0])
    elif stage=='exmr_features':
        directory=frame_dir(project,n)
        from exmr.preprocessing.body_features import SamBodyFeatures
        from .memory_mode import run as memory_run
        def body_features(low):
            model=acquire(model_key('exmr_sam',*[BY_KEY[key].path for key in ('flow_sam','flow_sam_config','flow_mhr')],low_memory=low),
                lambda:SamBodyFeatures(weights/'sam-3d-body-dinov3',batch_size=8,num_workers=0,low_memory=low))
            try:return model(directory,boxes,k)
            finally:release(model)
        body_receipt=project/'exmr_body_partial.json'
        body_valid=False
        # A hand-only phase reuses the validated body partial, including after
        # a forced body phase in this same pipeline. Force still rebuilds hands.
        if feature_phase is not None and (not force or feature_phase=='hand'):
            try:
                partial=json.loads(body_receipt.read_text(encoding='utf-8'))
                body_valid=partial['identity']==signature and partial['output']==file_stat(project/'exmr_body_features.pt')
                if body_valid:
                    values=load(project/'exmr_body_features.pt')
                    body_valid=tuple(values.shape)==(n,1024) and bool(torch.isfinite(values).all())
                    del values
            except (OSError,ValueError,KeyError,RuntimeError,EOFError):pass
        if not body_valid:
            emit('progress',current=0,total=n,message='ExMR: SAM body features')
            values=memory_run('ExMR body features',body_features,5.5,emit)
            save_tensor(values,project/'exmr_body_features.pt');del values;gc.collect();torch.cuda.empty_cache()
            if identity(project,stage)!=signature:raise ValueError('ExMR inputs changed during body feature extraction')
            atomic_json(body_receipt,dict(identity=signature,output=file_stat(project/'exmr_body_features.pt')))
        if feature_phase=='body':
            emit('complete',stage=stage,output=str(project/'exmr_body_features.pt'),cached=body_valid,elapsed=time.monotonic()-started)
            return
        prepare_mano(root,weights)
        from exmr.preprocessing.hand_features import HamerHandFeatures
        points=load(project/'exmr_keypoints.pt')
        hand_boxes=P.hand_boxes(points,'sapiens2')
        emit('progress',current=0,total=n,message='ExMR: left and right hand features')
        def hand_features(low):
            model=acquire(model_key('hamer',*[BY_KEY[key].path for key in ('hand_hamer','hand_mean','mano_right')],low_memory=low),
                lambda:HamerHandFeatures(weights/'hamer',batch_size=8,num_workers=0,low_memory=low))
            try:return model(directory,hand_boxes)
            finally:release(model)
        values=memory_run('ExMR hand features',hand_features,5.5,emit)
        save_tensor(values,project/'exmr_hand_features.pt')
    else:
        config=saved(scene_root(project));rotations=camera(project,emit)
        if len(rotations)!=n:raise ValueError('ExMR take camera and motion lengths differ')
        data=P.model_inputs(load(project/'exmr_keypoints.pt'),boxes,k,rotations,
            load(project/'exmr_body_features.pt'),load(project/'exmr_hand_features.pt'),'sapiens2')
        emit('progress',current=0,total=n,message='ExMR: reconstruct and refine full body, feet and fingers')
        from exmr.model.exmr import ExMR
        model=acquire(model_key('exmr',BY_KEY['exmr'].path,BY_KEY['smplx'].path),
            lambda:ExMR.from_pretrained(BY_KEY['exmr'].path,BY_KEY['smplx'].path))
        try:
            with torch.inference_mode():
                output=model(**{key:value[None].cuda() for key,value in data.items()},
                    static_cam=config['static_camera'],snap_to_ground=False,
                    smooth_incam=False,smooth_global=config['smoothing']!='none',fps=state['fps'],
                    normalize_world=False)
                native={space:{key:value[0].cpu() for key,value in output[space].items()} for space in ('smplx_incam','smplx_global')}
                native.update(static_conf_logits=output['static_conf_logits'][0].cpu(),K_fullimg=k,bbx_xyxy=boxes)
                del output
        finally:release(model)
        del model
        native['fps']=state['fps'];validate_native(native,n)
        import smplx
        body=smplx.SMPLX(str(BY_KEY['smplx'].path),flat_hand_mean=False,use_pca=False).eval()
        record={'smpl_params_global':canonical_params(native['smplx_global'],body),
                'smpl_params_incam':canonical_params(native['smplx_incam'],body),
                'net_outputs':{'static_conf_logits':native['static_conf_logits'][None]},'source_model':'exmr'}
        save_tensor(native,project/'exmr_results.pt')
        save_tensor(record,project/'hmr4d_results.pt')
        atomic_json(project/'initial_sources.json',{name:file_stat(project/name) for name in ('exmr_results.pt','exmr_keypoints.pt','exmr_body_features.pt','exmr_hand_features.pt','tracking.npz','exmr_camera_sources.json')})
    if identity(project,stage)!=signature:raise ValueError('ExMR inputs changed during processing; retry the stage')
    atomic_json(receipt,dict(identity=signature,outputs={name:file_stat(project/name) for name in outputs},
        peak_vram_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_vram_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
        elapsed=time.monotonic()-started,ground_snap=False,native_precision=True))
    pending.unlink(missing_ok=True)
    if stage=='exmr':(project/'exmr_pending.json').unlink(missing_ok=True)
    complete_stage(project,stage,project/outputs[0],time.monotonic()-started,False,emit)
