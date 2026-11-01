"""Isolated official FlowHMR inference. Never imported by viewport painting.

Native SMPL-H is retained. The upstream generator conditions on camera rotation,
not metric camera translation; scene placement is a separate constant offset.
"""
from pathlib import Path
import gc, hashlib, json, math, os, sys, uuid
import numpy as np
from .paths import ROOT, UPSTREAM
from .storage import atomic_json, sha256
from .model_session import set_memory_budget
from .compute_policy import gpu_memory_fraction
from .result_artifacts import file_stat
from .resource_cache import code_hash

VERSION=1
MAX_FRAMES=5000  # Pinned FlowHMR per-branch RoPE table; 30 fps.
ASSET_LINKS={
    'flowhmr':'checkpoints/flowhmr_latest/flowhmr_latest.ckpt',
    'flow_config':'checkpoints/flowhmr_latest/config.yml',
    'flow_sam':'ckpts/sam-3d-body-dinov3/model.ckpt',
    'flow_sam_config':'ckpts/sam-3d-body-dinov3/model_config.yaml',
    'flow_mhr':'ckpts/sam-3d-body-dinov3/assets/mhr_model.pt',
    'flow_vggt':'ckpts/vggt-omega/vggt_omega_1b_512.pt',
    'smplh':'assets/body_models/smplh/neutral/model.npz',
}


def scene_root(project):
    state=json.loads((project/'project.json').read_text(encoding='utf-8'))
    if state.get('people_root'):
        root=(project/state['people_root']).resolve()
        if not project.resolve().is_relative_to(root) or not (root/'people.json').is_file():
            raise ValueError('Invalid FlowHMR people scene path')
        return root
    return project


def camera_interval(frames,requested=0):
    if requested not in (-1,0,1,2,5,10,30):raise ValueError('Unsupported camera sampling interval')
    return requested or max(1,math.ceil((frames-1)/127))


def validate_length(frames):
    """Retain the pinned model's real context limit, without memory estimates."""
    if frames<1:raise ValueError('The motion clip is empty')
    if frames>MAX_FRAMES:
        raise ValueError(f'FlowHMR supports up to {MAX_FRAMES:,} frames ({MAX_FRAMES/30:.1f} seconds at 30 fps). Trim the timeline clip; full temporal context is retained.')


def activate():
    import torch
    from .models import BY_KEY,ready
    root=UPSTREAM/'FlowHMR'
    if not (root/'flowhmr/pipeline/v2m_inference.py').is_file():
        raise RuntimeError('The FlowHMR runtime is missing. Repair or reinstall RGB MoCap.')
    for key,name in ASSET_LINKS.items():
        model=BY_KEY[key]
        if not ready(model):continue
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.samefile(model.path):continue
        if target.exists():target.unlink()
        try:os.link(model.path,target)
        except OSError:
            # Models may live on another drive. Copy only during explicit setup,
            # never unexpectedly duplicate multi-GB weights at inference time.
            try:target.symlink_to(model.path)
            except OSError as error:raise RuntimeError('Keep the app and its models on the same drive, or enable Windows symbolic links.') from error
    os.chdir(root)
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    os.environ.setdefault('WANDB_MODE','disabled')
    # An upstream hub import would otherwise require a first-run network fetch.
    original=torch.hub.load
    def local_hub(repo,model,*args,**kwargs):
        if repo=='facebookresearch/dinov3':
            repo=str(root/'third_party/dinov3');kwargs['source']='local'
        return original(repo,model,*args,**kwargs)
    if not getattr(original, '_rgb_flow', False):
        local_hub._rgb_flow=True
        torch.hub.load=local_hub
    upstream_load=torch.load
    allowed={str((root/name).resolve()) for name in ASSET_LINKS.values()}
    def official_load(path,*args,**kwargs):
        if isinstance(path,(str,Path)) and str(Path(path).resolve()) in allowed:
            kwargs.setdefault('weights_only',False)
        return upstream_load(path,*args,**kwargs)
    if not getattr(upstream_load, '_rgb_flow', False):
        official_load._rgb_flow=True
        torch.load=official_load
    return root


def load_tensor(path):
    import torch
    return torch.load(path,map_location='cpu',weights_only=False,mmap=True)


def save_tensor(path,value):
    import torch
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:torch.save(value,temporary);os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def identity(project,stage):
    from .models import BY_KEY,STAGE_REQUIREMENTS
    state=json.loads((project/'project.json').read_text(encoding='utf-8'))
    names={'flow_camera':['input.mp4'], 'flow_features':['input.mp4','tracking.npz'],
           'flowhmr':['flow_tokens.pt','flow_camera.npz','flow_placement.json','tracking.npz'],
           'flow_export':['hmr4d_results.pt']}[stage]
    root=scene_root(project)
    root_state=json.loads((root/'project.json').read_text(encoding='utf-8'))
    selected=root_state.get('flow_options',{}).get('camera_interval',0)
    files={name:file_stat(project/name) for name in names if (project/name).exists()}
    # Full-scene camera is shared even when each person has shorter visible takes.
    if stage=='flow_camera':
        files={'root_input':file_stat(root/'input.mp4')}
        if root!=project and (root/'flow_camera.npz').exists():files['root_camera']=file_stat(root/'flow_camera.npz')
    import inspect
    # Export/rig improvements must not discard expensive camera/SAM/inference.
    functions={'flow_camera':(camera,camera_interval), 'flow_features':(features,_features), 'flowhmr':(infer,validate_motion)}
    if stage=='flow_export':
        code={name:sha256(ROOT/'mocap'/name) for name in ('flow_export.py','source_pose_fit.py','mixamo_rig.py','bvh_bind.py','mixamo_data.py')}
    else:
        code={'adapter':hashlib.sha256(''.join(inspect.getsource(f) for f in (*functions[stage],activate,load_tensor,save_tensor)).encode()).hexdigest()}
    upstream={'flow_camera':'flowhmr/utils/runtime_vggt_camera.py','flow_features':'flowhmr/utils/runtime_sam_features.py','flowhmr':'flowhmr/pipeline/pipeline_v2m.py'}.get(stage)
    if upstream:code[upstream]=sha256(UPSTREAM/'FlowHMR'/upstream)
    if stage in ('flow_camera','flow_features'):
        for name in ('memory_mode.py','gpu_offload.py','model_session.py'):code[name]=sha256(ROOT/'mocap'/name)
    if stage in ('flow_camera','flowhmr'):code['flow_camera.py']=sha256(ROOT/'mocap/flow_camera.py')
    if stage=='flow_features':
        for name in ('flow_sam.py','flow_fast_sam.py','flow_sam_batch.py','batch_prefetch.py'):code[name]=code_hash(ROOT/'mocap'/name)
    if stage=='flow_features':
        from .flow_sam import saved_fast_sam, source_identity, acceleration_options
        code['sam_source']=source_identity(saved_fast_sam(project,state))
        code['sam_acceleration']=acceleration_options(project,state)
    import torch
    return dict(version=VERSION,stage=stage,inputs=files,camera_interval=selected if stage=='flow_camera' else None,
        models={key:file_stat(BY_KEY[key].path) for key in STAGE_REQUIREMENTS[stage] if not (stage=='flow_camera' and selected==-1)},
        code=code,runtime=[str(torch.__version__),torch.version.cuda],frames=state['frames'],fps=state['fps'],take=state.get('person_frames'))


def validate_motion(output,n):
    import torch
    for key,shape in [('rot6d',(1,n,52,6)),('trans',(1,n,3))]:
        if key not in output or tuple(output[key].shape)!=shape or not torch.isfinite(output[key]).all():
            raise ValueError('FlowHMR returned invalid '+key)
    if tuple(output['shapes'].shape)!=(1,1,16) or not torch.isfinite(output['shapes']).all():
        raise ValueError('FlowHMR requires 16 finite SMPL-H shape coefficients')


def camera(project,emit,force=False):
    import torch
    root=scene_root(project)
    state=json.loads((root/'project.json').read_text(encoding='utf-8'));n=state['frames']
    interval=camera_interval(n,state.get('flow_options',{}).get('camera_interval',0))
    if root!=project:
        run_stage(root,'flow_camera','initial',emit,force=force)
        begin,end=json.loads((project/'project.json').read_text(encoding='utf-8'))['person_frames']
        with np.load(root/'flow_camera.npz',allow_pickle=False) as data:
            values={key:data[key][begin:end] for key in ('camera_RT_yup','K')}
        temporary=project/'flow_camera.partial.npz';np.savez(temporary,**values);os.replace(temporary,project/'flow_camera.npz')
        atomic_json(project/'flow_camera_sources.json',{'root_camera':file_stat(root/'flow_camera.npz')})
        return project/'flow_camera.npz'
    if interval==-1:
        from .flow_camera import static_camera
        emit('progress',current=n,total=n,message='Use explicit locked static camera; VGGT skipped')
        target=project/'flow_camera.partial.npz'
        np.savez(target,**static_camera(n,state['width'],state['height']))
        os.replace(target,project/'flow_camera.npz')
        return project/'flow_camera.npz'
    emit('progress',current=0,total=0,message=f'Estimate one scene camera ({interval}-frame sampling)')
    from flowhmr.utils.runtime_vggt_camera import extract_vggt_camera
    target=project/'flow_camera.partial.npz'
    target.unlink(missing_ok=True)  # upstream must not reuse a partial extraction
    from .memory_mode import run as memory_run
    from .gpu_offload import load_vggt,release
    def estimate(low):
        models=[]
        def loader(path,device,camera_only=True):
            model=load_vggt(path,device,camera_only,low_memory=low);models.append(model);return model
        try:
            target.unlink(missing_ok=True)
            return extract_vggt_camera(str(project/'input.mp4'),output_npz=str(target),frame_range=(0,n-1),
                expected_len=n,frame_interval=interval,keep_frames=False,camera_only=True,model_loader=loader)
        finally:
            for model in models:release(model)
            models.clear()
    samples=len(set(range(0,n,interval))|{n-1})
    memory_run('FlowHMR VGGT camera',estimate,6.+.1*samples,emit)
    with np.load(target,allow_pickle=False) as data:
        if data['camera_RT_yup'].shape!=(n,4,4) or data['K'].shape!=(n,3,3) or not all(np.isfinite(data[k]).all() for k in ('camera_RT_yup','K')):
            raise ValueError('VGGT-Omega returned incomplete camera data')
        r=data['camera_RT_yup'][:,:3,:3]
        if not np.allclose(r.swapaxes(-1,-2)@r,np.eye(3),atol=2e-4):raise ValueError('Invalid camera rotations')
    os.replace(target,project/'flow_camera.npz')
    return project/'flow_camera.npz'


def features(project,emit):
    from .memory_mode import run as memory_run
    from .flow_sam import acceleration_options
    state=json.loads((project/'project.json').read_text(encoding='utf-8'))
    estimate=4.5  # Admission for the one-frame fast path; batching adapts after loading.
    return memory_run('FlowHMR SAM features',lambda low:_features(project,emit,low),estimate,emit)


def _features(project,emit,low_memory):
    import cv2, torch
    from .flow_sam import build_extractor, saved_fast_sam, acceleration_options
    state=json.loads((project/'project.json').read_text(encoding='utf-8'));n=state['frames']
    with np.load(project/'tracking.npz',allow_pickle=False) as data:boxes=data['boxes'].copy()
    if boxes.shape!=(n,4):raise ValueError('Tracking and video frame counts differ')
    k=np.array([[math.hypot(state['width'],state['height']),0,state['width']/2],
                [0,math.hypot(state['width'],state['height']),state['height']/2],[0,0,1]],np.float32)
    fast=saved_fast_sam(project,state)
    options=acceleration_options(project,state)
    from .model_session import acquire, model_key
    from .models import BY_KEY
    extractor=acquire(model_key('flow_sam',*[BY_KEY[k].path for k in ('flow_sam','flow_sam_config','flow_mhr')],fast=fast,low_memory=low_memory),
        lambda:build_extractor(torch.device('cuda'),fast=fast,low_memory=low_memory))
    cap=cv2.VideoCapture(str(project/'input.mp4'));cap.set(cv2.CAP_PROP_N_THREADS,2) if hasattr(cv2,'CAP_PROP_N_THREADS') else None
    from .flow_sam import extract_tokens
    tokens=torch.empty((n,3072),dtype=torch.float32);anchors=[]
    try:
        from .flow_sam_batch import current_batch_size, extract_bounded
        batch_size = current_batch_size(n,state['width'],state['height'],enabled=options['sam_batch4'],
                                       low_memory=low_memory,gpu_crops=options['sam_gpu_crop'])
        batch_limit = batch_size
        if fast:
            emit('info', message=f"Fast SAM acceleration: {batch_size} frame(s) per batch; GPU hand crops {'on' if options['sam_gpu_crop'] else 'off'}.")
        if options['sam_batch4'] and low_memory:
            emit('info', message='Low Memory: using one frame per SAM batch to reduce VRAM. GPU hand crop selection is retained.')
        def cpu_batches():
            for begin in range(0,n,batch_size):
                count=min(batch_size,n-begin);images=[]
                for index in range(begin,begin+count):
                    ok,frame=cap.read()
                    if not ok:raise ValueError(f'Cannot decode FlowHMR frame {index+1}')
                    images.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                yield begin,count,images
        from .batch_prefetch import batch_source
        with batch_source(cpu_batches(), batch_bytes=batch_size*state['width']*state['height']*3) as batches:
            for begin,count,images in batches:
                if any(options.values()):
                    batch_limit = min(batch_limit,current_batch_size(n,state['width'],state['height'],
                        enabled=options['sam_batch4'],low_memory=low_memory,gpu_crops=options['sam_gpu_crop']))
                    values, outputs, batch_limit = extract_bounded(extractor, images, boxes[begin:begin+count],
                        torch.from_numpy(k[None]), gpu_crops=options['sam_gpu_crop'], placement=begin<30,
                        limit=batch_limit,emit=emit)
                else:
                    token, outputs = extract_tokens(extractor, images[0], boxes[begin:begin+1],
                        torch.from_numpy(k[None]), placement=begin<30)
                    values = token[None]
                if values.shape != (count,3072) or not torch.isfinite(values).all():
                    raise ValueError(f'Body/hand feature extraction failed at frame {begin+1}')
                tokens[begin:begin+count] = values.float()
                for offset, person in enumerate(outputs):
                    index = begin+offset
                    if index >= min(n,30):break
                    # MHR70 hips, metres in original OpenCV camera coordinates.
                    pelvis = person['pred_keypoints_3d'][[9,10]].mean(0)+person['pred_cam_t']
                    if np.isfinite(pelvis).all() and pelvis[2]>.05:anchors.append([index,*pelvis.tolist()])
                if begin//10 != (begin+count)//10 or begin+count==n:
                    emit('progress',current=begin+count,total=n,message='Extract body and hand features (Fast SAM)' if fast else 'Extract body and hand features (Original SAM)')
        if cap.read()[0]:raise ValueError('Video contains more frames than its project manifest')
    finally:
        cap.release()
        from .model_session import release
        release(extractor)
    save_tensor(project/'flow_tokens.pt',tokens)
    atomic_json(project/'flow_placement.json',dict(camera_pelvis=anchors,intrinsics=k.tolist(),method='Constant X/Z placement from SAM MHR70 pelvis; native FlowHMR height and motion retained'))
    return project/'flow_tokens.pt'


def infer(project,emit):
    import torch
    from flowhmr.pipeline.v2m_inference import build_pipeline,load_yaml,build_feature_dict
    from flowhmr.core.bodymodels.smpl_skeleton import SMPLSkeleton
    from .models import BY_KEY
    checkpoint=BY_KEY['flowhmr'].path
    tokens=load_tensor(project/'flow_tokens.pt');n=len(tokens)
    if tokens.shape!=(n,3072) or not torch.isfinite(tokens).all():raise ValueError('Invalid cached SAM features; rerun feature extraction.')
    with np.load(project/'flow_camera.npz',allow_pickle=False) as data:rt=torch.from_numpy(data['camera_RT_yup'].copy())
    if tuple(rt.shape)!=(n,4,4):raise ValueError('Camera and body frame counts differ')
    from .model_session import acquire,model_key
    def build():
        pipeline=build_pipeline(load_yaml(str(BY_KEY['flow_config'].path)),device='cuda',ckpt=str(checkpoint),validation_steps=20)
        weights=load_tensor(checkpoint)
        weights=weights.get('model_state_dict',weights.get('state_dict',weights))
        if any(name not in weights or tuple(weights[name].shape)!=tuple(value.shape) for name,value in pipeline.named_parameters()):
            raise ValueError('FlowHMR checkpoint is missing required learned parameters. Repair its checkpoint and configuration.')
        return pipeline
    pipeline=acquire(model_key('flowhmr',checkpoint,BY_KEY['flow_config'].path),build)
    emit('progress',current=0,total=20,message='FlowHMR: integrate 20 ODE steps')
    from .flow_camera import relative_camera
    with torch.inference_mode():
        output=pipeline.generate(feature=build_feature_dict(tokens,relative_camera(rt),torch.device('cuda')),seeds=[0],length=n,cfg_scale=1.0)
    output={key:value.detach().cpu() if isinstance(value,torch.Tensor) else value for key,value in output.items()}
    validate_motion(output,n)
    save_tensor(project/'flow_prediction.pt',output)  # untouched official prediction
    del pipeline;gc.collect();torch.cuda.empty_cache()
    placement=json.loads((project/'flow_placement.json').read_text(encoding='utf-8'))
    anchors=np.asarray(placement['camera_pelvis'],float)
    offset=np.zeros(3,np.float32)
    if len(anchors):
        skeleton=SMPLSkeleton(model_path=str(BY_KEY['smplh'].path))
        count=min(n,30)
        with torch.no_grad():
            joints=skeleton(dict(rot6d=output['rot6d'][0,:count],shapes=output['shapes'].reshape(1,16),trans=output['trans'][0,:count]))['keypoints3d'].numpy()
        differences=[]
        for row in anchors:
            index=int(row[0]);r=rt[index,:3,:3].numpy()
            differences.append(r.T@row[1:]-joints[index,0])
        offset[[0,2]]=np.median(differences,axis=0)[[0,2]]
    placed={key:value.clone() if isinstance(value,torch.Tensor) else value for key,value in output.items()}
    placed['trans']=placed['trans']+torch.from_numpy(offset)[None,None]
    placed['scene_offset']=torch.from_numpy(offset)
    placed['source_representation']='smplh';placed['flow_version']=VERSION
    save_tensor(project/'hmr4d_results.pt',placed)
    atomic_json(project/'initial_sources.json',{name:file_stat(project/name) for name in ('flow_prediction.pt','flow_tokens.pt','flow_camera.npz','flow_placement.json','tracking.npz')})
    return project/'hmr4d_results.pt'


def run_stage(project,stage,result,emit,force=False):
    import time,torch,psutil
    from .models import missing
    project=Path(project);started=time.monotonic()
    torch.set_num_threads(4)
    if stage in ('correct','rig','mesh','export','flow_export'):
        return run_post(project,stage,result,emit,force)
    if stage in ('rig','mesh','export'):stage='flow_export'
    if stage not in ('flow_camera','flow_features','flowhmr','flow_export'):
        raise ValueError('This processing stage is not compatible with FlowHMR')
    if result not in ('auto','initial','corrected_initial'):raise ValueError('FlowHMR supports original or corrected original motion only')
    static=(stage=='flow_camera' and json.loads((scene_root(project)/'project.json').read_text(encoding='utf-8')).get('flow_options',{}).get('camera_interval',0)==-1)
    absent=[] if static else missing(stage)
    if absent:raise ValueError('Open FlowHMR Model Manager and prepare: '+', '.join(m.name for m in absent))
    state=json.loads((project/'project.json').read_text(encoding='utf-8'))
    signature=identity(project,stage)
    receipt_path=project/(stage+'_receipt.json');pending=project/(stage+'_pending.json')
    cached=False
    if not force and not pending.exists() and receipt_path.exists():
        try:
            receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
            cached=receipt['identity']==signature and all(Path(name).name==name and file_stat(project/name)==stamp for name,stamp in receipt['outputs'].items())
            if cached and stage=='flow_export':
                from .mixamo_data import current
                from .result_artifacts import artifacts_current
                cached=current(project,'initial') and artifacts_current(project,'initial')
            if cached and stage=='flowhmr':validate_motion(load_tensor(project/'hmr4d_results.pt'),state['frames'])
        except Exception:cached=False  # A damaged cached tensor must recover without AI inference.
    if stage=='flowhmr' and not force:
        from .flow_placement import prediction_current,ensure
        if prediction_current(project):
            activate();ensure(project);cached=True
    if cached:
        emit('complete',stage=stage,output=str(project/next(iter(receipt['outputs']))),cached=True,elapsed=0);return
    root=scene_root(project);root_state=json.loads((root/'project.json').read_text(encoding='utf-8'))
    n=root_state['frames'] if stage=='flow_camera' else state['frames']
    interval=camera_interval(n,root_state.get('flow_options',{}).get('camera_interval',0))
    free=total=None
    if stage!='flow_export' and not static:
        if not torch.cuda.is_available():raise RuntimeError('FlowHMR requires a supported NVIDIA CUDA GPU.')
        free,total=torch.cuda.mem_get_info()
    validate_length(n)
    if free is not None:set_memory_budget(free,total)
    if not static:activate()
    atomic_json(pending,dict(stage=stage,started=time.time()))
    if stage=='flow_camera':output=camera(project,emit,force=force)
    elif stage=='flow_features':output=features(project,emit)
    elif stage=='flowhmr':
        output=infer(project,emit)
        from .flow_placement import ensure
        ensure(project,generated=True)
    else:
        from .flow_export import prepare
        output=prepare(project,emit)
    outputs=[output.name]
    if stage=='flow_camera' and root!=project:outputs.append('flow_camera_sources.json')
    if stage=='flow_features':outputs.append('flow_placement.json')
    if stage=='flowhmr':outputs+=['flow_prediction.pt','initial_sources.json','flow_world.json']
    committed_identity=identity(project,stage)
    if stage=='flow_features' and committed_identity!=signature:
        raise RuntimeError('SAM inputs or settings changed during extraction. Reconstruct 3D again; this extraction was not committed.')
    receipt=dict(identity=committed_identity,outputs={name:file_stat(project/name) for name in outputs})
    atomic_json(receipt_path,receipt);pending.unlink(missing_ok=True)
    latest=json.loads((project/'project.json').read_text(encoding='utf-8'))
    latest.setdefault('pipeline',{}).update(model='flowhmr',result='initial',htd_refine=False,footmr=False)
    latest.setdefault('stages',{})[stage]=dict(status='complete',elapsed=time.monotonic()-started,output=str(output))
    atomic_json(project/'project.json',latest)
    emit('complete',stage=stage,output=str(output),cached=False,elapsed=time.monotonic()-started,
         peak_ram_gib=psutil.Process().memory_info().peak_wset/1024**3,
         peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_initialized() else 0)


def run_post(project,stage,result,emit,force=False):
    """CPU-only contact/export stages, separately validated from AI caches."""
    import time,torch,psutil
    from .models import missing
    from .motion_tools import result_is_current,selected_export
    from .flow_export import prepare
    from .flow_correction import run,code_identity
    kind='initial' if result=='auto' else result
    if kind not in ('initial','corrected_initial'):raise ValueError('FlowHMR supports original or corrected original motion only')
    absent=missing('flow_export')
    if absent:raise ValueError('Import the SMPL-H neutral body model in FlowHMR Model Manager.')
    validate_length(json.loads((project/'project.json').read_text(encoding='utf-8'))['frames'])
    activate()
    from .flow_placement import ensure_group
    ensure_group(project)
    started=time.monotonic();cached=False
    if stage=='correct':
        if kind!='corrected_initial':raise ValueError('Select a corrected result')
        try:
            report=json.loads((project/(kind+'_correction.json')).read_text(encoding='utf-8'))
            from .correction import backend
            cached=(not force and result_is_current(project,kind) and report.get('code')==code_identity()
                    and report.get('backend_sha256')==sha256(Path(backend().__file__)))
            if cached:validate_motion(load_tensor(project/(kind+'.pt')),report['frames'])
        except Exception:cached=False  # Cache decoding failure is recoverable, not an inference failure.
        if not cached:
            activate();output=run(project,emit,kind)
        else:output=project/(kind+'.pt')
    else:
        if kind=='corrected_initial' and not result_is_current(project,kind):
            run(project,emit,kind)
        # Keep both selections available. Missing/corrupt payloads rebuild only
        # this inexpensive derived preview; source inference is never rerun.
        activate()
        for selected in (('initial',kind) if kind!='initial' else ('initial',)):
            current=selected_export(project,selected)
            if force or current is None:output=prepare(project,emit,selected)
            else:output=current
    latest=json.loads((project/'project.json').read_text(encoding='utf-8'))
    latest.setdefault('pipeline',{}).update(model='flowhmr',result=kind,htd_refine=False,footmr=False)
    latest.setdefault('stages',{})[stage]=dict(status='complete',elapsed=time.monotonic()-started,output=str(output))
    atomic_json(project/'project.json',latest)
    if stage in ('export','flow_export','rig','mesh'):(project/'flow_export_pending.json').unlink(missing_ok=True)
    emit('complete',stage=stage,output=str(output),cached=cached,elapsed=time.monotonic()-started,
        peak_ram_gib=psutil.Process().memory_info().peak_wset/1024**3)
