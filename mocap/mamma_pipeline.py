"""Cancellable, process-isolated multi-view research pipeline.

Official MAMMA code remains in its separately licensed user installation.
Completed inference is immutable; new attempts cannot replace displayed work.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
import uuid
from .storage import atomic_json
from .image_io import read_image, write_image, unicode_image_paths
from .result_artifacts import file_stat

STAGES = ('mamma_prepare', 'mamma_masks', 'mamma_landmarks', 'mamma_camera', 'mamma_fit', 'mamma_publish')


def emit(kind, **data):
    print('@@MOCAP@@'+json.dumps(dict(type=kind, **data), ensure_ascii=False), flush=True)


def request(path):
    path = Path(path)
    if path.stat().st_size > 2*1024**2:
        raise ValueError('Invalid MAMMA request size.')
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('version') != 1:
        raise ValueError('Unsupported MAMMA request.')
    from .mamma_inputs import video_paths
    paths = video_paths([v['path'] for v in value['inputs']])
    if len(paths) < 2:
        raise ValueError('MAMMA reconstruction requires at least two synchronized fixed-camera videos.')
    if not all(math.isfinite(float(value[k])) for k in ('start','end')) or not 0 <= float(value['start']) < float(value['end']):
        raise ValueError('Invalid common MAMMA trim range.')
    for entry,path in zip(value['inputs'],paths):
        if entry['id']!=path.stem or len(entry['id'])>120:
            raise ValueError('Use short, unique camera filenames matching their saved camera IDs.')
    options=value['options']
    if options.get('camera_mode') not in ('vggt','manual') or not isinstance(options.get('people',0),int) or not 0<=options.get('people',0)<=8:
        raise ValueError('Invalid MAMMA camera/person settings.')
    height=float(options.get('height_m',1.7))
    if not math.isfinite(height) or not .5<=height<=2.5:raise ValueError('Invalid reference height.')
    value['root'] = Path(value['root']).resolve()
    return value


def work_for(job):
    root = job['root']
    relative = job.get('work', '')
    work = (root/relative).resolve()
    if not relative or not work.is_relative_to(root/'mamma'):
        raise ValueError('Invalid MAMMA generation path.')
    return work


def validate_inputs(job):
    from .models import BY_KEY
    for value in job['identity']['inputs']:
        if file_stat(value['path'])!=value['stat']:
            raise ValueError('A source video changed during MAMMA processing. Start a new reconstruction.')
    for key,stamp in job['identity']['models'].items():
        if file_stat(BY_KEY[key].path)!=stamp:
            raise ValueError('A MAMMA model changed during processing. The previous result was preserved.')


def prepare(path, job):
    from .mamma_inputs import common_media, match_calibration
    from .mamma_runtime import configuration
    from .models import missing
    from .project_workspace import create
    from .video import executable, probe
    common = common_media(job['inputs'])
    if job['end'] > common['duration']+1e-4:
        raise ValueError('The common trim extends beyond a camera video.')
    n = round((job['end']-job['start'])*30)
    if not 4 <= n <= 1800:
        raise ValueError('Use 4–1,800 frames for the MAMMA integration; no frames were dropped.')
    if any(m.key not in {'flow_vggt','pear_geocalib'} or job['options']['camera_mode']!='manual' for m in missing('mamma_assets')):
        raise ValueError('Prepare the missing MAMMA models in Model Manager first.')
    options = job['options']
    if options['camera_mode'] == 'manual':
        match_calibration(options['calibration'], job['inputs'])
        for camera in options['calibration']['cameras'].values():
            if abs(camera['intrinsic_matrix'][0][1])>1e-6 or any(abs(float(v))>1e-9 for v in camera.get('distortion_coeffs',[])):
                raise ValueError('This camera adapter requires undistorted pinhole video with zero skew and zero lens distortion. It will not silently discard lens calibration.')
    runtime = configuration()
    from .mamma_runtime import verify_engine
    verify_engine(runtime)
    from .models import BY_KEY, STAGE_REQUIREMENTS
    identity = dict(version=1, inputs=[dict(path=str(Path(v['path']).resolve()),stat=file_stat(v['path']),id=v['id']) for v in job['inputs']],
                    start=job['start'], end=job['end'], frames=n, options=options,
                    engine=runtime['engine_files'], models={k:file_stat(BY_KEY[k].path) for k in STAGE_REQUIREMENTS['mamma_assets'] if options['camera_mode']!='manual' or k not in {'flow_vggt','pear_geocalib'}})
    if options['camera_mode']!='manual':
        from .mamma_gravity import support_identity
        try:identity['gravity_support']=support_identity()
        except (OSError,ValueError,KeyError,TypeError) as error:
            raise RuntimeError('MAMMA background gravity support is unavailable. Close RGB MoCap and run install.cmd, then prepare Scene Gravity / GeoCalib in Model Manager.') from error
    identity['runtime']=dict(packages=runtime['packages'],managed_id=runtime.get('managed_id'))
    # All connector code contributes, not just the model prediction script.
    from .paths import ROOT
    from .storage import sha256
    identity['code']={name:sha256(ROOT/'mocap'/name) for name in ('mamma_pipeline.py','mamma_camera.py','mamma_gravity.py','mamma_gravity_worker.py','mamma_inputs.py','image_io.py','memory_mode.py','gpu_offload.py','worker_progress.py','runtime_registry.py','pear/alignment_support.py')}
    token=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    slot=job['root']/'mamma'/'inference'/token
    cached=False
    try:
        done=json.loads((slot/'complete.json').read_text(encoding='utf-8'))
        cached=(done['identity']==identity and all(file_stat(slot/name)==stamp for name,stamp in done['outputs'].items()))
    except (OSError,ValueError,KeyError,TypeError):pass
    if slot.exists() and not cached:
        # Retain interrupted generations for diagnosis; never reuse mixed files.
        slot=slot.with_name(token+'-'+uuid.uuid4().hex[:12])
    slot.mkdir(parents=True,exist_ok=True)
    create(job['root'],Path(job['inputs'][0]['path']),dict(path=job['inputs'][0]['path'],**common),'mamma')
    job.update(work=slot.relative_to(job['root']).as_posix(), frames=n, identity=identity, cached=cached)
    serial={**job,'root':str(job['root'])}
    atomic_json(path,serial)
    if cached:
        emit('progress',current=1,total=1,message='Reusing verified MAMMA inference; original and applied edits are retained.')
        return
    atomic_json(slot/'pending.json',identity)
    videos=slot/'videos';videos.mkdir()
    entries=[]
    for i,entry in enumerate(job['inputs']):
        emit('progress',current=i,total=len(job['inputs']),message='Prepare synchronized camera '+entry['id'])
        output=videos/(entry['id']+'.mp4')
        actual=probe(Path(entry['path']))
        if any(actual[key]!=entry['metadata'][key] for key in ('width','height','frames','fps')):
            raise ValueError('Source video metadata changed since it was opened. Open the cameras again.')
        # Trim/resample on one common time grid; do not independently reset a
        # camera's start or stretch short footage to fill the other cameras.
        vf=f"trim=start={job['start']:.9f}:end={job['end']:.9f},setpts=PTS-STARTPTS,fps=30,scale=w='min(1920,iw)':h='min(1080,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2"
        subprocess.run([executable('ffmpeg'),'-hide_banner','-loglevel','error','-i',entry['path'],
            '-map','0:v:0','-an','-vf',vf,'-frames:v',str(n),'-fps_mode','cfr','-c:v','libx264',
            '-preset','fast','-crf','18','-pix_fmt','yuv420p','-threads','2',str(output)],
            check=True,creationflags=0x08000000)
        info=probe(output)
        if info['frames']!=n or abs(info['fps']-30)>.001:
            raise ValueError('Camera frame count differs from the common trim; synchronize or shorten the inputs.')
        entries.append(dict(id=entry['id'],path=str(output),size=[info['width'],info['height']],original_size=[entry['metadata']['width'],entry['metadata']['height']]))
    atomic_json(slot/'inputs.json',entries)
    emit('progress',current=len(entries),total=len(entries),message='Synchronized camera clips prepared')


def run_script(engine, section, filename, args):
    import runpy
    directory=engine/section
    sys.path[:0]=[str(directory),str(engine)]
    os.chdir(directory)
    sys.argv=[str(directory/filename),*map(str,args)]
    with unicode_image_paths():
        runpy.run_path(sys.argv[0],run_name='__main__')


def masks(engine, work, job):
    import cv2
    import open_clip
    import yaml
    from .models import BY_KEY
    original_clip=open_clip.create_model_and_transforms
    def local_clip(name,*args,**kwargs):
        if name!='ViT-B-32' or kwargs.get('pretrained')!='laion2b_s34b_b79k':
            raise ValueError('Unsupported MAMMA appearance model; no automatic downloads are allowed.')
        kwargs['pretrained']=str(BY_KEY['mamma_clip'].path)
        return original_clip(name,*args,**kwargs)
    open_clip.create_model_and_transforms=local_clip
    sys.path[:0]=[str(engine/'segmentation'),str(engine)]
    from core.pipeline import SegmentMultipleFrames
    from core.frame_source import VideoSource
    original=SegmentMultipleFrames._resolve_sam_video_source
    def resolve(self,frames,cam_data=None):
        if not isinstance(frames,VideoSource):return original(self,frames,cam_data)
        dest=Path(self.output_path)/frames.cam_name/'_sam_frames_lossless'
        dest.mkdir(parents=True,exist_ok=True)
        for i in range(len(frames)):
            if i%30==0:emit('progress',current=i,total=len(frames),message='Prepare SAM frames: '+frames.cam_name)
            if not write_image(str(dest/f'{i:06}.png'),frames.reader.read_bgr(i)):
                raise OSError('Cannot write a lossless SAM input frame.')
        return str(dest)
    SegmentMultipleFrames._resolve_sam_video_source=resolve
    names=[v['id'] for v in job['inputs']]
    config=yaml.safe_load((engine/'segmentation/configs/sam2.yaml').read_text(encoding='utf-8'))
    # Frozen settings from the completed occlusion trials: a single anatomical
    # identity anchor, bidirectional propagation and bounded CPU video storage.
    config.setdefault('matching',{}).update(max_anchors_per_id=1,init_anchor_enable=False)
    config.setdefault('sam',{}).update(propagate_reverse=True,offload_video_to_cpu=True,
        offload_state_to_cpu=True,offload_state_min_frames=1500,fp16_frames=False)
    config.setdefault('exports',{}).update(export_masked_outputs_mp4=False,export_prompt_overlays_mp4=False,
        export_prompt_similarity_mp4=False,debug_full_masks_npy=False,debug_crop_summary=False)
    config_path=work/'sam2.yaml';config_path.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    args=['--seq_name','motion','--out',work/'masks','--dataset_name','mamma_example',
          '--sam_version','sam2','--yolo-checkpoint',BY_KEY['mamma_yolo'].path,
          '--sam_checkpoint',BY_KEY['mamma_sam2'].path,'--cam_names',*names,
          '--videos_dir',work/'videos','--start',0,'--end',job['frames'],
          '--init_frame',min(30,job['frames']//2),'--cam_init',names[0],
          '--skip_masked_outputs','--skip_collage','--cfg',config_path]
    people=job['options'].get('people',0)
    if people:args+=['--expected_subjects',people]
    run_script(engine,'segmentation','run_ma_masks.py',args)
    raw=work/'masks/mamma_example/motion'
    cleaned=work/'clean-masks/motion'
    count=0
    import numpy as np
    for path in raw.glob('*/masks/*.png'):
        im=read_image(str(path),0)
        if im is None:raise ValueError('MAMMA produced an unreadable mask.')
        _,labels,stats,_=cv2.connectedComponentsWithStats((im>127).astype('uint8'),connectivity=8)
        sizes=stats[1:,cv2.CC_STAT_AREA]
        if len(sizes) and sizes.max()>=128:
            im[np.isin(labels,np.flatnonzero(sizes<8)+1)]=0
        out=cleaned/path.relative_to(raw);out.parent.mkdir(parents=True,exist_ok=True)
        if not write_image(str(out),im):raise OSError('Cannot write cleaned MAMMA mask.')
        count+=1
    if not count:raise ValueError('MAMMA produced no person masks. Check visibility and the initial camera.')


def landmarks(engine,work,job):
    from .models import BY_KEY
    os.environ['MAMMA_SPEED_VARIANT']='fast-cpu-ckpt-exact-roi'
    os.environ['MAMMA_SPEED_DIR']=str(work)
    run_script(engine,'landmarks','run_ma_2d.py',[
        '--config_path','configs/train/models_2d/config_mammanet_mask_512.yaml',
        '--weights',BY_KEY['mamma'].path,'--out_folder',work/'landmarks',
        '--seq_name','motion','--mask_path',work/'clean-masks','--downsampled-verts',BY_KEY['mamma_vertices'].path,
        '--cam_names',*[v['id'] for v in job['inputs']], '--videos_dir',work/'videos',
        '--start',0,'--end',job['frames'],'--no-save_cam_output'])


def fit(engine,work,job):
    import inspect
    import torch
    import numpy as np
    import shutil
    sys.path[:0]=[str(engine/'optimization'),str(engine)]
    from losses import losses
    original=losses.proj_pts_loss
    source=inspect.getsource(original)
    old='torch.clip(per_point_weight[cam_id][..., None], min=vis_clip_value)*pred_in_img'
    if source.count(old)!=1:raise RuntimeError('The configured MAMMA visibility loss is not the validated version.')
    scope=dict(original.__globals__)
    exec(compile(source.replace(old,'torch.clamp(per_point_weight[cam_id][..., None], min=0., max=1.)*pred_in_img'),'<MAMMA visibility bridge>','exec'),scope)
    unfloored=scope['proj_pts_loss']
    def weighted(pts2d,pts3d_pred,intrinsics,extrinsics,loss='mse',per_point_weight=None,vis_clip_value=.8,weight=1.):
        weights=per_point_weight
        if weights is not None:
            weights=[]
            for v in per_point_weight:
                q=((v.mean(-1)-.03)/.12).clamp(0,1);q=q*q*(3-2*q)
                if len(q)>1:
                    kernel=v.new_tensor([1.,4.,6.,4.,1.])[None,None]/16
                    for _ in range(2):
                        q=torch.nn.functional.conv1d(torch.nn.functional.pad(q[None,None],(2,2),mode='replicate'),kernel)[0,0]
                weights.append(v*q[:,None])
        return unfloored(pts2d,pts3d_pred,intrinsics,extrinsics,loss,weights,0.,weight)
    losses.proj_pts_loss=weighted
    from .models import BY_KEY
    os.environ['MAMMA_SPEED_VARIANT']='quiet-fit'
    # Upstream expects its body-part index arrays beside the registered D512
    # matrix. Derive the small NPZ from the user's installed upstream JSON;
    # do not put those restricted correspondence arrays in this repository.
    parts=json.loads((engine/'landmarks/lib/datasets/_smplx_512_body_parts.json').read_text(encoding='utf-8'))
    parts={k:np.asarray(v,dtype=np.int64) for k,v in parts.items()}
    if not {'body','left_head','right_head','left_hand','right_hand','left_feet','right_feet'}<=parts.keys() or any(v.ndim!=1 or np.any((v<0)|(v>=512)) for v in parts.values()):
        raise ValueError('Invalid upstream MAMMA body-part correspondence indices.')
    data=work/'model-data';data.mkdir(exist_ok=True)
    downsample=data/'verts_512.pkl'
    if not downsample.exists():shutil.copy2(BY_KEY['mamma_vertices'].path,downsample)
    np.savez(data/'smplx_512_body_parts.npz',**parts)
    run_script(engine,'optimization','run_ma_3d.py',[
        '--config_file','config_files/contact_configs/config_real_gmf_small_vals_detr_exp_no_vtemplate.yaml',
        '--seq_name','motion','--ma_2d_dir',work/'landmarks','--out_path',work/'fit',
        '--ma_cap_dir',work/'capture','--smplx-models',BY_KEY['mamma_body'].path.parent.parent,
        '--downsampled-verts',downsample,'--cam_names',*[v['id'] for v in job['inputs']],
        '--skip_detection_analysis'])
    outputs=list((work/'fit/motion').glob('smplx_params_body_id-*.npz'))
    if not outputs:raise ValueError('MAMMA did not produce a complete motion result.')
    outputs.extend([work/'camera-world.json',work/'inputs.json',*(work/'videos').glob('*.mp4'),
                    *(work/'landmarks/motion').glob('*.npz')])
    atomic_json(work/'complete.json',dict(identity=job['identity'],outputs={p.relative_to(work).as_posix():file_stat(p) for p in outputs}))
    (work/'pending.json').unlink(missing_ok=True)


def main(argv=None,engine=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=STAGES)
    parser.add_argument('--request',required=True,type=Path)
    args=parser.parse_args(argv);started=time.monotonic()
    try:
        job=request(args.request)
        emit('progress',current=0,total=0,message=args.stage.replace('mamma_','MAMMA: '))
        if args.stage=='mamma_prepare':prepare(args.request,job)
        elif args.stage=='mamma_publish':
            validate_inputs(job)
            from .mamma_publish import publish
            publish(job,emit)
        elif not job.get('cached'):
            validate_inputs(job)
            if engine is None:raise RuntimeError('MAMMA inference requires its separate configured environment.')
            from .paths import HOME
            import tempfile
            cache=HOME/'cache/mamma';cache.mkdir(parents=True,exist_ok=True)
            temp=cache/'temp';temp.mkdir(exist_ok=True);tempfile.tempdir=str(temp)
            os.environ.update(TEMP=str(temp),TMP=str(temp),YOLO_CONFIG_DIR=str(cache/'yolo'),
                HF_HOME=str(cache/'hf'),HF_HUB_OFFLINE='1',MPLCONFIGDIR=str(cache/'mpl'),MPLBACKEND='Agg',
                WANDB_MODE='disabled',NO_ALBUMENTATIONS_UPDATE='1')
            import torch,cv2
            from .compute_policy import configure_worker_cpu,gpu_memory_fraction
            torch.set_num_threads(configure_worker_cpu(args.stage));cv2.setNumThreads(1)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            if not torch.cuda.is_available():raise RuntimeError('MAMMA requires a supported CUDA GPU.')
            free,total=torch.cuda.mem_get_info();torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction(free,total))
            if args.stage!='mamma_camera':
                from .memory_mode import select_stage
                select_stage(args.stage,free,total,job['frames'],emit)
            work=work_for(job)
            if args.stage=='mamma_masks':masks(engine,work,job)
            elif args.stage=='mamma_landmarks':landmarks(engine,work,job)
            elif args.stage=='mamma_camera':
                from .mamma_camera import run
                run(engine,work,job,emit)
            elif args.stage=='mamma_fit':fit(engine,work,job)
            validate_inputs(job)
        emit('complete',stage=args.stage,elapsed=time.monotonic()-started,cached=job.get('cached',False),output=str(job['root']))
        return 0
    except Exception as error:
        traceback.print_exc()
        message=str(error)
        if isinstance(error,MemoryError) or 'out of memory' in message.lower():
            from .memory_mode import GUIDANCE
            message='Processing ran out of memory and stopped safely. '+GUIDANCE
        emit('failed',message=message)
        return 1
