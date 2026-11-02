"""Official masked DroidCalib/Metric3D camera estimation in a bounded worker."""
import gc,importlib.util,sys
import numpy as np


def estimate(pipe,emit):
    import torch
    from .prompthmr_camera_runtime import activate
    from .models import BY_KEY,ready
    from .prompthmr_models import CAMERA_LINKS,CAMERA_KEYS
    from .prompthmr_runtime import configuration
    from .prompthmr_pipeline import link_assets
    folder=activate();engine=configuration()['engine']
    if any(not ready(BY_KEY[k]) for k in CAMERA_KEYS):raise ValueError('Prepare the Moving Camera files in Model Manager.')
    link_assets(__import__('pathlib').Path(engine),CAMERA_LINKS)
    name='pipeline.camera._rgb_windows'
    spec=importlib.util.spec_from_file_location(name,folder/'masked_droid_slam.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    # No network fallback: use only the Model Manager's verified Metric3D file.
    from pipeline.yvanyin_metric3d_main import hubconf
    # The pinned xFormers flash kernels do not support every Windows GPU.
    # Keep the same attention equations/layout using Torch's native backend.
    from mono.model.backbones import ViT_DINO,ViT_DINO_reg
    def attention(q,k,v,attn_bias=None):
        if attn_bias is not None and not isinstance(attn_bias,torch.Tensor):raise ValueError('Unsupported Metric3D attention bias.')
        return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=attn_bias,dropout_p=0.).transpose(1,2)
    previous_attention=[(m,m.memory_efficient_attention) for m in (ViT_DINO,ViT_DINO_reg)]
    original=hubconf.torch.hub.load_state_dict_from_url
    def weights(url,*args,**kwargs):
        if url.endswith('/metric_depth_vit_large_800k.pth'):
            return torch.load(BY_KEY['phmr_metric3d'].path,map_location='cpu',weights_only=False)
        return original(url,*args,**kwargs)
    masks=torch.from_numpy(pipe.results['masks'])
    if masks.shape[0]!=len(pipe.images):raise ValueError('Camera mask/video frame count differs.')
    from pipeline.camera.slam_utils import est_calib,slam_args
    # Allocate for every possible selected keyframe plus trajectory-fill slots;
    # do not silently drop frames or overflow the upstream fixed-size buffer.
    slam_args.buffer=max(32,len(pipe.images)+16)
    emit('progress',current=0,total=0,message='Masked DroidCalib camera tracking and Metric3D scale')
    print(f'PromptHMR camera: DroidCalib + Metric3D ViT-Large; frames={len(pipe.images)}, depth batch=1, keyframe buffer={slam_args.buffer}.',flush=True)
    for m,_ in previous_attention:m.memory_efficient_attention=attention
    hubconf.torch.hub.load_state_dict_from_url=weights
    try:
        r,t,k=module.run_metric_slam(pipe.images,masks=masks,calib=est_calib(pipe.images),
            monodepth_method='metric3d_vit_large',use_depth_inp=False,stride=1,opt_intr=True,save_depth=False)
    finally:
        hubconf.torch.hub.load_state_dict_from_url=original
        for m,previous in previous_attention:m.memory_efficient_attention=previous
        gc.collect();torch.cuda.empty_cache()
    r=np.asarray(r);t=np.asarray(t);k=np.asarray(k)
    from .camera_tracks import valid_rotations
    valid_rotations(r)
    if r.shape!=(len(pipe.images),3,3) or t.shape!=(len(pipe.images),3) or not np.isfinite(t).all():raise ValueError('Invalid or incomplete moving-camera trajectory.')
    if k.shape!=(4,) or not np.isfinite(k).all() or min(k[:2])<=0:raise ValueError('Invalid moving-camera intrinsics.')
    pipe.results['camera']=dict(pred_cam_R=r,pred_cam_T=t,img_focal=k[0],img_center=k[2:])
    pipe.results['has_slam']=True
