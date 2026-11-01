"""FootMR adapter: official ankle weights, bounded images/attention, exact SMPL-X rotations.

Upstream: https://github.com/twehrbein/FootMR (noncommercial research/non-profit).
We load only foot_motion_refiner, not its jointly trained full-body regressor.
"""
from pathlib import Path
import gc
import json
import time

from .storage import atomic_json, sha256
from .motion_tools import RESULTS, result_sources


def resource_plan(free_vram, available_ram):
    """FP32 quality is fixed. Only image batch size and sequence core vary."""
    from .compute_policy import image_plan
    return image_plan('footmr',free_vram,available_ram)


def temporal_predict(model, obs, cliff, rotations, core=1024, halo=720):
    """Exact six-layer local attention context; bounded even for hour-long input.

    Each layer sees at most 120 neighboring tokens. 720 halo frames include
    every dependency of the retained core, including the shifted end masks.
    RoPE uses relative positions, so shifting an entire window is equivalent.
    """
    import torch
    length = len(obs)
    outputs = []
    device = next(model.parameters()).device
    first=0
    from .compute_policy import retry_cuda
    while first<length:
        def reduced(value):
            nonlocal core
            core=value
        def infer(size):
            stop=min(length,first+size);a,b=max(0,first-halo),min(length,stop+halo)
            pred=model(torch.tensor([b-a],device=device),obs[None,a:b].to(device),cliff[None,a:b].to(device),rotations[None,a:b].to(device))
            return pred[0,first-a:stop-a].cpu()
        part=retry_cuda(infer,min(core,length-first),reduced)
        outputs.append(part)
        first+=len(part)
    return torch.cat(outputs)


def extract_keypoints(project, plan, emit):
    """Stream decoded RGB crops; never load a whole source video into RAM/VRAM."""
    import imageio.v3 as iio
    import numpy as np
    import torch
    from hmr4d.utils.preproc.vitpose import VitPoseExtractor
    from hmr4d.utils.preproc.vitfeat_extractor import get_batch
    from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy
    from .models import BY_KEY
    from .engines import save_tensor
    target = project / "footmr_keypoints.pt"
    info_path = project / "footmr_keypoints.json"
    signature = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in
                 (project / "input.mp4", project / "tracking.npz", BY_KEY['vitpose_wholebody'].path)]
    # Include adapter code so preprocessing fixes invalidate observations.
    signature.append(sha256(Path(__file__)))
    from .paths import UPSTREAM
    signature.append(sha256(UPSTREAM / "FootMR/hmr4d/utils/preproc/vitpose_pytorch/src/vitpose_infer/model_builder.py"))
    signature = json.loads(json.dumps(signature))
    if target.exists() and info_path.exists() and json.loads(info_path.read_text(encoding='utf-8')) == signature:
        return torch.load(target, map_location="cpu", weights_only=False), True
    with np.load(project / "tracking.npz") as tracking:
        boxes = get_bbx_xys_from_xyxy(torch.from_numpy(tracking["boxes"]), base_enlarge=1.2)
    pose = VitPoseExtractor(number_joints=23, tqdm_leave=False)
    pose.batch_size = plan["image_batch"]
    reader = iio.imiter(str(project / "input.mp4"), plugin="pyav",
                       filter_sequence=[("scale", "iw*0.5:ih*0.5")], thread_count=2)
    results, first, batch = [], 0, plan["image_batch"]
    try:
        while first < len(boxes):
            frames = []
            for _ in range(min(batch, len(boxes)-first)):
                try:
                    image = next(reader)
                except StopIteration:
                    raise RuntimeError(f"Cannot read FootMR frame {first+len(frames)}")
                # Same FFmpeg scale/colorspace as upstream read_video_np, streamed.
                frames.append(image)
            # get_batch(np) expects full-resolution bbox coordinates for its own scale.
            crops, adjusted = get_batch(np.stack(frames), boxes[first:first+len(frames)]*.5,
                                        img_ds=1.0, path_type="np")
            adjusted *= 2
            offset = 0
            while offset < len(crops):
                count = min(batch, len(crops)-offset)
                from .compute_policy import retry_cuda
                def reduced(value):
                    nonlocal batch
                    batch=value
                    plan['image_batch']=batch
                    pose.batch_size=batch
                output=retry_cuda(lambda size:pose.extract(crops[offset:offset+size],adjusted[offset:offset+size]),count,reduced)
                results.append(output)
                offset += len(output)
            first += len(frames)
            emit("progress", stage="footmr", current=first, total=len(boxes),
                 message=f"FootMR: detecting foot keypoints (batch {batch})")
    finally:
        reader.close()
    keypoints = torch.cat(results)
    save_tensor(keypoints, target)
    atomic_json(info_path, signature)
    del pose
    gc.collect()
    torch.cuda.empty_cache()
    return keypoints, False


def run(project, emit, result="footmr_refined"):
    import torch
    import numpy as np
    import psutil
    from .engines import activate, link_models, save_tensor
    from .models import BY_KEY
    root = activate("FootMR")
    link_models(root)
    from hmr4d.network.footmr.foot_transformer import FootEncoderRoPE
    from hmr4d.utils.geo.hmr_cam import normalize_kp2d, compute_bbox_info_bedlam, get_bbx_xys_from_xyxy
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix
    free, total = torch.cuda.mem_get_info()
    plan = resource_plan(free, psutil.virtual_memory().available)
    started = time.monotonic()
    with torch.inference_mode():
        kp, cached = extract_keypoints(project, plan, emit)
        model = FootEncoderRoPE().eval()
        # The public checkpoint includes the jointly trained body regressor.
        # Only the strict, complete foot module is used; body motion is preserved.
        checkpoint = torch.load(BY_KEY["footmr"].path, map_location="cpu", weights_only=False)
        prefix = "pipeline.foot_motion_refiner."
        state = {k[len(prefix):]: v for k,v in checkpoint["state_dict"].items() if k.startswith(prefix)}
        model.load_state_dict(state, strict=True)
        del checkpoint, state
        model.cuda()
        with np.load(project / "tracking.npz") as tracking:
            boxes = get_bbx_xys_from_xyxy(torch.from_numpy(tracking["boxes"]), base_enlarge=1.2)
            K = torch.from_numpy(tracking["K"]).expand(len(boxes),3,3)
        obs = normalize_kp2d(kp, boxes)[:,15:23]
        cliff = compute_bbox_info_bedlam(boxes, K)
        initial = torch.load(project / "hmr4d_results.pt", map_location="cpu", weights_only=False)
        world_root = axis_angle_to_matrix(initial["smpl_params_global"]["global_orient"])
        cam_root = axis_angle_to_matrix(initial["smpl_params_incam"]["global_orient"])
        world_to_cam = cam_root @ world_root.transpose(-1,-2)
        parents = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
        written = []
        for kind in result_sources(result):
            if not kind.startswith("footmr_"):
                continue
            base = kind.removeprefix("footmr_")
            filename, key = RESULTS[base]
            record = torch.load(project / filename, map_location="cpu", weights_only=False)
            params = record[key]
            # HTD jointly optimizes camera extrinsics. Its camera must condition
            # its refined motion, rather than the original GVHMR camera.
            camera_rotation = record["T_w2c_optimized"][:,:3,:3] if base == "refined" else world_to_cam
            if len(params["body_pose"]) != len(kp):
                raise ValueError("FootMR input motion/video frame counts differ")
            local = axis_angle_to_matrix(torch.cat((params["global_orient"][:,None],params["body_pose"].reshape(-1,21,3)),1))
            rotations = [camera_rotation @ local[:,0]]
            for joint in range(1,22):
                rotations.append(rotations[parents[joint]] @ local[:,joint])
            global_rot = torch.stack(rotations,1)
            condition = matrix_to_rotation_6d(global_rot[:,[4,5,7,8]]).flatten(1)
            pred = temporal_predict(model, obs, cliff, condition, plan["core_frames"], plan["halo_frames"])
            ankle = rotation_6d_to_matrix(pred.reshape(-1,2,6))
            local_ankle = global_rot[:,[4,5]].transpose(-1,-2) @ ankle
            refined = {k:v.clone() for k,v in params.items()}
            refined["body_pose"][:,18:24] = matrix_to_axis_angle(local_ankle).flatten(1)
            if not all(torch.isfinite(v).all() for v in refined.values()):
                raise ValueError("FootMR produced non-finite motion")
            target = project / RESULTS[kind][0]
            save_tensor({"smpl_params_global":refined, "source":base, "method":"FootMR ankle-only adapter"}, target)
            sources = {name: [(project/name).stat().st_size, (project/name).stat().st_mtime_ns]
                       for name in dict.fromkeys((filename, "hmr4d_results.pt", "input.mp4", "tracking.npz"))}
            atomic_json(project / f"{kind}_sources.json", sources)
            written.append(kind)
            emit("progress", stage="footmr", current=len(written), total=2 if result.endswith("refined") else 1,
                 message="FootMR: refining ankle rotations after " + ("HTD" if base == "refined" else "GVHMR"))
    atomic_json(project / "footmr_report.json", {"order":"GVHMR -> HTD (if enabled) -> FootMR", "results":written,
        "plan":plan, "keypoints_cached":cached, "elapsed":time.monotonic()-started,
        "gpu":torch.cuda.get_device_name(), "free_vram_gib":free/1024**3,
        "foot_module_parameters":sum(p.numel() for p in model.parameters()),
        "changes":"Only SMPL-X ankle local rotations (body_pose[18:24]); root, shape, remaining pose unchanged",
        "foot_lock":False})
    return project / RESULTS[result][0]
