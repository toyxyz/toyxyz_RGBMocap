from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .storage import atomic_json
from .box_avatar import global_rotations
from .motion_tools import resolve_motion_result, RESULTS, result_sources, result_is_current, base_result, body_result

JOINT_NAMES = ["Hips", "LeftUpLeg", "RightUpLeg", "Spine", "LeftLeg", "RightLeg", "Spine1", "LeftFoot", "RightFoot",
               "Spine2", "LeftToeBase", "RightToeBase", "Neck", "LeftShoulder", "RightShoulder", "Head",
               "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand"]
PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])


def write_bvh(path: Path, rest_joints: np.ndarray, root_positions: np.ndarray, local_rotvec: np.ndarray, fps=30.0):
    """Export actual local SMPL-X rotations to a 22-joint, Y-up, cm BVH."""
    if local_rotvec.shape[1:] != (22, 3) or root_positions.shape != (len(local_rotvec), 3):
        raise ValueError("Expected [T,22,3] local rotations and [T,3] root positions")
    if rest_joints.shape != (22, 3) or fps <= 0:
        raise ValueError("Invalid skeleton or frame rate")
    if not all(np.isfinite(array).all() for array in (rest_joints, root_positions, local_rotvec)):
        raise ValueError("Motion contains non-finite values")
    order, lines = [], ["HIERARCHY"]

    def visit(index, depth):
        indent = "  " * depth
        order.append(index)
        lines.extend([f"{indent}{'ROOT' if index == 0 else 'JOINT'} {JOINT_NAMES[index]}", indent + "{"])
        offset = np.zeros(3) if index == 0 else (rest_joints[index] - rest_joints[PARENTS[index]]) * 100
        lines.append(indent + "  OFFSET " + " ".join(f"{x:.7f}" for x in offset))
        lines.append(indent + ("  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation" if index == 0 else "  CHANNELS 3 Zrotation Xrotation Yrotation"))
        children = np.flatnonzero(PARENTS == index)
        for child in children:
            visit(int(child), depth + 1)
        if not len(children):
            lines.extend([indent + "  End Site", indent + "  {", indent + "    OFFSET 0 0 0", indent + "  }"])
        lines.append(indent + "}")

    visit(0, 0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected")
        angles = Rotation.from_rotvec(local_rotvec.reshape(-1, 3)).as_euler("ZXY").reshape(-1, 22, 3)
    angles = np.rad2deg(np.unwrap(angles, axis=0))
    lines.extend(["MOTION", f"Frames: {len(local_rotvec)}", f"Frame Time: {1/fps:.10f}"])
    for frame in range(len(local_rotvec)):
        values = np.concatenate([root_positions[frame] * 100, angles[frame, order].reshape(-1)])
        lines.append(" ".join(f"{value:.7f}" for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def export_project(project: Path, result="auto"):
    from .motion_tools import require_supported_project
    if require_supported_project(project).key=="vidihand2":
        raise ValueError("Export the ViDiHand2 first-person skeleton as FBX.")
    from .engines import activate, link_models
    import json
    model=json.loads((project/"project.json").read_text(encoding="utf-8")).get("pipeline",{}).get("model")
    mamma=model=="mamma"
    if not mamma:
        root = activate("GVHMR")
        link_models(root)
    import torch
    if not mamma:
        from hmr4d.utils.smplx_utils import make_smplx
    from .result_artifacts import begin_update,commit_artifacts,file_stat
    from .models import BY_KEY
    from .mamma_body import body_path,model_for
    registered_body=body_path(project)
    model_stamp=file_stat(registered_body)
    body = model_for(project) if mamma else make_smplx("supermotion").eval()
    result = resolve_motion_result(project, result)
    import json
    fps=float(json.loads((project/'project.json').read_text(encoding='utf-8')).get('fps',30)) if (project/'project.json').exists() else 30.
    if not all(result_is_current(project, name) for name in result_sources(result)):
        raise ValueError("Motion result is incomplete or out of date. Apply Correction or reconstruct the selected motion before exporting.")
    source_stamps={name:file_stat(project/RESULTS[name][0]) for name in result_sources(result)}
    motions = [(name, torch.load(project / RESULTS[name][0], map_location="cpu", weights_only=False,mmap=True)[RESULTS[name][1]])
               for name in result_sources(result)]
    with torch.no_grad():
        for name, parameters in motions:
            begin_update(project,name,'export')
            joints = []
            for start in range(0, len(parameters["body_pose"]), 32):
                batch = {key: value[start:start + 32] for key, value in parameters.items()}
                from .hand_pose import evaluate
                joints.append(evaluate(body,batch).joints[:, :22].numpy())
            local = torch.cat([parameters["global_orient"][:, None], parameters["body_pose"].reshape(-1, 21, 3)], dim=1).numpy()
            np.savez_compressed(project / f"{name}_joints.npz", joints=np.concatenate(joints),
                                rotations=global_rotations(local), fps=np.array(fps))
            rest = body(body_pose=torch.zeros(1, 63), global_orient=torch.zeros(1, 3), transl=torch.zeros(1, 3),
                        betas=parameters["betas"][:1]).joints[0, :22].numpy()
            root_positions = parameters["transl"].numpy() + rest[0]
            write_bvh(project / f"{name}_motion.bvh", rest, root_positions, local,fps=fps)
    from .mesh_data import prepare_mesh_preview
    prepare_mesh_preview(project, kinds=[name for name, _ in motions])
    from .mixamo_data import prepare as prepare_target
    prepare_target(project,result,body=body)
    if file_stat(registered_body)!=model_stamp or not all(result_is_current(project,name) for name,_ in motions):
        raise ValueError('Motion inputs changed during export. Run export again.')
    for name,_ in motions:commit_artifacts(project,name,source_stamps[name],fps)
    state=json.loads((project/'project.json').read_text(encoding='utf-8')) if (project/'project.json').exists() else {}
    import shutil
    target = project / "motion.bvh"
    target_record=json.loads((project/f'{result}_mixamo.json').read_text(encoding='utf-8'))
    shutil.copy2(project / target_record['bvh_file'], target)
    selected = dict(motions)[result]
    label={"initial":"GVHMR original", "refined":"GVHMR + HTD-Refine",
        "footmr_initial":"GVHMR + FootMR", "footmr_refined":"GVHMR + HTD-Refine + FootMR"}[body_result(result)]
    exmr=state.get('pipeline',{}).get('model')=='exmr'
    if exmr:label='ExMR original'
    if mamma:label='MAMMA original'
    if state.get('pipeline',{}).get('model')=='prompthmr':label='PromptHMR original'
    from .motion_tools import has_handflow
    if has_handflow(result):label+=' + HandFlow'
    atomic_json(project / "export.json", {"format": "BVH", "frames": len(selected["body_pose"]), "fps": fps,
        "length_unit": "meter", "up_axis": "Y", "rotation_order": target_record.get("export_bind",{}).get("rotation_order_default","ZXY"), "joints": target_record['names'],
        "skeleton":"Humanoid52", "finger_motion":"MAMMA fitted body and finger motion" if mamma else "ExMR native finger inference" if exmr else "HandFlow finger inference; body and wrists preserved" if has_handflow(result) else "Source relaxed pose animated relative to open bind; finger inference is not enabled",
        "bind_pose":target_record.get('export_bind',{}),
        "result_kind": result, "source": label + (" + Motion Correction" if result.startswith('corrected_') else '') + " SMPL-X motion",
        "ground_truth_accuracy": None})
    return target
