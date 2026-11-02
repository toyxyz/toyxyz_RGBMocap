from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from .paths import ROOT
from .storage import atomic_json, sha256
from .resource_cache import code_hash
from .motion_tools import MOTION_TOOLS, resolve_motion_result, RESULTS, result_sources, result_options, result_is_current, base_result, body_result, has_handflow, require_supported_project, UnsupportedMotionTool

PREFIX = "@@MOCAP@@"
# This predecessor differs only in refinement cache dependency tracking. Accept
# its otherwise identical non-refinement identities once, then migrate them.
PRE_ALIGNMENT_WORKER = '3408ac3a41871bc4c28a7c245c40bebdf0da357a8a8d29e1441613405ad056f8'


def source_previews_ready(project,selected):
    """Legacy joint-only acceptance is insufficient for a full SMPL-X view."""
    from .result_artifacts import artifacts_current
    from .mesh_data import MESH_VERSION
    try:
        for kind in result_sources(selected):
            if not (project/f'{kind}_joints.npz').is_file():return False
            info=json.loads((project/f'{kind}_mesh.json').read_text(encoding='utf-8'))
            name=info['frames_file']
            if info.get('version')!=MESH_VERSION or Path(name).name!=name or not (project/name).is_file():return False
            if not artifacts_current(project,kind):return False
        return True
    except (OSError,ValueError,KeyError,TypeError):return False


def fingerprint(stage, project, result="auto", _worker_hash=None, _legacy_routing=False):
    import hashlib
    from .models import BY_KEY, STAGE_REQUIREMENTS
    inputs = {"track": ["input.mp4"], "pva": ["tracking.npz"], "features": ["input.mp4","tracking.npz"],
              "gvhmr": ["input.mp4","tracking.npz","features.pt","gvhmr_keypoints.pt","gvhmr_preprocess.json"], "refine": ["hmr4d_results.pt", "pva_results.pt", "camera.pt"],
              "footmr": ["input.mp4", "tracking.npz", "hmr4d_results.pt"],
              'handflow': ['input.mp4','tracking.npz','camera.pt','gvhmr_keypoints.pt'],
              "correct": ["hmr4d_results.pt"],
              "export": []}
    if stage in {"export", "footmr", "handflow", "correct"}:
        selected = resolve_motion_result(project, result)
        if stage == "export":
            inputs[stage] = [RESULTS[kind][0] for kind in result_sources(selected)]
        elif stage == "correct":
            inputs[stage]=list(dict.fromkeys(["hmr4d_results.pt",*[RESULTS[kind][0] for kind in result_sources(base_result(selected))]]))
        elif stage=='handflow':
            inputs[stage]+=[RESULTS[kind][0] for kind in result_sources(body_result(selected))]
        elif result_options(selected)[0]:
            inputs[stage].append("opt_results.pt")
    sources = [Path(__file__), Path(__file__).with_name("engines.py"), Path(__file__).with_name("export.py"), Path(__file__).with_name("mesh_data.py"), ROOT / "requirements-windows.txt",Path(__file__).with_name('compute_policy.py')]
    if stage in ('track','pva','features','gvhmr','refine','footmr','handflow'):
        sources.append(Path(__file__).with_name('memory_mode.py'))
    if stage == "footmr":
        sources.append(Path(__file__).with_name("footmr.py"))
        sources.append(ROOT / "third_party/FootMR/hmr4d/utils/preproc/vitpose_pytorch/src/vitpose_infer/model_builder.py")
    if stage=='handflow':
        sources.extend(Path(__file__).with_name(n) for n in ('handflow.py','handflow_engine.py','hand_pose.py'))
        sources.extend([ROOT/'third_party/HandFlow/utils/inference_utils.py',ROOT/'requirements-hand-windows.txt'])
    if stage == "correct":
        sources.append(Path(__file__).with_name('correction.py'))
        sources.append(Path(__file__).with_name('contact_trajectory.py'))
        sources.append(Path(__file__).with_name('multilevel_contacts.py'))
        sources.append(Path(__file__).with_name('hand_lock.py'))
        sources.extend(Path(__file__).with_name(name) for name in (
            'contact_arm_plan.py','palm_contact.py','contact_pose_plan.py','contact_temporal_root.py'))
        sources.append(Path(__file__).with_name('anatomical_ik.py'))
        sources.append(Path(__file__).with_name('mixamo_rig.py'))
        sources.append(Path(__file__).with_name('correction_settings.py'))
        sources.append(Path(__file__).with_name('foot_contacts.py'))
        sources.extend((ROOT/'third_party/KimodoMotionCorrection/bin').glob('*.pyd'))
    if stage in ('correct','export'):
        sources.append(Path(__file__).with_name('result_artifacts.py'))
    if stage == "pva":
        sources.append(Path(__file__).with_name("inference_data.py"))
    if stage == 'refine':
        sources.extend(ROOT/'third_party/HTD-Refine'/name for name in (
            'optimization/camera_alignment.py','optimization/pipeline.py','scripts/demo_optimization.py'))
    if stage in ('gvhmr','features'):
        sources.append(Path(__file__).with_name('model_session.py'))
        sources.extend((Path(__file__).with_name('inference_data.py'),
            Path(__file__).with_name('gvhmr_preprocess.py'),Path(__file__).with_name('gvhmr_attention.py'),
            ROOT/'third_party/GVHMR/hmr4d/network/gvhmr/relative_transformer.py',
            ROOT/'third_party/GVHMR/hmr4d/utils/preproc/vitpose_pytorch/src/vitpose_infer/model_builder.py',
            ROOT/'third_party/GVHMR/hmr4d/network/hmr2/__init__.py'))
    predecessor={'worker.py':'e22c2a149770ac22c22979ab9ab7d8fe58b25d5e19995b50e755bc0b07dee068',
        'export.py':'100eb6ef6a639aab9c0b011bd1dd85bd515c1a0ceaeeac83fd2ea831145c745a',
        'compute_policy.py':'ad9d1d9c0841eeeaaae64b1fcfb82753ff0b769ab0fab319957588cd770e12da'}
    # This revision adds only target-rig preparation/export. Retain the exact
    # preceding identity for unchanged inference and source-correction stages.
    non_generator={'worker.py':'6e3ffd307be19ee0659e68643162268501f1c56eda67212c11ed5d3c0a3320b8',
        'export.py':'efb4472ac879ddcf0dc4ae60b4c1d49dab10949351635320e239db3a3f660284',
        'result_artifacts.py':'563ac5467cb6d4e0a6abc603979227de5b541d9c153489f0aceccb227b1b3128',
        'compute_policy.py':'a5953e6dfe424c30c2992e5e84c3c78e2dc4217c3658f8a7ae622bafdefe484b'}
    if stage=='export':
        sources.extend(Path(__file__).with_name(name) for name in ('mixamo_rig.py','mixamo_data.py','mixamo_native.py','bvh_bind.py','source_pose_fit.py'))
    identity = {"stage": stage, "code": [(_worker_hash if _worker_hash and p == Path(__file__) else predecessor[p.name] if _legacy_routing and p.name in predecessor else non_generator[p.name] if stage!='export' and p.name in non_generator else code_hash(p)) for p in sources], "inputs": [], "models": []}
    if stage in ('gvhmr','features'):
        import torch
        identity['runtime']=[str(torch.__version__),torch.version.cuda,torch.backends.cudnn.version(),
                             torch.cuda.get_device_name() if torch.cuda.is_available() else 'no_cuda',
                             os.environ.get('RGB_MOCAP_WORKER_THREADS'),os.environ.get('RGB_MOCAP_COMPUTE_MODE')]
    if stage in {"export", "footmr", "handflow", "correct"}:
        selected=resolve_motion_result(project,result)
        identity["result"] = body_result(selected) if stage=='footmr' else base_result(selected) if stage=='handflow' else selected
    if stage=='correct':
        state=json.loads((project/'project.json').read_text(encoding='utf-8'))
        identity['options']=state.get('correction',{})
        identity['fps']=state.get('fps',30)
        from .foot_contacts import contact_revision
        identity['contact_edits']={base_result(kind):contact_revision(project,base_result(kind)) for kind in result_sources(selected) if kind.startswith('corrected_')}
    if stage=='export' and selected.startswith('corrected_'):
        from .foot_contacts import contact_revision
        identity['contact_edits']={base_result(kind):contact_revision(project,base_result(kind)) for kind in result_sources(selected) if kind.startswith('corrected_')}
    for name in inputs[stage]:
        path = project / name
        if path.exists():
            identity["inputs"].append((name, path.stat().st_size, path.stat().st_mtime_ns))
    for key in STAGE_REQUIREMENTS.get(stage, ()):
        path = BY_KEY[key].path
        identity["models"].append((key, path.stat().st_size, path.stat().st_mtime_ns))
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def emit(kind: str, **data) -> None:
    print(PREFIX + json.dumps({"type": kind, **data}, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["diagnose", "prepare", "download", "import_model", "mesh", "rig", "save_fbx", "track", "pva", "features", "gvhmr", "refine", "footmr", "handflow", "correct", "export", "people", "person", 'flow_camera','flow_features','flowhmr','flow_export','exmr_preprocess','exmr_features','exmr'])
    parser.add_argument("--project", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--model")
    parser.add_argument("--motion-tool", choices=list(MOTION_TOOLS))
    parser.add_argument("--result", choices=["auto", *RESULTS], default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true", help="Recompute a completed stage for verification.")
    parser.add_argument('--person-id',type=int)
    parser.add_argument('--first-frame',type=int,default=0)
    parser.add_argument('--end-frame',type=int)
    parser.add_argument('--floor',type=float,help='Manual world floor Y; omitted uses automatic height')
    from .correction_settings import PRESETS
    parser.add_argument('--correction-preset',choices=list(PRESETS),default='normal')
    parser.add_argument('--foot-lock',type=int,choices=(0,1),default=1)
    parser.add_argument('--hand-lock',type=int,choices=(0,1),default=0)
    parser.add_argument('--correction-mode',choices=['flat','multi_level'],default='flat')
    parser.add_argument('--pelvis-offset',type=float,default=0.)
    args = parser.parse_args()
    from .model_session import set_memory_budget
    from .compute_policy import configure_worker_cpu,gpu_memory_fraction,GIB
    threads=configure_worker_cpu(args.stage)
    os.environ['RGB_MOCAP_STAGE']=args.stage
    if args.project:
        args.project = args.project.resolve()
    started = time.monotonic()
    try:
        if args.stage == "diagnose":
            from .diagnostics import collect
            value = collect()
            if args.output:
                atomic_json(args.output, value)
            emit("diagnostics", result=value)
            return 0
        if args.stage == "prepare":
            from .video import prepare
            project = prepare(args.video.resolve(), args.start, args.end,motion_tool=args.motion_tool or 'gvhmr',project=args.project)
            manifest = json.loads((project / "project.json").read_text(encoding="utf-8"))
            if args.motion_tool=='flowhmr':
                manifest['flow_options']={**manifest.get('flow_options',{}),'camera_interval':int(os.environ.get('RGB_MOCAP_FLOW_INTERVAL','0')),'fast_sam':os.environ.get('RGB_MOCAP_FLOW_FAST_SAM','1')=='1','sam_batch4':os.environ.get('RGB_MOCAP_FLOW_BATCH4','1')=='1','sam_gpu_crop':os.environ.get('RGB_MOCAP_FLOW_GPU_CROP','1')=='1','sam_hand_stride4':os.environ.get('RGB_MOCAP_FLOW_HAND_STRIDE4','1')=='1'}
                manifest['flow_options']['hamer']=os.environ.get('RGB_MOCAP_FLOW_HAMER','0')=='1'
                if manifest['flow_options']['hamer']:manifest['flow_options'].update(fast_sam=True,sam_hand_stride4=True)
            if args.motion_tool=='exmr':
                from .exmr_options import options
                manifest['exmr_options']=options(json.loads(os.environ.get('RGB_MOCAP_EXMR_OPTIONS','{}')))
            if args.motion_tool:
                manifest.setdefault('pipeline',{})['model']=args.motion_tool
            if args.result != "auto":
                manifest["pipeline"] = {"model": args.motion_tool or "gvhmr", "result": args.result,
                                        "htd_refine": result_options(args.result)[0], "footmr": result_options(args.result)[1], 'handflow':has_handflow(args.result)}
                if args.result.startswith('corrected_'):
                    manifest['correction']={'auto_floor':args.floor is None,'floor':float(args.floor or 0),'preset':args.correction_preset,'pelvis_offset':args.pelvis_offset,'mode':args.correction_mode,'foot_lock':bool(args.foot_lock),'hand_lock':bool(args.hand_lock)}
            atomic_json(project / "project.json", manifest)
            emit("prepared", project=str(project), manifest={k: v for k, v in manifest.items() if k != "original_times"})
            return 0
        if args.stage == "download":
            from .models import download, BY_KEY
            emit("progress", current=0, total=0, message="Downloading " + BY_KEY[args.model].name)
            result = download(args.model)
            emit("downloaded", path=str(result), model=args.model)
            return 0
        if args.stage=='import_model':
            from .models import import_model_file
            result=import_model_file(args.model,args.video)
            emit('downloaded',path=str(result),model=args.model)
            return 0
        if args.stage=='save_fbx':
            from .fbx_export import run
            count=run(args.output,emit)
            emit('complete',stage='save_fbx',files=count,elapsed=time.monotonic()-started)
            return 0
        project = args.project.resolve()
        require_supported_project(project)
        if require_supported_project(project).key=='exmr' and args.stage not in ('exmr_preprocess','exmr_features','exmr','people','person','track','rig','mesh','export','correct'):
            raise ValueError('This stage is not compatible with ExMR. Its original prediction has been preserved.')
        if require_supported_project(project).key=='exmr' and args.stage.startswith('exmr'):
            from .exmr_backend import run_stage
            run_stage(project,args.stage,args.result,emit,force=args.force)
            return 0
        if require_supported_project(project).key=='flowhmr' and args.stage not in ('people','person','track'):
            from .flow_backend import run_stage
            run_stage(project,args.stage,args.result,emit,force=args.force)
            return 0
        if require_supported_project(project).key in ('mamma','prompthmr'):
            if args.stage=='export':
                from .export import export_project
                output=export_project(project,args.result)
            elif args.stage=='correct':
                from .correction import run as correct_mamma
                correct_mamma(project,emit,args.result);output=project/RESULTS[args.result][0]
            elif args.stage in ('mesh','rig'):
                from .export import export_project
                output=export_project(project,args.result)
            else:raise ValueError('Use this model\'s dedicated reconstruction pipeline; generic inference stages cannot process this project.')
            emit('complete',stage=args.stage,output=str(output),cached=False,elapsed=time.monotonic()-started)
            return 0
        if args.stage == 'rig':
            from .mixamo_data import prepare
            kind=resolve_motion_result(project,args.result)
            if source_previews_ready(project,kind):
                result=prepare(project,kind,emit)
            else:
                # Recover missing source mesh/joint payloads too, without AI
                # inference or substituting another source result.
                from .export import export_project
                export_project(project,kind)
                result=project/(kind+'_mixamo.json')
            import shutil
            target=json.loads(result.read_text(encoding='utf-8'))
            shutil.copy2(project/target['bvh_file'],project/'motion.bvh')
            export_path=project/'export.json'
            metadata=json.loads(export_path.read_text(encoding='utf-8')) if export_path.exists() else {}
            metadata.update(format='BVH',skeleton='Humanoid52',joints=target['names'],result_kind=kind,frames=target['frames'],fps=target['fps'],
                length_unit='meter',up_axis='Y',rotation_order=target.get('export_bind',{}).get('rotation_order_default','ZXY'),
                finger_motion='ExMR native finger inference' if require_supported_project(project).key=='exmr' else 'HandFlow finger inference; body and wrists preserved' if has_handflow(kind) else 'Source relaxed pose animated relative to open bind; finger inference is not enabled',bind_pose=target.get('export_bind',{}))
            atomic_json(export_path,metadata)
            import psutil
            emit('complete',stage='rig',output=str(result),cached=False,elapsed=time.monotonic()-started,
                peak_ram_gib=round(psutil.Process().memory_info().peak_wset/1024**3,3))
            return 0
        if args.stage == "mesh":
            from .mesh_data import prepare_mesh_preview
            result = prepare_mesh_preview(project)
            emit("complete", stage="mesh", output=str(result), cached=False, elapsed=time.monotonic()-started)
            return 0
        from .models import missing
        unavailable = missing(args.stage)
        if unavailable:
            emit("blocked", message="Prepare the required models first.", models=[m.key for m in unavailable])
            return 3
        manifest_path = project / "project.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if args.stage in ('people','person'):
            import psutil
            import torch
            torch.set_num_threads(threads)
            memory=psutil.virtual_memory()
            free=total=None
            if args.stage=='people':
                if not torch.cuda.is_available():raise RuntimeError('A supported NVIDIA CUDA GPU is required for person detection.')
                free,total=torch.cuda.mem_get_info()
            if free is not None:
                set_memory_budget(free,total)
                from .memory_mode import select_stage
                select_stage(args.stage,free,total,manifest.get('frames',0),emit)
            from .person_tracking import discover,prepare_person
            if args.stage=='people':
                output=discover(project,emit)
            else:
                output=prepare_person(project,args.person_id,args.first_frame,args.end_frame or manifest['frames'],args.result,
                    {'auto_floor':args.floor is None,'floor':float(args.floor or 0),'preset':args.correction_preset,'pelvis_offset':args.pelvis_offset,'mode':args.correction_mode,'foot_lock':bool(args.foot_lock),'hand_lock':bool(args.hand_lock)},emit)
            emit('complete',stage=args.stage,output=str(output),elapsed=time.monotonic()-started,cached=False,
                peak_ram_gib=round(psutil.Process().memory_info().peak_wset/1024**3,3),
                peak_cuda_allocated_gib=round(torch.cuda.max_memory_allocated()/1024**3,3) if torch.cuda.is_initialized() else 0)
            return 0
        outputs = {"track": "tracking.npz", "pva": "pva_results.pt", "features": "features.pt", "gvhmr": "hmr4d_results.pt", "refine": "opt_results.pt", "export": "motion.bvh"}
        if args.stage == "footmr":
            if not body_result(resolve_motion_result(project, args.result)).startswith("footmr_"):
                raise ValueError("FootMR requires --result footmr_initial or footmr_refined")
            outputs["footmr"] = RESULTS[body_result(args.result)][0]
        if args.stage=='handflow':
            if not has_handflow(args.result):raise ValueError('Select a HandFlow result.')
            outputs['handflow']=RESULTS[base_result(args.result)][0]
        if args.stage=='correct':
            if not args.result.startswith('corrected_'):raise ValueError('Select a corrected result')
            outputs['correct']=RESULTS[args.result][0]
        record = manifest.get("stages", {}).get(args.stage, {})
        expected = project / outputs[args.stage]
        signature = fingerprint(args.stage, project, args.result)
        companions = {"pva": ["keypoints.npz"], "features": ["gvhmr_keypoints.pt","gvhmr_preprocess.json"], "export": ["initial_joints.npz", "initial_motion.bvh", "export.json", "initial_mesh.json"]}
        if args.stage == "export":
            companions["export"] = ["export.json", *[f"{kind}_{suffix}" for kind in result_sources(resolve_motion_result(project, args.result))
                for suffix in ("joints.npz", "motion.bvh", "mesh.json")]]
        if args.stage == "footmr":
            companions["footmr"] = ["footmr_report.json", "footmr_keypoints.pt", *[RESULTS[kind][0] for kind in
                result_sources(args.result) if kind.startswith("footmr_")], *[f"{kind}_sources.json" for kind in
                result_sources(args.result) if kind.startswith("footmr_")]]
        if args.stage=='handflow':
            companions['handflow']=['handflow_report.json','handflow_prediction.json',
                *[RESULTS[k][0] for k in result_sources(base_result(args.result)) if has_handflow(k)],
                *[f'{k}_{suffix}.json' for k in result_sources(base_result(args.result)) if has_handflow(k) for suffix in ('sources','handflow')]]
            has_valid_hands=all(result_is_current(project,k) for k in result_sources(base_result(args.result)) if has_handflow(k))
        else:has_valid_hands=True
        if args.stage=='correct':
            companions['correct']=['correction_report.json',*[RESULTS[kind][0] for kind in result_sources(args.result) if kind.startswith('corrected_')],
                *[f'{kind}_{suffix}.json' for kind in result_sources(args.result) if kind.startswith('corrected_') for suffix in ('sources','correction')],
                *[f'{kind}_contacts.npz' for kind in result_sources(args.result) if kind.startswith('corrected_')]]
        has_companions = has_valid_hands and all((project / name).is_file() for name in companions.get(args.stage, ()))
        if has_companions and args.stage in ('correct','export'):
            from .result_artifacts import stage_results_current
            has_companions=stage_results_current(args.stage,project,args.result)
        cache_matches = record.get('fingerprint') == signature
        if not cache_matches:
            # Routing-only predecessor; existing GVHMR math is unchanged.
            cache_matches = record.get('fingerprint') == fingerprint(args.stage,project,args.result,_legacy_routing=True)
        if (not args.force and not cache_matches and args.stage != 'refine' and
                record.get('status') == 'complete' and expected.is_file() and has_companions):
            cache_matches = record.get('fingerprint') == fingerprint(args.stage,project,args.result,_worker_hash=PRE_ALIGNMENT_WORKER)
            if cache_matches:
                record['fingerprint'] = signature
                atomic_json(manifest_path,manifest)
        if not args.force and record.get("status") == "complete" and cache_matches and expected.is_file() and has_companions:
            emit("complete", stage=args.stage, output=str(expected), cached=True, elapsed=0)
            return 0
        import psutil
        import torch
        torch.set_num_threads(threads)
        free=None
        if args.stage in ('track','pva','features','gvhmr','refine','footmr','handflow'):
            if not torch.cuda.is_available():
                raise RuntimeError('A supported NVIDIA CUDA GPU is required for this model. Check System in Help for details.')
            free,total=torch.cuda.mem_get_info()
        memory=psutil.virtual_memory()
        if free is not None:
            set_memory_budget(free,total)
            from .memory_mode import select_stage
            select_stage(args.stage,free,total,manifest.get('frames',0),emit)
        from .memory_mode import effective_mode,requested_mode
        plan=dict(stage=args.stage,threads=threads,mode=effective_mode(),requested_mode=requested_mode(),
             frames=manifest.get('frames',0),available_ram_gib=round(memory.available/GIB,2),
             free_vram_gib=round(free/GIB,2) if free is not None else None)
        atomic_json(project/f'{args.stage}_resources.json',plan)
        emit('compute_plan',**plan)
        manifest.setdefault("stages", {})[args.stage] = {"status": "running", "engine_version": 1}
        atomic_json(manifest_path, manifest)
        if args.stage == "export":
            from .export import export_project
            result = export_project(project, result=args.result)
            selected = resolve_motion_result(project, args.result)
            manifest["pipeline"] = {"model": args.motion_tool or manifest.get('pipeline',{}).get('model','gvhmr'), "result": selected,
                                    "htd_refine": result_options(selected)[0], "footmr": result_options(selected)[1], 'handflow':has_handflow(selected)}
        elif args.stage == "footmr":
            from .footmr import run
            result = run(project, emit, body_result(args.result))
        elif args.stage=='handflow':
            from .handflow import run
            result=run(project,emit,args.result)
        elif args.stage=='correct':
            from .correction import run
            result=run(project,emit,args.result)
        else:
            from . import engines
            result = getattr(engines, args.stage)(project, emit)
        elapsed = time.monotonic() - started
        import torch
        import psutil
        memory = {"peak_ram_gib": round(psutil.Process().memory_info().peak_wset / 1024**3, 3)}
        if torch.cuda.is_initialized():
            memory.update(peak_cuda_allocated_gib=round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                          peak_cuda_reserved_gib=round(torch.cuda.max_memory_reserved() / 1024**3, 3))
        manifest["stages"][args.stage] = {"status": "complete", "output": str(result), "elapsed": elapsed, "engine_version": 1,
                                         "fingerprint": fingerprint(args.stage,project,args.result), **memory}
        atomic_json(manifest_path, manifest)
        emit("complete", stage=args.stage, output=str(result), elapsed=elapsed, cached=False, **memory)
        return 0
    except Exception as error:
        details = traceback.format_exc()
        print(details, file=sys.stderr, flush=True)
        if args.project and not isinstance(error,UnsupportedMotionTool):
            (args.project / f"{args.stage}_error.log").write_text(details, encoding="utf-8")
            state_path = args.project / "project.json"
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                state.setdefault("stages", {})[args.stage] = {"status": "failed", "error": str(error)}
                atomic_json(state_path, state)
        message=str(error)
        if isinstance(error,MemoryError) or 'out of memory' in message.lower():
            from .memory_mode import GUIDANCE
            message='Processing ran out of memory and stopped safely. '+GUIDANCE+' Completed stages can be reused.'
        emit("failed", stage=args.stage, message=message)
        return 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
