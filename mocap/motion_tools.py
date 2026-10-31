"""Available motion tools and their compatible optional processing stages."""
from dataclasses import dataclass
import json


@dataclass(frozen=True)
class MotionTool:
    key: str
    label: str
    reconstruction: tuple[str, ...]
    refinement: tuple[str, ...] = ()
    input_setup_only: bool = False
    live: bool = False

    def stages(self, refine=False, footmr=False, correction=False, handflow=False):
        if self.input_setup_only or self.live:
            return []
        if self.key in ('mamma','prompthmr'):return list(self.reconstruction)
        if self.key == 'flowhmr':
            return [*self.reconstruction, *(('correct',) if correction else ()), 'flow_export']
        return [*self.reconstruction, *(self.refinement if refine else ()), *(('footmr',) if footmr and self.key == 'gvhmr' else ()),
                *(("handflow",) if handflow and self.key == 'gvhmr' else ()),
                *(("correct",) if correction else ()), "export"]


MOTION_TOOLS = {"pear": MotionTool("pear", "PEAR · Live (Experimental)", (), live=True),"gvhmr": MotionTool("gvhmr", "GVHMR", ("track", "gvhmr"), ("pva", "refine")),
                "flowhmr": MotionTool("flowhmr", "FlowHMR", ("track", "flow_camera", "flow_features", "flowhmr")),
                "exmr": MotionTool("exmr", "ExMR", ("track", "exmr_preprocess", "exmr_features", "exmr")),
                "mamma": MotionTool("mamma", "MAMMA", ("mamma_prepare","mamma_masks","mamma_landmarks","mamma_camera","mamma_fit","mamma_publish")),
                "prompthmr": MotionTool("prompthmr", "PromptHMR", ("prompthmr_prepare","prompthmr_track","prompthmr_pose","prompthmr_world","prompthmr_publish"))}


class UnsupportedMotionTool(ValueError):
    pass


def require_supported_tool(model):
    if model not in MOTION_TOOLS:
        raise UnsupportedMotionTool(f"This project's motion model ({model}) is no longer supported. Its files have been preserved. Open a supported project or prepare the source video as a new project.")
    return MOTION_TOOLS[model]


def require_supported_project(project):
    from pathlib import Path
    path=Path(project)/'project.json'
    state=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    return require_offline_tool(state.get('pipeline',{}).get('model','gvhmr'))


def require_offline_tool(model):
    tool=require_supported_tool(model)
    if tool.live:raise UnsupportedMotionTool('PEAR is a transient live preview, not a saved reconstruction model. Project files were preserved.')
    return tool

# Derived results never replace either upstream prediction.
RESULTS = {
    "initial": ("hmr4d_results.pt", "smpl_params_global"),
    "refined": ("opt_results.pt", "smpl_params_global_optimized"),
    "footmr_initial": ("footmr_initial.pt", "smpl_params_global"),
    "footmr_refined": ("footmr_refined.pt", "smpl_params_global"),
}
BODY_RESULTS = dict(RESULTS)
RESULTS.update({'handflow_'+name: ('handflow_'+name+'.pt', 'smpl_params_global') for name in BODY_RESULTS})
BASE_RESULTS = dict(RESULTS)
RESULTS.update({"corrected_"+name: ("corrected_"+name+".pt", "smpl_params_global") for name in BASE_RESULTS})


def base_result(kind):
    return kind.removeprefix("corrected_")


def result_kind(htd=False, footmr=False, correction=False, handflow=False):
    return ("corrected_" if correction else "") + ('handflow_' if handflow else '') + ("footmr_" if footmr else "") + ("refined" if htd else "initial")


def has_handflow(kind):
    return base_result(kind).startswith('handflow_')


def body_result(kind):
    return base_result(kind).removeprefix('handflow_')


def result_options(kind):
    if kind not in RESULTS:
        raise ValueError("Unknown motion result: " + kind)
    return kind.endswith("refined"), body_result(kind).startswith("footmr_")


def result_sources(kind):
    """Preview variants produced by this run, including each enabled base."""
    htd, foot = result_options(kind)
    bases=["initial", *(["refined"] if htd else []),
            *(["footmr_initial"] if foot else []), *(["footmr_refined"] if htd and foot else [])]
    if has_handflow(kind):bases += ['handflow_'+name for name in bases.copy()]
    return bases + (["corrected_"+name for name in bases] if kind.startswith("corrected_") else [])


def result_is_current(project, kind, ignore_contacts=False):
    """Reject an add-on preview if its underlying reconstruction was replaced."""
    try:
        require_supported_project(project)
    except (OSError, ValueError, TypeError, KeyError):
        return False
    if (project/'exmr_pending.json').exists():return False
    if (project/'exmr_receipt.json').exists():
        try:
            from .models import BY_KEY
            from .result_artifacts import file_stat
            stamp=json.loads((project/'exmr_receipt.json').read_text(encoding='utf-8'))['identity']['models']['smplx']
            if BY_KEY['smplx'].path.exists() and file_stat(BY_KEY['smplx'].path)!=stamp:return False
        except (OSError,ValueError,KeyError,TypeError):return False
    if (project/'exmr_camera_sources.json').exists():
        try:
            from .result_artifacts import file_stat
            camera=json.loads((project/'exmr_camera_sources.json').read_text(encoding='utf-8'))
            root=(project/camera['root']).resolve()
            if not project.resolve().is_relative_to(root) or file_stat(root/'exmr_camera.npy')!=camera['root_stat']:return False
        except (OSError,ValueError,KeyError,TypeError):return False
    if (project/'flow_world_pending.json').exists():return False
    if (project/'flow_world.json').exists():
        from .flow_placement import current as placement_current
        if not placement_current(project):return False
    metadata = project / f"{kind}_sources.json"
    if has_handflow(kind) and not kind.startswith('corrected_'):
        from .handflow import result_current
        if not result_current(project,kind):return False
    if (project/'flowhmr_pending.json').exists():return False
    if (project/'flow_camera_sources.json').exists():
        try:
            state=json.loads((project/'project.json').read_text(encoding='utf-8'))
            root=(project/state['people_root']).resolve()
            if not project.resolve().is_relative_to(root):return False
            receipt=json.loads((project/'flow_camera_sources.json').read_text(encoding='utf-8'))
            camera=(root/'flow_camera.npz').stat()
            if receipt['root_camera']!=[camera.st_size,camera.st_mtime_ns] or (root/'flow_camera_pending.json').exists():return False
        except (OSError,ValueError,KeyError,TypeError):return False
    if not metadata.exists():
        return not kind.startswith("corrected_")  # Only original legacy projects.
    try:
        if kind.startswith("corrected_"):
            from .result_artifacts import pending_path,file_stat
            if pending_path(project,kind,'correct').exists():return False
            if not result_is_current(project,base_result(kind)):return False
            report=json.loads((project/f"{kind}_correction.json").read_text(encoding="utf-8"))
            if report.get('version',1)>=3:
                if file_stat(project/RESULTS[kind][0])!=report.get('output_stat'):return False
                if file_stat(project/f'{kind}_contacts.npz')!=report.get('contacts_stat'):return False
            state=json.loads((project/'project.json').read_text(encoding="utf-8"))
            from .correction_settings import correction_options,effective_options
            if correction_options(report.get('options',{}))!=effective_options(project,state):return False
            from .scene_floor import receipt_matches
            if not receipt_matches(project,state,report.get('shared_floor')):return False
            if not ignore_contacts:
                from .foot_contacts import contact_revision
                if report.get('contact_revision')!=contact_revision(project,base_result(kind)):return False
            if 'fps' in report and report['fps']!=state.get('fps',30):return False
            if 'model_stat' in report:
                from .models import BY_KEY
                # Already-baked previews remain usable without an imported model,
                # just as uncorrected previews do. Compare it when available.
                if state.get('pipeline',{}).get('model')=='mamma':
                    from .mamma_body import body_path
                    model_path=body_path(project)
                else:model_path=BY_KEY[report.get('model_key','smplx')].path
                if model_path.is_file():
                    model=model_path.stat()
                    if report['model_stat']!=[model.st_size,model.st_mtime_ns]:return False
        sources = json.loads(metadata.read_text(encoding="utf-8"))
        return all((project/name).is_file() and [(project/name).stat().st_size, (project/name).stat().st_mtime_ns] == stat
                   for name, stat in sources.items())
    except (OSError, ValueError,KeyError,TypeError):
        return False


def resolve_motion_result(project, result="auto"):
    if result not in {"auto", *RESULTS}:
        raise ValueError("Unknown motion result: " + result)
    return ("refined" if (project / "opt_results.pt").exists() else "initial") if result == "auto" else result


def selected_export(project, kind):
    """Only the target rig shown by Mixamo mode may be exported."""
    from .mixamo_data import current
    from .result_artifacts import artifacts_current
    if (project/'flow_prediction.pt').exists():
        from .flow_placement import current as placement_current
        if not placement_current(project):return None
    if not artifacts_current(project,kind):return None
    if current(project,kind) and result_is_current(project,kind):
        record=json.loads((project/f'{kind}_mixamo.json').read_text(encoding='utf-8'))
        return project/record['bvh_file']
    return None


def legacy_export(project, kind):
    """Internal compatibility access; never exposed as a Mixamo export."""
    from .result_artifacts import artifacts_current
    if (project/'flow_prediction.pt').exists():
        from .flow_placement import current as placement_current
        if not placement_current(project):return None
    if not artifacts_current(project,kind):return None
    path = project / f"{kind}_motion.bvh"
    if path.exists():
        return path
    metadata, legacy = project / "export.json", project / "motion.bvh"
    if metadata.exists() and legacy.exists():
        try:
            record = json.loads(metadata.read_text(encoding="utf-8"))
            source = record.get("result_kind", "refined" if "HTD" in record.get("source", "") else "initial")
            if source == kind:
                return legacy
        except (ValueError, OSError):
            pass
    return None
