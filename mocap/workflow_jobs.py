"""Workflow job arguments shared by the desktop and CLI."""
from pathlib import Path

from .queue_groups import grouped_jobs, join_single_pipeline

RESULT_STAGES = {'export', 'flow_export', 'footmr', 'handflow', 'correct'}
POST_STAGES = {'correct', 'export', 'flow_export'}


def correction_arguments(options, *, floor_last=False):
    floor = ['--floor', str(options['floor'])] if not options['auto_floor'] else []
    locks = ['--correction-preset', options['preset'], '--correction-mode', options['mode'],
             '--foot-lock', str(int(options['foot_lock'])), '--hand-lock', str(int(options['hand_lock'])),
             '--pelvis-offset', str(options['pelvis_offset'])]
    return locks + floor if floor_last else floor + locks


def video_jobs(video, start, end, model, stages, *, result=None, correction=None,
               workspace=None, interpolation=False, people_options=None):
    args = ['--video', str(video), '--start', str(start), '--end', str(end)]
    if interpolation:
        args.append('--rife-interpolation')
    if workspace is not None:
        args += ['--project', str(workspace)]
    if result or model != 'gvhmr':
        args += ['--motion-tool', model]
    if result:
        args += ['--result', result]
        if result.startswith('corrected_'):
            args += correction_arguments(correction)
    jobs = [('prepare', args)]
    for stage in stages:
        arguments = None
        if stage == 'sam31_people':
            arguments = people_options
        elif stage in RESULT_STAGES and result:
            arguments = ['--result', result]
        jobs.append((stage, arguments))
    return jobs


def next_job(jobs, project):
    remaining = join_single_pipeline(jobs, project)
    stage, arguments = remaining.pop(0)
    if arguments is None:
        arguments = ['--project', str(project)]
    elif stage in RESULT_STAGES | {'sam31_people','vidihand_arms'} and '--project' not in arguments:
        arguments = ['--project', str(project), *arguments]
    return stage, arguments, remaining


def person_jobs(root, ids, first, end, manifest, result, correction):
    fps = manifest['fps']
    begin = round((first - manifest['start']) * fps)
    finish = round((end - manifest['start']) * fps)
    return [('person', ['--project', str(root), '--person-id', str(person),
                        '--first-frame', str(begin), '--end-frame', str(finish),
                        '--result', result, *correction_arguments(correction, floor_last=True)])
            for person in ids]


def take_jobs(paths, stages, result):
    inference, post = [], []
    for path in paths:
        for stage in stages:
            queue = post if stage in POST_STAGES else inference
            queue.append((stage, ['--project', str(path), '--result', result]))
    return inference, post


def grouped_take_jobs(root, inference, pending, post):
    return grouped_jobs(inference, root) + pending + post


def export_row(project, kind, source, target, name, options, *, people_root=None, person_id=None):
    row = dict(project=str(project), kind=kind, source=str(source), target=str(target))
    row.update(rig_name=name, prepend_tpose=options.get('prepend_tpose', False),
               export_face=options.get('export_face', False), face_format=options.get('face_format', 'txt'),
               smooth_face=options.get('smooth_face', False), bvh_profile=options.get('bvh_profile', 'generic'))
    fmt = options.get('format', 'fbx')
    if fmt == 'fbx':
        if options.get('fbx_compatibility', False):
            row['fbx_compatibility'] = True
        for key in ('fbx_profile', 'fbx_rest_pose', 'ue5_target', 'ue5_in_place', 'ue5_include_mesh'):
            if key in options:
                row[key] = options[key]
    for key in ('place_at_origin', 'symmetric_rest'):
        row[key] = options.get(key, False) and fmt in ('fbx', 'bvh')
    row['origin_group'] = str(Path(project).resolve()) if person_id is None else f'person:{person_id}'
    for key, default in (('animate_camera', True), ('background_refine', False), ('camera_translation_scale', 1.)):
        row[key] = options.get(key, default)
    if people_root is not None:
        from .scene_export import context
        row['scene'] = context(people_root, project)
    return row


def scene_correction_jobs(root, people, kind, *, stages=('correct', 'export')):
    from .motion_tools import RESULTS, base_result, result_is_current, result_sources
    from .people_store import safe_child

    jobs, skipped = [], []
    bases = result_sources(base_result(kind))
    for person in people:
        for number, take in enumerate(person.get('takes', []), 1):
            project = safe_child(root, take['project'])
            available = all((project / RESULTS[base][0]).is_file() and result_is_current(project, base)
                            for base in bases)
            if available:
                jobs.extend((stage, ['--project', str(project), '--result', kind]) for stage in stages)
            elif person.get('checked'):
                skipped.append(f"{person['name']}, Take {number}")
    return jobs, skipped



def publish_output(source, target, *, replace=True):
    import os
    if replace:
        os.replace(source, target)
    elif os.name == 'nt':
        os.rename(source, target)
    else:
        os.link(source, target)
        source.unlink()



def sam31_arguments(prompt, time, points):
    import json
    return ['--sam31-options', json.dumps(dict(prompt=prompt, time=time, points=points), ensure_ascii=True)]
