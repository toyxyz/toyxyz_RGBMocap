"""Independent, cancellable project copies; no model or tensor imports."""
from pathlib import Path
import argparse
import json
import os
import shutil
import stat
import time
import uuid
from .storage import atomic_json
from .project_storage import manifest, storage_root, source_video


class SaveCancelled(Exception):
    pass


def validate_destination(source, target):
    target = Path(target).resolve()
    if target.exists():
        raise ValueError('Choose a new folder name. Existing folders are never overwritten.')
    if not target.parent.is_dir():
        raise ValueError('Choose an existing location for the new project folder.')
    if source is not None:
        source = Path(source).resolve()
        if target.is_relative_to(source) or source.is_relative_to(target):
            raise ValueError('Save the copy outside the current project folder.')
    return target


def files(root):
    # os.walk does not retain a clip-sized file inventory. Refuse junctions and
    # symbolic links rather than silently copying content outside the project.
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            path = Path(directory) / name
            stamp = path.lstat()
            if path.is_symlink() or getattr(stamp, 'st_file_attributes', 0) & 0x400:
                raise ValueError('Project contains a linked folder or file: ' + str(path))
            if name in names:
                if not stat.S_ISREG(stamp.st_mode):
                    raise ValueError('Unsupported project file: ' + str(path))
                yield path, stamp


def save_project(request, emit=lambda *a, **k: None, cancelled=lambda: False):
    source = Path(request['source']).resolve() if request.get('source') else None
    target = validate_destination(source, request['target'])
    active = Path(request['active']).resolve() if source and request.get('active') else None
    if active is not None and not active.is_relative_to(source):
        raise ValueError('Active project is outside the saved workspace.')
    if source:
        from .project_workspace import is_workspace
        grouped=is_workspace(source)
        if grouped:
            from .project_workspace import read
            state=read(source)
            media=source_video(source,state)
        else:
            from .motion_tools import require_supported_project
            require_supported_project(active)
            state = manifest(active)
            media = source_video(active, state)
    else:
        media = Path(request['video']).resolve()
        state = {}
        grouped=True
    # A missing original is recoverable from the project's prepared clip.
    offset = float(state.get('time_offset',state.get('storage', {}).get('time_offset', 0)))
    if not media.is_file() and source:
        if grouped:
            from .project_workspace import child
            relative=state.get('reference')
            fallback=child(source,relative) if relative else active
        else:fallback = source if (source/'input.mp4').is_file() else active
        if fallback is None:raise ValueError('The original video is missing and no prepared clip is available.')
        media = fallback / 'input.mp4'
        offset = float(manifest(fallback)['start'])
    if not media.is_file():
        raise ValueError('The project video is missing. Restore it before saving a copy.')
    media_internal = source is not None and media.is_relative_to(source)
    relative_media = media.relative_to(source).as_posix() if media_internal else 'media/' + media.name
    # Each camera video is a project asset. Copy external inputs once; retain
    # project-relative references so the saved folder can be moved intact.
    camera_entries=request.get('mamma_inputs') or state.get('mamma_inputs',[])
    camera_copies=[];saved_cameras=[]
    if camera_entries:
        from .mamma_inputs import resolved_entries
        for i,entry in enumerate(resolved_entries(source or target,camera_entries)):
            camera_path=Path(entry['path'])
            if not camera_path.is_file():raise ValueError('Camera video is missing: '+camera_path.name)
            internal=source is not None and camera_path.is_relative_to(source)
            relative=(relative_media if camera_path==media else camera_path.relative_to(source).as_posix()
                      if internal else f'media/cameras/{i+1:02d}/'+camera_path.name)
            if camera_path!=media and not internal:camera_copies.append((camera_path,relative))
            value=dict(entry,media=relative,path=str(target/relative))
            saved_cameras.append(value)
    staging = target.parent / ('.' + target.name + '.saving-' + uuid.uuid4().hex)
    def check():
        if cancelled():
            raise SaveCancelled('Project save cancelled. The current project was kept.')
    def signature(s):
        return s.st_size, s.st_mtime_ns
    def inode(s):
        # Find person hardlinks one clip into every person and take folder; copy it once.
        return (s.st_dev, s.st_ino) if s.st_nlink > 1 and s.st_ino else None
    last = 0.
    done = 0
    staged = {}
    def copy_file(path, dest, before):
        nonlocal done, last
        check()
        dest.parent.mkdir(parents=True, exist_ok=True)
        key = inode(before)
        if key in staged:
            try:
                os.link(staged[key], dest)
            except OSError:
                pass
            else:
                if signature(dest.stat()) != signature(before):
                    raise ValueError('Project files changed while saving. Try again when processing is idle.')
                return
        with path.open('rb') as src, dest.open('xb') as dst:
            while True:
                check()
                block = src.read(4 * 1024**2)
                if not block:
                    break
                dst.write(block)
                done += len(block)
                if time.monotonic() - last > .15:
                    emit('progress', phase='copy', current=done, total=total, message='Copying project files')
                    last = time.monotonic()
        if signature(before) != signature(path.stat()):
            raise ValueError('Project files changed while saving. Try again when processing is idle.')
        shutil.copystat(path, dest)
        if signature(dest.stat()) != signature(before):
            raise ValueError('The destination cannot preserve project file timestamps. Choose another drive.')
        if key:
            staged[key] = dest
    try:
        emit('progress', phase='scan', current=0, total=0, message='Checking project files and free space')
        total = 0
        counted = set()
        if source:
            for path, stamp in files(source):
                check()
                key = inode(stamp)
                if key in counted:
                    continue
                if key:
                    counted.add(key)
                total += stamp.st_size
        if not media_internal:
            total += media.stat().st_size
        total+=sum(path.stat().st_size for path,_ in camera_copies)
        if shutil.disk_usage(target.parent).free < total + 64 * 1024**2:
            raise OSError('Not enough free space to save this project.')
        check()
        staging.mkdir()
        if source:
            for path, stamp in files(source):
                copy_file(path, staging / path.relative_to(source), stamp)
        if not media_internal:
            copy_file(media, staging / relative_media, media.stat())
        for camera_path,relative in camera_copies:
            copy_file(camera_path,staging/relative,camera_path.stat())
        if not source:
            from . import video
            from .project_workspace import create
            emit('progress', phase='prepare', current=0, total=0, message='Saving the video project')
            create(staging,media,video.probe(media),request['model'])
            active_relative=None
        else:
            active_relative = active.relative_to(source) if active else None
        check()
        emit('progress', phase='commit', current=0, total=0, message='Finishing the saved project')
        if grouped:
            value=manifest(staging)
            value.update(name=target.name,media=relative_media,time_offset=offset,active_model=request.get('model',value['active_model']))
            if saved_cameras:
                value['mamma_inputs']=saved_cameras
                value['mamma_active_camera']=request.get('mamma_active_camera',state.get('mamma_active_camera',0))
            origins=value.setdefault('origins',[])
            if source and not any(p['path']==str(source) and p.get('relative','.')=='.' for p in origins):
                origins.append(dict(path=str(source),relative='.'))
            view=request.get('workspace_view')
            if view is None:
                fps=request.get('preview_fps',30.)
                view=dict(source_time=offset+request.get('frame',0)/fps)
                if request.get('trim'):view['trim_time']=[offset+t/fps for t in request['trim']]
            value['view']=view
            entry=value.setdefault('models',{}).setdefault(value['active_model'],{})
            entry['view']=view
            if request.get('workspace_settings'):entry['settings']=request['workspace_settings']
            if active_relative is not None:
                entry['active']=active_relative.as_posix();value['reference']=entry['active']
                path=staging/active_relative/'project.json'
                data=manifest(path.parent)
                data.setdefault('pipeline',{}).update(request.get('pipeline',{}))
                for key in ('correction','flow_options','exmr_options'):
                    if key in request:data[key]={**data.get(key,{}),**request[key]}
                atomic_json(path,data)
            atomic_json(staging/'project.json',value)
            check()
            if target.exists():raise ValueError('The destination appeared during saving. Choose another folder name.')
            staging.rename(target)
            emit('project_saved',project=str(target/active_relative) if active_relative is not None else str(target),root=str(target))
            return target
        origins = list(state.get('storage', {}).get('origins', []))
        if source:
            origins.append(str(source))
        origins = list(dict.fromkeys(origins))
        # Preserve all cache/receipt files byte-for-byte and keep their mtimes.
        # Only project manifests gain location metadata, not motion identities.
        for path, _ in files(staging):
            if path.name != 'project.json':
                continue
            check()
            value = json.loads(path.read_text(encoding='utf-8'))
            relative = path.parent.relative_to(staging)
            value['storage'] = dict(version=1, root=os.path.relpath(staging, path.parent),
                media=relative_media, time_offset=offset, origins=origins)
            if relative == active_relative:
                value.setdefault('pipeline', {}).update(request.get('pipeline', {}))
                if 'correction' in request:
                    value['correction'] = request['correction']
            if relative == active_relative or relative == Path('.'):
                for key in ('flow_options', 'exmr_options'):
                    if key in request:
                        value[key] = {**value.get(key,{}), **request[key]}
            atomic_json(path, value)
        view_frame=request.get('frame',0)
        view_trim=request.get('trim')
        if source is None:
            prepared=manifest(staging)
            scale=prepared['fps']/float(request.get('preview_fps',prepared['fps']))
            view_frame=min(prepared['frames']-1,round(view_frame*scale))
            if view_trim:
                view_trim=[round(view_trim[0]*scale),min(prepared['frames'],round(view_trim[1]*scale))]
        atomic_json(staging / 'workspace_state.json', dict(version=1, name=target.name,
            active=active_relative.as_posix(), frame=view_frame, trim=view_trim))
        check()
        if target.exists():
            raise ValueError('The destination appeared during saving. Choose another folder name.')
        staging.rename(target)
        emit('project_saved', project=str(target / active_relative), root=str(target))
        return target
    finally:
        # This is exclusively a fresh UUID directory created by this operation.
        if staging.exists() and staging.parent == target.parent and staging.name.startswith('.' + target.name + '.saving-'):
            shutil.rmtree(staging)


def main(emit):
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding='utf-8'))
    start = time.monotonic()
    try:
        output = save_project(request, emit, lambda: args.request.with_suffix('.cancel').exists())
        emit('complete', stage='save_project', elapsed=time.monotonic()-start, output=str(output))
    except SaveCancelled as error:
        emit('save_cancelled', message=str(error))
    except Exception as error:
        emit('failed', message=str(error))
        raise SystemExit(1)
