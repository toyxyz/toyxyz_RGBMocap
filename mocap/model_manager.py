"""Model Manager catalog, storage inventory and scoped removal."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
from uuid import uuid4

from .models import BY_KEY, MODELS_LIST, STAGE_REQUIREMENTS, ready
from .motion_tools import MOTION_TOOLS
from .paths import HOME, MODELS, ROOT, UPSTREAM
from .storage import atomic_json


def workflow_catalog():
    from .flow_hamer import MODEL_KEYS
    from .sam31_runtime import SUPPORTED

    catalog = {}
    for key, tool in MOTION_TOOLS.items():
        stages = tool.stages(refine=True, footmr=True, handflow=True)
        if key in ('mamma', 'prompthmr', 'pear', 'gemx'):
            stages = (*stages, key + '_assets')
        if key in SUPPORTED:
            stages = (*stages, 'people', 'sam31_people')
        if key == 'pear':
            stages = (*stages, 'pear_align')
        if key == 'prompthmr':
            stages = (*stages, 'prompthmr_camera_assets')
        keys = {model for stage in stages for model in STAGE_REQUIREMENTS.get(stage, ())}
        if key == 'flowhmr':
            keys.update(('flow_metadata', *MODEL_KEYS))
        if not tool.live:
            from .rife_video import MODEL_KEY
            keys.add(MODEL_KEY)
        catalog[key] = keys
    return catalog


def workflow_name(key):
    return 'PEAR' if key == 'pear' else MOTION_TOOLS[key].label


def model_consumers(catalog):
    return {
        model.key: tuple(workflow for workflow, keys in catalog.items() if model.key in keys)
        for model in MODELS_LIST
    }


def file_info(path, root):
    """Inspect an entry without accepting linked or redirected parent directories."""
    parts = path.relative_to(root).parts
    if not parts:
        raise ValueError('Expected a file below a managed folder.')
    current = root
    for part in ('', *parts[:-1]):
        current /= part
        info = current.lstat()
        if not stat.S_ISDIR(info.st_mode) or is_redirected(info):
            raise ValueError(f'Model storage path is redirected: {current}')
    return path.lstat()


def is_redirected(info):
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0))


@dataclass(frozen=True)
class FileEntry:
    path: Path
    root: Path
    identity: tuple
    size: int
    links: int
    modified: int
    alias: bool = False


def entry(path, root, *, allow_link=False):
    try:
        info = file_info(path, root)
    except FileNotFoundError:
        return None
    linked = is_redirected(info)
    if linked:
        if not allow_link or not stat.S_ISLNK(info.st_mode):
            raise ValueError(f'Model file is redirected or stored externally: {path}')
    elif not stat.S_ISREG(info.st_mode):
        raise ValueError(f'Model path is not a regular file: {path}')
    identity = (info.st_dev, info.st_ino) if info.st_ino else (path,)
    return FileEntry(path, root, identity, 0 if linked else info.st_size,
                     info.st_nlink, info.st_mtime_ns, linked)


def runtime_aliases(model):
    from .exmr_backend import LINKS as exmr_links
    from .flow_backend import ASSET_LINKS
    from .prompthmr_models import CAMERA_LINKS, LINKS as prompt_links

    key = model.key
    if key != 'person_reid' and not key.startswith('phmr_'):
        for engine in ('GVHMR', 'HTD-Refine', 'FootMR'):
            yield UPSTREAM / engine / 'inputs/checkpoints' / model.relative, ROOT
    if key in ASSET_LINKS:
        yield UPSTREAM / 'FlowHMR' / ASSET_LINKS[key], ROOT
    if key in exmr_links:
        yield HOME / 'cache/exmr/assets' / exmr_links[key], HOME
    relative = prompt_links.get(key) or CAMERA_LINKS.get(key)
    if relative:
        # Managed PromptHMR environments can keep one or more engine versions.
        parent = ROOT / 'runtime_prompthmr'
        if parent.is_dir() and not is_redirected(parent.lstat()):
            for version in parent.iterdir():
                yield version / 'engine' / relative, ROOT


def linked_entries(model, payload):
    for path, root in runtime_aliases(model):
        try:
            candidate = entry(path, root, allow_link=True)
            if candidate and (
                (candidate.alias and path.resolve() == payload.path)
                or (not candidate.alias and candidate.identity == payload.identity)
            ):
                yield candidate
        except (OSError, ValueError, RuntimeError):
            # Independent copies and paths outside the managed tree are preserved.
            continue


def receipt_path(model):
    return model.path.with_suffix(model.path.suffix + '.receipt.json')


def installed_files(model):
    if model.key != 'pear_geometry':
        payload = entry(model.path, HOME)
        if payload is None:
            return ()
        receipt = entry(receipt_path(model), HOME)
        return (payload, *((receipt,) if receipt else ()), *linked_entries(model, payload))

    from .pear_models import geometry_folder, spec
    pointer = entry(model.path, HOME)
    if pointer is None:
        return ()
    folder = geometry_folder()
    if folder.parent.parent != MODELS / 'pear' or not folder.parent.name.startswith('geometry-') or folder.name != 'assets':
        raise ValueError('PEAR geometry is outside its managed generation.')
    return (pointer, *(entry(folder / asset['path'], HOME) for asset in spec()['files']))


def installed_models():
    # Include broken installations and externally linked paths so users can see them.
    return [model for model in MODELS_LIST if os.path.lexists(model.path)]


def physical_bytes(files, reclaim=False):
    counts = {}
    for item in {item.path: item for item in files if item is not None and not item.alias}.values():
        size, count, links = counts.get(item.identity, (item.size, 0, item.links))
        counts[item.identity] = (size, count + 1, max(links, item.links))
    return sum(size for size, count, links in counts.values()
               if not reclaim or (0 < links <= count))


def file_size(size):
    if size < 1024:
        return f'{size} B'
    for unit in ('KiB', 'MiB', 'GiB', 'TiB'):
        size /= 1024
        if size < 1024 or unit == 'TiB':
            return f'{size:.2f} {unit}'


def removal_plan(keys):
    keys = set(keys)
    retained = {model.path for model in MODELS_LIST if model.key not in keys}
    files = {}
    for key in sorted(keys):
        model = BY_KEY[key]
        installed = installed_files(model)
        if not installed or any(item is None for item in installed):
            raise ValueError(f'{model.name} changed. Refresh the list and try again.')
        if any(item.path in retained for item in installed):
            raise ValueError(f'{model.name} shares its storage path with another registered model.')
        for item in installed:
            files[item.path] = item
    return tuple(files.values())


def journal_path():
    return HOME / 'models/.rgbmocap-removal.json'


def recovery_roots(keys, paths):
    allowed = {}
    for key in keys:
        model = BY_KEY[key]
        allowed[model.path] = HOME
        if key != 'pear_geometry':
            allowed[receipt_path(model)] = HOME
            allowed.update(runtime_aliases(model))
            continue
        from .pear_models import spec
        assets = {Path(item['path']) for item in spec()['files']}
        pear = MODELS / 'pear'
        for path in paths:
            if path.is_relative_to(pear):
                parts = path.relative_to(pear).parts
                if (len(parts) > 2 and parts[0].startswith('geometry-') and parts[1] == 'assets'
                        and Path(*parts[2:]) in assets):
                    allowed[path] = HOME
    return allowed


def fingerprint(item):
    device, file_id = item.identity if len(item.identity) == 2 else (0, 0)
    return [device, file_id, item.size, item.modified, item.alias]


def recover_removal():
    journal = journal_path()
    if not entry(journal, HOME):
        return
    state = json.loads(journal.read_text(encoding='utf-8'))
    txid = state['id']
    if state.get('version') != 1 or len(txid) != 32 or any(c not in '0123456789abcdef' for c in txid):
        raise ValueError('Invalid model removal record.')
    records = state['files']
    paths = [Path(record['path']) for record in records]
    allowed = recovery_roots(state['keys'], paths)
    if len(paths) != len(set(paths)) or any(path not in allowed or '..' in path.parts for path in paths):
        raise ValueError('Model removal record leaves managed storage.')
    folders = set()
    for index in reversed(range(len(paths))):
        path = paths[index]
        root = allowed[path]
        stage = path.with_name(f'.rgbmocap-remove-{txid}-{index}')
        current = entry(stage, root, allow_link=True)
        if current is None:
            continue
        if fingerprint(current) != records[index]['fingerprint']:
            raise ValueError(f'Model removal recovery found a changed file: {stage}')
        if state['committed']:
            stage.unlink()
            if path.is_relative_to(MODELS):
                folders.add(path.parent)
        elif entry(path, root, allow_link=True):
            raise ValueError(f'Model removal recovery found a replacement: {path}')
        else:
            stage.rename(path)
    for folder in sorted(folders, key=lambda path: len(path.parts), reverse=True):
        while folder != MODELS:
            try:
                folder.rmdir()
            except OSError:
                break
            folder = folder.parent
    journal.unlink()
    return True


def remove_models(keys, expected):
    journal = journal_path()
    if entry(journal, HOME):
        raise ValueError('A previous model removal needs recovery.')
    if removal_plan(keys) != expected:
        raise ValueError('Model files changed after confirmation. Review the updated list.')
    if not expected:
        return
    txid = uuid4().hex
    moves = [(item, item.path.with_name(f'.rgbmocap-remove-{txid}-{index}'))
             for index, item in enumerate(expected)]
    state = dict(version=1, id=txid, committed=False, keys=sorted(keys), files=[
        dict(path=str(item.path), fingerprint=fingerprint(item)) for item in expected
    ])
    atomic_json(journal, state)
    try:
        for item, stage in moves:
            if entry(item.path, item.root, allow_link=item.alias) != item or os.path.lexists(stage):
                raise ValueError(f'Model file changed: {item.path}')
            item.path.rename(stage)
        atomic_json(journal, {**state, 'committed': True})
    except (OSError, ValueError):
        recover_removal()
        raise
    recover_removal()
