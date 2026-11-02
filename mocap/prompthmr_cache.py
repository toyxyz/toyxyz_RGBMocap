"""Persistent PromptHMR downloads; inference intermediates stay per generation."""
from pathlib import Path
import itertools
import os
import shutil
import uuid

from .paths import HOME
from .storage import sha256

# The installed torchvision DEFAULT weights (the hash prefix is in its URL).
DEEPLAB = 'deeplabv3_resnet50_coco-cd0a2569.pth'
DEEPLAB_BYTES = 168312152
DEEPLAB_SHA256 = 'cd0a25694c4a0f7106b38f4938bf90a874f2f241cc410b8f63c7024399538f06'

# Only these reviewed fixed-camera-equivalent revisions reuse the preceding
# identity. Moving-camera preparation explicitly uses the actual code hashes.
# Any future pipeline edit invalidates normally until separately reviewed.
PIPELINE_CACHE_ONLY = {
    '062f3d4f4e79fb8b4998257f73bf69a916a31c4138b31622774c5b64d791ad85':
    'b3098bbc038f9f5e25c9b9e96c9ec4fa2116226f74c645089bd328014f024e7d',
    'd1075279956f9443c3ff80694a9388fc9e1f2d74b3af5f9dab0714c50c2a7687':
    'b3098bbc038f9f5e25c9b9e96c9ec4fa2116226f74c645089bd328014f024e7d',
    # Resource Safety wording only; no fixed-camera inference changes.
    '82b1d570030f4858683064d10a214be1ded4b1d93de190d8971ad0847ac5314e':
    'b3098bbc038f9f5e25c9b9e96c9ec4fa2116226f74c645089bd328014f024e7d',
}


def inference_code_hash(path):
    digest = sha256(path)
    return PIPELINE_CACHE_ONLY.get(digest, digest) if path.name == 'prompthmr_pipeline.py' else digest


def valid_deeplab(path):
    try:
        return path.is_file() and path.stat().st_size == DEEPLAB_BYTES and sha256(path) == DEEPLAB_SHA256
    except OSError:
        return False


def child_directories(parent, limit):
    try:
        with os.scandir(parent) as entries:
            for entry in itertools.islice(entries, limit):
                try:
                    path = Path(entry.path)
                    if entry.is_dir(follow_symlinks=False) and path.resolve().is_relative_to(parent.resolve()):
                        yield path
                except OSError:
                    continue
    except OSError:
        return


def legacy_checkpoints(work, home):
    """Bound the search to known generation layouts, never walk project payloads."""
    yield work / 'torch-cache/hub/checkpoints' / DEEPLAB
    parents = [work.parent]
    projects = home / 'projects'
    parents.extend(project / 'prompthmr/inference' for project in child_directories(projects, 512))
    remaining = 512
    seen = set()
    for parent in parents:
        parent = parent.resolve()
        if parent in seen or not parent.is_dir():
            continue
        seen.add(parent)
        for generation in child_directories(parent, remaining):
            if remaining <= 0:
                return
            remaining -= 1
            yield generation / 'torch-cache/hub/checkpoints' / DEEPLAB


def configure_downloads(work, *, home=HOME):
    """Reuse validated old weights without moving/deleting any project files."""
    import torch
    work, home = Path(work).resolve(), Path(home).resolve()
    cache = home / 'cache/prompthmr'
    torch_home = cache / 'torch'
    hub = torch_home / 'hub'
    target = hub / 'checkpoints' / DEEPLAB
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not valid_deeplab(target):
            raise ValueError('PromptHMR shared DeepLab cache failed SHA-256 verification: ' + str(target))
    else:
        remaining_checks = 4  # Bound legacy hashing to about 642 MiB.
        for source in legacy_checkpoints(work, home):
            try:
                if source.stat().st_size != DEEPLAB_BYTES:
                    continue
            except OSError:
                continue
            if remaining_checks == 0:
                break
            remaining_checks -= 1
            if not valid_deeplab(source):
                continue
            temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
            try:
                try:
                    os.link(source, temporary)
                except OSError:
                    shutil.copy2(source, temporary)
                if not valid_deeplab(temporary):
                    continue
                # Another worker may have populated the shared cache meanwhile.
                if target.exists():
                    if not valid_deeplab(target):
                        raise ValueError('PromptHMR shared DeepLab cache changed during reuse.')
                else:
                    os.replace(temporary, target)
                print('PromptHMR: reused SHA-256 verified DeepLab weights from an existing project.', flush=True)
                break
            finally:
                temporary.unlink(missing_ok=True)
    os.environ.update(TORCH_HOME=str(torch_home), HF_HOME=str(cache / 'hf'))
    # Override any previously initialized torch.hub directory as well as env.
    torch.hub.set_dir(str(hub))
    print('PromptHMR: shared model download cache: ' + str(hub), flush=True)
    return hub
