"""Install an optional GEM-X runtime using the existing Windows CPython base."""

import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from installer import engine
from mocap.gemx_runtime import LOCK, configuration, runtime_id
from mocap.storage import atomic_json, sha256
from tools.pinned_download import download, valid
from tools.session_lock import app_lock
from tools.setup_mamma import extract_source


def reuse_dlls(base, target, pins):
    """Share only verified identical files and retain the staged copy on failure."""
    shared = 0
    for row in pins:
        name = row['name']
        if not re.fullmatch(r'[\w.-]+\.dll', name):
            raise ValueError('Invalid CUDA DLL pin.')
        dll, old = target / name, base / name
        if not dll.is_file() or not old.is_file() or old.is_symlink():
            continue
        if getattr(old.stat(), 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            continue
        if not valid(old, row) or not valid(dll, row):
            continue
        link = dll.with_suffix('.gemx-link')
        try:
            os.link(old, link)
            if not valid(link, row):
                raise OSError('CUDA DLL changed while linking.')
            os.replace(link, dll)
        except OSError:
            link.unlink(missing_ok=True)
        else:
            shared += dll.stat().st_size
    return shared


def wheel(root, row, cache):
    if 'bundled' in row:
        file = root / '.setup/bootstrap-git' / row['bundled']
        if not valid(file, row):
            raise ValueError('Bundled wheel is missing or changed: ' + row['filename'])
        return file
    file = cache / row['filename']
    if not valid(file, row):
        print('Downloading ' + row['filename'], flush=True)
        part = download(cache, row['filename'], row)
        os.replace(part, file)
    return file


def setup(root=ROOT):
    if os.name != 'nt':
        raise RuntimeError('GEM-X setup is available on Windows.')
    root = Path(root).resolve()
    lock = json.loads((root / LOCK).read_text(encoding='utf-8'))
    base_lock = json.loads((root / 'installer/runtime-lock.json').read_text(encoding='utf-8'))
    base = root / 'runtime_exmr/python.exe'
    uv = root / '.setup/bootstrap-git/uv.exe'
    if not base.is_file() or not uv.is_file() or sha256(uv) != base_lock['uv_sha256']:
        raise RuntimeError('Run install.cmd first to prepare ExMR Python and the installer.')
    logs = root / '.setup/logs'
    logs.mkdir(parents=True, exist_ok=True)
    cache = root / '.setup/gemx-downloads'
    cache.mkdir(parents=True, exist_ok=True)

    def run(args, label):
        print('GEM-X setup: ' + label, flush=True)
        engine.run(args, root, None, logs / ('gemx-' + label + '.log'))

    def verify(folder):
        report = logs / 'gemx-verification.json'
        run([folder / 'Scripts/python.exe', '-I', '-B', '-X', 'utf8',
             root / 'installer/verify_gemx.py', folder, root / LOCK, report], 'verify')
        return json.loads(report.read_text(encoding='utf-8'))

    with app_lock(root):
        tag = runtime_id(root)
        parent = root / 'runtime_gemx'
        if parent.resolve().parent != root:
            raise ValueError('GEM-X runtime folder leaves the installation.')
        registration = root / '.setup/gemx-runtime.json'
        previous = None
        try:
            generation = json.loads(registration.read_text(encoding='utf-8'))['generation']
            if re.fullmatch(r'[0-9a-f]{12}-[0-9a-f]{8}', generation):
                candidate = parent / generation
                is_link = getattr(candidate.lstat(), 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
                if candidate.is_dir() and not is_link and (candidate / '.gemx-managed').is_file():
                    previous = candidate
        except (OSError, ValueError, KeyError, TypeError):
            pass
        try:
            existing = configuration(root)
            verify(existing['folder'])
        except (OSError, ValueError, RuntimeError) as error:
            print(str(error), flush=True)
        else:
            print('Verified GEM-X runtime retained.', flush=True)
            return
        if shutil.disk_usage(root).free < 12 * 1024**3:
            raise OSError('Allow 12 GiB of temporary free space for GEM-X runtime setup.')
        parent.mkdir(exist_ok=True)
        folder = parent / (tag[:12] + '-' + uuid.uuid4().hex[:8])
        folder.mkdir()
        committed = False
        try:
            run([base, '-I', '-B', '-m', 'venv', '--copies', '--without-pip', folder], 'python')
            target = folder / 'Lib/site-packages'
            requirements = folder / 'packages.txt'
            requirements.write_text('\n'.join(
                wheel(root, row, cache).as_uri() + ' --hash=sha256:' + row['sha256']
                for row in lock['packages']) + '\n', encoding='utf-8')
            run([uv, 'pip', 'install', '--python', folder / 'Scripts/python.exe', '--target', target,
                 '--offline', '--no-index', '--no-deps', '--require-hashes', '--only-binary', ':all:',
                 '--link-mode', 'copy', '--no-cache', '-r', requirements], 'packages')
            requirements.unlink()
            source = lock['source']
            archive = download(cache, 'gem-x-' + source['commit'], source)
            extract_source(archive, {**source, 'archive_root': 'GEM-X-' + source['commit']},
                           folder / 'engine')
            shared = reuse_dlls(base.parent / 'Lib/site-packages/torch/lib', target / 'torch/lib',
                                lock['shared_cuda_dlls'])
            report = verify(folder)
            if runtime_id(root) != tag:
                raise RuntimeError('GEM-X setup sources changed. Run setup again.')
            report.update(generation=folder.name, identity=tag, shared_bytes=shared)
            (folder / '.gemx-managed').touch(exist_ok=False)
            atomic_json(registration, report)
            committed = True
            print(f'GEM-X ready. Shared {shared / 1e9:.2f} GB of verified CUDA files.', flush=True)
            if previous is not None:
                try:
                    engine.remove_owned(parent, previous)
                except (OSError, ValueError) as error:
                    print('Previous GEM-X generation retained: ' + str(error), flush=True)
        finally:
            if not committed:
                engine.remove_owned(parent, folder)


if __name__ == '__main__':
    try:
        setup()
    except (OSError, ValueError, RuntimeError) as error:
        print('GEM-X setup failed: ' + str(error), file=sys.stderr)
        raise SystemExit(1) from None
