#!/usr/bin/env python3
"""Linux counterpart of install.cmd (unofficial port).

Builds the same separated runtimes as installer/engine.py, from the same
version pins in installer/runtime-lock.json, using Linux wheels. The lock's
SHA-256 values are for Windows wheels, so Linux wheels are resolved by exact
version from PyPI and the PyTorch cu128 index instead.

The on-disk layout keeps the names the application expects:
  runtime/python.exe        -> bin/python3.10   (symlink)
  runtime/Lib               -> lib/python3.10   (symlink)
  tools/bin/ffmpeg.exe      -> ffmpeg           (symlink)
  runtime_character/blender.exe -> blender      (symlink)
so application code needs no path changes.

Standard library only. Run with any Python >= 3.10:  python3 tools/install_linux.py
"""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, sys, tarfile, urllib.request, zipfile

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / '.setup'
RES = SETUP / 'bootstrap-git'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
TARGETS = {'shared': 'runtime', 'gui': 'runtime_gui', 'flow': 'runtime_flow', 'hand': 'runtime_hand',
           'htd': 'runtime_htd', 'exmr': 'runtime_exmr'}
CONTRACTS = {'lock_sha256': 'installer/runtime-lock.json', 'registry_sha256': 'mocap/runtime_registry.py',
             'support_sha256': 'installer/git-support-lock.json', 'assets_sha256': 'installer/external-assets.json'}
SUPPORT_ROOTS = ('runtime_character', 'tools/bin', 'third_party/KimodoMotionCorrection/bin')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    temp = Path(path).with_name(Path(path).name + '.partial')
    temp.write_text(json.dumps(value, indent=2), encoding='utf8')
    os.replace(temp, path)


def env():
    e = {k: v for k, v in os.environ.items() if not k.upper().startswith(('PYTHON', 'PIP_', 'UV_'))}
    e.update(UV_PYTHON_INSTALL_DIR=str(SETUP / 'python'), UV_CACHE_DIR=str(SETUP / 'uv-cache'), UV_NO_CONFIG='1',
             UV_PYTHON_PREFERENCE='only-managed', PYTHONDONTWRITEBYTECODE='1', RGB_MOCAP_DATA=str(ROOT))
    return e


def run(command, **kwargs):
    print('  $ ' + ' '.join(str(x) for x in command)[:300], flush=True)
    return subprocess.run([str(x) for x in command], cwd=ROOT, env=env(), check=True, **kwargs)


def fetch(spec, dest):
    """Download one pinned archive to dest, verifying its SHA-256."""
    dest = Path(dest)
    if dest.is_file() and sha(dest) == spec['sha256']:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + '.partial')
    print('Downloading ' + spec['url'], flush=True)
    run(['curl', '-fL', '--retry', '3', '-C', '-', '-o', partial, spec['url']])
    if sha(partial) != spec['sha256']:
        partial.unlink()
        raise RuntimeError('Download does not match its pinned SHA-256: ' + spec['url'])
    os.replace(partial, dest)
    return dest


def extract(archive, root_name, destination):
    """Extract a tar archive's single top folder into destination."""
    destination = Path(destination)
    stage = destination.with_name(destination.name + '.extracting')
    if stage.exists():
        shutil.rmtree(stage)
    with tarfile.open(archive) as tar:
        tar.extractall(stage, filter='data')
    inner = stage / root_name
    if not inner.is_dir():
        raise RuntimeError('Unexpected archive layout: ' + str(archive))
    os.replace(inner, destination)
    shutil.rmtree(stage)


def uv_binary(pins):
    uv = RES / 'uv'
    archive = fetch(pins['uv'], RES / 'uv.tar.gz')
    if not uv.is_file():
        with tarfile.open(archive) as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith('/uv'))
            member.name = 'uv'
            tar.extract(member, RES, filter='data')
    return uv


def bundled_wheels(lock):
    """Rebuild the pure-Python wheels kept as source in installer/runtime_sources."""
    out = {}
    for name in sorted({p['bundled'] for g in lock['groups'] for p in g['packages'] if p.get('bundled')}):
        source = ROOT / 'installer/runtime_sources' / Path(name).stem
        wheel = RES / name
        wheel.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(wheel, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(source.rglob('*')):
                if p.is_file() and p.name != 'SOURCE_WHEEL.json':
                    info = zipfile.ZipInfo(p.relative_to(source).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o644 << 16
                    z.writestr(info, p.read_bytes())
        out[name] = wheel
    return out


def group_ids(lock):
    result = {}
    for g in lock['groups']:
        value = dict(group=g, base_id=result.get(g.get('base')))
        result[g['key']] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return result


def index_args(pins):
    return ['--index-url', 'https://pypi.org/simple', '--extra-index-url', pins['torch_index'],
            '--index-strategy', 'unsafe-best-match']


def linux_extras(uv, group, known, pins):
    """Packages Linux wheels need that the Windows lock does not list (CUDA libraries, triton)."""
    wanted = [p['name'] + '==' + p['version'] for p in group['packages'] if p['name'] in ('torch', 'torchvision')]
    if not wanted:
        return []
    out = subprocess.run([str(uv), 'pip', 'compile', '-', '--python-version', group['python'][:4], '--no-header',
                          '--no-annotate', '--python-platform', 'x86_64-manylinux_2_28', *index_args(pins)],
                         input='\n'.join(wanted) + '\n', text=True, capture_output=True, env=env(), cwd=ROOT)
    if out.returncode:
        raise RuntimeError('Could not resolve the Linux PyTorch dependencies:\n' + out.stderr[-2000:])
    extras = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '==' not in line:
            continue
        name = line.split('==')[0].lower().replace('_', '-')
        if name not in known:
            extras.append(line)
    return extras


def python_home(version):
    home = SETUP / 'python' / ('cpython-' + version + '-linux-x86_64-gnu')
    if not (home / 'bin').is_dir():
        raise RuntimeError('The private Python download is incomplete: ' + version)
    return home


def install_group(uv, group, lock, wheels, pins):
    key = group['key']
    dest = ROOT / group['target']
    short = group['python'].rsplit('.', 1)[0]
    if dest.exists():
        shutil.rmtree(dest)
    if key in ('shared', 'exmr'):
        shutil.copytree(python_home(group['python']), dest, symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        (dest / 'python.exe').symlink_to('bin/python' + short)
        (dest / 'Lib').symlink_to('lib/python' + short)
        python = dest / 'bin' / ('python' + short)
        target = dest / 'lib' / ('python' + short) / 'site-packages'
    else:
        python = ROOT / 'runtime/bin/python3.10'
        target = dest
        target.mkdir()
    base = next((g for g in lock['groups'] if g['key'] == group.get('base')), None)
    known = {p['name'] for p in group['packages']} | ({p['name'] for p in base['packages']} if base else set())
    lines = []
    for p in group['packages']:
        lines.append(str(wheels[p['bundled']]) if p.get('bundled') else p['name'] + '==' + p['version'])
    extras = linux_extras(uv, group, known, pins)
    if extras:
        print('  Linux-only additions for ' + key + ': ' + ', '.join(extras), flush=True)
    req = SETUP / (key + '-linux-requirements.txt')
    req.write_text('\n'.join(lines + extras) + '\n', encoding='utf8')
    run([uv, 'pip', 'install', '--python', python, '--target', target, '--no-deps', '--only-binary', ':all:',
         '--link-mode', 'copy', *index_args(pins), '-r', req])
    return extras


def verify_group(group):
    python = ROOT / ('runtime_exmr' if group['key'] == 'exmr' else 'runtime') / 'python.exe'
    request = SETUP / (group['key'] + '-verify-input.json')
    atomic(request, group)
    out = subprocess.run([str(python), '-I', '-X', 'utf8', str(ROOT / 'installer/verify_runtime.py'), str(ROOT),
                          str(request), str(ROOT / 'mocap/runtime_registry.py'), str(ROOT)],
                         cwd=ROOT, env=env(), text=True, capture_output=True)
    if out.returncode:
        raise RuntimeError('Runtime verification failed for ' + group['key'] + ':\n' + (out.stdout + out.stderr)[-3000:])
    return json.loads(out.stdout.strip().splitlines()[-1])


def runtimes(uv, lock, pins, force):
    done_file = SETUP / 'linux-runtimes.json'
    try:
        done = json.loads(done_file.read_text(encoding='utf8'))
    except (OSError, ValueError):
        done = {}
    ids = group_ids(lock)
    wheels = bundled_wheels(lock)
    reports = {}
    for version in sorted({g['python'] for g in lock['groups']}):
        run([uv, 'python', 'install', version, '--install-dir', SETUP / 'python', '--no-bin'])
    for index, group in enumerate(lock['groups'], 1):
        key = group['key']
        if TARGETS.get(key) != group['target']:
            raise ValueError('Invalid runtime target')
        dest = ROOT / group['target']
        record = done.get(key)
        base_ok = not group.get('base') or group['base'] in reports
        if not force and record and record['id'] == ids[key] and dest.is_dir() and base_ok and not record.get('rebuilt_base'):
            try:
                reports[key] = verify_group(group)
                print('[%d/%d] Keeping verified %s runtime' % (index, len(lock['groups']), key), flush=True)
                continue
            except RuntimeError as exc:
                print('Repair required for ' + key + ': ' + str(exc)[-300:], flush=True)
        if dest.exists() and not record:
            raise RuntimeError('Existing unmanaged runtime folder was preserved: ' + str(dest))
        print('[%d/%d] Installing %s runtime' % (index, len(lock['groups']), key), flush=True)
        extras = install_group(uv, group, lock, wheels, pins)
        # Record ownership before verification so a failed group can be repaired by a rerun.
        done[key] = dict(id='incomplete', extras=extras)
        atomic(done_file, done)
        reports[key] = verify_group(group)
        done[key] = dict(id=ids[key], extras=extras)
        atomic(done_file, done)
        if key in ('shared',):
            # Overlays are bound to this exact base; rebuild them after it changes.
            for other in lock['groups']:
                if other.get('base') == key and other['key'] in done:
                    done[other['key']]['id'] = 'stale-base'
            atomic(done_file, done)
        print('  %s runtime verified (%d files)' % (key, reports[key]['checked_files']), flush=True)
    return reports, ids


def support(pins):
    cache = RES / 'support-downloads'
    blender = ROOT / 'runtime_character'
    marker = blender / '.linux-port.json'
    if not (marker.is_file() and json.loads(marker.read_text())['sha256'] == pins['blender']['sha256'] and (blender / 'blender').is_file()):
        archive = fetch(pins['blender'], cache / 'blender-4.5.0-linux-x64.tar.xz')
        if blender.exists():
            if not marker.is_file():
                raise RuntimeError('Existing unmanaged runtime_character folder was preserved.')
            shutil.rmtree(blender)
        print('Extracting Blender 4.5.0...', flush=True)
        extract(archive, pins['blender']['archive_root'], blender)
        (blender / 'blender.exe').symlink_to('blender')
        atomic(marker, dict(sha256=pins['blender']['sha256'], url=pins['blender']['url']))
    version = subprocess.run([str(blender / 'blender.exe'), '--version'], capture_output=True, text=True).stdout.splitlines()[0]
    print('  ' + version, flush=True)
    bin_dir = ROOT / 'tools/bin'
    marker = bin_dir / '.linux-port.json'
    if not (marker.is_file() and json.loads(marker.read_text())['sha256'] == pins['ffmpeg']['sha256'] and (bin_dir / 'ffmpeg').is_file()):
        archive = fetch(pins['ffmpeg'], cache / 'ffmpeg-7.1-linux64-gpl.tar.xz')
        stage = SETUP / 'ffmpeg-stage'
        if stage.exists():
            shutil.rmtree(stage)
        extract(archive, pins['ffmpeg']['archive_root'], stage)
        bin_dir.mkdir(parents=True, exist_ok=True)
        (bin_dir / 'notices').mkdir(exist_ok=True)
        for name in ('ffmpeg', 'ffprobe'):
            shutil.copy2(stage / 'bin' / name, bin_dir / name)
            link = bin_dir / (name + '.exe')
            link.unlink(missing_ok=True)
            link.symlink_to(name)
        for path in stage.iterdir():
            if path.is_file():
                shutil.copy2(path, bin_dir / 'notices' / path.name)
        shutil.rmtree(stage)
        atomic(marker, dict(sha256=pins['ffmpeg']['sha256'], url=pins['ffmpeg']['url']))
    print('  ' + subprocess.run([str(bin_dir / 'ffmpeg.exe'), '-version'], capture_output=True, text=True).stdout.splitlines()[0], flush=True)
    # Native motion correction: the checked-in binary is a Windows .pyd, so build the same source here.
    out = ROOT / 'third_party/KimodoMotionCorrection/bin'
    source = ROOT / 'third_party/KimodoMotionCorrection'
    identity = hashlib.sha256(b''.join(p.read_bytes() for p in sorted([*source.glob('src/cpp/**/*'), source / 'CMakeLists.txt']) if p.is_file())).hexdigest()
    marker = out / '.linux-port.json'
    if not (marker.is_file() and json.loads(marker.read_text())['source'] == identity and list(out.glob('_motion_correction*.so'))):
        if not shutil.which('cmake') or not shutil.which('g++'):
            raise RuntimeError('Building motion correction needs cmake and g++ (sudo apt install build-essential cmake).')
        build = SETUP / 'motion-correction-build'
        if build.exists():
            shutil.rmtree(build)
        python = python_home('3.10.22') / 'bin/python3.10'
        print('Building native motion correction...', flush=True)
        log = SETUP / 'motion-correction-build.log'
        with log.open('w') as stream:
            subprocess.run(['cmake', '-S', str(source), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                            '-DPython3_EXECUTABLE=' + str(python), '-DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON'],
                           check=True, stdout=stream, stderr=subprocess.STDOUT)
            subprocess.run(['cmake', '--build', str(build), '--target', '_motion_correction', '--parallel', '4'],
                           check=True, stdout=stream, stderr=subprocess.STDOUT)
        built = list(build.glob('_motion_correction*.so'))
        if len(built) != 1:
            raise RuntimeError('Motion correction build produced no module; see ' + str(log))
        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob('_motion_correction*'):
            old.unlink()
        shutil.copy2(built[0], out / built[0].name)
        atomic(marker, dict(source=identity, module=built[0].name))
    code = ("import sys;sys.path.insert(0,sys.argv[1]);from mocap.correction import backend;"
            "assert backend().rgb_mocap_anatomical_version==1;print('  Native motion correction verified')")
    run([ROOT / 'runtime/python.exe', '-I', '-X', 'utf8', '-B', '-c', code, ROOT])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--force', action='store_true', help='Rebuild every runtime even when it verifies.')
    parser.add_argument('--skip-optional', action='store_true', help='Skip PEAR and camera-alignment support.')
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise SystemExit('This installer is for Linux. Use install.cmd on Windows.')
    RES.mkdir(parents=True, exist_ok=True)
    from session_lock import app_lock
    from stage_assets import stage as stage_assets
    pins = json.loads((ROOT / 'installer/linux-support-lock.json').read_text(encoding='utf8'))
    lock = json.loads((ROOT / 'installer/runtime-lock.json').read_text(encoding='utf8'))
    with app_lock(ROOT):
        pending = SETUP / 'git-sync.pending'
        pending.write_text('Run install.sh to complete or retry.\n', encoding='utf8')
        free = shutil.disk_usage(ROOT).free
        print('Free disk: %.1f GiB' % (free / 1024 ** 3), flush=True)
        if not (ROOT / 'SETUP_RECEIPT.json').exists() and free < 30 * 1024 ** 3:
            raise OSError('Insufficient free space for the runtime installation (about 30 GiB needed).')
        uv = uv_binary(pins)
        reports, ids = runtimes(uv, lock, pins, args.force)
        support(pins)
        print('Fetching auxiliary body-model data...', flush=True)
        stage_assets(None, ROOT, download=True)
        for name in ('models', 'projects'):
            (ROOT / name).mkdir(exist_ok=True)
        atomic(ROOT / 'SETUP_RECEIPT.json', dict(schema=1, id=lock['id'], version=lock['version'], runtimes=reports,
                                                 group_ids=ids, models_included=False, platform='linux'))
        inventory = []
        for name in sorted(SUPPORT_ROOTS):
            for path in sorted((ROOT / name).rglob('*')):
                if path.is_file() and not path.is_symlink():
                    st = path.stat()
                    inventory.append(dict(path=path.relative_to(ROOT).as_posix(), stat=[st.st_size, st.st_mtime_ns]))
        atomic(SETUP / 'git-runtime-state.json', dict(**{k: sha(ROOT / v) for k, v in CONTRACTS.items()},
                                                     support_inventory=inventory, platform='linux'))
        pending.unlink(missing_ok=True)
        if not args.skip_optional:
            from tools.optional_support import prepare
            prepare(ROOT)
    print('Done. Start the app with ./RGBMoCap.sh', flush=True)


if __name__ == '__main__':
    main()
