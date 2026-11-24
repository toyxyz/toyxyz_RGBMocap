"""Install a prebuilt Live Link sender or build it against the installed engine."""
from pathlib import Path
import argparse
from hashlib import sha256
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mocap.unreal_stream import source_identity, source_tree_identity, UnrealStream
from tools.package_unreal_sender import BINARIES, METADATA, MODULE, read_package
from tools.session_lock import app_lock


def find_engine(explicit=None, package=None):
    candidates = [Path(explicit)] if explicit else []
    if not explicit:
        base = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Epic Games'
        candidates = sorted(base.glob('UE_5.*'), reverse=True)
    for path in candidates:
        version = path / 'Engine/Build/Build.version'
        if version.is_file():
            data = json.loads(version.read_text(encoding='utf-8'))
            if (data['MajorVersion'], data['MinorVersion']) < (5, 6):
                continue
            if package is not None:
                number = '.'.join(str(data[key]) for key in ('MajorVersion', 'MinorVersion', 'PatchVersion'))
                metadata = path / 'Engine' / BINARIES / METADATA
                if (number != package['engine'] or not metadata.is_file()
                        or json.loads(metadata.read_text(encoding='utf-8-sig')).get('BuildId') != package['build_id']):
                    continue
            return path.resolve()
    if package is not None:
        raise RuntimeError(f"Install Epic Launcher UE {package['engine']} Win64 "
                           f"(BuildId {package['build_id']}) or pass its folder with --engine.")
    raise RuntimeError('Unreal Engine 5.6 or later was not found. Use --engine "D:/Epic Games/UE_5.6".')


def register(runtime, engine, project, frozen_source, engine_build, build_id=None):
    receipt = runtime / 'bridge.json'
    module = project.parent / BINARIES / MODULE
    if not module.is_file():
        raise RuntimeError('Build finished without the Live Link sender module.')
    if source_identity(ROOT) != frozen_source or source_tree_identity(project.parent) != frozen_source:
        raise RuntimeError('Live Link source changed during setup. Run setup again.')
    if sha256((engine / 'Engine/Build/Build.version').read_bytes()).hexdigest() != engine_build:
        raise RuntimeError('Unreal Engine changed during setup. Run setup again.')
    data = dict(version=1, engine=str(engine), source=frozen_source,
                project=project.relative_to(runtime).as_posix(),
                module=sha256(module.read_bytes()).hexdigest(),
                engine_build=engine_build)
    if build_id is not None:
        data['build_id'] = build_id
    previous_receipt = receipt.read_bytes() if receipt.is_file() else None
    pending = receipt.with_suffix('.tmp')
    pending.write_text(json.dumps(data, indent=2), encoding='utf-8')
    pending.replace(receipt)
    sender = None
    try:
        print('Verifying the local provider startup...', flush=True)
        sender = UnrealStream(ROOT)
        deadline = time.monotonic() + 120
        while not sender.ready:
            error = sender.error()
            if error:
                raise RuntimeError(error)
            if time.monotonic() > deadline:
                raise RuntimeError('The provider did not start within 120 seconds.')
            time.sleep(.1)
    except BaseException:
        if previous_receipt is None:
            receipt.unlink(missing_ok=True)
        else:
            pending.write_bytes(previous_receipt)
            pending.replace(receipt)
        raise
    finally:
        if sender is not None:
            sender.close()
            sender.thread.join(8)


def install_prebuilt(runtime, engine, frozen_source, engine_build, package, payload):
    project_dir = Path(tempfile.mkdtemp(prefix='Prebuilt-', dir=runtime))
    try:
        shutil.copytree(ROOT / 'developer-build/unreal-live-link', project_dir, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('Binaries', 'Intermediate', 'Saved', 'DerivedDataCache'))
        binaries = project_dir / BINARIES
        binaries.mkdir(parents=True)
        for name, raw in payload.items():
            (binaries / name).write_bytes(raw)
        register(runtime, engine, project_dir / 'RGBMoCapLiveLink.uproject', frozen_source,
                 engine_build, build_id=package['build_id'])
    except BaseException:
        shutil.rmtree(project_dir, ignore_errors=True)
        raise


def build_sender(runtime, engine, frozen_source, engine_build, keep_build_cache):
    ubt = engine / 'Engine/Binaries/DotNET/UnrealBuildTool/UnrealBuildTool.dll'
    hosts = list((engine / 'Engine/Binaries/ThirdParty/DotNet').glob('*/win-x64/dotnet.exe'))
    if not ubt.is_file() or not hosts:
        raise RuntimeError('Install the Unreal Engine editor and C++ build support first.')
    build_key = sha256(f'{frozen_source}:{engine_build}'.encode('ascii')).hexdigest()[:12]
    project_dir = runtime / ('Bridge-' + build_key)
    shutil.copytree(ROOT / 'developer-build/unreal-live-link', project_dir, dirs_exist_ok=True)
    project = project_dir / 'RGBMoCapLiveLink.uproject'
    if source_tree_identity(project_dir) != frozen_source or source_identity(ROOT) != frozen_source:
        raise RuntimeError('Live Link source changed during setup. Retry when editing is finished.')
    print('Building optional Live Link sender using ' + str(engine), flush=True)
    print('Requires Visual Studio C++ tools and Windows SDK. No models are downloaded.', flush=True)
    # Call the native .NET host directly to preserve international paths.
    subprocess.run([str(sorted(hosts)[-1]), str(ubt), 'RGBMoCapLiveLinkEditor', 'Win64', 'Development',
                    '-Project='+str(project), '-WaitMutex', '-NoHotReloadFromIDE'],
                   cwd=project_dir, check=True)
    register(runtime, engine, project, frozen_source, engine_build)
    if not keep_build_cache:
        intermediate = (project_dir / 'Intermediate').resolve()
        if not intermediate.is_relative_to(runtime.resolve()):
            raise RuntimeError('Build cache is outside the optional runtime.')
        if intermediate.is_dir():
            shutil.rmtree(intermediate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path)
    parser.add_argument('--prebuilt', type=Path, help='RGB MoCap sender ZIP for an exact Epic Launcher build')
    parser.add_argument('--keep-build-cache', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'win32':
        raise RuntimeError('This optional bridge currently supports Windows only.')
    with app_lock(ROOT):
        frozen_source = source_identity(ROOT)
        if args.prebuilt is not None:
            package, payload = read_package(args.prebuilt, frozen_source)
        else:
            package = None
        engine = find_engine(args.engine, package)
        if not (engine / 'Engine/Binaries/Win64/UnrealEditor-Cmd.exe').is_file():
            raise RuntimeError('Install the matching Unreal Engine editor first.')
        engine_build = sha256((engine / 'Engine/Build/Build.version').read_bytes()).hexdigest()
        runtime = ROOT / 'runtime_unreal_stream'
        runtime.mkdir(exist_ok=True)
        if package is None:
            build_sender(runtime, engine, frozen_source, engine_build, args.keep_build_cache)
        else:
            print(f"Installing prebuilt sender for UE {package['engine']} (BuildId {package['build_id']})", flush=True)
            install_prebuilt(runtime, engine, frozen_source, engine_build, package, payload)
    print('Live Link support is ready. In Stream choose Unreal Live Link, then check people.', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        print('Live Link setup failed: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
