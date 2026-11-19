"""Build optional Live Link support against the user's installed Unreal Engine."""
from pathlib import Path
import argparse
from hashlib import sha256
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mocap.unreal_stream import source_identity, source_tree_identity, UnrealStream


def find_engine(explicit=None):
    candidates = [Path(explicit)] if explicit else []
    if not explicit:
        base = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Epic Games'
        candidates = sorted(base.glob('UE_5.*'), reverse=True)
    for path in candidates:
        version = path / 'Engine/Build/Build.version'
        if version.is_file():
            data = json.loads(version.read_text(encoding='utf-8'))
            if (data['MajorVersion'], data['MinorVersion']) >= (5, 7):
                return path.resolve()
    raise RuntimeError('Unreal Engine 5.7+ was not found. Use --engine "D:/Epic Games/UE_5.7". Only 5.7.4 has been tested.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path)
    parser.add_argument('--keep-build-cache', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'win32':
        raise RuntimeError('This optional bridge currently supports Windows only.')
    engine = find_engine(args.engine)
    executable = engine / 'Engine/Binaries/Win64/UnrealEditor-Cmd.exe'
    ubt = engine / 'Engine/Binaries/DotNET/UnrealBuildTool/UnrealBuildTool.dll'
    hosts = list((engine / 'Engine/Binaries/ThirdParty/DotNet').glob('*/win-x64/dotnet.exe'))
    if not executable.is_file() or not ubt.is_file() or not hosts:
        raise RuntimeError('Install the Unreal Engine editor and C++ build support first.')
    runtime = ROOT / 'runtime_unreal_stream'
    runtime.mkdir(exist_ok=True)
    receipt = runtime / 'bridge.json'
    # Failed builds must never leave a previously verified receipt advertising new source.
    frozen_source = source_identity(ROOT)
    project_dir = runtime / ('Bridge-' + frozen_source[:12])
    shutil.copytree(ROOT / 'developer-build/unreal-live-link', project_dir, dirs_exist_ok=True)
    project = project_dir / 'RGBMoCapLiveLink.uproject'
    if source_tree_identity(project_dir) != frozen_source or source_identity(ROOT) != frozen_source:
        raise RuntimeError('Live Link source changed during setup. Retry when editing is finished.')
    print('Building optional Live Link sender using ' + str(engine), flush=True)
    print('Requires Visual Studio 2022 C++ tools and Windows SDK. No models are downloaded.', flush=True)
    # Call the native .NET host directly: no CMD reparsing of international paths.
    subprocess.run([str(sorted(hosts)[-1]), str(ubt), 'RGBMoCapLiveLinkEditor', 'Win64', 'Development',
                    '-Project='+str(project), '-WaitMutex', '-NoHotReloadFromIDE'],
                   cwd=project_dir, check=True)
    module = project_dir / 'Binaries/Win64/UnrealEditor-RGBMoCapLiveLink.dll'
    if not module.is_file():
        raise RuntimeError('Build finished without the Live Link sender module.')
    if source_identity(ROOT) != frozen_source or source_tree_identity(project_dir) != frozen_source:
        raise RuntimeError('Live Link source changed during the build. Run setup again.')
    data = dict(version=1, engine=str(engine), source=frozen_source,
                project=project.relative_to(runtime).as_posix(),
                module=sha256(module.read_bytes()).hexdigest(),
                engine_build=sha256((engine/'Engine/Build/Build.version').read_bytes()).hexdigest())
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
    if not args.keep_build_cache:
        intermediate = (project_dir / 'Intermediate').resolve()
        if not intermediate.is_relative_to(runtime.resolve()):
            raise RuntimeError('Build cache is outside the optional runtime.')
        if intermediate.is_dir():
            shutil.rmtree(intermediate)
    print('Live Link support built. In Stream choose Unreal Live Link, then check people.', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print('Live Link setup failed: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
