"""Install a pinned Blender build as an optional Linux support component."""
import json
import os
from pathlib import Path
import subprocess
import tarfile
from tempfile import TemporaryDirectory

from pinned_download import download


def verify(executable, version):
    result = subprocess.run([str(executable), '--version'], capture_output=True,
                            text=True, encoding='utf-8', errors='replace', timeout=30)
    if result.returncode or result.stdout.split()[:2] != ['Blender', version]:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError("Blender verification failed. Check the Linux guide's system libraries. " + detail)


def prepare(root):
    pin = json.loads((root / 'installer/linux-blender-lock.json').read_text(encoding='utf-8'))
    archive_root = pin['archive_root']
    if not archive_root or Path(archive_root).name != archive_root or archive_root in ('.', '..'):
        raise ValueError('Invalid Blender archive root')

    parent = root / 'runtime_character'
    if parent.is_symlink():
        raise RuntimeError('The character runtime path is a symbolic link.')
    destination = parent / 'blender-linux'
    receipt = '.rgbmocap-blender.json'
    if os.path.lexists(destination):
        if destination.is_symlink() or not (destination / receipt).is_file():
            raise RuntimeError('Existing unmanaged Blender was preserved. Use RGBMOCAP_BLENDER to select it.')
        if json.loads((destination / receipt).read_text(encoding='utf-8')) != pin:
            raise RuntimeError('A different managed Blender is installed. See docs/LINUX.md.')
        verify(destination / 'blender', pin['version'])
        print('Keeping managed Blender ' + pin['version'], flush=True)
        return

    artifact = download(root / '.setup/linux-blender-downloads', archive_root, pin,
                        filename=archive_root + '.tar.xz', allowed_hosts={'download.blender.org'})
    parent.mkdir(exist_ok=True)

    def archive_filter(member, staging):
        if member.name != archive_root and not member.name.startswith(archive_root + '/'):
            raise tarfile.FilterError('Blender archive member outside its release directory')
        return tarfile.data_filter(member, staging)

    with TemporaryDirectory(prefix='.blender-stage-', dir=parent) as temporary:
        with tarfile.open(artifact, mode='r:xz') as archive:
            archive.extractall(temporary, filter=archive_filter)
        build = Path(temporary) / archive_root
        verify(build / 'blender', pin['version'])
        (build / receipt).write_text(json.dumps(pin, indent=2), encoding='utf-8')
        if os.path.lexists(destination):
            raise RuntimeError('The Blender destination was created during setup and was preserved.')
        build.rename(destination)
    print('Installed Blender ' + pin['version'] + ' at ' + str(destination), flush=True)
