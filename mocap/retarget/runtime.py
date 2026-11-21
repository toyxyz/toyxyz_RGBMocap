"""Dedicated character I/O runtime; never changes an AI environment."""
import os,shutil
from pathlib import Path
from mocap.paths import EXE,ROOT

def blender():
    name = 'blender' + EXE
    candidates = []
    explicit = os.environ.get('RGBMOCAP_BLENDER')
    if explicit:
        override = Path(explicit)
        candidates.extend((override, override / name))
    if not EXE:
        candidates.append(ROOT / 'runtime_character/blender-linux/blender')
    candidates.append(ROOT / 'runtime_character' / name)
    found=shutil.which('blender')
    if found:candidates.append(Path(found))
    # Development discovery only; packaged applications use the managed runtime.
    if EXE and not getattr(__import__('sys'),'frozen',False):
        candidates+=sorted(Path('C:/BlenderLauncher/Blender/stable').glob('blender-4.5*/blender.exe'))
        candidates+=sorted(Path('C:/Program Files/Blender Foundation').glob('Blender 4.5/blender.exe'))
    for p in candidates:
        if p.is_file():return p.resolve()
    if EXE == '':
        raise RuntimeError('Blender is missing. Run ./install.sh --with-blender or set RGBMOCAP_BLENDER to Blender 4.5 LTS.')
    raise RuntimeError('Character runtime is missing. Install the character-enabled application package, or set RGBMOCAP_BLENDER to Blender 4.5 LTS blender.exe.')
