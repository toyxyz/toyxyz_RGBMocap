"""Dedicated character I/O runtime; never changes an AI environment."""
import os,shutil
from pathlib import Path
from mocap.paths import EXE,ROOT

def blender():
    candidates=[Path(os.environ.get('RGBMOCAP_BLENDER',''))/('blender'+EXE),ROOT/'runtime_character'/('blender'+EXE)]
    explicit=os.environ.get('RGBMOCAP_BLENDER')
    if explicit:candidates.insert(0,Path(explicit))
    found=shutil.which('blender')
    if found:candidates.append(Path(found))
    # Development discovery only; packaged applications use the managed runtime.
    if not getattr(__import__('sys'),'frozen',False):
        candidates+=sorted(Path('C:/BlenderLauncher/Blender/stable').glob('blender-4.5*/blender.exe'))
        candidates+=sorted(Path('C:/Program Files/Blender Foundation').glob('Blender 4.5/blender.exe'))
    for p in candidates:
        if p.is_file():return p.resolve()
    raise RuntimeError('Character runtime is missing. Install the character-enabled application package, or set RGBMOCAP_BLENDER to Blender 4.5 LTS blender.exe.')
