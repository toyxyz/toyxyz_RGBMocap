"""Dedicated character I/O runtime; never changes an AI environment."""
import os,shutil
from pathlib import Path
from mocap.paths import ROOT
from mocap.platform_compat import IS_WINDOWS,exe_name

def blender():
    name=exe_name('blender')
    candidates=[ROOT/'runtime_character'/name]
    explicit=os.environ.get('RGBMOCAP_BLENDER')
    # RGBMOCAP_BLENDER may name the executable itself or the folder that contains it.
    if explicit:candidates[:0]=[Path(explicit),Path(explicit)/name]
    found=shutil.which('blender')
    if found:candidates.append(Path(found))
    # Development discovery only; packaged applications use the managed runtime.
    if IS_WINDOWS and not getattr(__import__('sys'),'frozen',False):
        candidates+=sorted(Path('C:/BlenderLauncher/Blender/stable').glob('blender-4.5*/blender.exe'))
        candidates+=sorted(Path('C:/Program Files/Blender Foundation').glob('Blender 4.5/blender.exe'))
    for p in candidates:
        if p.is_file():return p.resolve()
    raise RuntimeError('Character runtime is missing. Install the character-enabled application package, or set RGBMOCAP_BLENDER to Blender 4.5 LTS ('+name+').')
