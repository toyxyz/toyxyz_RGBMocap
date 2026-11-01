"""Build a fresh app/Setup from this source tree, without a prior release."""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys
from runtime_support import ROOT,run,sha,application_version,cache_dependency_files
from stage_assets import stage
from build_setup import build as setup
from build_character_runtime import discover as discover_blender,stage as stage_blender

def source_resources():
    rows=json.loads((ROOT/'installer/source-resources.json').read_text(encoding='utf-8'))['files']
    allowed=set()
    for row in rows:
        path=(ROOT/row['path']).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('Missing or changed add-on material source: '+row['path'])
        allowed.add(path)
    return allowed

def copy_tree(source,dest):
    allowed=source_resources()
    excluded={'__pycache__','.git','checkpoints','ckpts','outputs','inputs','cache','.cache','build','dist','bin','tests','experiments'}
    forbidden={'.pt','.pth','.ckpt','.safetensors','.onnx','.npz','.npy','.pkl','.mp4','.avi','.fbx','.bvh','.blend','.exe','.dll','.pyd','.pyc','.pyo'}
    def ignore(folder,names):
        return [n for n in names if n.lower() in excluded or (Path(folder)/n).is_symlink() or (Path(n).suffix.lower() in forbidden and (Path(folder)/n).resolve() not in allowed) or n.lower() in {'agents.md','claude.md','gemini.md','codex.md'}]
    shutil.copytree(source,dest,ignore=ignore,dirs_exist_ok=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--work',type=Path,default=ROOT/'build/release')
    p.add_argument('--output',type=Path,default=ROOT/'releases/RGBMoCap-setup-windows-x64.exe')
    p.add_argument('--asset-root',type=Path,default=ROOT,help='Root containing acquired third_party auxiliary files')
    p.add_argument('--ffmpeg-dir',type=Path,help='Folder containing ffmpeg.exe and ffprobe.exe; defaults to PATH')
    p.add_argument('--blender-dir',type=Path,help='Complete Blender 4.5 LTS Windows folder, or its blender.exe')
    p.add_argument('--native-module',type=Path,help='An existing module compiled from this reviewed source for CPython 3.10 x64')
    p.add_argument('--native-deps',type=Path,help='Optional local pybind11/Eigen source cache for CMake')
    p.add_argument('--uv',type=Path)
    p.add_argument('--app-only',action='store_true')
    a=p.parse_args()
    if sys.platform!='win32' or sys.version_info[:2]!=(3,10):raise RuntimeError('Build using the shared Windows Python 3.10 environment.')
    cache_inputs=cache_dependency_files(ROOT)
    blender,blender_version=discover_blender(a.blender_dir)
    source_resources()
    work=a.work.resolve();work.mkdir(parents=True,exist_ok=True)
    version=application_version();version_tuple=tuple(map(int,version.split('.')))+(0,)
    version_file=work/'windows-version.txt'
    version_file.write_text("VSVersionInfo(ffi=FixedFileInfo(filevers=%r, prodvers=%r, mask=0x3f, flags=0, OS=0x40004, fileType=1, subtype=0, date=(0,0)), kids=[StringFileInfo([StringTable('040904B0', [StringStruct('FileVersion', '%s'), StringStruct('ProductVersion', '%s'), StringStruct('ProductName', 'RGB MoCap'), StringStruct('FileDescription', 'RGB MoCap'), StringStruct('OriginalFilename', 'RGBMoCap.exe')])]), VarFileInfo([VarStruct('Translation', [1033,1200])])])" % (version_tuple,version_tuple,version,version),encoding='utf8')
    # Always use fresh staging; never merge an old build or touch installed data.
    stage_root=work/('app-'+__import__('uuid').uuid4().hex[:10]);stage_root.mkdir()
    buildcmd=[sys.executable,'-m','PyInstaller','--noconfirm','--onedir','--windowed','--name','RGBMoCap',
        '--distpath',str(stage_root),'--workpath',str(work/'pyinstaller'),'--specpath',str(work),'--version-file',str(version_file)]
    for name in ['torch','torchvision','timm','lightning','pytorch_lightning','scipy','matplotlib']:
        buildcmd+=['--exclude-module',name]
    subprocess.run(buildcmd+['app.py'],cwd=ROOT,check=True)
    app=stage_root/'RGBMoCap'
    for name in ['worker_boot.py','scene_boot.py','README.md','LICENSE','THIRD_PARTY_NOTICES.md']:
        shutil.copy2(ROOT/name,app/name)
    for source in cache_inputs:shutil.copy2(source,app/source.name)
    for name in ['mocap','native_compat','assets','third_party','docs']:
        copy_tree(ROOT/name,app/name)
    # Auxiliary body-derived data is supplied separately, hash checked and never in Git.
    stage(a.asset_root,app)
    character=stage_blender(blender,app)
    character['version']=blender_version
    native=a.native_module
    if native is None:
        command=[sys.executable,str(ROOT/'tools/build_motion_correction.py'),'--build-dir',str(work/'native'),'--output-dir',str(work/'native-output')]
        if a.native_deps:command+=['--deps',str(a.native_deps.resolve())]
        subprocess.run(command,cwd=ROOT,check=True)
        choices=list((work/'native-output').glob('_motion_correction*.pyd'))
        if len(choices)!=1:raise RuntimeError('Expected one native correction module')
        native=choices[0]
    if not native.is_file() or 'cp310' not in native.name:raise ValueError('A CPython 3.10 Windows native module is required')
    target=app/'third_party/KimodoMotionCorrection/bin';target.mkdir(parents=True,exist_ok=True)
    shutil.copy2(native,target/native.name)
    video=app/'tools/bin';video.mkdir(parents=True,exist_ok=True)
    for name in ['ffmpeg','ffprobe']:
        source=(a.ffmpeg_dir/(name+'.exe')) if a.ffmpeg_dir else Path(shutil.which(name) or '')
        if not source.is_file():raise FileNotFoundError('Install FFmpeg and put ffmpeg/ffprobe on PATH, or use --ffmpeg-dir')
        shutil.copy2(source,video/(name+'.exe'))
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    source_commit=commit.stdout.strip() if commit.returncode==0 else None
    (app/'BUILD_INFO.json').write_text(json.dumps(dict(version=version,source_build=True,source_commit=source_commit,runtime_setup=True,models_bundled=False,character_runtime=character),indent=2), encoding='utf-8')
    manifest=[dict(path=f.relative_to(app).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in app.rglob('*') if f.is_file()]
    (work/'application-manifest.json').write_text(json.dumps(dict(app=str(app),files=manifest),indent=2), encoding='utf-8')
    print('APP_READY',app,flush=True)
    if not a.app_only:
        report=setup(app,a.output,work/'setup',a.uv)
        (work/'build-report.json').write_text(json.dumps(report,indent=2), encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
