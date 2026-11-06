"""Fetch pinned upstream support archives and stage them without touching active files."""
from pathlib import Path, PurePosixPath
import hashlib, json, os, shutil, stat, urllib.request, zipfile

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def valid(path,spec):
    return path.is_file() and path.stat().st_size==spec['bytes'] and digest(path)==spec['sha256']

def download(cache,name,spec):
    from tools.pinned_download import download as fetch
    return fetch(cache,name,spec)

def safe_members(archive,root):
    total=0
    for count,item in enumerate(archive.infolist(),1):
        if count>100000:raise ValueError('Support archive has too many files')
        name=item.filename
        path=PurePosixPath(name)
        if '\\' in name or path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0]!=root or ':' in name:
            raise ValueError('Unsafe support archive member: '+name)
        if stat.S_ISLNK(item.external_attr >> 16):raise ValueError('Archive links are not allowed')
        total+=item.file_size
        if total>3*1024**3:raise ValueError('Support archive is too large')
        yield item,path

def extract(archive_path,root,destination,select):
    count=0
    with zipfile.ZipFile(archive_path) as archive:
        for item,path in safe_members(archive,root):
            if item.is_dir() or not select(path.parts[1:]):continue
            relative=Path(*path.parts[1:]); target=destination/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(item) as src,target.open('wb') as out:shutil.copyfileobj(src,out,1024*1024)
            count+=1
    if not count:raise ValueError('Support archive contained no expected files')

def stage(destination,cache,lock):
    cache.mkdir(parents=True,exist_ok=True)
    native=ROOT/lock['correction']['path']
    if not valid(native,lock['correction']):raise ValueError('Prebuilt motion correction module is missing or has the wrong hash')
    blender=download(cache,'blender-4.5.0',lock['blender'])
    ffmpeg=download(cache,'ffmpeg-7.1.1',lock['ffmpeg'])
    out=destination/'runtime_character'; extract(blender,lock['blender']['archive_root'],out,lambda parts:bool(parts))
    if not (out/'blender.exe').is_file():raise ValueError('Blender executable missing in archive')
    ffroot=destination/'ffmpeg-stage'; extract(ffmpeg,lock['ffmpeg']['archive_root'],ffroot,lambda p:bool(p) and (p[-1] in ('ffmpeg.exe','ffprobe.exe') or p[-1].lower() in ('license','license.txt','copying','readme.txt')))
    binout=destination/'tools/bin';binout.mkdir(parents=True)
    for name in ('ffmpeg.exe','ffprobe.exe'):
        matches=list(ffroot.rglob(name))
        if len(matches)!=1:raise ValueError('Expected one '+name+' in archive')
        shutil.copy2(matches[0],binout/name)
    notices=binout/'notices';notices.mkdir()
    for path in ffroot.rglob('*'):
        if path.is_file() and path.name not in ('ffmpeg.exe','ffprobe.exe'):
            shutil.copy2(path,notices/path.name)
    target=destination/'third_party/KimodoMotionCorrection/bin';target.mkdir(parents=True)
    shutil.copy2(native,target/native.name)
    print('Upstream support archives verified and staged.',flush=True)
