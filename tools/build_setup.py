"""Build Setup directly from fresh application files; no previous release needed."""
from pathlib import Path
import argparse,hashlib,json,shutil,struct,subprocess,zipfile
from runtime_support import ROOT,prepared_lock,find_uv,sha,application_version,cache_dependency_files

def build(app,output,work,uv=None):
    app=Path(app).resolve();output=Path(output).resolve();work=Path(work).resolve();work.mkdir(parents=True,exist_ok=True)
    if not (app/'RGBMoCap.exe').is_file():raise FileNotFoundError('Build the GUI application first')
    cache_dependency_files(app)
    version=application_version()
    if json.loads((app/'BUILD_INFO.json').read_text(encoding='utf-8'))['version']!=version:raise ValueError('Application and installer versions do not match')
    lock=prepared_lock(work/'resources');uv=find_uv(uv)
    if lock['version']!=version:raise ValueError('Release metadata and runtime lock versions do not match')
    if sha(uv)!=lock['uv_sha256']:raise ValueError('Pinned uv helper hash mismatch')
    manifest=[];payload=work/'payload.zip'
    with zipfile.ZipFile(payload,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for p in sorted(app.rglob('*')):
            if not p.is_file():continue
            if '__pycache__' in p.relative_to(app).parts or p.suffix.lower() in {'.pyc','.pyo'}:continue
            relative=p.relative_to(app).as_posix()
            if relative.split('/')[0] in {'models','projects','runtime','runtime_flow','runtime_hand','runtime_htd','runtime_exmr'}:raise ValueError('User/runtime data in application payload')
            manifest.append(dict(path=relative,bytes=p.stat().st_size,sha256=sha(p)));z.write(p,'app/'+relative)
        z.writestr('app-manifest.json',json.dumps(dict(files=manifest),indent=2))
        for name in ['engine.py','verify_runtime.py']:
            z.write(ROOT/'installer'/name,name)
        z.write(ROOT/'mocap/runtime_registry.py','runtime_registry.py');z.write(uv,'uv.exe')
        z.write(work/'resources/runtime-lock.json','runtime-lock.json')
        for name in sorted({p['bundled'] for g in lock['groups'] for p in g['packages'] if p.get('bundled')}):z.write(work/'resources'/name,name)
        for p in (ROOT/'installer/licenses').iterdir():
            if p.is_file():z.write(p,'licenses/'+p.name)
    summary={k:lock[k] for k in ['version','download_bytes','installed_bytes_estimate','fresh_free_bytes_required','id']}
    character_bytes=sum(row['bytes'] for row in manifest if row['path'].startswith('runtime_character/'))
    summary['installed_bytes_estimate']+=character_bytes
    summary['fresh_free_bytes_required']+=character_bytes*2
    (work/'setup.json').write_text(json.dumps(summary), encoding='utf-8')
    compiler=Path('C:/Windows/Microsoft.NET/Framework64/v4.0.30319/csc.exe')
    if not compiler.is_file():raise FileNotFoundError('The Windows .NET Framework C# compiler is required')
    stub=work/'Setup.stub.exe'
    assembly=work/'Setup.Version.cs'
    assembly.write_text('[assembly:System.Reflection.AssemblyVersion("'+version+'.0")]\n[assembly:System.Reflection.AssemblyFileVersion("'+version+'.0")]\n[assembly:System.Reflection.AssemblyInformationalVersion("'+version+'") ]\n[assembly:System.Reflection.AssemblyProduct("RGB MoCap")]\n[assembly:System.Reflection.AssemblyTitle("RGB MoCap Setup")]\n',encoding='utf8')
    subprocess.run([str(compiler),'/nologo','/target:winexe','/platform:x64','/optimize+','/codepage:65001','/out:'+str(stub),
        '/win32manifest:'+str(ROOT/'installer/Setup.manifest'),'/resource:'+str(work/'setup.json')+',setup.json',
        '/r:System.Windows.Forms.dll','/r:System.Drawing.dll','/r:System.Web.Extensions.dll','/r:System.IO.Compression.dll',
        '/r:System.IO.Compression.FileSystem.dll',str(ROOT/'installer/Setup.cs'),str(ROOT/'installer/FolderPicker.cs'),str(assembly)],check=True)
    output.parent.mkdir(parents=True,exist_ok=True);temp=output.with_suffix('.exe.partial')
    with temp.open('wb') as f:
        for p in [stub,payload]:
            with p.open('rb') as s:shutil.copyfileobj(s,f,4*1024**2)
        f.write(struct.pack('<Q',payload.stat().st_size));f.write(bytes.fromhex(sha(payload)));f.write(b'RGBMOCAPSETUPV1!')
    temp.replace(output);output.with_suffix('.exe.sha256').write_text(sha(output)+'  '+output.name+'\n',encoding='ascii')
    return dict(setup=str(output),bytes=output.stat().st_size,sha256=sha(output),runtime_lock=lock['id'],files=len(manifest),character_runtime_bytes=character_bytes,installed_bytes_estimate=summary['installed_bytes_estimate'],free_bytes_required=summary['fresh_free_bytes_required'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--app',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'releases/RGBMoCap-setup-windows-x64.exe');p.add_argument('--work',type=Path,default=ROOT/'build/setup');p.add_argument('--uv',type=Path)
    a=p.parse_args();print(json.dumps(build(a.app,a.output,a.work,a.uv),indent=2))

if __name__=='__main__':main()
