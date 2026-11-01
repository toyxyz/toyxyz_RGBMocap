"""Transactional import/retarget jobs. A failed job never replaces a usable character."""
import json,os,shutil,subprocess,sys,time
from pathlib import Path
from mocap.storage import atomic_json
from .runtime import blender

def report(text):print('CHARACTER: '+text,flush=True)

def invoke(mode,out,*args):
    command=[str(blender()),'-b','--factory-startup','--disable-autoexec','--threads','3','--python-exit-code','1','--python',str(Path(__file__).with_name('blender_entry.py')),'--',mode,str(out),*map(str,args)]
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    with (out/'blender.log').open('a',encoding='utf8') as log:
        p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf8',errors='replace',env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        tail=[]
        for line in p.stdout:
            log.write(line);tail=(tail+[line])[-25:]
            if line.startswith('CHARACTER:'):print(line.strip(),flush=True)
        p.stdout.close()
        if p.wait():raise RuntimeError('Character processing failed:\n'+''.join(tail)[-2500:])

def main(args):
    # Packaged workers run with -I, which ignores PYTHONIOENCODING/PYTHONUTF8.
    # QProcess decodes these pipes as UTF-8, regardless of the Windows locale.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    request=Path(args[args.index('--request')+1]);job=json.loads(request.read_text(encoding='utf8'))
    out=Path(job['output']);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    if job['mode']=='adopt':
        rows=[]
        for c in job['characters']:
            report('Saving imported character '+c['name'])
            shutil.copytree(Path(job['source'])/c['asset'],out/c['id']/'asset')
            rows.append(dict(c,asset=job['prefix']+'/'+c['id']+'/asset'))
        result=dict(characters=rows)
    elif job['mode']=='import':
        invoke('import',out,job['file'])
        result=json.loads((out/'import.json').read_text(encoding='utf8'))
    elif job['mode']=='export':
        for i,(take,target) in enumerate(zip(job['takes'],job['files'])):
            report(f'Exporting character take {i+1}/{len(job["takes"])}')
            folder=Path(job['animation'])/take['folder']
            options=dict(format=job.get('format','fbx'),include_mesh=job.get('include_mesh',True))
            invoke('export',out,folder,target,json.dumps(take.get('transform')),job['source_name'],json.dumps(options))
        result=dict(files=job['files'])
    else:
        from .source import prepare,stamp
        from .pipeline import run
        asset=Path(job['asset']);kind=json.loads((asset/'import.json').read_text(encoding='utf-8'))['kind'];takes=[]
        for i,take in enumerate(job['takes']):
            folder=out/f'take-{i}';folder.mkdir()
            report(f'Take {i+1}/{len(job["takes"])}: preparing selected source')
            for name in [f'{kind}-rest.blend',f'{kind}-rig.json',f'{kind}-canonical.npz',f'{kind}-contact-geometry.npz','import.json']:
                shutil.copy2(asset/name,folder/name)
            receipt=prepare(Path(take['project']),job['result'],folder)
            report(f'Take {i+1}/{len(job["takes"])}: solving contact retargeting')
            diagnostic=run(folder,kind)
            report(f'Take {i+1}/{len(job["takes"])}: baking body bones')
            invoke('bake',folder,kind)
            if stamp(map(Path,receipt['source_stats']))!=receipt['source_stats']:
                raise RuntimeError('Source changed during retargeting. Apply your changes and retarget again.')
            atomic_json(folder/'receipt.json',dict(receipt,diagnostic=diagnostic))
            takes.append(dict(folder=f'take-{i}',start=take['start'],frames=receipt['frames'],fps=receipt['fps'],transform=take.get('transform'),source=take['project']))
        result=dict(takes=takes,source_name=job['source_name'],result=job['result'])
    result['seconds']=time.perf_counter()-start
    atomic_json(out/'complete.json',result)
    report('Complete')

if __name__=='__main__':main(sys.argv[1:])
