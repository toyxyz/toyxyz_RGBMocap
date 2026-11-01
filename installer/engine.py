"""Standard-library installer engine. Runs only from Setup's private Python.

Every path is scoped to the Setup folder. Runtime installation is staged and
verified before commit; models, projects and application caches are never moved.
"""
from __future__ import annotations
import argparse, ctypes, hashlib, importlib.metadata, json, msvcrt, os
from pathlib import Path
import re, shutil, stat, subprocess, sys, time, urllib.error, urllib.parse, urllib.request, zipfile

PREFIX='@@SETUP@@'
TARGETS={'shared':'runtime','gui':'runtime_gui','flow':'runtime_flow','hand':'runtime_hand','htd':'runtime_htd','exmr':'runtime_exmr'}
APP_TOP={'RGBMoCap.exe','_internal','assets','mocap','native_compat','third_party','tools','docs','runtime_character',
         'worker_boot.py','scene_boot.py','README.md','LICENSE','THIRD_PARTY_NOTICES.md',
         'requirements-windows.txt','requirements-flow-windows.txt','requirements-hand-windows.txt','BUILD_INFO.json'}

class Cancelled(Exception):pass

def emit(kind='progress',**data):
    print(PREFIX+json.dumps(dict(type=kind,**data),ensure_ascii=False),flush=True)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def atomic_json(path,data):
    path=Path(path);p=path.with_name(path.name+'.partial');p.write_text(json.dumps(data,indent=2),encoding='utf8');os.replace(p,path)

def inside(root,path):
    root=Path(root).resolve();p=Path(path).resolve()
    if p==root or not p.is_relative_to(root):raise ValueError('Path leaves the installer folder: '+str(path))
    return p

def remove_owned(setup,path):
    p=inside(setup,path)
    if p.exists():
        if p.is_symlink() or p.stat().st_file_attributes&stat.FILE_ATTRIBUTE_REPARSE_POINT:raise ValueError('Unexpected installer link')
        shutil.rmtree(p)

def clean_env(root):
    env={k:v for k,v in os.environ.items() if not k.upper().startswith(('PYTHON','PIP_','UV_'))}
    temp=inside(root,root/'.setup/tmp');temp.mkdir(parents=True,exist_ok=True)
    env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',
        UV_PYTHON_INSTALL_DIR=str(root/'.setup/python'),UV_CACHE_DIR=str(root/'.setup/uv-cache'),
        UV_NO_CONFIG='1',UV_NO_PROGRESS='1',UV_PYTHON_NO_REGISTRY='1',
        UV_PYTHON_PREFERENCE='only-managed',TEMP=str(temp),TMP=str(temp),
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',RGB_MOCAP_DATA=str(root))
    return env

def check_cancel(cancel):
    if cancel and Path(cancel).exists():raise Cancelled('Installation cancelled. Verified downloads are kept for retry.')

def run(command,root,cancel,log):
    check_cancel(cancel)
    with Path(log).open('w',encoding='utf8') as output:
        p=subprocess.Popen([str(x) for x in command],cwd=root,env=clean_env(root),stdout=output,stderr=subprocess.STDOUT,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            while p.poll() is None:
                check_cancel(cancel);time.sleep(.2)
        except BaseException:
            subprocess.run([str(Path(os.environ['SystemRoot'])/'System32/taskkill.exe'),'/PID',str(p.pid),'/T','/F'],
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
            p.wait(timeout=30);raise
    if p.returncode:
        tail=Path(log).read_text(encoding='utf8',errors='replace')[-3500:]
        raise RuntimeError('A setup step failed. '+tail)

def validate_package(p):
    if not re.fullmatch('[0-9a-f]{64}',p['sha256']):raise ValueError('Invalid package digest')
    if Path(p['filename']).name!=p['filename'] or not p['filename'].endswith('.whl'):raise ValueError('Invalid wheel filename')
    if p.get('url'):
        u=urllib.parse.urlsplit(p['url'])
        if u.scheme!='https' or u.hostname not in {'files.pythonhosted.org','download.pytorch.org','download-r2.pytorch.org'}:
            raise ValueError('Unapproved package source')

def download(p,cache,cancel,completed,total):
    validate_package(p);folder=inside(cache,cache/p['sha256']);folder.mkdir(parents=True,exist_ok=True)
    dest=folder/p['filename'];partial=dest.with_name(dest.name+'.partial')
    if dest.is_file() and dest.stat().st_size==p['bytes'] and sha(dest)==p['sha256']:
        emit(message='Using verified download: '+p['name'],phase='download',current=completed+p['bytes'],total=total);return dest
    for attempt in range(3):
        check_cancel(cancel);start=partial.stat().st_size if partial.exists() else 0
        headers={'User-Agent':'RGBMoCap-Setup/1'}
        if start:headers['Range']=f'bytes={start}-'
        try:
            request=urllib.request.Request(p['url'],headers=headers)
            with urllib.request.urlopen(request,timeout=25) as response:
                resume=response.status==206 and response.headers.get('Content-Range','').startswith(f'bytes {start}-')
                if not resume:start=0
                count=start;last=0.;began=time.monotonic()
                with partial.open('ab' if resume else 'wb') as f:
                    while True:
                        check_cancel(cancel);chunk=response.read(1024**2)
                        if not chunk:break
                        f.write(chunk);count+=len(chunk)
                        if time.monotonic()-last>.2:
                            elapsed=time.monotonic()-began;extra={}
                            if elapsed>=2 and count>start:
                                extra['remaining_seconds']=max(0,(total-completed-count)*elapsed/(count-start))
                            emit(message='Downloading '+p['name'],phase='download',current=min(total,completed+count),total=total,**extra);last=time.monotonic()
            if partial.stat().st_size!=p['bytes'] or sha(partial)!=p['sha256']:
                partial.unlink(missing_ok=True);raise ValueError('Download verification failed: '+p['name'])
            os.replace(partial,dest);return dest
        except (OSError,ValueError,urllib.error.HTTPError) as e:
            if attempt==2:raise
            if isinstance(e,urllib.error.HTTPError) and e.code==416:partial.unlink(missing_ok=True)
            emit(message='Retrying '+p['name'],phase='download',current=completed,total=total)

def python_home(root,version):
    p=inside(root,root/'.setup/python'/f'cpython-{version}-windows-x86_64-none')
    if not (p/'python.exe').is_file():raise RuntimeError('The private Python download is incomplete: '+version)
    return p

def verify_group(stage,group,root,cancel,logs):
    py=stage/'runtime/python.exe' if group['key']!='exmr' else stage/'runtime_exmr/python.exe'
    if not py.is_file() and group.get('base'):
        py=root/'runtime/python.exe'
    report=logs/(group['key']+'-verify.log')
    scripts=Path(__file__).resolve().parent
    request=logs/(group['key']+'-verify-input.json');atomic_json(request,group)
    run([py,'-I','-X','utf8',scripts/'verify_runtime.py',stage,request,scripts/'runtime_registry.py',root],root,cancel,report)
    lines=report.read_text(encoding='utf8').strip().splitlines();return json.loads(lines[-1])


def group_ids(lock):
    result={}
    for g in lock.get('groups',[]):
        base=g.get('base')
        value=dict(group=g,base_id=result.get(base))
        result[g['key']]=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    return result


def reusable_groups(root,lock,cancel,logs,checked=None,failed=None):
    try: old=json.loads((root/'SETUP_RECEIPT.json').read_text(encoding='utf8'))
    except (OSError,ValueError): return {}
    ids=group_ids(lock);result={}
    for g in lock['groups']:
        if old.get('group_ids',{}).get(g['key'])!=ids[g['key']]: continue
        # An overlay can only be reused with its validated base.
        if g.get('base') and g['base'] not in result: continue
        if g['key'] in (failed or set()): continue
        try:
            result[g['key']]=(checked or {}).get(g['key']) or verify_group(root,g,root,cancel,logs)
            emit(message='Keeping verified '+g['key']+' runtime',phase='verify')
        except Cancelled: raise
        except Exception as exc: emit(message='Repair required for '+g['key']+': '+str(exc)[-240:],phase='verify')
    return result

def prune_runtime(folder):
    # The same non-runtime exclusions used in the validated portable release.
    for current,dirs,files in os.walk(folder):
        for name in files:
            p=Path(current)/name
            if p.suffix.lower() in {'.lib','.exp','.pdb','.obj','.o','.a','.ilk','.pyc','.pyo'}:p.unlink()

def install_runtimes(root,resources,lock,cancel,reused=None):
    setup=inside(root,root/'.setup');stage=inside(setup,setup/'staging'/lock['id']);stage.mkdir(parents=True,exist_ok=True)
    logs=setup/'logs';logs.mkdir(exist_ok=True);cache=setup/'downloads';cache.mkdir(exist_ok=True);uv=resources/'uv.exe'
    if sha(uv)!=lock['uv_sha256']:raise ValueError('Setup helper verification failed')
    reused=reused or {}; pending=[g for g in lock['groups'] if g['key'] not in reused]
    for version in sorted({g['python'] for g in pending}):
        emit(message='Preparing private Python '+version,phase='python',current=0,total=0)
        run([uv,'python','install',version,'--install-dir',setup/'python','--no-bin','--no-registry','--no-config','--no-progress','--cache-dir',setup/'uv-cache'],root,cancel,logs/('python-'+version+'.log'))
    unique={p['sha256']:p for g in pending for p in g['packages']};local={};total=sum(p['bytes'] for p in unique.values() if p.get('url'));completed=0
    for digest,p in unique.items():
        if p.get('bundled'):
            wheel=inside(resources,resources/p['bundled'])
            if sha(wheel)!=digest:raise ValueError('Bundled compatibility file is damaged')
            local[digest]=wheel
        else:
            local[digest]=download(p,cache,cancel,completed,total);completed+=p['bytes']
    reports=dict(reused)
    for index,g in enumerate(lock['groups']):
        check_cancel(cancel);key=g['key']
        if TARGETS.get(key)!=g['target']:raise ValueError('Invalid runtime target')
        if key in reused: continue
        dest=inside(stage,stage/g['target']);receipt=stage/('complete-'+key+'.json')
        if receipt.is_file():
            try:
                reports[key]=verify_group(stage,g,root,cancel,logs);continue
            except Cancelled:raise
            except Exception:receipt.unlink()
        if dest.exists():remove_owned(stage,dest)
        emit(message='Installing '+('the common runtime' if key=='shared' else key+' runtime')+f' ({index+1}/{len(lock["groups"])})',phase='install',current=0,total=0)
        if key in ('shared','exmr'):
            shutil.copytree(python_home(root,g['python']),dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            py=dest/'python.exe';target=dest/'Lib/site-packages';target.mkdir(parents=True,exist_ok=True)
        else:
            py=stage/'runtime/python.exe'
            if not py.is_file(): py=root/'runtime/python.exe'
            target=dest;target.mkdir()
        req=stage/(key+'-wheels.txt')
        req.write_text('\n'.join(local[p['sha256']].as_uri()+' --hash=sha256:'+p['sha256'] for p in g['packages'])+'\n',encoding='utf8')
        run([uv,'pip','install','--python',py,'--target',target,'--no-deps','--offline','--no-index',
             '--no-config','--require-hashes','--only-binary',':all:','--link-mode','copy','--no-cache','-r',req],root,cancel,logs/(key+'-install.log'))
        prune_runtime(dest);reports[key]=verify_group(stage,g,root,cancel,logs);atomic_json(receipt,reports[key])
        emit(message=key+' runtime verified',phase='install',current=index+1,total=len(lock['groups']))
    return stage,reports

def commit(root,resources,stage,lock,reports,cancel):
    check_cancel(cancel);app=resources/'app';manifest=json.loads((resources/'app-manifest.json').read_text(encoding='utf8'))
    for e in manifest['files']:
        rel=Path(e['path'])
        if rel.parts[0] not in APP_TOP or rel.is_absolute() or '..' in rel.parts:raise ValueError('Invalid application payload path')
        if sha(inside(app,app/rel))!=e['sha256']:raise ValueError('Application payload verification failed')
    if not (root/'BUILD_INFO.json').is_file() and not (root/'SETUP_RECEIPT.json').is_file():
        conflicts=[p for p in ('mocap','_internal','third_party',*TARGETS.values(),'RGBMoCap.exe') if (root/p).exists()]
        if conflicts:raise RuntimeError('This folder contains unrelated application files. Move Setup to a dedicated folder.')
    emit(message='Finishing installation. Please keep Setup open.',phase='commit',current=0,total=0,committing=True)
    backup=inside(root,root/'.setup/backups'/lock['id']);backup.mkdir(parents=True,exist_ok=True);moved=[];files=[]
    try:
        for name in TARGETS.values():
            if not (stage/name).exists(): continue
            target=inside(root,root/name);old=inside(backup,backup/name)
            if target.exists():
                if old.exists():raise RuntimeError('A previous recovery is pending. Retry Setup.')
                os.replace(target,old)
            moved.append((target,old));os.replace(stage/name,target)
        for e in manifest['files']:
            rel=Path(e['path']);src=app/rel;dst=inside(root,root/rel);saved=inside(backup,backup/'app'/rel)
            if dst.is_file() and sha(dst)==e['sha256']:continue
            dst.parent.mkdir(parents=True,exist_ok=True);saved.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists():os.replace(dst,saved)
            files.append((dst,saved));shutil.copy2(src,dst)
        # Verify the committed layout with the real native correction backend.
        py=root/'runtime/python.exe';log=root/'.setup/logs/native-verify.log'
        code="import sys;sys.path.insert(0,sys.argv[1]);from mocap.correction import backend;assert backend().rgb_mocap_anatomical_version==1;from mocap.flow_sam import source_identity;source_identity(True);print('Native correction and application sources verified')"
        run([py,'-B','-c',code,str(root)],root,None,log)
        for name in ('models','projects'):(root/name).mkdir(exist_ok=True)
        atomic_json(root/'SETUP_RECEIPT.json',dict(schema=1,id=lock['id'],version=lock['version'],runtimes=reports,
            group_ids=group_ids(lock),installed_at=time.time(),runtime_sources='Pinned public PyPI/PyTorch wheels, private managed Python',models_included=False))
    except BaseException:
        for dst,saved in reversed(files):
            if dst.exists():dst.unlink()
            if saved.exists():os.replace(saved,dst)
        for target,old in reversed(moved):
            if target.exists():os.replace(target,stage/target.name)
            if old.exists():os.replace(old,target)
        raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--resources',type=Path,required=True);p.add_argument('--cancel',type=Path);args=p.parse_args()
    root=args.root.resolve();resources=args.resources.resolve();setup=root/'.setup'
    if setup.exists() and (setup.is_symlink() or setup.stat().st_file_attributes&stat.FILE_ATTRIBUTE_REPARSE_POINT):raise ValueError('Installer folder must not be a link')
    setup.mkdir(exist_ok=True);inside(setup,resources);lock=json.loads((resources/'runtime-lock.json').read_text(encoding='utf8'))
    # A retry can reuse or clear its own existing temporary files. Existing
    # application runtimes are renamed on commit, not copied a second time.
    reusable=0
    for folder in (setup/'downloads',setup/'staging'/lock['id'],setup/'python',setup/'tmp'):
        if folder.exists():
            inside(setup,folder)
            reusable+=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file() and not p.is_symlink())
    required=max(2*1024**3,lock['fresh_free_bytes_required']-reusable)
    with (setup/'install.lock').open('a+b') as handle:
        handle.seek(0);handle.write(b'0');handle.flush();handle.seek(0)
        msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        try:
            checked={};failed=set()
            existing=root/'SETUP_RECEIPT.json'
            if existing.is_file() and json.loads(existing.read_text(encoding='utf8')).get('id')==lock['id']:
                try:
                    emit(message='Checking the existing installation...',phase='verify',current=0,total=0)
                    logs=setup/'logs';logs.mkdir(exist_ok=True)
                    for g in lock['groups']:
                        try: checked[g['key']]=verify_group(root,g,root,args.cancel,logs)
                        except Cancelled: raise
                        except Exception:
                            failed.add(g['key']);raise
                    app=json.loads((resources/'app-manifest.json').read_text(encoding='utf8'))
                    if not all((root/e['path']).is_file() and sha(root/e['path'])==e['sha256'] for e in app['files']):raise ValueError('App repair required')
                    emit('complete',message='Existing application and all runtimes verified.');return
                except Cancelled:raise
                except Exception:emit(message='Repairing the installation...',phase='repair',current=0,total=0)
            if shutil.disk_usage(root).free<required:raise OSError('Not enough free space for a safe installation or repair.')
            logs=setup/'logs';logs.mkdir(exist_ok=True)
            reused=reusable_groups(root,lock,args.cancel,logs,checked,failed)
            stage,reports=install_runtimes(root,resources,lock,args.cancel,reused);commit(root,resources,stage,lock,reports,args.cancel)
        finally:handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
    emit('complete',message='All runtimes are installed and verified.',receipt=str(root/'SETUP_RECEIPT.json'))

if __name__=='__main__':
    try:main()
    except Cancelled as e:emit('cancelled',message=str(e));sys.exit(2)
    except Exception as e:emit('error',message=str(e));sys.exit(1)
