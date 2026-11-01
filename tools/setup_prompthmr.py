"""Install an immutable, separate PromptHMR environment from verified sources."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from functools import lru_cache
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from installer import engine
from installer.prompthmr_adapters import adapt_engine,install_python_source
from tools.setup_mamma import compiler,fetch_package,extract_source,quoted
from tools.support_components import download
from tools.session_lock import app_lock
from mocap.storage import atomic_json


@lru_cache(maxsize=4)
def identity(root=ROOT):
    names=('installer/prompthmr-runtime-lock.json','installer/prompthmr_adapters.py','installer/verify_prompthmr.py')
    return hashlib.sha256(json.dumps({n:engine.sha(Path(root)/n) for n in names},sort_keys=True).encode()).hexdigest()


def run(command,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf8') as out:
        env=engine.clean_env(ROOT);cache=ROOT/'cache/prompthmr-setup'
        env.update(YOLO_CONFIG_DIR=str(cache/'yolo'),MPLCONFIGDIR=str(cache/'mpl'),HF_HOME=str(cache/'hf'),TORCH_HOME=str(cache/'torch'),HF_HUB_OFFLINE='1')
        p=subprocess.Popen(list(map(str,command)),cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,creationflags=0x08000000)
        started=notice=time.monotonic()
        try:
            while p.poll() is None:
                time.sleep(.2)
                if time.monotonic()-notice>=15:
                    notice=time.monotonic();print('PromptHMR setup: '+log.stem+' is working ('+str(round(notice-started))+' s).',flush=True)
        except BaseException:
            subprocess.run([str(Path(os.environ['SystemRoot'])/'System32/taskkill.exe'),'/PID',str(p.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=0x08000000)
            p.wait(timeout=30);raise
    if p.returncode:raise RuntimeError('PromptHMR setup failed: '+log.read_text(encoding='utf8',errors='replace')[-4000:])


def verify(folder):
    report=folder/'verification.json'
    run([folder/'python.exe','-I','-X','utf8',ROOT/'installer/verify_prompthmr.py','--lock',ROOT/'installer/prompthmr-runtime-lock.json','--output',report],ROOT/'.setup/logs/prompthmr-verify.log')
    return json.loads(report.read_text(encoding='utf8'))


def setup():
    lock=json.loads((ROOT/'installer/prompthmr-runtime-lock.json').read_text(encoding='utf-8'))
    uv=ROOT/'.setup/bootstrap-git/uv.exe'
    if not uv.is_file() or engine.sha(uv)!=lock['uv_sha256']:raise RuntimeError('Run install.cmd first.')
    with app_lock(ROOT):
        tag=identity();config=ROOT/'.setup/prompthmr-runtime.json'
        if config.is_file():
            try:
                old=json.loads(config.read_text(encoding='utf8'));folder=Path(old['python']).parent
                if old.get('managed_id')==tag and folder.resolve().is_relative_to((ROOT/'runtime_prompthmr').resolve()):
                    report=verify(folder)
                    from mocap.prompthmr_runtime import configure
                    configure(folder,tag)
                    print('Verified PromptHMR runtime retained. Models and projects were not changed.',flush=True)
                    return report
            except Exception as e:print('Existing PromptHMR generation needs repair: '+str(e),flush=True)
        cuda,vc=compiler()
        if shutil.disk_usage(ROOT).free<18*1024**3:raise OSError('Allow approximately 18 GiB free for PromptHMR setup, excluding model files and compiler tools.')
        setup=ROOT/'.setup';cache=setup/'prompthmr-downloads';cache.mkdir(exist_ok=True)
        generation=tag[:20]+'-'+os.urandom(4).hex();stage=setup/'prompthmr-staging'/generation;stage.mkdir(parents=True)
        promoted=None;committed=False
        try:
            logs=setup/'logs';logs.mkdir(exist_ok=True)
            print('PromptHMR: separate pinned environment; %.2f GiB package downloads. No SDF or MAMMA engine required.'%(sum(p['bytes'] for p in lock['packages'])/2**30),flush=True)
            run([uv,'python','install',lock['python'],'--install-dir',setup/'python','--no-bin','--no-registry','--no-config'],logs/'prompthmr-python.log')
            shutil.copytree(engine.python_home(ROOT,lock['python']),stage,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            py=stage/'python.exe';target=stage/'Lib/site-packages'
            # Cache identical downloaded wheels once; never share writable installed
            # packages with MAMMA or the other model profiles.
            for row in lock['packages']:
                previous=setup/'mamma-downloads'/row['sha256']/row['filename']
                local=cache/row['sha256']/row['filename']
                if not local.exists() and previous.is_file() and previous.stat().st_size==row['bytes'] and engine.sha(previous)==row['sha256']:
                    local.parent.mkdir(exist_ok=True)
                    try:os.link(previous,local)
                    except OSError:shutil.copy2(previous,local)
            local={p['name']:fetch_package(p,cache) for p in lock['packages']}
            req=stage/'locked-wheels.txt'
            req.write_text('\n'.join(local[p['name']].as_uri()+' --hash=sha256:'+p['sha256'] for p in lock['packages'] if p['kind']=='wheel')+'\n',encoding='utf8')
            run([uv,'pip','install','--python',py,'--target',target,'--no-deps','--offline','--no-index','--require-hashes','--only-binary',':all:','--link-mode','copy','--no-cache','-r',req],logs/'prompthmr-packages.log')
            for p in lock['packages']:
                if p['kind']=='source':
                    run([uv,'pip','install','--python',py,'--target',target,'--no-deps','--no-build-isolation','--offline','--no-index','--link-mode','copy',local[p['name']]],logs/('prompthmr-'+p['name']+'.log'))
            sources=stage/'sources';sources.mkdir()
            for name,spec in lock['sources'].items():
                archive=download(cache,name+'-'+spec['commit'],spec)
                extract_source(archive,spec,sources/name)
            adapt_engine(sources/'prompthmr');os.replace(sources/'prompthmr',stage/'engine')
            install_python_source(sources/'detectron2',target,'detectron2')
            run([uv,'pip','install','--python',py,'--target',target,'--no-deps','--no-build-isolation','--no-index','--offline','--link-mode','copy',sources/'chumpy'],logs/'prompthmr-chumpy.log')
            cap=subprocess.run([str(py),'-I','-X','utf8','-c',"import torch;assert torch.cuda.is_available();print('.'.join(map(str,torch.cuda.get_device_capability())))"],capture_output=True,text=True,encoding='utf-8',errors='replace',check=True,creationflags=0x08000000).stdout.strip()
            if not __import__('re').fullmatch(r'\d+\.\d+',cap):raise ValueError('Invalid CUDA architecture.')
            build=stage/'build-native.cmd'
            lines=['@echo off','setlocal DisableDelayedExpansion','chcp 65001 >nul','call '+quoted(vc),'if errorlevel 1 exit /b 1',
                'set "CUDA_HOME='+str(cuda).replace('%','%%')+'"','set "TORCH_CUDA_ARCH_LIST='+cap+'"',
                'set "MAX_JOBS=2"','set "DISTUTILS_USE_SDK=1"','set "MSSdk=1"','set "FORCE_CUDA=1"','set "SAM2_BUILD_ALLOW_ERRORS=0"','set "SAM2_BUILD_CUDA=1"',
                ' '.join(quoted(x) for x in [uv,'pip','install','--python',py,'--target',target,'--no-deps','--no-build-isolation','--no-index','--offline','--no-cache','--link-mode','copy',sources/'sam2']),'if errorlevel 1 exit /b 1','exit /b 0']
            build.write_text('\n'.join(lines)+'\n',encoding='utf8')
            run([os.environ.get('COMSPEC','cmd.exe'),'/d','/c',build],logs/'prompthmr-native.log')
            engine.prune_runtime(stage);verify(stage)
            dest=ROOT/'runtime_prompthmr'/generation;dest.parent.mkdir(exist_ok=True);os.replace(stage,dest);promoted=dest
            report=verify(dest)
            identity.cache_clear()
            if identity()!=tag:raise RuntimeError('PromptHMR runtime sources changed during setup. Retry setup; the previous runtime was preserved.')
            report.update(installed_bytes=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),runtime_id=tag)
            atomic_json(dest/'ready.json',report)
            from mocap.prompthmr_runtime import configure
            configure(dest,tag);committed=True
            atomic_json(setup/'prompthmr-install-report.json',report)
            print('PromptHMR ready: %.2f GiB installed. Prepare weights in Model Manager.'%(report['installed_bytes']/2**30),flush=True)
            return report
        finally:
            if stage.exists():engine.remove_owned(setup,stage)
            if promoted is not None and not committed:engine.remove_owned(ROOT/'runtime_prompthmr',promoted)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--accept-licenses',action='store_true',required=True)
    try:parser.parse_args();setup()
    except Exception as error:print('PromptHMR setup failed: '+str(error),file=sys.stderr);raise SystemExit(1)
