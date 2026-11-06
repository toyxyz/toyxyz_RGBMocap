"""Git installer: independent support repair and transaction-safe base promotion."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,shutil,stat,sys,urllib.request,uuid,zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
from runtime_support import cache_dependency_files,prepared_lock,sha
from session_lock import app_lock
from git_transaction import begin,promote,recover,store
from runtime_verification import reuse as reuse_verification,save as save_verification,inventory as runtime_inventory

CONTRACTS={'lock_sha256':'installer/runtime-lock.json','registry_sha256':'mocap/runtime_registry.py',
           'support_sha256':'installer/git-support-lock.json','assets_sha256':'installer/external-assets.json'}

def source_snapshot():
    names=[*CONTRACTS.values(),'installer/engine.py','installer/verify_runtime.py',
           'tools/runtime_support.py','tools/git_managed_install.py','tools/runtime_verification.py',
           'tools/stage_assets.py','tools/support_components.py','tools/pinned_download.py']
    names.extend(p.relative_to(ROOT).as_posix() for p in (ROOT/'installer/runtime_sources').rglob('*') if p.is_file() and p.suffix not in ('.pyc','.pyo'))
    return {name:sha(ROOT/name) for name in names}

def assert_sources(snapshot):
    if source_snapshot()!=snapshot:
        raise RuntimeError('Installation source changed while setup was running. Close other updates/editors and rerun install.cmd.')

def atomic(path,value):
    temp=path.with_name(path.name+'.partial');temp.write_text(json.dumps(value,indent=2),encoding='utf8');os.replace(temp,path)

def setup():
    parser=argparse.ArgumentParser();parser.add_argument('--assets-root',type=Path,default=None)
    parser.add_argument('--verify-all',action='store_true',help='Recheck all installed runtime RECORD hashes and imports, ignoring recent verification metadata.')
    args=parser.parse_args();folder=ROOT/'.setup';resources=folder/'bootstrap-git';resources.mkdir(parents=True,exist_ok=True)
    cache_dependency_files(ROOT)
    from stage_assets import stage as stage_assets
    with app_lock(ROOT):
        pending=folder/'git-sync.pending';pending.write_text('Run install.cmd to complete or retry.\n',encoding='utf8')
        frozen=source_snapshot()
        lock=prepared_lock(resources)
        for origin,name in [(ROOT/'installer/engine.py','engine.py'),(ROOT/'installer/verify_runtime.py','verify_runtime.py'),(ROOT/'mocap/runtime_registry.py','runtime_registry.py')]:
            shutil.copy2(origin,resources/name)
        spec=importlib.util.spec_from_file_location('git_setup_engine',resources/'engine.py');engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
        if recover(ROOT,engine.remove_owned):print('Recovered an interrupted runtime update.',flush=True)
        available=shutil.disk_usage(ROOT).free
        print('Free disk: %.1f GiB; fresh runtime installation estimate: %.1f GiB' % (available/1024**3,lock['fresh_free_bytes_required']/1024**3),flush=True)
        if not (ROOT/'SETUP_RECEIPT.json').exists() and available<lock['fresh_free_bytes_required']:
            raise OSError('Insufficient free space for the runtime installation.')
        support_path=ROOT/'installer/git-support-lock.json'
        support=json.loads(support_path.read_text(encoding='utf8'))
        roots={'runtime_character','tools/bin','third_party/KimodoMotionCorrection/bin'}
        state_file=folder/'git-runtime-state.json'
        try:
            previous=json.loads(state_file.read_text(encoding='utf8'))
            inventory=previous['support_inventory']
            def unchanged(row):
                path=ROOT/row['path'];st=path.stat()
                return path.is_file() and not (st.st_file_attributes&stat.FILE_ATTRIBUTE_REPARSE_POINT) and [st.st_size,st.st_mtime_ns]==row['stat']
            support_ready=(previous['support_sha256']==sha(support_path) and bool(inventory) and all(unchanged(row) for row in inventory))
        except (OSError,ValueError,KeyError,TypeError):support_ready=False
        support_stage=folder/('git-support-'+uuid.uuid4().hex)
        if not support_ready:
            support_stage.mkdir()
            from support_components import stage as stage_support
            stage_support(support_stage, resources/'support-downloads', support)
        if args.assets_root:stage_assets(args.assets_root,ROOT)
        else:stage_assets(None,ROOT,download=True)
        logs=folder/'logs';logs.mkdir(exist_ok=True)
        ids=engine.group_ids(lock)
        checked,retained=reuse_verification(ROOT,lock,ids,args.verify_all)
        reused=engine.reusable_groups(ROOT,lock,None,logs,checked=checked)
        stage,reports=engine.install_runtimes(ROOT,resources,lock,None,reused)
        names=[*engine.TARGETS.values(),*sorted(roots)]
        if not (ROOT/'SETUP_RECEIPT.json').is_file():
            conflicts=[name for name in names if (ROOT/name).exists()]
            if conflicts:raise RuntimeError('Existing unmanaged runtime files were preserved: '+', '.join(conflicts))
        for name in names:
            dest=ROOT/name
            if dest.exists() and (dest.is_symlink() or dest.stat().st_file_attributes&stat.FILE_ATTRIBUTE_REPARSE_POINT):
                raise RuntimeError('A runtime path is a link; refusing to replace it: '+str(dest))
        receipt=ROOT/'SETUP_RECEIPT.json';state=folder/'git-runtime-state.json'
        assert_sources(frozen)
        for group in lock['groups']:
            key=group['key']
            if key in retained and runtime_inventory(ROOT,group['target'])!=retained[key]['files']:
                raise RuntimeError('Runtime files changed during setup: '+key+'. Rerun install.cmd --verify-all.')
        journal,transaction=begin(ROOT)
        try:
            for name in names:
                if name in {engine.TARGETS[key] for key in reused}:continue
                incoming=(stage/name) if name in engine.TARGETS.values() else (support_stage/name)
                if not incoming.exists():continue
                promote(ROOT,journal,transaction,name,incoming)
            atomic(receipt,dict(schema=1,id=lock['id'],version=lock['version'],runtimes=reports,group_ids=engine.group_ids(lock),models_included=False))
            inventory=[]
            for name in sorted(roots):
                for path in sorted((ROOT/name).rglob('*')):
                    if path.is_file():
                        st=path.stat();inventory.append(dict(path=path.relative_to(ROOT).as_posix(),stat=[st.st_size,st.st_mtime_ns]))
            atomic(state,dict(**{key:frozen[name] for key,name in CONTRACTS.items()},support_inventory=inventory))
            assert_sources(frozen)
            transaction['committed']=True;store(journal,transaction)
        except BaseException:
            recover(ROOT,engine.remove_owned)
            raise
        recover(ROOT,engine.remove_owned)
        # A failure in an independent live/camera capability must not mark the
        # already committed, compatible base installation as incomplete.
        pending.unlink(missing_ok=True)
        if support_stage.exists():engine.remove_owned(folder,support_stage)
        try:save_verification(ROOT,lock,ids,reports,retained)
        except (OSError,ValueError) as exc:
            print('Recent verification metadata could not be saved; the next install will fully verify: '+str(exc),flush=True)
        from tools.optional_support import prepare as prepare_support
        prepare_support(ROOT)

if __name__=='__main__':setup()
