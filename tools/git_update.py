"""Update a clean branch without destroying local work or skipping dependency repair."""
import os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from session_lock import app_lock

def git(*args):
    return subprocess.run(['git','-C',str(ROOT),*args],capture_output=True,text=True,encoding='utf8',errors='replace',check=True).stdout.strip()

def main():
    with app_lock(ROOT):
        if git('status','--porcelain','--untracked-files=no'):raise RuntimeError('Tracked source files have local changes. Commit or back up changes before updating.')
        if git('branch','--show-current')!='main':raise RuntimeError('Switch to the intended main branch before updating.')
        before=git('rev-parse','HEAD');pending=ROOT/'.setup/git-sync.pending'
        pending.parent.mkdir(parents=True,exist_ok=True)
        previous_pending=pending.read_bytes() if pending.exists() else None
        pending.write_text('Update requires installation and validation.\n',encoding='utf8')
        try:
            print('Updating the current branch...',flush=True)
            subprocess.run(['git','-C',str(ROOT),'pull','--ff-only'],check=True)
        except BaseException:
            if git('rev-parse','HEAD')==before:
                if previous_pending is None:pending.unlink(missing_ok=True)
                else:pending.write_bytes(previous_pending)
            raise
    command=[str(Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'),'-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'tools/git_bootstrap.ps1'),*sys.argv[1:]]
    print('Checking the updated runtime configuration...',flush=True)
    subprocess.run(command,check=True,cwd=ROOT)

def cli():
    # Keep completion handling in memory: Git can replace update.cmd on disk.
    result=0
    cancelled=False
    try:
        main()
    except KeyboardInterrupt:
        print('Update cancelled.',file=sys.stderr,flush=True)
        result=130
        cancelled=True
    except Exception as exc:
        print('Update failed:',exc,file=sys.stderr,flush=True)
        result=exc.returncode if isinstance(exc,subprocess.CalledProcessError) else 1
    if result:
        print('Update stopped; review the message above.',file=sys.stderr,flush=True)
    if not cancelled and (len(sys.argv)==1 or not sys.argv[1]):
        try:
            subprocess.run([str(Path(os.environ['SystemRoot'])/'System32/cmd.exe'),'/d','/c','pause'],check=False)
        except (OSError,KeyboardInterrupt):
            pass
    # sys.exit needs a signed C int; preserve Windows high-bit process statuses.
    return result-0x100000000 if result>0x7fffffff else result

if __name__=='__main__':
    sys.exit(cli())
