"""Import exact auxiliary body data separately; never bundle it into Git."""
from pathlib import Path
import argparse,json,shutil,urllib.request
from runtime_support import ROOT,sha

def stage(source,dest,download=False):
    source=Path(source).resolve() if source else None;dest=Path(dest).resolve();rows=json.loads((ROOT/'installer/external-assets.json').read_text(encoding='utf-8'))['files'];copied=[]
    for r in rows:
        target=dest/r['path'];original=(source/r['path']) if source else target
        if not target.resolve().is_relative_to(dest) or (source and not original.resolve().is_relative_to(source)):raise ValueError('Auxiliary asset leaves its root')
        if download and (not target.is_file() or sha(target)!=r['sha256']):
            target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_name(target.name+'.partial');last=None
            for url in r.get('urls',[]):
                try:
                    with urllib.request.urlopen(url,timeout=45) as response,temporary.open('wb') as stream:shutil.copyfileobj(response,stream)
                    if temporary.stat().st_size!=r['bytes'] or sha(temporary)!=r['sha256']:raise ValueError('Upstream auxiliary hash mismatch')
                    temporary.replace(target);last=None;break
                except Exception as exc:last=exc;temporary.unlink(missing_ok=True)
            if last or not target.is_file():raise RuntimeError('Download unavailable; acquire the exact upstream file manually: '+r['path']) from last
            original=target
        if not original.is_file():raise FileNotFoundError('Acquire the matching upstream auxiliary data: '+str(original))
        if original.stat().st_size!=r['bytes'] or sha(original)!=r['sha256']:raise ValueError('Wrong auxiliary asset: '+r['path'])
        target.parent.mkdir(parents=True,exist_ok=True)
        if original!=target:shutil.copy2(original,target)
        copied.append(r['path'])
    return copied

def main():
    p=argparse.ArgumentParser(description='Use an upstream checkout/extraction root containing third_party/<tool>/...; follow the provider terms.')
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--from-root',type=Path);group.add_argument('--download',action='store_true')
    p.add_argument('--dest',type=Path,default=ROOT)
    a=p.parse_args();print('Imported',len(stage(a.from_root,a.dest,a.download)),'verified auxiliary files. Model weights remain separate.')

if __name__=='__main__':main()
