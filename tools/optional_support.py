"""Prepare independent capabilities after the base installation has committed."""
from pathlib import Path
import json
import os


def prepare(root):
    from tools.setup_pear import prepare as pear
    from tools.setup_pear_alignment import prepare as alignment
    path = Path(root) / '.setup/git-components.json'
    rows = {}

    def record():
        temp = path.with_suffix('.partial')
        temp.write_text(json.dumps(dict(version=1, components=rows), indent=2), encoding='utf8')
        os.replace(temp, path)

    for name, action in [('PEAR', lambda: pear(align=False)), ('Camera alignment', alignment)]:
        rows[name] = dict(status='preparing')
        record()
        try:
            action()
        except Exception as exc:
            rows[name] = dict(status='unavailable', error=str(exc)[-1500:])
            print(name + ' support unavailable: ' + str(exc), flush=True)
        else:
            rows[name] = dict(status='ready')
        record()
    failures = [name for name, row in rows.items() if row['status'] != 'ready']
    if failures:
        print('Installation complete with warnings. The base app is ready. Unavailable: '
              + ', '.join(failures) + '. Rerun install.cmd to retry; see .setup/git-components.json. '
              'Automatic MAMMA gravity requires Camera alignment support.', flush=True)
    else:
        print('Installation complete. Existing models and projects were retained.', flush=True)
    return rows
