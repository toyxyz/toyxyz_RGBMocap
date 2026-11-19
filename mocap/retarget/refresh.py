"""Select baked characters affected by an explicit correction batch."""
import os
from pathlib import Path
from mocap.motion_tools import base_result


def correction_refreshes(characters, jobs, project):
    changed = {}
    for stage, args in jobs:
        if stage not in ('correct', 'gemx_correct'):
            continue
        args = args or []
        path = args[args.index('--project') + 1] if '--project' in args else project
        kind = args[args.index('--result') + 1]
        changed[os.path.normcase(str(Path(path).resolve()))] = kind
    plans = []
    for character in characters:
        takes = character.get('takes', [])
        kinds = {changed.get(os.path.normcase(str(Path(t['source']).resolve())))
                 for t in takes if t.get('source')}
        kinds.discard(None)
        kinds = {k for k in kinds if base_result(k) == base_result(character.get('result', ''))}
        if not kinds:
            continue
        if len(kinds) != 1 or any(not t.get('source') for t in takes):
            raise ValueError('Cannot identify the source of a character. Retarget it again.')
        plans.append(dict(id=character['id'], result=kinds.pop(),
                          source_name=character.get('source_name', 'Person'),
                          takes=[dict(project=t['source'], start=t['start'],
                                      transform=t.get('transform')) for t in takes]))
    return plans
