"""Read the attributed, bundled UE5 Third Person reference rigs."""
import json

from .paths import ROOT

BUNDLE = ROOT / 'third_party' / 'ue5_mannequin'
TARGETS = ('manny', 'quinn')


def load_profiles():
    from .ue5_retarget import target_rig
    path = BUNDLE / 'reference.json'
    try:
        if path.stat().st_size > 512 * 1024:
            raise ValueError('The mannequin reference is too large.')
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('version') != 1 or set(data['profiles']) != set(TARGETS):
            raise ValueError('Invalid mannequin reference.')
        return {target: target_rig(data['profiles'][target]) for target in TARGETS}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise ValueError('The bundled UE5 mannequin reference is missing or invalid. Repair the RGB MoCap installation.') from error


def skin_for(target):
    if target not in TARGETS:
        raise ValueError('Choose Manny or Quinn as the FBX target.')
    path = BUNDLE / (target + '.npz')
    if not path.is_file():
        raise ValueError('The selected mannequin mesh is missing. Repair the RGB MoCap installation.')
    return path
