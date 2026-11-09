"""Native SOMA names in the app's humanoid role order."""

BODY = (
    'Hips', 'LeftLeg', 'RightLeg', 'Spine1', 'LeftShin', 'RightShin', 'Spine2',
    'LeftFoot', 'RightFoot', 'Chest', 'LeftToeBase', 'RightToeBase', 'Neck1',
    'LeftShoulder', 'RightShoulder', 'Head', 'LeftArm', 'RightArm',
    'LeftForeArm', 'RightForeArm', 'LeftHand', 'RightHand',
)


def finger_mapping(kind, bones, semantic):
    suffix = {name.split(':')[-1]: name for name in bones}
    digits = {'Thumb': '親指', 'Index': '人指', 'Middle': '中指', 'Ring': '薬指', 'Pinky': '小指'}
    mapping = {}
    for side, lr in (('Left', 'L'), ('Right', 'R')):
        for finger, japanese in digits.items():
            thumb = finger == 'Thumb'
            for segment, part in enumerate(('Proximal', 'Intermediate', 'Distal'), 1):
                if kind == 'mixamo':
                    target = suffix.get(f'{side}Hand{finger}{segment}')
                elif kind == 'vrm':
                    digit = 'Little' if finger == 'Pinky' else finger
                    target = semantic.get(side.lower() + digit + part)
                else:
                    number = '０１２３'[segment - int(thumb)]
                    target = f'{japanese}{number}.{lr}'
                if target in bones:
                    # SOMA's non-thumb chains include a separate metacarpal.
                    mapping[target] = f'{side}Hand{finger}{segment + int(not thumb)}'
    return mapping
