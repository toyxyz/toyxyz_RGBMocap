"""Exact source-only compatibility for already committed target previews.

Both hashes must match a reviewed pair. Future math changes therefore cannot
reuse this exception simply because an old preview has the same version.
"""
from functools import lru_cache
from .result_artifacts import file_stat
from .storage import sha256

# Only child storage metadata changed; detection/tracking math is identical.
DISCOVERY_STORAGE_COMPAT = {'096f62724e78996b2c944db45a2a3151e03a5a72d92ea7aad2ae000c796e2135': 'fc7db0d237d87776f0f65388b615e9c00968f1b06c988f332ca91558e7d0fd2d',
    # Media hard links keep source timestamps (Linux NTFS); detection is unchanged.
    '92138a2919704d18aa94f4138f8a96b6b7a6191f4a5f09a8f207c550fafc0600': '39f6422998f9d77a501cd4ac88c2a62cf312cf2be0f01129e9c0e7b8a0dd2b5f'}


def discovery_identity(path):
    digest=sha256(path)
    return DISCOVERY_STORAGE_COMPAT.get(digest,digest)

COMPATIBLE_CODE = {
    'source_surface.py': {'2dc605a8d8381e11439200ed3885ab3d7208e76790c394ecd693b16963bab0fc': 'fbe6b1dd944e63b337d683a738e9a5a5be7e14a4c48718dde0820f1d37e7dfbb'},
    # Only the native module's platform suffix changed; correction math is identical.
    'correction.py': {'7b25e1eb57ed810bdf995a918d7b7633e7e558c25f982ba273f7d64783105ec3': '18b52409527dea9e7513850f7936d8662f09c93f00d64d8452b449881b2af1b8'},
    'mixamo_data.py': {
        # Project relocation changes path resolution only; fitting is unchanged.
        'e256fdca2b356c1002d011f85942529e6af16d84631985d9b303c3c938196eb6': 'fa10d8fbd61e8a6946038491abe23409ee34ffda0efbe0d5a093275de44dbd40',
        '92e16a81406a5ea8a9453db53a3dbc38eb8869e0b8ac294704d0f1252c510be5': '707e3e2e86c55aa5f468b0801ce02af637dbafec0fc0638f5c3771cadda42549',
        # FlowHMR adds a metadata-selected model key and native vertex count;
        # defaults retain the exact GVHMR loading, fitting and export math.
        '0f3d28c977ca5c7ee910e5f7647a223feaef0071db7177e9ef1ef404bf283bd1': '707e3e2e86c55aa5f468b0801ce02af637dbafec0fc0638f5c3771cadda42549',
        # Only the SMPL-H raw topology cache key was made shareable on GPU.
        '5b19f3e14986fbbb97b69e757d301c5e803f5ca45a5a35f98f9d8401c4c31338': '707e3e2e86c55aa5f468b0801ce02af637dbafec0fc0638f5c3771cadda42549',
        '8e039845858b9980690e5e864a8a36c3410dafb325b097e223507cf6c60293b0': '707e3e2e86c55aa5f468b0801ce02af637dbafec0fc0638f5c3771cadda42549',
        # HandFlow adds a full-finger branch. Legacy source tensors have no
        # 45-D hand fields, so their complete body/finger math is unchanged.
        'e83ef02aedfdf09fafa6bbb4d4d318fb6d3206ddb42325fc36ab8fa5b84ec659': '707e3e2e86c55aa5f468b0801ce02af637dbafec0fc0638f5c3771cadda42549',
    },
}


@lru_cache(maxsize=16)
def _digest(path, stamp):
    return sha256(path)


def code_stats_current(record):
    from .paths import ROOT
    stamps=record.get('code_stats',{})
    if not stamps:return False
    for name,old_stat in stamps.items():
        if '/' in name or '\\' in name:return False
        path=ROOT/'mocap'/name
        try:
            stamp=file_stat(path)
        except OSError:
            return False
        if stamp==old_stat:continue
        approved=COMPATIBLE_CODE.get(name,{}).get(record.get('code',{}).get(name))
        if approved is None or _digest(path,tuple(stamp))!=approved:return False
    return True
