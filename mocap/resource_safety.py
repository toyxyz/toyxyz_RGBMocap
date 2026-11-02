"""Local execution preference, independent of motion/project identities."""
import json
import os
from pathlib import Path

ENV = 'RGB_MOCAP_RESOURCE_SAFETY'
MAX_BYTES = 64 * 1024
HELP = ('Stop when RAM headroom is low and limit VRAM allocation. Disable to bypass '
        'these limits; actual memory shortages may cause processing failure or system '
        'slowdowns. Automatic batching, Low Memory and OOM recovery remain enabled.')


def enabled():
    # Only an explicit opt-out disables protection, including isolated workers.
    return os.environ.get(ENV) != '0'


def path(root=None):
    if root is None:
        from .paths import HOME
        root = HOME
    return Path(root) / 'config/local.json'


def read_settings(target):
    if not target.exists():return {}
    if target.stat().st_size > MAX_BYTES:raise ValueError('Local settings exceed the size limit.')
    value = json.loads(target.read_text(encoding='utf-8'))
    if not isinstance(value, dict):raise ValueError('Invalid local settings.')
    return value


def load(root=None):
    try:
        value = read_settings(path(root)).get('resource_safety', True)
        return value if isinstance(value, bool) else True
    except (OSError, ValueError):return True


def save(value, root=None):
    if not isinstance(value, bool):raise ValueError('Invalid Resource Safety selection.')
    from .storage import atomic_json
    target = path(root);settings = read_settings(target)
    settings['resource_safety'] = value
    if len(json.dumps(settings, ensure_ascii=False, indent=2).encode('utf-8')) > MAX_BYTES:
        raise ValueError('Local settings exceed the size limit.')
    atomic_json(target, settings)


def message(value):
    return ('Resource Safety: ON. RAM headroom stopping and VRAM allocation limits are enabled.' if value else
            'Resource Safety: OFF. RAM headroom stopping and VRAM allocation limits are disabled. '
            'Automatic batching, Low Memory and actual OOM recovery remain enabled.')
