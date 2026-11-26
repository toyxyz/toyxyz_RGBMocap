"""Remembered window choices in the Resource Safety config/local.json.

Installation-local conveniences only: never project, inference or export-request data.
A missing, unreadable or unoffered value keeps the window's own default.
"""

MODEL = 'motion_model'
DEFAULT_MODEL = 'exmr'
OPTIONAL_RUNTIMES = ('mamma', 'prompthmr', 'gemx')


def saved(key):
    from .resource_safety import path, read_settings
    try:return read_settings(path()).get(key)
    except (OSError, ValueError):return None


def offered(combo, value):
    # Same type and value, unlike findData's QVariant conversion: "true" or 1 must not select a True item, nor True a 1.
    return next((i for i in range(combo.count()) if type(combo.itemData(i)) is type(value) and combo.itemData(i) == value), -1)


def control_values(controls):
    return {name: control.currentData() if hasattr(control, 'findData') else control.isChecked()
            for name, control in controls.items()}


def apply_choices(controls, values):
    """Apply the offered choices and retain any unavailable combo-box values."""
    kept = {}
    if not isinstance(values, dict):return kept
    for name, control in controls.items():
        value = values.get(name)
        if not isinstance(value, (bool, str)):continue
        if hasattr(control, 'findData'):
            index = offered(control, value)
            if index >= 0:control.setCurrentIndex(index)
            else:kept[name] = (value, control.currentData())
        elif isinstance(value, bool):control.setChecked(value)
    return kept


def restore(controls, key):
    return apply_choices(controls, saved(key))


def remember(controls, key, kept=None):
    from .resource_safety import update
    values = control_values(controls)
    # A choice this window could not offer, such as Biped under GEM-X, survives unless the user changed that control.
    for name, (value, fallback) in (kept or {}).items():
        if values.get(name) == fallback:values[name] = value
    update({key: values})


def runtime_ready(model):
    if model=='vidihand2':
        from .first_person.runtime import load_settings
        try:load_settings()
        except (OSError,ValueError,TypeError):return False
        return True
    if model not in OPTIONAL_RUNTIMES:return True
    from importlib import import_module
    try:import_module('.' + model + '_runtime', __package__).configuration()
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, ImportError):return False
    return True


def start_model():
    """The last offline model picked in the selector, or ExMR when none is saved or its runtime is missing."""
    from .motion_tools import MOTION_TOOLS
    model = saved(MODEL)
    tool = MOTION_TOOLS.get(model) if isinstance(model, str) else None
    return model if tool is not None and runtime_ready(model) else DEFAULT_MODEL


def remember_model(model):
    from .motion_tools import MOTION_TOOLS
    from .resource_safety import update
    if model not in MOTION_TOOLS or saved(MODEL) == model:return
    update({MODEL: model})
