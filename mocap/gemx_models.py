"""Official public GEM-X/SOMA-X assets and the verified YOLOX ONNX import."""

import json
from pathlib import Path

from .paths import ROOT

_manifest = json.loads((ROOT / 'installer/gemx-models.json').read_text(encoding='utf-8'))
ASSETS = _manifest['models']
KEYS = tuple(row['key'] for row in ASSETS)
INSTRUCTIONS = {
    'gemx_detector': ('Download All extracts the official YOLOX archive automatically. For manual import, '
                      'open the supported ZIP and select the extracted end2end.onnx.'),
}


def download_model(key, progress):
    import shutil
    from tempfile import TemporaryDirectory
    from zipfile import ZipFile
    from .models import BY_KEY, import_model_file, ready
    from .paths import HOME
    from tools.pinned_download import download

    model = BY_KEY[key]
    if ready(model):
        return model.path
    cache = HOME / 'cache/gemx-downloads'
    if key != 'gemx_detector':
        pin = dict(url=model.download_url, bytes=model.minimum_bytes, sha256=model.expected_sha256)
        if progress:
            progress('progress', current=0, total=0, message='Downloading ' + model.name)
        file = download(cache, key, pin)
        installed = import_model_file(key, file)
        file.unlink(missing_ok=True)
        return installed
    if progress:
        progress('progress', current=0, total=0, message='Downloading official YOLOX detector archive')
    archive = download(cache, 'yolox-humanart', _manifest['detector_archive'])
    with ZipFile(archive) as data, TemporaryDirectory(dir=cache) as folder:
        entries = [entry for entry in data.infolist() if Path(entry.filename).name == 'end2end.onnx']
        if len(entries) != 1 or entries[0].file_size != model.minimum_bytes:
            raise ValueError('Unexpected official YOLOX archive layout.')
        path = Path(folder) / 'end2end.onnx'
        with data.open(entries[0]) as source, path.open('wb') as output:
            shutil.copyfileobj(source, output, 1024**2)
        installed = import_model_file(model.key, path)
    archive.unlink(missing_ok=True)
    return installed
