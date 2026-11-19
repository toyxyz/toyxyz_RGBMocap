"""Package the authored Unreal plugin with verified local matching binaries."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / 'unreal_plugin/RGBMoCapMetaHuman'
MODULES = ('RGBMoCapMetaHuman', 'RGBMoCapMetaHumanEditor')
PRIVATE = re.compile(r'(?:[A-Z]:[\\/]+Users[\\/]+|/(?:home|Users)/|[A-Z]:[\\/]+[^\x00\r\n]*[\\/]Plugins[\\/]RGBMoCapMetaHuman[\\/]|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)', re.I)


def audit(name, raw):
    for encoding in ('latin1', 'utf-16-le'):
        if PRIVATE.search(raw.decode(encoding, errors='ignore')):
            raise ValueError('Private path, machine identifier or credential pattern in ' + name)
    if name.endswith(('.pdb', '.log', '.uasset', '.umap', '.npz', '.mp4')):
        raise ValueError('Forbidden plugin payload: ' + name)


def package(build, output):
    build = Path(build).resolve()
    output = Path(output).resolve()
    descriptor = json.loads((PLUGIN / 'RGBMoCapMetaHuman.uplugin').read_text())
    payload = {}
    for path in sorted(PLUGIN.rglob('*')):
        if not path.is_file():
            continue
        name = path.relative_to(PLUGIN).as_posix()
        allowed = name in ('RGBMoCapMetaHuman.uplugin', 'README.md', 'LICENSE.txt', 'SECURITY.md', 'VALIDATION.md') or (
            name.startswith('Source/') and path.suffix in ('.h', '.cpp', '.cs'))
        if not allowed or path.is_symlink():
            raise ValueError('File outside the plugin source allowlist: ' + name)
        raw = path.read_bytes()
        if name.startswith('Source/') or name.endswith('.uplugin'):
            candidate = build / name
            if candidate.read_bytes() != raw:
                raise ValueError('Build source differs from repository: ' + name)
        audit(name, raw)
        payload[name] = raw
    modules = json.loads((build / 'Binaries/Win64/UnrealEditor.modules').read_text())
    if set(modules['Modules']) != set(MODULES) or modules['BuildId'] != '47537391':
        raise ValueError('Expected the verified stock UE 5.7.4 build and exactly two plugin modules')
    for module in MODULES:
        filename = 'UnrealEditor-' + module + '.dll'
        if modules['Modules'][module] != filename:
            raise ValueError('Unexpected plugin module filename')
        name = 'Binaries/Win64/' + filename
        raw = (build / name).read_bytes()
        if raw[:2] != b'MZ':
            raise ValueError('Invalid native module')
        audit(name, raw)
        payload[name] = raw
    payload['Binaries/Win64/UnrealEditor.modules'] = (json.dumps(modules, indent=2) + '\n').encode()
    manifest = dict(schema=1, version=descriptor['VersionName'], engine='5.7.4', platform='Win64', build_id=modules['BuildId'],
                    files=[dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) for name, raw in sorted(payload.items())])
    payload['PACKAGE_MANIFEST.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    if output.exists():
        raise ValueError('Choose a new archive path; existing releases are never overwritten')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(payload.items()):
            info = zipfile.ZipInfo('RGBMoCapMetaHuman/' + name, date_time=(2026, 11, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Plugin archive CRC verification failed')
        for name, raw in payload.items():
            if archive.read('RGBMoCapMetaHuman/' + name) != raw:
                raise ValueError('Plugin archive readback differs')
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name + '.sha256').write_text(digest + '  ' + output.name + '\n', encoding='ascii')
    return dict(archive=str(output), version=descriptor['VersionName'], bytes=output.stat().st_size, sha256=digest, payload_files=len(payload))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-plugin', type=Path, required=True, help='Locally compiled plugin root; Source must match this repository')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.build_plugin, args.output), indent=2))


if __name__ == '__main__':
    main()
