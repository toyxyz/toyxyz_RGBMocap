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
STOCK_BUILD_IDS = {'5.6.1': '43139311', '5.7.4': '47537391', '5.8.4': '55116800'}
PRIVATE = re.compile(r'(?:[A-Z]:[\\/]+Users[\\/]+|/(?:home|Users)/|[A-Z]:[\\/]+[^\x00\r\n]*[\\/]Plugins[\\/]RGBMoCapMetaHuman[\\/]|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)', re.I)


def audit(name, raw):
    for encoding in ('latin1', 'utf-16-le'):
        if PRIVATE.search(raw.decode(encoding, errors='ignore')):
            raise ValueError('Private path, machine identifier or credential pattern in ' + name)
    if name.endswith(('.pdb', '.log', '.uasset', '.umap', '.npz', '.mp4')):
        raise ValueError('Forbidden plugin payload: ' + name)


def package(build, output, engine_version='5.7.4'):
    build = Path(build).resolve()
    output = Path(output).resolve()
    build_id = STOCK_BUILD_IDS[engine_version]
    descriptor_name = 'RGBMoCapMetaHuman.uplugin'
    descriptor = json.loads((PLUGIN / descriptor_name).read_text(encoding='utf-8'))
    descriptor_version = engine_version.rsplit('.', 1)[0] + '.0'
    # AutomationTool rewrites the descriptor for the host engine.
    built_descriptor = json.loads((build / descriptor_name).read_text(encoding='utf-8-sig'))
    identity_fields = ('FileVersion', 'Version', 'VersionName', 'Modules', 'Plugins', 'CanContainContent')
    if (any(built_descriptor.get(key) != descriptor[key] for key in identity_fields)
            or built_descriptor.get('EngineVersion', descriptor_version) != descriptor_version):
        raise ValueError('Build descriptor does not match the source and selected engine')
    descriptor['EngineVersion'] = descriptor_version
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
        if name == descriptor_name:
            raw = (json.dumps(descriptor, indent=2) + '\n').encode('utf-8')
        elif name.startswith('Source/') and (build / name).read_bytes() != raw:
            raise ValueError('Build source differs from repository: ' + name)
        audit(name, raw)
        payload[name] = raw
    modules = json.loads((build / 'Binaries/Win64/UnrealEditor.modules').read_text())
    if set(modules['Modules']) != set(MODULES) or modules['BuildId'] != build_id:
        raise ValueError(f'Expected stock UE {engine_version} BuildId {build_id} and exactly two plugin modules')
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
    module_manifest = {'BuildId': build_id, 'Modules': modules['Modules']}
    payload['Binaries/Win64/UnrealEditor.modules'] = (json.dumps(module_manifest, indent=2) + '\n').encode()
    manifest = dict(schema=1, version=descriptor['VersionName'], engine=engine_version, platform='Win64', build_id=modules['BuildId'],
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
    parser.add_argument('--engine-version', choices=STOCK_BUILD_IDS, default='5.7.4',
                        help='Exact stock Unreal version used to build the plugin (default: 5.7.4)')
    args = parser.parse_args()
    print(json.dumps(package(args.build_plugin, args.output, args.engine_version), indent=2))


if __name__ == '__main__':
    main()
