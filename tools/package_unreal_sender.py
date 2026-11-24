"""Package only the RGB MoCap sender module for an exact installed UE build."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))

from tools.package_unreal_plugin import STOCK_BUILD_IDS, audit


BINARIES = Path('Binaries/Win64')
MODULE = 'UnrealEditor-RGBMoCapLiveLink.dll'
METADATA = 'UnrealEditor.modules'
MANIFEST = 'SENDER_MANIFEST.json'
LICENSE = 'LICENSE.txt'
LIMIT = 2 * 1024 * 1024


def module_manifest(build_id):
    return {'BuildId': build_id, 'Modules': {'RGBMoCapLiveLink': MODULE}}


def fingerprints(payload):
    return {name: {'bytes': len(raw), 'sha256': sha256(raw).hexdigest()}
            for name, raw in payload.items()}


def check_payload(payload, build_id):
    if not payload[MODULE].startswith(b'MZ'):
        raise ValueError('The sender archive does not contain a Windows DLL.')
    if json.loads(payload[METADATA]) != module_manifest(build_id):
        raise ValueError('The sender module does not match the package BuildId.')
    for name, raw in payload.items():
        audit(name, raw)


def read_package(path, source):
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        expected = {MANIFEST, MODULE, METADATA}
        if LICENSE in archive.namelist():
            expected.add(LICENSE)
        if (len(members) != len(expected) or {m.filename for m in members} != expected
                or any(m.file_size > LIMIT for m in members)):
            raise ValueError('Unexpected files or file size in the sender archive.')
        manifest = json.loads(archive.read(MANIFEST))
        files = {name: archive.read(name) for name in expected - {MANIFEST}}
    if not isinstance(manifest, dict) or not isinstance(manifest.get('engine'), str):
        raise ValueError('Invalid sender manifest.')
    version = manifest['engine']
    build_id = STOCK_BUILD_IDS.get(version)
    if (manifest.get('schema') != 1 or manifest.get('platform') != 'Win64'
            or not build_id or manifest.get('build_id') != build_id):
        raise ValueError('Unsupported Unreal build in sender manifest.')
    if manifest.get('source') != source:
        raise ValueError('The sender archive requires a different RGB MoCap source revision.')
    if manifest.get('files') != fingerprints(files):
        raise ValueError('The sender archive failed its file integrity check.')
    payload = {name: files[name] for name in (MODULE, METADATA)}
    check_payload(payload, build_id)
    return manifest, payload


def package(build, output, version):
    from mocap.unreal_stream import SOURCE, source_tree_identity

    source = source_tree_identity(ROOT / SOURCE)
    if source_tree_identity(build) != source:
        raise ValueError('The compiled sender project does not match the repository source.')
    build_id = STOCK_BUILD_IDS[version]
    payload = {name: (build / BINARIES / name).read_bytes() for name in (MODULE, METADATA)}
    check_payload(payload, build_id)
    files = {**payload, LICENSE: (ROOT / 'LICENSE').read_bytes()}
    manifest = dict(schema=1, engine=version, build_id=build_id, platform='Win64', source=source,
                    files=fingerprints(files))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted({**files, MANIFEST: (json.dumps(manifest, indent=2) + '\n').encode()}.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 25, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)
    if read_package(output, source) != (manifest, payload):
        raise ValueError('The sender archive readback differs.')
    digest = sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name + '.sha256').write_text(f'{digest}  {output.name}\n', encoding='ascii')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--engine-version', choices=STOCK_BUILD_IDS, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.build, args.output, args.engine_version), indent=2))
