"""Scoped PEAR assets; registered geometry is imported locally, never bundled."""
from pathlib import Path
import json,os,zipfile,uuid,shutil
from .paths import ROOT,MODELS
from .storage import sha256,atomic_json

REV='513a74e70a6b4bdecc90ac84ef989c17fe415a9e'
ASSETS=[dict(key='pear',name='Live Pose / PEAR',relative='pear/pear_model.pt',url='https://huggingface.co/BestWJH/PEAR_models',
    minimum_bytes=2685908343,download_url=f'https://huggingface.co/BestWJH/PEAR_models/resolve/{REV}/pear_model.pt',
    expected_sha256='be82dfa06e7b0608c6440058dfa0794d9b2ceee69f6e5b09bf41bb0076abeb18'),
    dict(key='pear_yolo',name='PEAR Person Detection / YOLOv8s',relative='yolo/yolov8s.pt',
        url='https://github.com/ultralytics/assets/releases/tag/v8.3.0',minimum_bytes=22588772,
        download_url='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt',
        expected_sha256='1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a'),
    dict(key='pear_geometry',name='PEAR / SMPL-X 2020 + FLAME Geometry',relative='pear/geometry.json',url='https://github.com/Pixel-Talk/PEAR#-preparation',minimum_bytes=1)]
ASSETS.append(dict(key='pear_geocalib',name='Scene Gravity / GeoCalib',relative='pear/geocalib-pinhole.tar',url='https://github.com/cvg/GeoCalib',minimum_bytes=116074121,download_url='https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar',expected_sha256='86d6aeacd8bbd974c59ce39f61854e00d36911c732ad89be471476fd708722ac'))
KEYS=('pear','pear_geometry','pear_yolo','flow_vggt','pear_geocalib')
INSTRUCTIONS={'pear_geocalib':'Shared by PEAR Camera Align and automatic MAMMA background gravity. MAMMA applies it before 3D fitting and requires it only without manual calibration. PEAR uses it for automatic Camera Align, including one camera. Alignment is optional for tracking: if its support or models are missing, PEAR continues in the selected camera view with a warning, without fusion or aligned foot lock. Download geocalib-pinhole.tar from the official v1.0 release, then import the original TAR file here without extracting it. Code: Apache-2.0; checkpoint: CC BY 4.0. See the official GeoCalib LICENSE-WEIGHTS. Existing VGGT weights are reused.',
    'pear':'Download pear_model.pt from the linked official checkpoint, then import it here.',
    'pear_geometry':'1. Review and accept the SMPL-X and FLAME provider terms; register or sign in where required. PEAR needs SMPL-X 2020 and FLAME 2020, not the standard SMPL-X v1.1 or MAMMA locked-head models.\n2. Open PEAR Preparation below. Use its official Google Drive assets bundle link to obtain assets.zip.\n3. Import the original assets.zip here. The app validates its SHA-256 and extracts only the required geometry files. The source ZIP is left unchanged and is not copied into the models folder.\n4. The geometry bundle has separate provider restrictions; it is not redistributed by RGB MoCap.'}

def spec():return json.loads((ROOT/'installer/pear-assets.json').read_text(encoding='utf8'))

def geometry_folder():
    state=json.loads((MODELS/'pear/geometry.json').read_text(encoding='utf8'))
    folder=(MODELS/'pear'/state['folder']).resolve()
    if not folder.is_relative_to((MODELS/'pear').resolve()):raise ValueError('Invalid PEAR geometry path')
    expected=spec()
    if state.get('bundle_sha256')!=expected['bundle_sha256']:raise ValueError('PEAR geometry bundle changed')
    for item in expected['files']:
        p=(folder/item['path']).resolve()
        if not p.is_relative_to(folder):raise ValueError('Invalid asset path')
        st=p.stat()
        if state['files'].get(item['path'])!=[st.st_size,st.st_mtime_ns] or st.st_size!=item['bytes']:raise ValueError('PEAR geometry changed; import the bundle again')
    return folder

def geometry_ready():
    try:geometry_folder();return True
    except (OSError,ValueError,KeyError,TypeError):return False

def import_bundle(source):
    source=Path(source);expected=spec()
    if source.stat().st_size!=expected['bundle_bytes'] or sha256(source)!=expected['bundle_sha256']:raise ValueError('Select the original assets.zip linked by PEAR Preparation; the bundle hash does not match.')
    target=MODELS/'pear'/('geometry-'+uuid.uuid4().hex)/'assets';target.mkdir(parents=True)
    try:
        linked={}
        with zipfile.ZipFile(source) as z:
            names={i.filename.replace('\\','/'):i for i in z.infolist() if not i.is_dir()}
            for item in expected['files']:
                matches=[k for k in names if k==item['member']]
                if len(matches)!=1:raise ValueError('Missing or ambiguous PEAR asset: '+item['path'])
                out=(target/item['path']).resolve()
                if not out.is_relative_to(target.resolve()):raise ValueError('Unsafe asset path')
                out.parent.mkdir(parents=True,exist_ok=True)
                prior=linked.get(item['sha256'])
                if prior is not None:
                    try:os.link(prior,out)
                    except OSError:shutil.copy2(prior,out)
                else:
                    with z.open(names[matches[0]]) as inp,out.open('wb') as stream:shutil.copyfileobj(inp,stream,1024**2)
                if out.stat().st_size!=item['bytes'] or sha256(out)!=item['sha256']:raise ValueError('Invalid PEAR geometry: '+item['path'])
                linked[item['sha256']]=out
        files={i['path']:[(target/i['path']).stat().st_size,(target/i['path']).stat().st_mtime_ns] for i in expected['files']}
        receipt=MODELS/'pear/geometry.json';atomic_json(receipt,dict(version=1,folder=target.relative_to(MODELS/'pear').as_posix(),bundle_sha256=expected['bundle_sha256'],files=files))
        return receipt
    except Exception:
        # Preserve old generations and a failed staging folder for diagnosis.
        raise
