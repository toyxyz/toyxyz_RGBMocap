from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .paths import MODELS
from .storage import atomic_json, sha256


@dataclass(frozen=True)
class Model:
    key: str
    name: str
    relative: str
    url: str
    drive_id: str | None = None
    minimum_bytes: int = 1_000_000
    download_url: str | None = None
    expected_sha256: str | None = None

    @property
    def path(self) -> Path:
        return MODELS / self.relative


MODELS_LIST = [
    Model("person_reid", "Person Identity / OSNet", "reid/osnet_x0_25_msmt17.pth",
          "https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO", "1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e", 1_000_000,
          expected_sha256='cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18'),
    Model("yolo", "Actor Detection / YOLO", "yolo/yolov8x.pt", "https://github.com/ultralytics/assets/releases", "11-GzfivV5LHFIBu9g4Zp74dt0CktHL7l", 100_000_000,
          expected_sha256="c4d5a3f000d771762f03fc8b57ebd0aae324aeaefdd6e68492a9c4470f2d1e8b"),
    Model("gvhmr", "Initial Motion / GVHMR", "gvhmr/gvhmr_siga24_release.ckpt", "https://github.com/zju3dv/GVHMR", "1c9iCeKFN4Kr6cMPJ9Ss6Jdc3SZFnO5NP", 10_000_000,
          expected_sha256="4fae7da2de388d5da3514cb27a2d003f364dacb280e9cf88972b710e589c6b91"),
    Model("hmr2", "Image Features / HMR2", "hmr2/epoch=10-step=25000.ckpt", "https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md", "1X5hvVqvqI9tvjUCb2oAlZxtgIKD9kvsc", 100_000_000,
          expected_sha256="2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e"),
    Model("vitpose", "Joint Estimation / ViTPose", "vitpose/vitpose-h-multi-coco.pth", "https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md", "1sR8xZD9wrZczdDVo6zKscNLwvarIRhP5", 100_000_000,
          expected_sha256="50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc"),
    Model("pva", "Temporal Refinement / PVA-Net", "pvanet.pt", "https://github.com/ant-research/HTD-Refine", "1dffyj3cdH1Myw2-tzgKTXryaO72Gbys1", 3_000_000_000,
          expected_sha256="a6e5f95a287a45dc1e4944fe7381e7d0c5fe02d4009c5794b09717077a4e1a26"),
    Model("footmr", "Ankle Refinement / FootMR", "footmr/footmr_checkpoint.ckpt", "https://github.com/twehrbein/FootMR", minimum_bytes=178_872_284,
          download_url="https://cloud.tnt.uni-hannover.de/index.php/s/tpLX3F6Mz4FqHaD/download?path=%2Fcheckpoints%2Ffootmr&files=footmr_checkpoint.ckpt",
          expected_sha256="2d31d8b5f7079c86dc176472909d4b3c14db801f0bcd9f99571637d0a860407a"),
    Model("vitpose_wholebody", "Foot Keypoints / ViTPose WholeBody", "vitpose/vitpose-h-wholebody.pth", "https://github.com/twehrbein/FootMR/blob/main/docs/INSTALL.md", minimum_bytes=2_549_087_447,
          download_url="https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/wholebody/vitpose-h-wholebody.pth",
          expected_sha256="dbed01fd5bb221610bf26434ec63426025f76eaca46f6177db71c9771a43316c"),
    Model("smplx", "Body Model / SMPL-X Neutral", "body_models/smplx/SMPLX_NEUTRAL.npz", "https://smpl-x.is.tue.mpg.de/", minimum_bytes=1_000_000),
    Model("smpl", "Body Model / SMPL Neutral", "body_models/smpl/SMPL_NEUTRAL.pkl", "https://smpl.is.tue.mpg.de/", minimum_bytes=1_000_000),
]
# Frozen upstream revisions match the separately validated Windows inference.
FLOW_REV = '1eb105e554e0cfc1846ea5feac8c2d516a36d319'
SAM_REV = '11aaa346c7204874a1cbafe3d39a979080b2c55a'
MODELS_LIST.extend([
    Model('flowhmr', 'Motion / FlowHMR latest', 'flowhmr/flowhmr_latest.ckpt', 'https://github.com/flowhmr/flowhmr', minimum_bytes=1841702021,
          download_url=f'https://huggingface.co/fafsaf1/flowhmr-0.46B/resolve/{FLOW_REV}/flowhmr_latest/flowhmr_latest.ckpt', expected_sha256='b7eb06b119feaf5ad7287bafdb2fff38d42dfc22ea9d256f8cdd2ada26dfc53a'),
    Model('flow_config', 'FlowHMR configuration', 'flowhmr/config.yml', 'https://github.com/flowhmr/flowhmr', minimum_bytes=3067,
          download_url=f'https://huggingface.co/fafsaf1/flowhmr-0.46B/resolve/{FLOW_REV}/flowhmr_latest/config.yml', expected_sha256='5deae01800c031975558bf9675819e8d1e24803414f692b8b5c2196b266c64d7'),
    Model('flow_metadata', 'FlowHMR checkpoint metadata', 'flowhmr/config.json', 'https://huggingface.co/fafsaf1/flowhmr-0.46B', minimum_bytes=25,
          download_url=f'https://huggingface.co/fafsaf1/flowhmr-0.46B/resolve/{FLOW_REV}/config.json', expected_sha256='dd387080edacff4092e6d663e3d7ff04b848700711b113ad3cec9681bf11b8c1'),
    Model('flow_sam', 'Body / Hand Features / SAM 3D Body', 'flowhmr/sam/model.ckpt', 'https://huggingface.co/facebook/sam-3d-body-dinov3', minimum_bytes=2109129346,
          expected_sha256='b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf'),
    Model('flow_sam_config', 'SAM 3D Body configuration', 'flowhmr/sam/model_config.yaml', 'https://huggingface.co/facebook/sam-3d-body-dinov3', minimum_bytes=1488,
          expected_sha256='1012fc3f39cb5e90e3f8fbadf7bded31604bfafdce0321d17a7c1a2d3f08b88d'),
    Model('flow_mhr', 'SAM Body Geometry / MHR', 'flowhmr/sam/assets/mhr_model.pt', 'https://huggingface.co/facebook/sam-3d-body-dinov3', minimum_bytes=696110248,
          expected_sha256='352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc'),
    Model('flow_vggt', 'Scene Camera / VGGT-Omega 512', 'flowhmr/vggt_omega_1b_512.pt', 'https://huggingface.co/facebook/VGGT-Omega', minimum_bytes=4576706117,
          expected_sha256='c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934'),
    Model('smplh', 'Body Model / AMASS SMPL-H Neutral', 'body_models/smplh/neutral/model.npz', 'https://mano.is.tue.mpg.de/index.html', minimum_bytes=80000000,
          expected_sha256='a60a7e29d33f09ef1a6352907e2f485e161f471330ca40592764374063e751df'),
])
MODELS_LIST.extend([
    Model('handflow', 'Finger Motion / HandFlow', 'handflow/handflow_denoiser.pt', 'https://github.com/mxxu00/HandFlow', minimum_bytes=667907131,
          download_url='https://huggingface.co/mxxu00/HandFlow/resolve/main/handflow_denoiser.pt',expected_sha256='2fbc4e1fa7a60f469a6ac94933a6e6dc8a86a0e1fc13bd7cd81c430c79acfcda'),
    Model('handflow_stats', 'HandFlow normalization', 'handflow/normalization_stats.npz', 'https://github.com/mxxu00/HandFlow', minimum_bytes=2008,
          download_url='https://huggingface.co/mxxu00/HandFlow/resolve/main/normalization_stats.npz',expected_sha256='7313334e6b9537fa57ec9763e83f36dfd4998e7d1de09aa3fb21c5bfca8e92c4'),
    Model('hand_hamer', 'Hand Features / HaMeR', 'handflow/hamer.ckpt', 'https://github.com/geopavlakos/hamer', minimum_bytes=2689536166,
          download_url='https://huggingface.co/spaces/geopavlakos/HaMeR/resolve/main/_DATA/hamer_ckpts/checkpoints/hamer.ckpt',expected_sha256='e5cc06f294d88a92dee24e603480aab04de532b49f0e08200804ee7d90e16f53'),
    Model('hand_config', 'HaMeR configuration', 'handflow/model_config.yaml', 'https://github.com/geopavlakos/hamer', minimum_bytes=2084,
          download_url='https://huggingface.co/spaces/geopavlakos/HaMeR/resolve/main/_DATA/hamer_ckpts/model_config.yaml',expected_sha256='2217f6bcf27ae4c074e2a33a3409d822bbfd02931373deedb9f71287adeabe16'),
    Model('hand_mean', 'HaMeR mean parameters', 'handflow/mano_mean_params.npz', 'https://github.com/geopavlakos/hamer', minimum_bytes=1178,
          download_url='https://huggingface.co/spaces/geopavlakos/HaMeR/resolve/main/_DATA/data/mano_mean_params.npz',expected_sha256='efc0ec58e4a5cef78f3abfb4e8f91623b8950be9eff8b8e0dbb0d036ebc63988'),
    Model('hand_detector', 'Hand Detection / WiLoR YOLO', 'handflow/detector.pt', 'https://github.com/rolpotamias/WiLoR', minimum_bytes=53582271,
          download_url='https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt',expected_sha256='5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033'),
    Model('mano_right', 'Hand Model / MANO Right', 'body_models/mano/MANO_RIGHT.pkl', 'https://mano.is.tue.mpg.de/index.html', minimum_bytes=3821356,
          expected_sha256='45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767'),
])
SAM_MANUAL_FILES = {'flow_sam': 'model.ckpt', 'flow_sam_config': 'model_config.yaml', 'flow_mhr': 'assets/mhr_model.pt'}
EXMR_REV = '8debd2e34262fcc63384f7e04d39ca784b0aec21'
MODELS_LIST.extend([
    Model('exmr','Motion / ExMR','exmr/exmr.safetensors','https://huggingface.co/twehrbein/ExMR',minimum_bytes=235766520,
          download_url=f'https://huggingface.co/twehrbein/ExMR/resolve/{EXMR_REV}/exmr.safetensors',
          expected_sha256='8d3947310c30444ec59dae4662574d7a32bf83af30320eeac87146529ff2fda5'),
    Model('exmr_config','ExMR configuration','exmr/config.json','https://huggingface.co/twehrbein/ExMR',minimum_bytes=180,
          download_url=f'https://huggingface.co/twehrbein/ExMR/resolve/{EXMR_REV}/config.json',
          expected_sha256='b29ad86d7916a7f54154c1a20fb07d3bc9d794da4d671c88ba0cf75a5cec8c43'),
    Model('exmr_sapiens','Body / Hand / Foot Keypoints / Sapiens2 1B','exmr/sapiens2_1b_pose.safetensors','https://huggingface.co/facebook/sapiens2-pose-1b',minimum_bytes=6079194752,
          download_url='https://huggingface.co/facebook/sapiens2-pose-1b/resolve/f5fed8b97b99698d5eea1d14ff0855d0b4c3f000/sapiens2_1b_pose.safetensors',
          expected_sha256='2dab7014a17e99e460c18817325a71dd7a81ce48d87027f01c2ee7d7b3af969f'),
])
from .mamma_models import ASSETS as MAMMA_ASSETS, KEYS as MAMMA_KEYS, INSTRUCTIONS as MAMMA_INSTRUCTIONS
MODELS_LIST.extend(Model(**entry) for entry in MAMMA_ASSETS)
from .prompthmr_models import ASSETS as PHMR_ASSETS,KEYS as PHMR_KEYS,CAMERA_KEYS as PHMR_CAMERA_KEYS,INSTRUCTIONS as PHMR_INSTRUCTIONS
from .pear_models import ASSETS as PEAR_ASSETS,KEYS as PEAR_KEYS,INSTRUCTIONS as PEAR_INSTRUCTIONS
from .gemx_models import ASSETS as GEMX_ASSETS,KEYS as GEMX_KEYS,INSTRUCTIONS as GEMX_INSTRUCTIONS
MODELS_LIST.extend(Model(**entry) for entry in PEAR_ASSETS)
MODELS_LIST.extend(Model(**entry) for entry in PHMR_ASSETS)
MODELS_LIST.extend(Model(**entry) for entry in GEMX_ASSETS)
BY_KEY = {model.key: model for model in MODELS_LIST}
if len(BY_KEY)!=len(MODELS_LIST):raise ValueError('Duplicate model identifiers in the model registry.')
MANUAL_INSTRUCTIONS = {
    'flow_metadata': 'Download config.json from the root of the supported FlowHMR Hugging Face revision. Auto Download can fetch this small public file. Import the original JSON in this row. This checkpoint metadata is separate from flowhmr_latest/config.yml; do not replace the inference YAML with it.',
    'exmr_config': 'Download config.json using the Supported file link below. It is the official ExMR checkpoint metadata, pinned to the same revision as exmr.safetensors. Return to the ExMR configuration row and click Import. Import the original JSON file without editing or renaming it. Auto Download can obtain this small public file directly.',
    'mano_right': '1. Open the Recommended download link below, or click Open Download Page. This opens the official download server for the exact archive. Sign in and accept the provider terms; register on the official MANO website if needed.\n2. Download mano_v1_2.zip. If browsing the MANO website instead, this is Models & Code > Models & Code.\n3. Extract the ZIP and open mano_v1_2/models.\n4. Return to Hand Model / MANO Right, click Import and select MANO_RIGHT.pkl. Keep this PKL file intact.\nHandFlow uses a right-hand canonical model for both hands, reflecting the left result back. The separate SMPL-H downloads cannot replace this hand model.',
    'smplx': '1. Open the SMPL-X download page below. Register or sign in and accept the provider terms.\n2. In the "SMPL-X Model" section, choose "Download SMPL-X v1.1 (NPZ+PKL, 830 MB)". Choose the standard v1.1 download listed above.\n3. Download models_smplx_v1_1.zip and extract the outer ZIP archive.\n4. Inside the extracted folder, open models, then smplx. Choose SMPLX_NEUTRAL.npz. Leave this NPZ file intact.\n5. Return to the Body Model / SMPL-X Neutral row and click Import. Select that single NPZ file; Missing changes to Ready after validation.',

    "pva": "Download pvanet.pt using the official Google Drive link below, then import that file here. Google Drive may temporarily limit downloads of this public file. If the limit also affects your browser, retry later. You can import a previously downloaded official copy. The app does not sign in or use account tokens.",
    'smplh': "1. Open the Recommended AMASS download link below, or click Open Download Page. This opens the official download server. Sign in and accept the provider terms; register on the official MANO website if needed.\n2. Download smplh.tar.xz. On the MANO website this is \"Extended SMPL+H model\" marked \"used in AMASS project\" under Models & Code. This is the recommended model for this app.\n3. Extract smplh.tar.xz. If your extractor first produces a TAR file, extract that file as well.\n4. Open the extracted smplh folder, then neutral. Select model.npz and leave this NPZ file intact.\n5. Return to Body Model / AMASS SMPL-H Neutral and click Import. Select smplh/neutral/model.npz. This is the neutral model with 16 shape components.",
    'flow_vggt': '1. Open facebook/VGGT-Omega on Hugging Face. Sign in, review the conditions and request access if required.\n2. Wait until your account has access, then open Files and versions.\n3. Find vggt_omega_1b_512.pt and download that file using its download button. The file link below opens it directly after access is granted.\n4. Return to the Scene Camera / VGGT-Omega 512 row and click Import. Select the downloaded PT file.\nThe app does not need your account password or access token.',
    'flowhmr': '1. Open the supported file link below. It selects flowhmr_latest at the app-supported revision.\n2. Download flowhmr_latest.ckpt. If browsing the repository instead, open flowhmr_latest, not flowhmr_base.\n3. Return to Motion / FlowHMR latest and click Import to select the CKPT file. Also import the matching config.yml in its separate configuration row.',
    'flow_config': '1. Open the supported file link below. It selects config.yml inside flowhmr_latest at the app-supported revision.\n2. Save that file. If your browser displays its contents, use the file download button or Save As to save config.yml.\n3. Return to FlowHMR configuration and click Import. Use the configuration paired with the supported flowhmr_latest checkpoint.',

}

MANUAL_INSTRUCTIONS.update(MAMMA_INSTRUCTIONS)
MANUAL_INSTRUCTIONS.update(PHMR_INSTRUCTIONS)
MANUAL_INSTRUCTIONS.update(PEAR_INSTRUCTIONS)
MANUAL_INSTRUCTIONS.update(GEMX_INSTRUCTIONS)

for _key in ('mano_right','smplh'):
    MANUAL_INSTRUCTIONS[_key] += ('\nIf the MANO website shows 502 Bad Gateway, use the Recommended download link below '
        'to open the official download server and sign in there. If sign-in or downloading still fails, '
        'retry later or import an official copy you already downloaded. This app does not need your password or access token.')

for _key, _file in SAM_MANUAL_FILES.items():
    MANUAL_INSTRUCTIONS[_key] = (
        '1. Open facebook/sam-3d-body-dinov3 on Hugging Face. Sign in and review its provider conditions. Complete the access request if required.\n'
        '2. Once access is granted, open the Supported file link below. It selects the app-supported revision and exact file.\n'
        '3. Download ' + _file + ' using the file download button. This is a single file; do not download another SAM model family.\n'
        '4. Return to this model row and click Import. Select ' + Path(_file).name + '.\n'
        '5. Repeat for the other two SAM rows: model.ckpt, model_config.yaml and assets/mhr_model.pt are all required. Use the same supported revision for all three.\n'
        'The app does not need your account password or access token.')

MANUAL_DETAILS = {
    'exmr':dict(selection='twehrbein/ExMR > Files and versions > exmr.safetensors',location='exmr.safetensors',avoid='FootMR standalone weights, config.json (use the separate ExMR configuration row), or repository ZIPs.'),
    'exmr_config':dict(selection='twehrbein/ExMR > Files and versions > config.json',location='config.json',avoid='Edited JSON, a different revision, an HTML page, or exmr.safetensors. Use the Supported file link for the matching revision.'),
    'exmr_sapiens':dict(selection='facebook/sapiens2-pose-1b > Files and versions > sapiens2_1b_pose.safetensors',location='sapiens2_1b_pose.safetensors',avoid='Sapiens v1, 0.3B models, segmentation, depth or normal checkpoints.'),
    'mano_right': dict(selection='Models & Code > Models & Code (mano_v1_2.zip)',location='mano_v1_2/models/MANO_RIGHT.pkl',avoid='SMPL+H archives, MANO_LEFT.pkl, or the outer ZIP file.'),
    'smplx': dict(selection='SMPL-X Model > Download SMPL-X v1.1 (NPZ+PKL, 830 MB)',
        location='models/smplx/SMPLX_NEUTRAL.npz',
        avoid='Removed-head-bun variants, Julia models, SMPL-X 2020, VPoser, male/female models, SMPLX_NEUTRAL.pkl, or the outer ZIP archive. Choose the standard v1.1 neutral NPZ listed above for GVHMR.'),
    'smplh': dict(selection='Extended SMPL+H model (used in AMASS project)',
        location='smplh/neutral/model.npz'),
    'flow_vggt': dict(selection='Files and versions > vggt_omega_1b_512.pt',
        location='vggt_omega_1b_512.pt (import the file itself)',
        avoid='256_text, 416, the original VGGT checkpoint, or a repository ZIP. They cannot replace the required Omega 512 checkpoint.'),
    'flowhmr': dict(selection='flowhmr_latest > flowhmr_latest.ckpt',
        location='flowhmr_latest/flowhmr_latest.ckpt',avoid='flowhmr_base or another checkpoint revision.'),
    'flow_config': dict(selection='flowhmr_latest > config.yml',
        location='flowhmr_latest/config.yml',avoid='The flowhmr_base configuration, config.json, an HTML page, or a configuration from a different revision.'),
    'flow_metadata': dict(selection='Files and versions > config.json (repository root)',
        location='config.json',avoid='config.yml, edited JSON, an HTML page, or a different revision.'),
}
for _key, _file in SAM_MANUAL_FILES.items():
    MANUAL_DETAILS[_key] = dict(selection='facebook/sam-3d-body-dinov3 > ' + _file,
        location=_file,avoid='SAM image segmentation, SAM 3D Objects, standalone DINOv3 weights, or files from a different revision. Import each of the three supported SAM files in its own matching row.')

def manual_download_page(key: str) -> str:
    model = BY_KEY[key]
    body_pages = {'smplx':'https://smpl-x.is.tue.mpg.de/download.php',
                  'smplh':'https://download.is.tue.mpg.de/download.php?domain=mano&resume=1&sfile=smplh.tar.xz',
                  'mano_right':'https://download.is.tue.mpg.de/download.php?domain=mano&resume=1&sfile=mano_v1_2.zip',
                  'smpl':'https://smpl.is.tue.mpg.de/download.php'}
    if key in body_pages:
        return body_pages[key]
    if key in SAM_MANUAL_FILES:
        return model.url + '/blob/' + SAM_REV + '/' + SAM_MANUAL_FILES[key]
    if key == 'flow_vggt':
        return model.url + '/blob/main/vggt_omega_1b_512.pt'
    if model.drive_id:
        return 'https://drive.google.com/file/d/' + model.drive_id + '/view'
    if model.download_url:
        return (model.download_url.replace('/resolve/', '/blob/', 1)
                if model.download_url.startswith('https://huggingface.co/') else model.download_url)
    return model.url


def manual_instructions_html(key: str) -> str:
    """Local download guidance with official, user-opened web links."""
    from html import escape
    model = BY_KEY[key]
    fallback = ('Download the neutral body model from its official website, accept its terms, '
        'extract the archive, then use Import to select ' + model.path.name + '.'
        if key in {'smplx','smpl'} else
        'Download ' + model.path.name + ' using the source links below. '
        'Return to this row and choose Import to register the downloaded file. '
        'Only the supported model version will pass validation.')
    instructions = MANUAL_INSTRUCTIONS.get(key, fallback)
    links = [('Official website', model.url)]
    if key in {'smplx','smpl'}:
        links.append(('Download page (sign-in required)', manual_download_page(key)))
    if key in {'mamma','mamma_vertices'}:
        links.append(('MAMMA Data (registration and sign-in required)', 'https://mamma.is.tue.mpg.de/download.php'))
    if model.drive_id:
        links.append(('Download file: ' + model.path.name, 'https://drive.google.com/file/d/' + model.drive_id + '/view'))
    elif model.download_url:
        links.append(('Download file: ' + model.path.name, model.download_url))
    if key == 'smplh':
        links.insert(0,('Recommended AMASS download: smplh.tar.xz (sign-in required)',
            manual_download_page(key)))
    if key == 'mano_right':
        links.insert(0,('Recommended download: mano_v1_2.zip (sign-in required)',
            manual_download_page(key)))
    if key == 'flow_vggt':
        links.append(('Files: vggt_omega_1b_512.pt',
                      model.url + '/blob/main/vggt_omega_1b_512.pt'))
    if key in SAM_MANUAL_FILES:
        links.append(('Supported file: ' + SAM_MANUAL_FILES[key],
                      model.url + '/blob/' + SAM_REV + '/' + SAM_MANUAL_FILES[key]))
    details = MANUAL_DETAILS.get(key, {})
    body = '<p><b>File to import:</b> <code>' + escape(model.path.name) + '</code></p>'
    if details.get('selection'):
        body += '<p><b>Choose on the website:</b><br/>' + escape(details['selection']) + '</p>'
    body += '<p><b>Steps</b><br/>' + escape(instructions).replace('\n', '<br/>') + '</p>'
    if details.get('location'):
        body += '<p><b>File location after download / extraction:</b><br/><code>' + escape(details['location']) + '</code></p>'
    if details.get('avoid'):
        body += '<p><b>Do not select:</b><br/>' + escape(details['avoid']) + '</p>'
    return body + ''.join(
        '<p><a style="color:#84c9ff" href="' + escape(url, quote=True) + '">' +
        escape(title) + '</a></p>' for title, url in links)


STAGE_REQUIREMENTS = {
    'gemx_assets': GEMX_KEYS,
    'gemx_reconstruct': GEMX_KEYS,
    'prompthmr_camera_assets': PHMR_CAMERA_KEYS,
    'pear_assets': ('pear','pear_geometry','pear_yolo'),
    'pear_align': ('flow_vggt','pear_geocalib'),
    "people": ("yolo", "person_reid"),
    "track": ("yolo",), "pva": ("pva",),
    "features": ("hmr2", "vitpose"),
    "gvhmr": ("hmr2", "vitpose", "gvhmr", "smplx"),
    "refine": ("smplx",), "export": ("smplx",), "correct": ("smplx",),
    "footmr": ("footmr", "vitpose_wholebody"),
    'handflow': ('handflow','handflow_stats','hand_hamer','hand_config','hand_mean','hand_detector','mano_right','smplx'),
    'flow_camera': ('flow_vggt',),
    'flow_features': ('flow_sam', 'flow_sam_config', 'flow_mhr'),
    'flowhmr': ('flowhmr', 'flow_config', 'smplh'),
    'flow_export': ('smplh',),
    'exmr_preprocess': ('exmr_sapiens',),
    'exmr_features': ('flow_sam','flow_sam_config','flow_mhr','hand_hamer','hand_mean','mano_right'),
    'exmr': ('exmr','exmr_config','smplx'),
    'mamma_assets': MAMMA_KEYS,
    **{stage:PHMR_KEYS for stage in ('prompthmr_assets','prompthmr_prepare','prompthmr_track','prompthmr_pose','prompthmr_world','prompthmr_publish')},
    'mamma_prepare': MAMMA_KEYS,
    'mamma_masks': ('mamma_sam2','mamma_yolo','mamma_clip'),
    'mamma_landmarks': ('mamma','mamma_vertices'),
    'mamma_camera': ('flow_vggt','pear_geocalib'),
    'mamma_fit': ('mamma_vertices','mamma_body','mamma_body_male','mamma_body_female'),
    'mamma_publish': ('mamma_body','mamma_vertices'),
}


def ready(model: Model) -> bool:
    if model.key=='pear_geometry':
        from .pear_models import geometry_ready
        return geometry_ready()
    valid=model.path.is_file() and model.path.stat().st_size >= model.minimum_bytes
    if valid and (model.key.startswith(('flow','hand','exmr','mamma','phmr','pear','gemx')) or model.key in ('smplh','mano_right')):
        import json
        try:
            receipt=json.loads(model.path.with_suffix(model.path.suffix+'.receipt.json').read_text(encoding='utf-8'))
            stat=model.path.stat()
            valid=receipt.get('stat')==[stat.st_size, stat.st_mtime_ns] and (not model.expected_sha256 or receipt.get('sha256')==model.expected_sha256)
        except (OSError,ValueError):valid=False
    return valid


def missing(stage: str,tool=None) -> list[Model]:
    if tool=='flowhmr' and stage in ('correct','export','rig','mesh'):stage='flow_export'
    return [BY_KEY[key] for key in STAGE_REQUIREMENTS.get(stage, ()) if not ready(BY_KEY[key])]


def download(key: str, progress=None) -> Path:
    if key in GEMX_KEYS:
        from .gemx_models import download_model
        return download_model(key, progress)
    import gdown
    if key == 'flowhmr':
        # Hugging Face checkpoint metadata uses the same pinned revision. It is
        # not the inference YAML and must not invalidate existing motion caches.
        download('flow_metadata', progress)
    model = BY_KEY[key]
    if not model.drive_id and not model.download_url:
        raise ValueError("Obtain this model from its official website, then import it.")
    if ready(model):
        return model.path
    model.path.parent.mkdir(parents=True, exist_ok=True)
    part = model.path.with_suffix(model.path.suffix + ".part")
    source = model.url
    def receive(response):
        import time
        try:total=int(getattr(response,'headers',{}).get('Content-Length',0))
        except (TypeError,ValueError):total=0
        current=0;last=time.monotonic()
        if progress:progress('progress',current=0,total=total,message='Downloading '+model.name,unit='B')
        with part.open('wb') as output:
            for chunk in response.iter_content(8*1024*1024):
                output.write(chunk);current+=len(chunk)
                now=time.monotonic()
                if progress and (now-last>=.25 or current==total):
                    progress('progress',current=current,total=total,message='Downloading '+model.name,unit='B');last=now
    try:
        if model.download_url:
            import requests
            source = model.download_url
            with requests.get(source, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                receive(response)
            result = str(part)
        else:
            result = gdown.download(id=model.drive_id, output=str(part), quiet=False, resume=True)
    except gdown.exceptions.FileURLRetrievalError as error:
        if key == "pva":
            detail = str(error).lower()
            reason = ("Google Drive's public download limit was reached" if any(
                word in detail for word in ("too many users", "quota", "24 hours"))
                else "Google Drive could not provide the public file")
            raise RuntimeError(reason + ": PVA-Net (pvanet.pt). "
                "Completed downloads are retained. Use "
                "Manual Download / Import in the PVA-Net row to import an official local copy, "
                "or retry Download All later. The app does not sign in.") from None
        if key not in {"gvhmr", "hmr2", "vitpose"}:
            raise
        # Same public mirror used by the official README-linked demo.
        import requests
        source = "https://huggingface.co/camenduru/GVHMR/resolve/main/" + model.relative.replace("=", "%3D")
        with requests.get(source, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            receive(response)
        result = str(part)
    if result is None or part.stat().st_size < model.minimum_bytes:
        raise RuntimeError(f"Model download did not complete: {model.name}")
    if progress:progress('progress',current=0,total=0,message='Verifying '+model.name)
    checksum = sha256(part)
    if model.expected_sha256 and checksum != model.expected_sha256:
        raise RuntimeError("The downloaded model checksum does not match: " + model.name)
    if key == "gvhmr" and checksum != "4fae7da2de388d5da3514cb27a2d003f364dacb280e9cf88972b710e589c6b91":
        raise RuntimeError("The GVHMR model file hash does not match.")
    part.replace(model.path)
    atomic_json(model.path.with_suffix(model.path.suffix + ".receipt.json"), {
        "source": source, "drive_id": model.drive_id, "bytes": model.path.stat().st_size,
        "sha256": checksum,
        'stat': [model.path.stat().st_size, model.path.stat().st_mtime_ns],
    })
    return model.path


def import_model_file(key: str, source: Path) -> Path:
    """Worker-only validated, atomic registration; never unpickles imports."""
    import os, uuid
    if key=='pear_geometry':
        from .pear_models import import_bundle
        return import_bundle(source)
    model=BY_KEY[key]; source=Path(source).resolve()
    if source.stat().st_size < model.minimum_bytes:
        raise ValueError('The file is too small for '+model.name)
    if key=='smplh':
        import numpy as np, zipfile
        with zipfile.ZipFile(source) as archive:
            if sum(item.file_size for item in archive.infolist()) > 256*1024**2:
                raise ValueError('Select the 16-shape AMASS neutral SMPL-H model, not a 300-shape model.')
        with np.load(source,allow_pickle=False) as data:
            shapes={'v_template':(6890,3),'shapedirs':(6890,3,16),'weights':(6890,52),'J_regressor':(52,6890),'kintree_table':(2,52),'f':(13776,3)}
            for name,shape in shapes.items():
                if name not in data or data[name].shape!=shape or not np.isfinite(data[name]).all():
                    raise ValueError('Select neutral/model.npz from the AMASS SMPL-H archive (16 shape components).')
    elif key not in {'smplx','smpl'} and not model.expected_sha256:
        raise ValueError('This model cannot be imported.')
    elif key in {'smplx','smpl'}:
        if source.name.lower() != model.path.name.lower():
            raise ValueError('Select ' + model.path.name + ' for ' + model.name + '.')
        return import_body_model(source)
    model.path.parent.mkdir(parents=True,exist_ok=True)
    temporary=model.path.with_name(model.path.name+'.'+uuid.uuid4().hex+'.part')
    try:
        # Validate the bytes actually copied; a changing source cannot publish.
        import hashlib
        checksum=hashlib.sha256()
        with source.open('rb') as incoming, temporary.open('wb') as output:
            for chunk in iter(lambda:incoming.read(8*1024**2),b''):
                output.write(chunk);checksum.update(chunk)
        digest=checksum.hexdigest()
        if model.expected_sha256 and digest!=model.expected_sha256:
            raise ValueError('This file does not match the supported official '+model.name+'. Check the filename and download version.')
        if key=='smplh':
            # Validate the stable copy again if the source was replaced meanwhile.
            with np.load(temporary,allow_pickle=False) as data:
                for name,shape in shapes.items():
                    if data[name].shape!=shape:raise ValueError('Body model changed during import.')
        os.replace(temporary,model.path)
        atomic_json(model.path.with_suffix(model.path.suffix+'.receipt.json'),dict(source=str(source),sha256=digest,stat=[model.path.stat().st_size,model.path.stat().st_mtime_ns],manual_import=True))
        license_file=source.parent/'LICENSE.txt'
        if license_file.exists() and license_file.resolve()!=(model.path.parent/'LICENSE.txt').resolve():shutil.copy2(license_file,model.path.parent/'LICENSE.txt')
    finally:temporary.unlink(missing_ok=True)
    return model.path


def import_body_model(source: Path) -> Path:
    candidates = [model for model in MODELS_LIST if model.key in {"smplx", "smpl"} and model.path.name.lower() == source.name.lower()]
    if not candidates:
        raise ValueError("Select SMPLX_NEUTRAL.npz or SMPL_NEUTRAL.pkl.")
    model = candidates[0]
    if source.stat().st_size < model.minimum_bytes:
        raise ValueError("The file is too small. Select an extracted body model file.")
    if model.key == "smplx":
        import numpy as np
        with np.load(source, allow_pickle=False) as data:
            required = {"v_template", "shapedirs", "posedirs", "kintree_table"}
            if not required.issubset(data.files):
                raise ValueError("The selected file does not match the required SMPL-X model format.")
    model.path.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != model.path.resolve():
        shutil.copy2(source, model.path)
    return model.path
