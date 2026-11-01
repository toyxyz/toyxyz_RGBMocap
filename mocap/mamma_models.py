"""MAMMA assets, separate from the three single-view body-model families."""

ASSETS = (
    dict(key='mamma', name='Multi-view Motion / MAMMA', relative='mamma/mamma_mask_full_cvpr.ckpt',
         url='https://mamma.is.tue.mpg.de/', minimum_bytes=1621536317,
         expected_sha256='3fd6793cf72f9994eded544d902a30b2cd5e9e5d225c1be110980ee4cadd8b29'),
    dict(key='mamma_vertices', name='MAMMA Surface Landmarks / verts_512', relative='mamma/verts_512.pkl',
         url='https://mamma.is.tue.mpg.de/', minimum_bytes=21453208,
         expected_sha256='85c6acd5618a2b93d96640285a6f3e1e02383a046140d882663e6de5a6185c82'),
    dict(key='mamma_body', name='MAMMA Body / Locked-head SMPL-X Neutral', relative='mamma/smplx/SMPLX_NEUTRAL.npz',
         url='https://smpl-x.is.tue.mpg.de/', minimum_bytes=137106406,
         expected_sha256='43d8f3a1375d7c5baae207870a5d51def0f7e6b507df709b4937598b5e7d965d'),
    dict(key='mamma_sam2', name='MAMMA Masks / SAM 2.1 Hiera Large', relative='mamma/sam2.1_hiera_large.pt',
         url='https://github.com/facebookresearch/sam2', minimum_bytes=898083611,
         download_url='https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt',
         expected_sha256='2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318'),
    dict(key='mamma_yolo', name='MAMMA Person Detection / YOLO12x', relative='mamma/yolo12x.pt',
         url='https://github.com/ultralytics/ultralytics', minimum_bytes=119322638,
         download_url='https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12x.pt',
         expected_sha256='682ce8dadee004dbe964950f1bf3eda451671815a6ed62db80b398916b9b7c6f'),
    dict(key='mamma_clip', name='MAMMA View Matching / OpenCLIP ViT-B-32', relative='mamma/open_clip_model.safetensors',
         url='https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K', minimum_bytes=605143316,
         download_url='https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/1a25a446712ba5ee05982a381eed697ef9b435cf/open_clip_model.safetensors',
         expected_sha256='ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6'),
    dict(key='mamma_body_male', name='MAMMA Body Initialization / Locked-head Male', relative='mamma/smplx/SMPLX_MALE.npz',
         url='https://smpl-x.is.tue.mpg.de/', minimum_bytes=137147214,
         expected_sha256='44e57050d127538e20e10e27f8274ba9ca3dd09c992e954d3d2b5ae5e5bb8313'),
    dict(key='mamma_body_female', name='MAMMA Body Initialization / Locked-head Female', relative='mamma/smplx/SMPLX_FEMALE.npz',
         url='https://smpl-x.is.tue.mpg.de/', minimum_bytes=137193916,
         expected_sha256='a32f91b24e13023b979ec0b5c8e9086f5918197132329c8a258ad1f1c6c557f6'),
)
KEYS = tuple(entry['key'] for entry in ASSETS)+('flow_vggt','pear_geocalib')

INSTRUCTIONS = {
    'mamma': '1. Open the official MAMMA website below and click Data.\n2. Register for an account if needed, review and accept the provider terms, then sign in.\n3. On the download page, download mamma_mask_full_cvpr.ckpt.\n4. Return to this window, click Import and select the single CKPT file.\nThe app does not request your password or account token.',
    'mamma_vertices': '1. Open the official MAMMA website below and click Data.\n2. Register for an account if needed, review and accept the provider terms, then sign in.\n3. On the download page, download verts_512.pkl.\n4. Return to this window, click Import and select the PKL file intact.\nThis is registered surface correspondence data, not an ordinary SMPL-X body model. Import checks its supported SHA-256 without executing the file.',
    'mamma_body': 'Register at the SMPL-X website and accept its terms. Download "SMPL-X with removed head bun (NPZ, 392 MB)". Extract it and import the locked-head SMPLX_NEUTRAL.npz. The standard SMPL-X v1.1 model used by GVHMR/ExMR is different and cannot replace this file. MAMMA data and source code have separate non-commercial and redistribution restrictions.',
}
for gender in ('male', 'female'):
    INSTRUCTIONS['mamma_body_'+gender] = INSTRUCTIONS['mamma_body'].replace('SMPLX_NEUTRAL.npz', 'SMPLX_'+gender.upper()+'.npz')
