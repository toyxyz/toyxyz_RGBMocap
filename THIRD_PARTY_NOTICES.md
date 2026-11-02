# Third-party notices

Original RGB MoCap code and documentation use the root [MIT License](LICENSE).
That grant does not relicense third-party code, upstream-derived modifications,
model weights or body data. This is a short guide to the major components;
retained license files and provider agreements contain the full terms.

## AI tools and models

Character import, baking and portable scene export use a separate, unmodified Blender 4.5 LTS runtime
(GPL; bundled license and source information in `runtime_character`). The scoped
FBX sampler in `mocap/retarget/selective_fbx.py` retains Blender's GPL terms.
The VRM add-on, mmd_tools and OpenCC retain their own notices under
`third_party/character_import` (VRM_LICENSE.txt, VRM_MIT.txt, VRM_NOTICE.txt,
MMD_LICENSE.txt, OPENCC_LICENSE.txt). Imported character files and textures remain the user's assets;
they are not bundled with the application.

| Component | License / essential conditions |
| --- | --- |
| [GVHMR](https://github.com/zju3dv/GVHMR) | Educational, research and nonprofit use only; derived changes must be open-source and noncommercial. [License](third_party/GVHMR/LICENSE). |
| [ExMR](https://github.com/twehrbein/ExMR) / [FootMR](https://github.com/twehrbein/FootMR) | GVHMR-derived educational, research and nonprofit terms. [ExMR license](third_party/ExMR/LICENSE), [FootMR license](third_party/FootMR/LICENSE). |
| [FlowHMR](https://github.com/flowhmr/flowhmr) | Apache-2.0 source. [License](third_party/FlowHMR/LICENSE). Supporting models have separate terms below. |
| [PromptHMR](https://github.com/yufu-wang/PromptHMR) (optional single-video runtime) | [Non-commercial scientific research license](https://github.com/yufu-wang/PromptHMR/blob/main/LICENSE). Restricted source is downloaded into the user's separate runtime, not bundled in this repository. Weights, SMPL data and supporting checkpoints retain their own terms. SAM2/Detectron2/ViTPose and runtime dependencies retain their notices. See [setup and use](docs/models/PROMPTHMR.md). |
| [HTD-Refine](https://github.com/ant-research/HTD-Refine) | AGPL-3.0-only. [License](third_party/HTD-Refine/LICENSE). PVA-Net/checkpoints retain their applicable model terms. |
| [HandFlow](https://github.com/mxxu00/HandFlow) / [HaMeR](https://github.com/geopavlakos/hamer) | MIT source, with retained notices for upstream-derived parts. [HandFlow license](third_party/HandFlow/LICENSE), [HaMeR license](third_party/HaMeR/LICENSE.md). Weights and WiLoR detector files retain provider terms. |
| [Sapiens2](https://github.com/facebookresearch/sapiens2) | Sapiens2 License covers its software and models. [License](third_party/ExMR/third_party/sapiens2/LICENSE.md). |
| [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) / MHR assets | SAM source uses the [SAM License](third_party/FlowHMR/third_party/sam-3d-body/LICENSE). Feature weights and MHR geometry retain provider terms; obtain access and import supported assets locally. |
| [DINOv3](https://github.com/facebookresearch/dinov3) | Separate [DINOv3 License](third_party/FlowHMR/third_party/dinov3/LICENSE.md) for its code/models. |
| [Fast SAM 3D Body](https://github.com/yangtiming/Fast-SAM-3D-Body) | Fork contributions: [MIT](third_party/FlowHMR/third_party/fast-sam-3d-body/LICENSE); upstream-derived code: [SAM License](third_party/FlowHMR/third_party/fast-sam-3d-body/SAM_LICENSE). Uses the same separately licensed model assets. |
| [MAMMA](https://mamma.is.tue.mpg.de/) (optional multi-view runtime) | Registered noncommercial/restricted-use software and weights, with separate locked-head SMPL-X body-data terms. Obtain the code and assets directly under the provider agreement; neither is bundled in this repository. The original app connector does not relicense them. Restricted MAMMA and pytorch_sdf source/native modules are fetched and built on the user's PC, not bundled. SAM 2 uses Apache-2.0; its weights and OpenCLIP weights retain their provider terms. See [MAMMA setup](docs/MAMMA.md). |
| [VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega) | [FAIR Noncommercial Research License](third_party/FlowHMR/third_party/vggt-omega/LICENSE). Official checkpoint requires provider access and local import. |
| [SMPL / SMPL-X](https://smpl-x.is.tue.mpg.de/modellicense.html), [SMPL-H / MANO](https://mano.is.tue.mpg.de/license.html) | Registered body/hand models and model-derived data retain their provider restrictions. The installed smplx software has separate noncommercial Model & Software terms. Obtain the required files under your own agreement; registration does not authorize public redistribution. |
| [Kimodo MotionCorrection](https://github.com/nv-tlabs/kimodo) | [Apache-2.0](third_party/KimodoMotionCorrection/LICENSE). The correction backend is used; Kimodo motion generation is not included. |
| [Ultralytics YOLO](https://www.ultralytics.com/license) | AGPL-3.0, including pretrained models by default, or a separate applicable provider agreement. |
| [HMR2](https://github.com/shubham-goel/4D-Humans), [ViTPose](https://github.com/ViTAE-Transformer/ViTPose), [OSNet](https://github.com/KaiyangZhou/deep-person-reid) | MIT, Apache-2.0 and MIT source respectively, subject to retained component notices. [OSNet license](third_party/OSNet/LICENSE). Checkpoint conditions remain separate. |

## GUI and video runtime

| Component | License / essential conditions |
| --- | --- |
| [Qt for Python / PySide6](https://doc.qt.io/qtforpython-6/licenses.html) | LGPLv3/GPLv3/commercial routes as applicable to the included components. Retain notices and satisfy the applicable replacement/relinking requirements. |
| [FFmpeg](https://ffmpeg.org/legal.html) / [PyAV](https://github.com/PyAV-Org/PyAV) | PyAV's wrapper is BSD. This build's FFmpeg executables and PyAV FFmpeg libraries identify as GPLv3+; applicable source and redistribution obligations remain. |

Other runtime packages retain their included licenses. Supporting notices include
[PyTorch3D transforms (BSD)](native_compat/LICENSE),
[Noto Sans KR (SIL OFL)](assets/fonts/OFL.txt), and
[Eigen / pybind11 notices](third_party/KimodoMotionCorrection/notices).

## Attribution and local adaptations

- GVHMR: copyright 2022–2023, 3D Vision Group, Zhejiang University.
  ExMR / FootMR: copyright 2025–2026, Leibniz University Hannover, with retained
  GVHMR notices. Follow the upstream publication/citation instructions, including
  Wehrbein and Rosenhahn (ExMR, arXiv:2610.26213; FootMR, 3DV 2026),
  and the original GVHMR work. [Papers and BibTeX](docs/CITATIONS.md) cover
  the supported motion models, PEAR live preview and optional refinement models.
- Kimodo MotionCorrection: copyright 2026, NVIDIA Corporation & Affiliates.
  Local changes add explicit contact targets, bounded solving and anatomical guards.
- OSNet: Kaiyang Zhou; original MIT notices and model-zoo provenance are retained.
- Windows adapters add model import, selected-person processing, bounded workers,
  shared-scene placement, preview/export and compatibility fixes. Upstream-derived
  portions keep their original terms, including the SAM-derived portions of
  `mocap/flow_sam.py`. Checkpoints and original inference outputs are kept separate
  from contact correction. Source revisions are recorded in retained
  `RGB_MOCAP_PROVENANCE.json` files and compatibility patches in
  `tools/prepare_upstream.py`.

Code-license labels do not establish permission for every weight or training
dataset. AGPL obligations, noncommercial restrictions and model redistribution
permissions are not waived by separate worker processes or separate labeling.
Keep original notices and any required corresponding source/build materials when
distributing. The private development folder may contain registered models,
derived caches and user projects; it is not a cleared public release package.

## Optional PEAR live support

[PEAR](https://github.com/Pixel-Talk/PEAR) source uses Apache-2.0. Setup fetches a
pinned upstream revision and preserves its license with the locally adapted
source. The connector omits unused training/render imports, uses native Torch
constructor helpers and reuses the shared runtime. PyTorch3D pure-Python utilities
retain their upstream BSD license/notices; RoMa retains its package license.
YOLOv8s/Ultralytics retain their existing AGPL/provider terms. PEAR weights,
SMPL-X 2020 and FLAME geometry retain separate provider conditions; registered
body data is neither committed nor redistributed by RGB MoCap. See the
[PEAR guide](docs/models/PEAR.md) and [PEAR paper citation](docs/CITATIONS.md#pear).


PEAR's automatic Camera Align and MAMMA's automatic pre-fitting gravity use [GeoCalib](https://github.com/cvg/GeoCalib)
(Apache-2.0), with original license retained beside the downloaded source;
Kornia/Kornia-rs keep their wheel licenses. Setup pins sources/wheels in
`installer/pear-alignment.json` and installs a separate support overlay over the
existing Flow runtime. VGGT retains its existing upstream/model terms; its weight
file is reused, not bundled again. GeoCalib's public checkpoint is obtained
separately through Model Manager under CC BY 4.0; the upstream
LICENSE-WEIGHTS is retained with support files. No TensorRT engine/runtime is distributed.
