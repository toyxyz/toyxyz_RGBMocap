# PromptHMR

[Model index](README.md) · [User guide](../USER_GUIDE.md) ·
[Official project](https://yufu-wang.github.io/phmr-page/) ·
[Official code](https://github.com/yufu-wang/PromptHMR)

PromptHMR reconstructs multiple people together from a **single video**. In our
subjective tests, it could reproduce relative positions and interactions more
accurately than independently reconstructed actors, but its **overall motion
quality was lower than ExMR**. It **does not support finger tracking**; the app
uses a fixed neutral finger pose. These observations are not a universal accuracy
ranking. Depth errors, sliding, identity changes and body intersections remain
possible. Interaction Prompt does not create a contact lock.

## Single-video interaction example

The input frame shows a kicking foot contacting the other person's torso.
The supplied reconstructions illustrate how the models place the two people
relative to each other.

**Input video frame**

![Input frame showing a kick to the other person's torso](../images/interaction-contact-input.png)

| PromptHMR | ExMR |
| --- | --- |
| ![PromptHMR reconstruction of foot-to-torso contact](../images/prompthmr-interaction-contact.png) | ![ExMR reconstruction with different foot-to-torso placement](../images/exmr-interaction-contact.png) |

In this example, PromptHMR more closely reproduces the foot-to-torso contact
visible in the input, while ExMR leaves a different relative placement at the
contact point. This is a qualitative example, not a measured contact-accuracy
benchmark. The viewport angles differ, and the images do not establish an exact
3D contact distance or guarantee similar results on other footage.

## Install the separate runtime

Keep PromptHMR separate from the base installation. Its package versions and
local SAM2 CUDA build need their own environment, compiler tools and license
review. A PromptHMR setup failure must not prevent the base models from working.
`install.cmd` prepares ExMR/FlowHMR/GVHMR; it does not install PromptHMR or MAMMA.

1. Complete `install.cmd` for the base app.
2. Install [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/)
   with **Desktop development with C++** and the Windows SDK, plus
   [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive).
   The NVIDIA driver alone does not include the compiler.
3. Close RGB MoCap and run `Setup_PromptHMR.cmd`. Review the provider license
   prompt. In Command Prompt, run `Setup_PromptHMR.cmd` from the app folder
   (PowerShell: `.\Setup_PromptHMR.cmd`). Setup downloads pinned sources/packages,
   builds SAM2 for the installed
   GPU and verifies real CUDA operations before registering the environment.
4. For moving footage, choose **Y** at the additional moving-camera setup prompt. This builds DroidCalib/LieTorch for the installed GPU and verifies CUDA operations. The existing PromptHMR Python/Torch environment is reused; fixed-camera support remains available if the optional camera build fails. Review the linked DroidCalib AGPL and dependency terms before choosing Y.
5. Reopen the app, select **PromptHMR**, and prepare the required files in **Model Manager**. The two rows named **Moving Camera** are optional for Fixed Camera.
   Google Drive may require a manual browser download; use **Manual Download... >
   Import** if automatic downloading fails. **Runtime Setup...** shows instructions
   and copies the command; it does not install anything inside the running app.
   A runtime already marked Ready does not need reinstalling before each use.

The setup report measured **6.32 GiB** on the test PC, excluding weights and
compiler tools. The later active-runtime inventory excluding caches measured
**6.22 GiB**. Package downloads total about **3.55 GiB**. Allow approximately
**18 GiB free during setup**, excluding models and compiler tools. Retained
download archives and project intermediates need additional space.

PromptHMR-specific models/support files total approximately **6.09 GiB**,
excluding the registered body files below. Standard SMPL-X Neutral is shared
with ExMR/GVHMR. SMPL Neutral is also required by PromptHMR; it is optional
for the three base workflows. Both live in `models/body_models/`. Import those licensed files
if absent; they are not bundled. MAMMA's locked-head body cannot substitute.
Adding PromptHMR to a complete base installation measured **6.13 GiB** of model
files including SMPL Neutral, plus **6.22 GiB** of cache-free runtime, for about
**12.34 GiB additional**. See the [all-model inventory](../USER_GUIDE.md#installed-size).
PromptHMR's ViTPose COCO-25, SAM2 tiny and detector checkpoints differ from
similarly named files in other workflows. Imports verify the supported hashes.

Select **PromptHMR > Model Manager** for the 14 base model/body files and two
optional Moving Camera rows. Locations below are relative to `models/`; obtain
the exact files listed by the manager.

| Component | Files or folder |
| --- | --- |
| Image model and configuration | `prompthmr/pretrain/phmr/checkpoint.ckpt`, `config.yaml` |
| Video model and configuration | `prompthmr/pretrain/phmr_vid/prhmr_release_002.ckpt`, `prhmr_release_002.yaml` |
| Tracking detector / masks | `prompthmr/pretrain/keypoint_rcnn_5ad38f.pkl`, `sam2_hiera_tiny.pt` |
| Keypoints / camera calibration | `prompthmr/pretrain/vitpose-h-coco_25.pth`, `camcalib_sa_biased_l2.ckpt` |
| Body correspondence / means / regressors | Four files under `prompthmr/body_models/`, listed individually in Model Manager |
| Registered standard bodies | `body_models/smplx/SMPLX_NEUTRAL.npz`, `body_models/smpl/SMPL_NEUTRAL.pkl` |

### Shared supporting-model download cache

The upstream tracking backend also uses torchvision DeepLab segmentation
weights (about 161 MiB) and Torch Hub source files. These supplementary downloads
are stored once under `cache/prompthmr/torch/hub/` and reused across videos and
reconstruction attempts. Hugging Face downloads use `cache/prompthmr/hf/`.
Readiness counts 14 base files in Fixed Camera and 16 files in Moving Camera;
these supplementary caches are not part of either count.
The first use of an empty cache may require internet access.

The connector first checks known old project `torch-cache/hub/checkpoints`
locations and imports the supported DeepLab weights only after full SHA-256
verification. It hardlinks when possible, otherwise copies once. Existing
projects and their caches are preserved; downloaded Python Hub source trees
are not imported from old projects. The Console/CMD prints the shared cache path
and successful weight reuse. An invalid shared DeepLab file produces a checksum
error instead of being loaded silently.

After confirming a successful updated run and closing the app, old
`<project>/prompthmr/inference/<generation>/torch-cache/` folders may be removed
to reclaim duplicate downloads. Keep the other inference/result files. Deleting
the new shared cache is also allowed, but its downloads will be needed again.
This cache reuse needs no runtime reinstall. Moving-camera inference has a
separate optional native setup and two managed weights, described below.

On one volume, the connector hardlinks verified weights into the private engine
without storing another copy. If hardlinks are unavailable, it reports the copy
and uses extra disk space. Installed Python packages remain separate.

This profile uses Python **3.11.17**, Torch **2.7.1+cu128**, torchvision
**0.22.1+cu128**, NumPy **1.26.4** and timm **0.9.0**. Torch overlaps with other
profiles, but Python ABI/package requirements differ. Installed packages remain
isolated for update safety; identical verified MAMMA download archives can be
reused. The current locks share 147 identical package archives, about 3.39 GiB
of downloads. This does not mean installed packages can safely be merged.
MAMMA and SDF are not required. Do not combine these environments.

`install.cmd`/`update.cmd` leave PromptHMR optional. A changed PromptHMR runtime
lock prompts for `Setup_PromptHMR.cmd`; other runtimes/projects are preserved.

### Update or repair

Close RGB MoCap before `update.cmd`. After updating, rerun `Setup_PromptHMR.cmd`
only when its runtime is missing, needs repair or the app reports changed
dependencies. It verifies and retains a matching environment, or prepares a new
generation before switching to it. Keep models, projects and `.setup` state.

If setup fails, read the last CMD error and the corresponding
`.setup/logs/prompthmr-*.log`. Check free space and the documented C++/CUDA tools;
then retry the same setup. If weights are Missing, use Model Manager instead
of rebuilding the runtime. If native GPU verification fails after moving to
another GPU architecture, run setup on that PC; do not copy compiled modules
from an arbitrary machine or edit registration files to mark them Ready.

## Moving-camera setup and behavior

Close the app and rerun `Setup_PromptHMR.cmd`, choosing **Y** for moving-camera
support. The base runtime is verified and retained. A separate, versioned native
support folder is built over its pinned Python 3.11 / Torch 2.7.1 CUDA 12.8;
other model runtimes are not changed. No Linux, WSL or Docker is required.
The verified installed camera extension measured **18.6 MiB** on the test PC,
excluding build archives and weights.
Use Visual Studio 2022 C++ Build Tools / Windows SDK and CUDA Toolkit 12.8.
Allow an additional **6 GiB free for building**, excluding models and compiler
tools. Downloaded build archives are retained separately from installed modules.

Prepare **DroidCalib** (16,049,525 bytes) and **Metric3D ViT-Large**
(1,647,972,663 bytes) in Model Manager. Together they add **1.55 GiB**
(1.66 GB) of weights, separate from the 14 base model/body files. Imports and
downloads use the declared SHA-256 hashes. Tracking masks, PromptHMR weights,
SPEC gravity and registered bodies are reused. Metric3D reads the verified file
directly; it does not download another copy into each video generation.

Select Moving Camera and **Reconstruct 3D**. The official masked DroidCalib
path estimates background camera poses and refines a constant focal length;
Metric3D provides an estimated metric scale. Native world fitting and
post-optimization use this trajectory, preserving the joint multi-person scene.
The usual optional app correction is applied afterward. The saved track also
drives Camera Preview, source-video bone overlays, viewport camera gizmos and
portable Blender camera animation. Display controls do not rerun inference.

Moving-camera results have separate inference identities, including camera
modules and model stamps. Changing the mode requires reconstruction; the previous
result stays available until replacement succeeds. Missing/modified support or
failed tracking produces an error rather than silently substituting a fixed
camera. These are estimated poses/scale, not measured camera calibration. Foot
height, drift, missed masks and occlusion can still affect the motion.

Windows adaptations retain the SLAM equations and full frame sequence, use
64-bit tensor indices, avoid CPU/CUDA kernel-name collisions, and use native
Torch SDPA for Metric3D attention. Depth batches are one image in both resource
modes; the keyframe buffer covers the selected clip plus trajectory-fill slots.
RAM/VRAM limits follow [Resource Safety](../PERFORMANCE.md#resource-safety); actual OOM handling remains active; long trims can need much more
memory. Additional PromptHMR block offloading is not implemented.

Camera setup logs: `.setup/logs/prompthmr-camera-*.log`.
Registration: `.setup/prompthmr-camera-runtime.json`. Never edit it to bypass a
failed build. Rerun setup on a new GPU architecture. Providers' source/licenses
are downloaded locally; compiled camera modules and weights are not in Git.

## Controls and supported workflow

- **One video**. **Camera > Fixed Camera** is the default and preserves the previous workflow. **Moving Camera / DroidCalib + Metric3D** tracks camera rotation and translation from the background. It requires the optional camera build and two additional models. Continuous focal-length changes/zoom, shot cuts, blur and poor background texture can break tracking or scale.
- **People** defaults to **2**. **0** detects automatically, up to eight. This is
  a maximum count, not an identity selector. The intended people should be
  visible near the start; background people and occlusion can confuse tracking.
- **Interaction Prompt** defaults on, asking the image model to consider people
  together. Off follows the official default video path. On was not uniformly
  better in our comparison. Changes apply on reconstruction.
- Click **Reconstruct 3D**. PromptHMR performs SAM2 tracking itself; **Find person**
  is hidden. Image/video inference, gravity/world placement and native
  post-optimization are retained. The app does not independently space or
  rescale actors. Progress/activity and cancellation use the normal worker UI.
- SMPL-X body motion is generated. The official video settings used here
  **do not generate animated fingers**. Missing zero-matrix hand placeholders
  are replaced with the registered body's fixed relaxed hand pose. Hand contact
  correction is not finger inference.
- Check People rows to show/hide actors and select a row to edit contacts.
  **Enable Correction** switches original/corrected motion. **Apply Correction**
  does not rerun PromptHMR and refreshes linked characters. Correction may
  intentionally change poses. Export through **File > Export...** (FBX, BVH or Blender), or retarget
  to an imported character.

The working video is sampled at 30 fps. The current adapter accepts **4–1,800
working frames** and **up to eight people**. Available RAM/VRAM may require less.
These are app limits, not claimed upstream limits. Long unobserved absences are
not displayed as stationary poses or replaced with another actor.

Both Automatic and Low Memory keep the validated **one-image batch**, model
precision, full temporal context and native post-optimization. Additional
PromptHMR block offloading is not implemented. RAM/Windows headroom stopping
and CUDA allocator budgets follow [Resource Safety](../PERFORMANCE.md#resource-safety),
default on. Actual allocation-error handling remains active in either setting. Low Memory is not a promise of a smaller PromptHMR footprint; trim or
close other GPU applications when needed.

Windows adapters retain strict checkpoint loading, avoid discarded initialization
downloads, and replace a failing xFormers decoder kernel with PyTorch SDPA.
Verification on one RTX 5090 PC is not full Linux-output equivalence or
other-GPU validation. Setup tests SAM2 connected components, torchvision CUDA
NMS and native attention on the user's GPU.

## Recorded validation

A moving-camera test used 80 frames at 30 fps (2.67 seconds), one person and
the existing RTX 5090 test PC. The installed Windows path completed preparation,
fresh tracking, DroidCalib/Metric3D, motion inference, native world optimization
and app correction/export preparation in **130.3 seconds** including process
startup. Stage wall times were 1.4 / 33.2 / 57.2 / 31.0 / 7.4 seconds.
Peak PyTorch allocated VRAM was **6.73 GiB**, reserved **6.93 GiB**; sampled
worker-tree RAM peaked at **7.42 GiB**. Other GPU allocations can add to that.
This short sample is not a long-video capacity or camera-accuracy benchmark.
The saved 80-frame camera track loaded in the app and exported as Blender
camera animation with synchronized relative video. A two-person fixed-camera
regression and reuse of an older fixed-camera inference receipt also passed.

A 422-frame, 14.1-second, two-person fixed-camera clip completed in **3 min 56 sec**
on the documented RTX 5090 test PC, with Interaction Prompt and app correction
enabled. Preparation took 4.8 s; tracking 73.5 s; camera/keypoints/motion 103.0 s;
world/post-optimization 33.0 s; correction/preview/bone preparation 21.9 s.
Peak worker working set was **8.74 GiB**. Peak PyTorch allocated VRAM was
**4.61 GiB**, with **7.83 GiB reserved**; total GPU use can be higher. These are
one-run observations, not minimum requirements or a direct speed ranking against
other models. Public FBX writing was checked separately.

Original/corrected motion for both actors was exported and imported into Blender
as 52-bone rigs with bone-only animation and separate rest transforms. Save As,
reopening and Mixamo character retargeting passed. Native playback had no sampled
video/timeline frame mismatch during a short 12-second check; frame-swap intervals
were 31 ms median, 47 ms at the 95th percentile and 47 ms maximum. Three repeated
opens kept handle/thread counts stable. This short check does not establish
universal playback performance or leak freedom. Fresh two-second inference/export
smoke tests also passed for ExMR, FlowHMR and GVHMR. See [validation](../VALIDATION.md).

### Refreshing older finger poses

Earlier results could contain a fixed, severely curled hand: the upstream video
path filled missing finger rotations with zero matrices, which became 120-degree
rotations during axis-angle conversion. The app now treats these as missing
hands and uses a fixed registered-neutral pose. Native inference files remain
unchanged; this is not newly inferred finger animation.

For an existing result showing the legacy notice, keep the original video,
models and settings, then **Reconstruct 3D**. A verified native inference cache
is reused and the app republishes motion, correction and export previews; a
missing or changed cache can require inference. **Prepare Preview** alone does
not replace already published hand parameters. No runtime reinstall is needed.

Two short two-person clips (32 and 85 frames) were checked. Body/root/shape and
corrected-body tensors remained exact, body/wrist joint positions were unchanged,
and four Blender FBX imports retained fixed local finger rotations.

## Files and agent/developer guide

| Item | Location |
| --- | --- |
| Weights | `models/prompthmr/`, shared `models/body_models/` |
| Runtime | `runtime_prompthmr/<generation>/` |
| Runtime registration | `.setup/prompthmr-runtime.json` |
| Optional camera extension / registration | `runtime_prompthmr/camera-<generation>/`, `.setup/prompthmr-camera-runtime.json` |
| Shared supporting downloads | `cache/prompthmr/torch/hub/`, `cache/prompthmr/hf/` |
| Setup logs / retained archives | `.setup/logs/prompthmr-*.log`, `.setup/prompthmr-downloads/` |
| Native inference | `<project>/prompthmr/inference/<generation>/results.pkl` |
| Published actors/takes | `<project>/prompthmr/results/<generation>/` |
| Lock / Windows adaptations / CUDA verification | `installer/prompthmr-runtime-lock.json`, `installer/prompthmr_adapters.py`, `installer/verify_prompthmr.py` |
| Setup implementation | `tools/setup_prompthmr.py` |
| Camera build / verification | `tools/setup_prompthmr_camera.py`, `installer/prompthmr-camera-lock.json`, `installer/prompthmr_camera_adapters.py`, `installer/verify_prompthmr_camera.py` |
| App connector | `mocap/prompthmr_pipeline.py`, `prompthmr_cache.py`, `prompthmr_publish.py`, `prompthmr_scene.py`, `prompthmr_ui.py`, `prompthmr_runtime.py` |

Native inference, derived correction and export generations remain separate.
Cancelled/incomplete new work does not replace the saved active project. Verified
complete inference can be reused; changed models/inputs/adapters invalidate it.
Tracking/mask intermediates can be much larger than final motion. Do not delete
files from saved projects to repair them, or edit a ready runtime in place.

For agent-assisted setup, ask: “Read AGENTS.md and docs/models/PROMPTHMR.md.
Check Windows C++/CUDA prerequisites, run Setup_PromptHMR.cmd after my required
license review, verify its CUDA checks and help import the exact model files.
Preserve the other runtimes, models and projects.” Do not put credentials in chat.

## License

[PromptHMR's license](https://github.com/yufu-wang/PromptHMR/blob/main/LICENSE)
restricts use to non-commercial scientific research. RGB MoCap's MIT license
does not relicense it, supporting checkpoints or registered body data. Restricted
source is downloaded into the user's private runtime; weights/body models are not
bundled. SAM2, Detectron2, ViTPose and other components retain their own terms.

## Citation

[PromptHMR paper](https://arxiv.org/abs/2504.06397) · [Research citations and BibTeX](../CITATIONS.md#prompthmr)
