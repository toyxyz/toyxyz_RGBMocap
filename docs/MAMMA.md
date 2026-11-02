# MAMMA multi-view reconstruction

MAMMA reconstructs SMPL-X human motion from multiple camera views, including
multi-person interactions. The [official input guide](https://github.com/cuevhv/mamma/blob/main/docs/YOUR-DATA.md)
organizes footage and calibration by camera; it does not define four cameras as
the maximum. The [official dataset documentation](https://github.com/cuevhv/mamma/blob/main/docs/DATASETS.md)
includes 16- and 32-camera captures as well as four-iPhone captures. The four-view
example is not a general model limit.

**RGB MoCap implementation:** reconstruction accepts two or more synchronized
fixed-camera videos without a fixed camera-count maximum. Available RAM and
VRAM determine practical capacity. Existing system-memory monitoring, CUDA
allocator budgeting and OOM failure handling remain active; removing the input
count limit does not guarantee that every camera/clip combination fits memory.
Four columns in the thumbnail strip are only a layout choice; more views add
rows. Input/preparation/save/restore was checked with up to 65 tiny synthetic
camera files, not 65-view AI fitting. Full model inference has only the scope
recorded in [validation](VALIDATION.md); official multi-camera datasets are not
evidence that every such workload fits this adapter or a particular GPU.
See [all model guides](models/README.md) for the other capture workflows.

It has a separate optional runtime; ExMR, FlowHMR and GVHMR keep their existing
environments. Run the normal `install.cmd` first. PromptHMR is another independent
optional workflow: neither model requires the other's setup. The same C++/CUDA
build tools can be used, but their installed Python packages and body-model
requirements remain separate. See [PromptHMR](models/PROMPTHMR.md).

## Before installing

Obtain the official [MAMMA code and registered assets](https://mamma.is.tue.mpg.de/)
under its own license. The MAMMA software, weights, dataset and locked-head
SMPL-X body data have separate restricted-use terms. They are not redistributed
in this repository; the root MIT license covers the original connector only.
This integration does not grant commercial-use or redistribution permission.

Use 64-bit Windows and a compatible NVIDIA GPU/driver. This path builds native
CUDA components locally. Install the following tools separately before running
the MAMMA setup:

- **Visual Studio 2022 Build Tools**, with **Desktop development with C++** and
  a Windows SDK. Select the 2022 release.
- **[NVIDIA CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive)**.
  The NVIDIA driver alone does not include the CUDA compiler.

Compiler tools add disk usage beyond the runtime and may require administrator
installation. The app and its runtime are installed in the existing RGB MoCap
folder. Only one Windows/NVIDIA PC has been tested; a clean Windows installation
and other GPUs have not been verified.

| Item | Storage |
| --- | ---: |
| Active runtime excluding caches / earlier setup-report size | 6.11 GiB / 6.19 GiB |
| Model files, when the shared VGGT-Omega model is already installed | About 3.53 GiB additional |
| Model files including VGGT-Omega | About 7.80 GiB |
| Free space needed during runtime setup, before models and compiler tools | About 18 GiB |

The setup downloads approximately 3.57 GiB of package/source archives, excluding
Python and models. Retained downloads, videos and generated projects use extra
space. These figures are separate from the base installation's storage estimate.
The cache-free additional runtime/model total is about **9.65 GiB** when
VGGT is already installed. GeoCalib adds about 0.108 GiB of weights if PEAR has
not already installed that shared checkpoint. The existing base camera-alignment
support is reused; MAMMA's Python 3.11 environment is not upgraded.
See the [all-model inventory](USER_GUIDE.md#installed-size)
for shared-file accounting and setup versus installed-space differences.

## Install the runtime

1. Complete the [base installation](USER_GUIDE.md#install) with `install.cmd`,
   then install the compiler tools above.
2. Save your project and close RGB MoCap. In its folder, run:

   ```cmd
   Setup_MAMMA.cmd
   ```

3. Read and accept the upstream license prompt if its terms permit your use.
   Setup downloads hash-pinned packages and official source, compiles SAM 2/SDF
   for this PC's GPU, tests actual CUDA kernels and checks installed package
   hashes before enabling the runtime. Provider passwords are not requested.
4. Run `RGBMoCap.cmd`, select **MAMMA** in **AI Motion Model**, and open
   **Model Manager**. **Runtime Setup...**
   shows readiness, prerequisites and a command you can copy. Prepare models
   separately as described below.

Keep the setup window open until it reports completion. **Runtime Setup...**
shows instructions and can copy the command; it does not run installation inside
the app. You do not need to install Python packages by hand. The installer uses
its separate pinned Python 3.11/PyTorch CUDA environment without replacing the
other model runtimes.

MAMMA and pytorch_sdf prohibit ordinary redistribution under their provider
terms. Consequently, their source/native modules are downloaded and built on
the user's machine rather than shipped as prebuilt files by RGB MoCap.

## Prepare model files

Select **MAMMA** before opening **Model Manager**. Use automatic download for
public files. For registered files, choose **Manual Download...**, register or
sign in on the provider's site and accept its terms, then use **Import...** in
that window or in the model list. The app does not ask for provider passwords.

The manager lists ten assets. Automatic cameras require both VGGT-Omega and
GeoCalib; manual cameras do not require either checkpoint. The paths below are relative to the app's
`models/` folder; importing places the file in its expected location.

| Asset | How to obtain it | Destination inside `models/` |
| --- | --- | --- |
| MAMMA checkpoint | [MAMMA website](https://mamma.is.tue.mpg.de/), manual download | `mamma/mamma_mask_full_cvpr.ckpt` |
| Surface landmarks | MAMMA website, manual download | `mamma/verts_512.pkl` |
| Locked-head SMPL-X Neutral, Male and Female (three files) | [SMPL-X website](https://smpl-x.is.tue.mpg.de/), manual download | `mamma/smplx/SMPLX_NEUTRAL.npz`, `SMPLX_MALE.npz`, `SMPLX_FEMALE.npz` |
| SAM 2.1 Hiera Large | Automatic download | `mamma/sam2.1_hiera_large.pt` |
| YOLO12x | Automatic download | `mamma/yolo12x.pt` |
| OpenCLIP ViT-B/32 | Automatic download | `mamma/open_clip_model.safetensors` |
| VGGT-Omega 512 | Model Manager's manual download instructions; shared with FlowHMR | `flowhmr/vggt_omega_1b_512.pt` |
| Scene Gravity / GeoCalib | Automatic download; shared with PEAR | `pear/geocalib-pinhole.tar` |

For the MAMMA checkpoint and surface landmarks, open the
[official MAMMA website](https://mamma.is.tue.mpg.de/) and click **Data**.
Register if needed, review and accept the provider terms, then sign in to access
the download page. Download `mamma_mask_full_cvpr.ckpt` and `verts_512.pkl`.
Return to Model Manager and use **Import** in each matching row, or directly
from its **Manual Download** window. The app does not need your password or token.

On the separate SMPL-X site, choose **"SMPL-X with removed head bun (NPZ, 392 MB)"**.
Extract the archive and import each of its three NPZ files. The ordinary SMPL-X
v1.1 files used by other models cannot replace these locked-head files.
Supported files are checked against expected SHA-256 hashes. If an automatic
download fails, use the row's manual download/import actions. With compatible
**Manual Camera Data**, the VGGT weight is not required; the other eight assets
are still needed.

## Prepare your videos

- Use two or more fixed-camera views of the **same performance**, with overlapping views
  of the people. Unrelated videos are not multi-view input.
- Synchronize their starting times before opening them. The app does not align
  audio or provide separate camera start offsets.
- Use distinct, short filenames such as `camera_01.mp4` and `camera_02.mp4`.
  Filenames without extensions identify cameras, including in manual calibration.
- Ensure the intended people are visible in the first camera, which initializes
  their identities. A view in which a person is completely hidden supplies little
  useful evidence for that person.

The app uses a common 30 fps timeline, limited by the shortest input. The current
adapter accepts 4–1,800 common frames (up to 60 seconds at 30 fps) and at most
eight people. Start with a short trimmed section to check the cameras and masks
before processing a long clip. These are adapter limits, not a guarantee that
every clip fits memory; the full 60-second limit has not been validated.

## Reconstruct and export

1. Select **MAMMA** and use **File > Open Video** to select two or more
   synchronized, fixed-camera videos of the same performance.
2. Click camera thumbnails to change the main preview. Up to four thumbnails
   fit per row. One common 30 fps timeline and trim applies to every video;
   duration is limited by the shortest input. Drag the ends of the timeline's
   **Video** track to trim all views together without modifying the originals.
3. Keep **VGGT-Omega + GeoCalib (Automatic)** or use **Import Camera Data...**. **People: 0**
   automatically detects the count; specify 1–8 if someone is missed or background people interfere.
   MAMMA finds people during reconstruction, so **Find person** is hidden.
   Enter only the number in **Reference Height (m)**, for example `1.70` for
   170 cm. It sets approximate metric scale from the initial head-to-sole distance.
   This still assumes a suitable reference pose and height; crouching, occlusion
   or uncertain landmarks can bias scale. It is not used to infer the vertical axis.
4. **Reconstruct 3D** prepares aspect-preserving inputs within 1920×1080, creates
   SAM 2.1 masks, predicts FP32 landmarks, estimates/loads cameras, runs the full
   MAMMA body fit, then prepares native people, correction and export bones.
   Mask fragments below eight pixels are removed when a substantial component
   exists; soft visibility weights reduce the influence of occluded views.
5. Review the people together in one shared world. Check/uncheck **People** rows
   to show or hide them; select a row to edit that person's contacts.
   **Enable Correction** is on
   by default; turning it off selects the preserved original motion. Edit contact
   timing and use **Apply Correction** as for the other models. These operations
   do not need new MAMMA inference. Foot correction is on and hand correction is
   off by default; enable **Hand** when palm support is needed. The viewport
   **Lock** checkbox only shows or hides contact markers.
6. Use **File > Export...** for the displayed, applied motion. Choose FBX or BVH
   for skeleton animation, or Blender for a portable project. Blender also offers
   **MAMMA whole scene: body meshes and all cameras** for checked people, baked
   PC2 surfaces, reference skeletons and every synchronized camera/video.
   Whole-scene export preserves the common timeline without an added T-pose;
   keep its `videos/` and `meshes/` folders beside the `.blend`. See
   [portable Blender export](USER_GUIDE.md#portable-blender-projects). Skeleton export uses the
   existing 52-bone source-export convention. Use **File > Save Project As** to
   keep a named copy; reopen its `project.json` with **File > Open Project**.

Resource monitoring remains active. Fitting can take several minutes even for
a short clip; an animated progress indicator is used when the stage cannot
report a meaningful percentage. The app uses SAM 2.1 and FP32 fitting; SAM 3.1
standalone tests remain separate. No quantization, TF32, iteration reduction or
physics simulation is enabled in this path.

Low Memory does not select smaller MAMMA camera sets, fewer fitting iterations
or a lower-precision model. It keeps VGGT camera transformer blocks on CPU and
transfers them for inference. Automatic uses the fast camera path when GPU memory
is sufficient, selects offloading when needed, and can recover once from a camera
CUDA OOM. Decisions and Low Memory advice appear in Console and the CMD window.

The Automatic VGGT camera fast path shares FlowHMR's meta-initialized,
direct-assignment checkpoint loader. It reduces temporary model loading work;
landmarks, masks, fitting and camera calculations are unchanged. Low Memory and
Automatic fallback retain copy loading because the experimental assign/offload
combination increased Windows private allocation. See the
[camera-loading measurements and scope](PERFORMANCE.md#vggt-camera-checkpoint-loading).
Offloading may be slower and use more RAM; masks, landmarks and fitting retain
their existing bounded settings. The shared allocator budget and system-memory guard
still apply; shorten the common clip or reduce input views yourself if needed.

## Applied acceleration and resource modes

The supported path uses exact-coordinate ROI landmark preprocessing, CPU
checkpoint loading and the supplied masks instead of an unnecessary landmark
detector. It omits unused mask/collage/debug videos and reduces fitting
diagnostics. VGGT samples up to three times per fixed camera at its 512 input
setting and omits unneeded depth output. Matching completed inference is reused;
contact edits and correction do not require another MAMMA inference.

Earlier four-view, two-person, 30-frame tests on the RTX 5090 test PC measured:

| Applied change | Stage time before → after | Time reduction |
| --- | ---: | ---: |
| Exact ROI preprocessing + CPU checkpoint loading, FP32 landmarks | 104.43 → 57.18 s | 45% |
| Omit unused outputs during mask processing | 129.02 → 92.39 s | 28% |

These isolated stage gains must not be added together or treated as a whole-job
speedup. Reducing fitting diagnostics did not show a reliable fitting-stage gain.
The latest ExMR/FlowHMR CPU prefetch and connected-worker changes do not apply
to MAMMA, which already processes multiple people together.

**Low Memory offloads VGGT only**, not the full landmark/fitting pipeline. A
controlled shared-VGGT test with nine images from three cameras measured
allocated VRAM **6.47 → 3.14 GiB** (52% less), RAM working set **8.75 → 8.85 GiB**,
and warm inference **0.225 → 0.751 s** (3.3× as long). That is a camera-component
measurement, not a 52% saving for all of MAMMA.

A separate paired app-mode check used two views, two people and 30 frames:
RAM was **8.15 → 8.12 GiB**, and total time **241.61 → 222.00 s**. Whole-pipeline
VRAM was not recorded for that pair, so its VRAM saving is **not established**.
Startup/cache variation affects these single-pass times; Low Memory is not a
reliable speed improvement. It can require more RAM. The mode does not reduce
views, fitting iterations or precision, and cannot guarantee that fitting fits.
See [common benchmark conditions and memory definitions](PERFORMANCE.md).

## Automatic camera orientation

VGGT-Omega estimates relative fixed-camera placement and intrinsics. GeoCalib
estimates background gravity at the sampled times, with the VGGT focal length
held fixed and detected people excluded from the geometric confidence maps.
The camera rotations carry these directions into one common world. A
camera-balanced spherical estimate rejects unreliable or conflicting samples.
This does not guarantee that every background provides reliable gravity.

The resulting vertical axis aligns the camera world **before MAMMA 3D fitting**.
There is no additional GeoCalib rotation after fitting and no automatic fallback
to a head-to-foot vertical axis. If background gravity cannot be estimated
reliably, processing stops with an actionable message to import manual calibration
or use clearer background views. GeoCalib does not measure absolute floor height:
the initial sole datum and reference-height metric scale remain estimates.

GeoCalib runs sequentially in the already installed Flow/Python 3.10 camera
support overlay. Its packages are not imported into MAMMA's Python 3.11 process.
If support is missing, close the app and run `install.cmd`; obtain its shared
checkpoint through Model Manager. `Setup_MAMMA.cmd` still prepares the separate
MAMMA/SAM 2/SDF runtime. Existing saved results are not changed just by opening
them; an explicit new reconstruction uses the new camera-generation identity.

## Manual cameras

Selecting Manual Camera Data bypasses VGGT-Omega and GeoCalib, and disables the
reference-height control: supplied calibration defines orientation and scale.
With automatic cameras, Reference Height accepts 0.50–2.50 m
(default 1.70 m); it is an approximate scene-scale assumption, not a measurement.
Reconstruct after changing camera, count or height settings; Apply Correction
does not re-estimate cameras or poses.

### JSON example and required fields

Open [mamma-camera-example.json](samples/mamma-camera-example.json) for a complete,
importable two-camera example. It is an original RGB MoCap **synthetic format
example**, covered by this repository's MIT license. Its numbers are not measured
calibration for MAMMA's videos or your own footage. Changing camera names alone
does not calibrate a video; use VGGT when you do not have matching calibration.

| Field | Required structure and meaning |
| --- | --- |
| `units` | `"metres"`; translations use metres. |
| `extrinsics_convention` | `"world_to_camera"`. |
| `cameras` | Object containing one entry for every opened video. Each key must exactly match its filename without the extension: `A001.mp4` → `A001`. Remove unused camera entries when opening only a subset of views. |
| `intrinsic_matrix` | 3×3 matrix `[[fx,0,cx],[0,fy,cy],[0,0,1]]`. Focal lengths `fx,fy` and optical center `cx,cy` are in pixels for the original input resolution. |
| `extrinsics_matrix` | 3×4 matrix `[R \| t]`, or 4×4 with last row `[0,0,0,1]`. `R` is a proper rotation; `t` is a translation in metres. Maps a world point to camera coordinates: `p_camera = R @ p_world + t`. |
| `image_size` | `[width,height]` in pixels, matching the original opened video, not the app's resized working copy. |
| `distortion_coeffs` | Optional `[k1,k2,p1,p2]` or `[k1,k2,p1,p2,k3]`. Omitted means zero. The current app adapter requires all zeros and undistorted pinhole footage. |

The example has `A001` and `B001` entries at 1920×1080. It describes two hypothetical
cameras at `[0,-4,1.6]` and `[4,0,1.6]` metres, looking toward the world origin.
Camera position is **not** the last column of the extrinsic matrix:
`camera_position_world = -R.T @ t`.

Use a **Z-up world** (ground at world Z=0) and OpenCV-style camera axes: X right,
Y down, Z forward. Zero intrinsic skew and zero lens distortion are required;
unsupported calibration is rejected rather than silently discarded. Intrinsics
are explicitly resized to the working input dimensions. Import is limited to
1 MiB per calibration file.

Open the matching videos, choose **Manual Camera Data**, click **Import Camera
Data...**, select the JSON, then **Reconstruct 3D**. A camera-name or resolution
mismatch is rejected. The app cannot confirm that supplied calibration measures
the correct real camera arrangement. VGGT ignores supplied manual calibration.

### Official MAMMA sample calibration

The official four-camera `pushing_and_lifting_from_ground` demo pairs its
`A001.mp4`, `B001.mp4`, `C001.mp4` and `D001.mp4` videos with
[iphones_outdoors.yaml](https://github.com/cuevhv/mamma/blob/588492f18876e2ed6888b2d26929047cb6b575e7/configs/examples/calib/iphones_outdoors.yaml).
This is **YAML**, not a camera JSON. Obtain it directly from MAMMA under its
[upstream license](https://github.com/cuevhv/mamma/blob/588492f18876e2ed6888b2d26929047cb6b575e7/LICENSE);
RGB MoCap does not redistribute the sample's original or converted calibration.
Use the corresponding original-resolution videos; if you independently resize
them, scale the calibration's focal lengths and optical center as well as its
resolution. The official sample specifies 3840×2160.

For MAMMA YAML, each entry under `cameras` has `intrinsics: [fx,fy,cx,cy]`,
`resolution: [width,height]`, `translation` (camera position in the world), and
`rotation_quaternion` (camera-to-world, Hamilton order `[w,x,y,z]`). The app
converts this to world-to-camera matrices. JSON uses the matrix convention above.
MAMMA's `configs/examples/captures/*.json` files describe input datasets and
processing paths; they are **not camera calibration files** for this button.

## Recorded processing time and memory

On the tested RTX 5090 32 GB / Ryzen 9 9950X / 96 GB RAM PC, a manual app test
used three synchronized camera videos resized within 1920×1080, two people,
SAM 2.1 masks and automatic VGGT cameras. The common trim was 413 frames at
30 fps (13.77 seconds).

| Input length | Videos | People | Total through correction/preview | Peak VRAM (GiB) | Peak RAM (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 13.8 seconds | 3 | 2 | 11m 52s | 6.47 | 9.11 |

Masks took 257.80 s, landmarks 89.72 s, cameras 14.42 s, 3D fitting 294.42 s,
and correction/preview 31.97 s. The 712.08-second total also includes video
preparation/startup. Peak VRAM is PyTorch allocated memory; peak RAM is worker
working set, measured across masks, landmarks, cameras and fitting. They exclude
unmeasured preparation/correction/preview peaks and are not whole-PC usage.
No fatal traceback or OOM appeared in that log.

This historical run predates the latest Low Memory changes. It uses different
input and stages from the [single-video model records](USER_GUIDE.md#recorded-processing-time-and-vram),
so it is not a direct speed ranking or a minimum hardware requirement. Camera
count, resolution, clip length, people and hardware can change both time and memory.

## Saved data and limitations

**Save Project As** copies all camera videos and project-private body data.
Keep the whole project folder together. MAMMA data lives under
`<project>/mamma/inference/<generation>/` (working videos, masks, landmarks,
cameras, raw fitted parameters) and `mamma/results/<generation>/` (native
people/takes, contacts, previews and exports). Deleting these folders removes
saved reconstruction work, not just disposable thumbnails.

Prepared MAMMA people can also be used as character-retargeting sources. The
retargeter reads the actor's project-private body, not the ordinary SMPL-X body
used by ExMR/GVHMR. That extra standard body is not required for MAMMA-only
retargeting. See [character import, retargeting and export](USER_GUIDE.md#character-retargeting).

Each actor retains all sixteen shape coefficients exactly: the final six are
folded into its private body template for the editor's ten-slot skin format.
Animated poses, joint centers and hand means are preserved. Original raw motion
is retained, while correction is saved separately. Body-derived files remain
restricted registered data and must not be published with source code.

Verified inference outputs are reused only with matching source, model, engine
and connector identities plus output receipts. Interrupted attempts retain a
pending generation; a retry cannot publish mixed outputs. Camera/person changes
require reconstruction and leave the previous displayed motion in place.

Occlusion, segmentation/identity mistakes, weak camera baselines and uncertain
scale can still produce poor poses. Contact correction reduces sliding but does
not repair lost identities, bad camera geometry or missing observations. This
integration was tested on one Windows/NVIDIA PC; see the
[recorded processing and memory](#recorded-processing-time-and-memory).
Clean Windows, other GPUs and the full 60-second limit have
not been validated.

## Updates and troubleshooting

- **Runtime missing or out of date:** save your work, close the app and run
  `Setup_MAMMA.cmd`. `update.cmd` maintains the base runtimes; it does not install
  this optional runtime. Re-run the MAMMA setup when its dependency notice asks
  you to, or after changing GPU architecture.
- **Compilation or download failure:** check the setup message, free space,
  compiler prerequisites and `.setup/logs/mamma-*.log`, then retry. Verified
  downloads can be reused. A failed setup retains the previously enabled runtime,
  models and projects. Do not delete active `runtime_mamma/` or `.setup/` state.
- **Models not ready:** use the exact supported file and **Import...**. Recheck
  that the SMPL-X archive is the locked-head version and that all three bodies
  were imported. Manual camera mode can run without the VGGT weight.
- **Poor poses or jitter:** inspect camera overlap, synchronization, identity
  and masks first. Try a clearer first camera or an explicit **People** count.
  Camera, people-count or reference-height changes require **Reconstruct 3D**;
  **Apply Correction** only updates contact correction.

The verified runtime is stored in an immutable generation under
`runtime_mamma/`, referenced by `.setup/mamma-runtime.json`. It is configured for
this installation's paths; copying that folder alone is not a supported setup
method. For development environments and detailed checks, see
[the agent installation/native-build runbook](MAMMA_AGENT_SETUP.md),
[Build from source](BUILD.md) and [Validation](VALIDATION.md).

## Citation

[MAMMA paper](https://arxiv.org/abs/2506.13040) · [Research citations and BibTeX](CITATIONS.md#mamma)
