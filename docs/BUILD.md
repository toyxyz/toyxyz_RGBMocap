# Windows development guide

The Git source distribution uses `install.cmd`, `RGBMoCap.cmd`, and `update.cmd`
for installation, launch and updates. For current model files and supported
controls, see [model-specific guides](models/README.md). For the native motion-correction module see
[developer build instructions](../developer-build/motion-correction/README.md).

## Prerequisites

- Native 64-bit Windows and a compatible NVIDIA driver for inference.
- An initial Python 3.10+ installation; `setup_dev.py` obtains pinned Python
  3.10.22 and 3.12.10 through uv 0.13.1.
- Visual Studio 2022 Build Tools, Desktop development with C++, Windows SDK and
  CMake. Git is needed if CMake fetches its pinned pybind11/Eigen source.
- FFmpeg and ffprobe on PATH, or a folder passed with `--ffmpeg-dir`.
- The complete [Blender 4.5 LTS Windows distribution](https://www.blender.org/download/lts/4-5/),
  installed or extracted separately. Keep its Python, libraries and license files.
- Internet access for dependencies; enough disk space for downloads and builds.

No Ubuntu, WSL or Docker is required. The three single-video workflows use the
reviewed wheel stack without a system CUDA Toolkit. Optional MAMMA native builds
require CUDA Toolkit 12.8 and the C++ tools described below. GPU compatibility
still depends on the NVIDIA driver and installed wheels.

Optional PromptHMR has a separate pinned Python 3.11 environment installed with
`Setup_PromptHMR.cmd`. It also needs C++ tools and CUDA Toolkit 12.8 to build
SAM2. Choosing Y for its optional moving-camera support also builds
DroidCalib/LieTorch over that same environment and verifies actual CUDA operations;
MAMMA/SDF is not required. Read [its setup/developer notes](models/PROMPTHMR.md)
before changing its lock, adapters, native modules or model paths.

## Environments

Run `python tools/setup_dev.py --plan` to review groups. Then run the same command
without `--plan` to install them. `--profiles shared exmr` prepares only those
profiles. Shared is added automatically for overlay groups. Every process returns
a failure if downloads, hashes or installation fail.

| Development folder | Purpose |
| --- | --- |
| `.venv` | GUI, GVHMR and shared CPU correction |
| `.venv-flow` | FlowHMR overlay on the shared dependency base |
| `.venv-hand` | HandFlow overlay |
| `.venv-htd` | HTD overlay |
| `.venv-exmr` | Independent ExMR Python 3.12 environment |

The authoritative source is `installer/runtime-lock.json`, not a broad pip
resolver over one combined environment. Small pure-Python packages without a
matching public wheel are retained as source under `installer/runtime_sources/`.
Build tools reconstruct wheels locally and write a derived lock under `build/`.
The source lock is never silently replaced from the developer's installed pip set.

The root `requirements-windows.txt`, `requirements-flow-windows.txt` and
`requirements-hand-windows.txt` are historical cache-fingerprint inputs, retained
byte-for-byte so existing worker receipts remain comparable. Keep them in the
source checkout. Do not use them to install managed runtimes; use
`tools/setup_dev.py` and the runtime lock above.

When changing model dependencies, review its profile and overlay relationship in
`mocap/runtime_registry.py`, update the corresponding lock deliberately and rerun
model-specific checks. Do not upgrade a shared dependency for a new model without
checking existing models.

## Editing application features

| Area | Main files |
| --- | --- |
| Workspace and controls | `mocap/gui.py`, `workspace_ui.py`, `people_ui.py`, `tooltips.py` |
| Model list and file import | `mocap/motion_tools.py`, `models.py`, `model_guide.py` |
| Worker routing | `mocap/worker.py`, `worker_progress.py`, `runtime_registry.py` |
| Model processing | `mocap/exmr_backend.py`, `flow_backend.py`, `engines.py` |
| Contact correction | `mocap/correction.py`, `contact_trajectory.py`, `contact_pose_plan.py`, `contact_arm_plan.py` |
| Skeleton and surface fitting | `mocap/source_animation.py`, `source_surface.py`, `surface_roll.py` |
| Preview and playback | `mocap/viewport.py`, `preview_data.py`, `playback.py`, `video_preview.py` |
| Source export | `mocap/source_export_dialog.py`, `fbx_export.py`, `fbx_writer.py`, `scene_export.py`, `export_copy.py`, `biped_bvh.py` |
| Blender bundles | `mocap/blend_export.py`, `blend_multicam_export.py`, `blend_export_worker.py`, `blend_support.py`, `blend_export_scene.py`, `blend_multicam_scene.py` |
| Camera preview/overlay | `mocap/camera_tracks.py`, `camera_ui.py`, `camera_reference.py`, `camera_background.py`, `camera_background_worker.py` |
| Git setup/update | `install.cmd`, `update.cmd`, `tools/git_bootstrap.ps1`, `tools/git_managed_install.py`, `tools/git_update.py`, `installer/runtime-lock.json` |

Run `app.py` from `.venv` while editing. Worker changes are loaded by their own
profile processes; C++ changes require rebuilding the native correction module.
Keep source inference, contact correction and derived preview/export generations
consistent. A change to painting or labels should not silently rerun inference.

Character import/baking runs in Blender's own Python, separately from the five AI
profiles. For development, point `RGBMOCAP_BLENDER` to its executable:

```powershell
$env:RGBMOCAP_BLENDER = 'C:\Tools\Blender-4.5\blender.exe'
.venv\Scripts\python.exe app.py
```

Character UI and processing live in `mocap/character_ui.py`, `character_data.py`
and `mocap/retarget/`. Standard Mixamo, VRM and PMX mappings are automatic.
Custom bone mapping and hair/clothing simulation are not implemented.

## Required model and auxiliary data

Use Model Manager for weights, SMPL-X/SMPL-H/MANO and gated supporting models.
Registration and license acceptance remain the user's responsibility.

Some upstream auxiliary regressors, mean-pose and mapping arrays are also kept
out of Git. Exact paths and hashes are in `installer/external-assets.json`.
Obtain the matching public auxiliary files from the pinned upstream repositories,
retaining their license conditions:

```powershell
python tools/stage_assets.py --download
```

This downloads only the small hash-pinned files listed in the auxiliary manifest,
not inference weights or registered body models. If an upstream link is unavailable,
obtain the exact files from the versions in `third_party/SOURCES.json` and import
an existing properly acquired tree with:

```powershell
python tools/stage_assets.py --from-root D:\AcquiredAssets
```

That folder must contain paths such as
`third_party\FlowHMR\assets\body_models\smpl_neutral_J_regressor.pt`.
The importer checks exact sizes and SHA256 values. It does not import weights or
registered body models. These locally acquired arrays remain ignored by Git.
Keep the source package's pinned version when acquiring them; do not replace
arrays with similarly named files from another body model.

`third_party/character_import` includes the actual format add-on source, licenses,
OpenCC dictionaries and editable shader libraries. The five small `.blend` shader
libraries and ten standard toon ramps are explicitly listed and hash checked in
`installer/source-resources.json`; they are not sample characters or motion files.
OpenCC's pure-Python wheel is retained as unpacked source under
`installer/runtime_sources/` and rebuilt in temporary staging during runtime setup. The add-ons
are loaded by the worker; do not install these source folders through Blender's
extension installer.

## Native motion-correction module

Native CMake compilation is defined in
`third_party/KimodoMotionCorrection/CMakeLists.txt`. Source changes to that backend
require a rebuild; see the [developer build instructions](../developer-build/motion-correction/README.md).
The core currently uses AVX compiler options; do not claim compatibility with
processors that have not been checked.

## Before publishing source changes

Test `install.cmd`, `RGBMoCap.cmd`, and `update.cmd` in a separate writable
checkout. Check failure/retry, local edits, model preservation, a short clip
for each supported model, correction, preview, character export and Blender
import where applicable. A source or smoke check does not replace hardware and
full-motion testing. See [Source validation](VALIDATION.md) for the recorded scope.

Inspect staged files and keep environments, models, external arrays, caches,
installers and personal projects out of Git. The reviewed prebuilt correction
module and licensed importer resources are explicit exceptions. Preserve
third-party licenses, corresponding source/build materials and model terms; see
[Third-party notices](../THIRD_PARTY_NOTICES.md). The Git installation downloads
Blender and FFmpeg from their pinned upstream locations.

Git attributes preserve exact source bytes for the source manifest, hash-pinned
runtime registry and unpacked wheel contents. Avoid automatic line-ending rewrites
of those files. When intentionally changing the registry, review and refresh the
lock's registry hash; the development bootstrap rejects an unreviewed mismatch.

## Optional MAMMA runtime

Follow the [agent installation and native-build runbook](MAMMA_AGENT_SETUP.md)
for prerequisite checks, staged/manual builds, CUDA tests and recovery. For
ordinary use, prefer the automated setup over manually assembling packages.

`Setup_MAMMA.cmd` calls `tools/setup_mamma.py` after license acceptance.
`installer/mamma-runtime-lock.json` pins Python 3.11.17, PyTorch 2.7.1+cu128,
all package artifacts and the official MAMMA/SAM 2/pytorch_sdf commits by SHA-256.
The six core runtime profiles stay unchanged. The optional setup needs Visual
Studio 2022 C++ Build Tools with the Windows SDK and CUDA Toolkit 12.8.

`installer/mamma_adapters.py` applies Windows portability and the validated
FP32 masked-landmark/quiet-fit adapters to directly downloaded upstream source.
The original helper code is in `installer/mamma_helpers/`. Restricted upstream
source and compiled SDF modules must not be added to this repository.
Both native SAM 2 and SDF are built for the installed GPU architecture, then
checked by `installer/verify_mamma.py` (real CUDA kernels, SDF gradients and
installed package file hashes). A GPU architecture change requires setup again.

Verified generations live in ignored `runtime_mamma/<generation>/`; the active
pointer is `.setup/mamma-runtime.json`. Download/build/verification failure
preserves the old pointer. Setup stages first and checks again after relocation.
Changing the optional lock/adapters invalidates this runtime only: the UI requests
`Setup_MAMMA.cmd`, while other models and saved results stay available. Keep
private benchmark scripts, logs and derived models outside the source checkout.

## PEAR support

`install.cmd` and `update.cmd` prepare PEAR support with `tools/setup_pear.py`
after committing the base shared Python/Torch runtime. The helper runs under the
installer's existing application lock; it must not acquire that lock a second time.
It prepares pinned
upstream PEAR source and pure-Python support in `runtime_pear/`, rather than
changing base dependencies. No native PyTorch3D build is used. Source URLs and
hashes are in `installer/pear-sources.json`; explicit import adaptations are in
`installer/pear_adapters.py`. Geometry imports use `installer/pear-assets.json`
for exact archive/member hashes. See [PEAR](models/PEAR.md). Models, generated
support folders, diagnostics and private regression artifacts are excluded from Git.
The active receipt binds the support/adapters to the installed shared-runtime group.
Matching verified support is retained without downloading or extracting it again;
a source/base change or damaged support stages and CUDA-verifies a new generation
before replacing the pointer. Failure preserves prior support generations. Base
completion is independent: a post-commit PEAR failure reports a warning rather
than leaving the base install retry marker. No weights or registered geometry
are acquired by this step.

`tools/setup_pear_alignment.py` additionally stages the pinned GeoCalib/Kornia
support from `installer/pear-alignment.json`, over the existing Flow runtime,
without upgrading shared packages. It verifies imports/CUDA availability and
writes a separate `.setup/pear-alignment.json` inventory bound to the installed
shared/Flow group IDs and runtime directory creation stamps. A replaced or changed
base requires support revalidation. VGGT/GeoCalib model
weights are obtained in Model Manager. This support enables camera alignment;
if it is unavailable, PEAR can still track the selected camera with a warning,
provided its base support and model files are ready.

MAMMA automatic cameras reuse this overlay through the `flow_mamma_gravity`
child worker. Its numeric gravity result is passed back to MAMMA before 3D
fitting; cp310 GeoCalib/Kornia packages must not enter MAMMA's Python 3.11
environment. Missing or unreliable gravity is an actionable failure for MAMMA
automatic cameras. Manual camera data bypasses this support and both camera
checkpoints. A separate GeoCalib package installation into MAMMA is not needed.

## Base update and recovery

Keep the existing bootstrap, pinned separate profiles and promotion journal.
`tools/git_managed_install.py` freezes source dependency/installer inputs, recovers
an interrupted promotion, verifies/stages the base, rechecks the frozen inputs,
then commits receipts and finishes journal cleanup. Only then clear the global
pending marker. PEAR and camera alignment are independently attempted through
`tools/optional_support.py`; `.setup/git-components.json` records their readiness
and errors. A support warning is not a successful capability verification.
Cancellation must propagate; mandatory base errors remain fatal.
If a live runtime is reused, discard its leftover installer staging directory
before installing dependents, and never promote that stale directory. Otherwise
an interrupted run could supply an unverified staged interpreter or replace an
already verified live runtime. This cleanup is confined to installer staging.

`tools/runtime_verification.py` caches successful full checks per profile for
at most seven days. Cache keys include the group/base dependency identity and
verifier code; reports must match the committed receipt. A bounded metadata
inventory includes files added/removed and size/mtime/ctime changes, ignoring
bytecode caches. Recheck reused inventories before base commit. A fast check
never extends the full-check date. Missing, changed, expired or malformed caches
fall back to full RECORD/import verification. `--verify-all` bypasses reuse.
This is an ordinary-change detector, not protection against hostile metadata
forgery. Keep the explicit full-audit path and do not broaden numerical cache
invalidation for installer-only changes.

Support archives use `tools/pinned_download.py`: default TLS verification,
HTTPS-only redirects, three transient-error attempts, bounded backoff and a
size-aware total deadline, a finite socket timeout, strict range validation and
a final complete size/hash check. Partial files are associated with the exact
URL/size/hash; an ignored Range restarts safely. Certificate/permanent HTTP or
pin failures do not loop indefinitely. Existing base-wheel and optional-runtime
package download paths are not all replaced by this helper.

Fault tests must distinguish failure before base commit from failure afterward.
Include real CMD argument forwarding/local Git fast-forward, retained pending
state after failed pulls, interrupted promotions, changed source inputs, partial
downloads, certificate/hash failures and user-data preservation. Keep tests and
their generated files outside the public source tree. See [validation](VALIDATION.md#installation-recovery-and-recent-verification-2026-11-06).

Changed upstream files receive an adaptation notice. Each support generation
also records pinned source URLs/hashes and the adapter hash in
`RGB_MOCAP_PROVENANCE.json`, alongside the retained provider license files.
