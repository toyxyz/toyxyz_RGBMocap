# RGB MoCap: agent operating and development guide

This guide is for agents helping users operate or modify RGB MoCap. It describes the Git/CMD version. User requests and newer verified behavior take precedence over historical benchmarks. The English GUI is the supported user interface; do not assume a general command-line inference or retargeting API exists. Avoid asking for the user's videos or model weights if the existing local project is sufficient.

## Start and update

- Windows 64-bit, compatible NVIDIA GPU/driver and OpenGL 4.3 are required for the documented inference workflow. Other systems and GPUs were not comprehensively verified; see [validation](VALIDATION.md).
- In a writable checkout, run `install.cmd` once. It sets up separate pinned profiles for ExMR, FlowHMR and GVHMR and their supported add-ons, PEAR support over the shared runtime, Blender 4.5, FFmpeg and the prebuilt native motion-correction module. It does not install only the selected model's runtime or obtain AI weights/body data. Downloading and disk space are required. Models are obtained separately. Optional MAMMA, PromptHMR and GEM-X have separate setup commands.
- Run `RGBMoCap.cmd`. Keep its command window open; launch errors, processing messages and app Console output are also sent there, but long messages can be truncated and a saturated output queue can drop messages. Within the GUI, **Help > Check System** checks the system.
- For Linux, use the [Linux installation guide](LINUX.md). If the AppImage
  stops at native motion correction build, check CMake 3.24 or newer and the
  C++17 compiler in the user's launch environment. The published preview
  compiles this module on first launch. Have the user retry the AppImage and
  check its setup log. See [AppImage build troubleshooting](LINUX.md#appimage-setup-cannot-find-cmake).
- Base worker commands use explicit `-X utf8`, including isolated `-I` launches. Isolated mode ignores `PYTHONIOENCODING`/`PYTHONUTF8` from the parent environment. Any redirected/discarded text log stream must also specify UTF-8. Keep byte-oriented PEAR IPC unchanged.
- Close the app before `update.cmd`. The updater requires a clean tracked worktree and the `main` branch, uses `git pull --ff-only`, then verifies/repairs runtime configuration. It refuses tracked local edits. A plain `git pull` does not perform runtime repair; use `install.cmd` after dependency changes when needed. Do not discard local edits or force an update.
- `update.cmd` buffers its final command block before Git can replace it. Reporting/pause belongs to `tools/git_update.py`; failures preserve the subprocess status (ordinary exceptions use 1, Python-intercepted cancellation uses 130). Keep delayed expansion off for paths/arguments. The final `CALL` expands only a numeric exit status after clearing an environment shadow; never reparse paths or arguments with `CALL`. Already-running older launchers remain vulnerable during their first transition. From an open CMD in the app folder, `runtime\python.exe -I -X utf8 -B tools\git_update.py` bypasses the old wrapper. A stray command error after reported installation completion calls for rerunning the updated launcher, not deleting setup markers or models. See [update recovery](USER_GUIDE.md#installation-verification-and-repair) and the preceding transition note. Forced console termination/deletion of the running launcher is outside this guarantee.
- For setup failure, read the message, check connectivity and free space, then retry `install.cmd`. Preserve models, projects and `.setup` state. Do not repeatedly reinstall or redownload models without evidence.
- Base completion and optional PEAR/camera-alignment readiness are separate. A post-commit support failure reports completion with warnings; it must not recreate the global base pending marker. Inspect `.setup/git-components.json` and the actual capability configuration. Both support checks require CUDA. For GPU warnings, check driver detection with `nvidia-smi` and verify CUDA in **Help > Check System** before retrying `install.cmd`. Camera-alignment preparation is independent of PEAR and also serves MAMMA automatic gravity. Mandatory base failures and unresolved promotion journals remain blocking. Never delete a pending marker blindly.
- Recent full runtime checks are reused for up to seven days with matching group/verifier identities and an unchanged bounded file inventory, including additions/removals. Reuse does not refresh the full-check date. `install.cmd --verify-all` or `update.cmd --verify-all` forces the existing RECORD hashes/import checks. Missing/expired metadata, dependency changes and detected file changes trigger full checks for affected groups/dependents. Metadata is a change-detection heuristic, not an authenticity guarantee. Do not mark a runtime verified without an actual successful check.
- `tools/pinned_download.py` supplies bounded HTTPS retries/resume for support archives. Keep certificate verification, HTTPS redirects, exact size and SHA-256, validated Content-Range, cancellation propagation and finite attempt/time limits. Ordinary base wheel downloads and optional package fetching retain their own paths; do not claim this replaces every network call. Source contracts are frozen/rechecked during base installation. A failed Git pull must preserve any pre-existing incomplete-install marker.
- Base installation currently estimates at least 32.4 GiB free before models; ExMR/FlowHMR/GVHMR and add-ons were previously measured around 29 GiB of weights. Optional MAMMA, PromptHMR and GEM-X add separate runtimes and model files. With GEM-X CUDA sharing, the six offline model/runtime sets total about **72.46 GiB (77.8 GB)** before optional PromptHMR moving-camera support, excluding caches, projects, downloads and compiler tools, counting shared files once. Camera support adds about **1.57 GiB** (18.6 MiB modules and 1.55 GiB weights). See the [installed-size table](USER_GUIDE.md#installed-size); installed size and temporary setup headroom are different. Measurements are not guarantees.

Multi-person reconstruction groups supported ExMR, FlowHMR and GVHMR phases
into sequential, bounded worker requests. `mocap/queue_groups.py` builds the queue;
`batch_worker.py` validates explicit project paths and runs one `ModelSession`
per phase. ExMR connects its keypoint, body-feature, hand-feature and motion
phases in one worker, for one or multiple actors. FlowHMR connects features and
motion for one actor; its existing multi-person phase grouping remains separate.
GVHMR runs its phases in one worker with a new closed scope for each.
The session retains at most one model and clears transient tokens.
It closes on completion/error; cancellation terminates the worker tree. Keep
per-actor receipts and pending markers, separate runtimes, precision, full temporal
context and optional add-on behavior intact. A fresh CUDA OOM recovery discards
the cached model before its one supported Low Memory retry. Do not replace this
with an indefinite global pool or parallel GPU workers.

`mocap/batch_prefetch.py` overlaps CPU image preparation with inference using
one producer thread and at most two prepared batches. Automatic enables it only
with sufficient CPU/RAM headroom; Low Memory disables it. Prepared batch storage
is additionally bounded by a 256 MiB estimate. Always close the producer before
closing its video capture or retrying an operation. ExMR body feature extraction
skips only the final unused pose readouts after the collected token is complete;
keep intermediate pose feedback, keypoint-token updates and the generic SAM path.
HaMeR's normal decoder/readout path remains intact. Do not change precision,
batch size, resolution or temporal context to claim further speedups.

This overlaps existing reads/crops; it does not introduce a persistent disk
cache. The two prepared batches are in addition to the batch being consumed.
The 256 MiB prepared-storage estimate is not a hard cap on whole-process RAM;
retain actual memory monitoring and deterministic producer cleanup.

For FlowHMR query-batch before/after time, allocated VRAM, worker RAM and
precision limits, use the [measured comparison table](PERFORMANCE.md#flowhmr-query-batches-and-four-frame-hands-2026-11-03).
Keep warm generator measurements separate from fresh feature jobs and the
additional HaMeR child pass. HaMeR is an optional finger-accuracy refinement,
not a guarantee of better results for every video.

## Models and model files

The unified Model Manager builds its workflow tabs and **Used by** labels from
the stage requirements plus shared discovery and optional features.
**Installed Models > Dedicated only** filters to assets used by one workflow.
Every workflow tab and **Installed Models** open the same confirmation for
**Delete Selected**. **Cancel** is the default.
Removal identifies exact catalog paths and confirmed runtime hardlinks or
symbolic links. It stages the files on the same volume and records the work for
recovery when the manager next opens. Keep projects, external source files and
unverified runtime copies out of removal. On this branch, an in-place model
stays protected even if its path is under the configured models folder. The
configured models folder is the managed root for ordinary copies. Check the
affected workflows and planned paths shown in the confirmation first.

ExMR's integrated v0.1.5 uses public `pipeline.hand_boxes`, `model_inputs` and
`ExMR.from_pretrained` with direct model forward. Keep stage-scoped model reuse
and the existing Low Memory/prefetch paths. Its upstream toe IK limits target
displacement to 0.03 m per axis. RGB MoCap passes `normalize_world=False` through
an explicit local option to preserve the existing world/camera frame. SimpleVO
uses a documented frame-reader/count adapter over the canonical working video;
do not switch it to JPEG pixels or extract duplicate frames just for camera
enumeration. Keep the app's shared contact policy and manual timing precedence:
upstream `foot_contacts` is not connected to correction. API/solver source hashes
and provenance are part of stage receipts. Model Manager also downloads the
official `models/exmr/config.json` (180 bytes) from the same pinned Hugging Face
revision as the weights, with SHA-256 validation and a separate import row.
It is checkpoint metadata, not app settings. Existing weight and runtime pins
are unchanged. See [validation](VALIDATION.md#exmr-public-api-and-bounded-toe-ik-2026-10-31).

FlowHMR Model Manager also exposes `models/flowhmr/config.json` (25 bytes),
downloaded with the checkpoint from the same pinned Hugging Face revision and
SHA-256 checked. It is download-accounting metadata, separate from the inference
`config.yml`; it is not a new weight, runtime dependency or inference receipt
input. Do not replace the YAML or invalidate motion to add this metadata.

| Model | Practical use | Hands / options |
| --- | --- | --- |
| ExMR | Default; generally fast, good quality in the project's subjective tests. SMPL-X. | Fingers included; static/moving camera and smoothing options. |
| FlowHMR | Slower; best overall quality in those tests, not a universal ranking. SMPL-H. | Fingers included; Fast SAM and camera/sampling choices. Moving camera needs VGGT-Omega. |
| GVHMR | Lower overall quality in those tests but sometimes better for unusual poses such as backflips. SMPL-X. | Fingers via optional HandFlow. HTD refines body motion; FootMR refines ankles. Both are separate from contact correction. |
| MAMMA | Optional multi-view SMPL-X capture from two or more synchronized fixed cameras, without a fixed camera-count maximum. Practical capacity depends on RAM/VRAM. Not part of the single-video quality ranking. | Fingers included; separate setup and locked-head body assets. VGGT-Omega + GeoCalib or compatible manual calibration. |
| PromptHMR | Multiple people reconstructed jointly from one video; can improve relative placement and interactions. Overall motion quality was lower than ExMR in our subjective tests. Fixed/moving-camera Windows paths, up to eight people; placement can still be wrong. | No finger tracking; fixed neutral fingers. Setup_PromptHMR.cmd; People defaults to 2, Interaction Prompt on. |
| GEM-X | One fixed-camera video and one automatic primary person. Fast processing with relatively low overall motion quality in the project tests. Native77 SOMA at 30 fps for trims up to 60 seconds. | Body and fingers, foot and hand correction, native FBX/BVH and Blender export, and character retarget. [Setup_GEMX.cmd](models/GEMX.md#setup), then Model Manager. |
| PEAR | Live single-person bone preview from a webcam or the video timeline; transient, with lightweight live foot lock and no recording/export or offline correction. | Fingers included; Start/Stop, fixed lengths/Reset Bones, One Euro filter, Rotation and default-on World Position. Shared support is installed by install.cmd. |

Start with the [model guide index](models/README.md), then [ExMR](models/EXMR.md), [FlowHMR](models/FLOWHMR.md), [GVHMR](models/GVHMR.md), [MAMMA](MAMMA.md), [PromptHMR](models/PROMPTHMR.md) or [GEM-X](models/GEMX.md) for exact setup and controls. The [user guide](USER_GUIDE.md#recorded-processing-time-and-vram) includes observed timings, conditions and VRAM; these are historical measurements, not minimum specifications. FlowHMR has a 5,000-frame per-take limit and available memory may require less. MAMMA retains its 4–1,800-frame common trim and eight-person limits; there is no fixed camera-count maximum. Four-column thumbnails are a layout, not admission control. Do not change numerical precision, temporal context or model settings merely to improve a benchmark.

Use **Model Manager** for the selected model and add-ons. Automatic download may fail, especially on Google Drive. Gated models require the user to register, accept upstream terms and obtain files; use **Manual Download...** and **Import...**. Do not handle provider passwords or redistribute licensed weights. Imports validate files; several model receipts tie validity to file size, timestamp and expected hash. If a copied file is not marked ready, import it again rather than assuming it is missing or editing receipts.

### Installation routes

| Command | Scope | Additional prerequisites |
| --- | --- | --- |
| `install.cmd` | Base GUI, ExMR/FlowHMR/GVHMR and supported add-ons, PEAR shared support, Blender, FFmpeg, native correction | See the base installation guide; no PromptHMR/MAMMA native build |
| `Setup_MAMMA.cmd` | Optional independent MAMMA runtime | VS 2022 C++ tools/Windows SDK, CUDA Toolkit 12.8, provider license review |
| `Setup_PromptHMR.cmd` | Optional independent PromptHMR runtime | Same build tools, separate provider license review; MAMMA/SDF not required |
| `Setup_GEMX.cmd` | Optional GEM-X Windows runtime | Reuses the pinned ExMR CPython and compatible CUDA DLLs, keeps its own inference packages, verifies CUDA. No compiler required. |
| `Setup_SAM31.cmd` (Windows), `./setup_sam31.sh` (Linux source) | Optional SAM 3.1 Find person tracker for ExMR, FlowHMR and GVHMR | Base ExMR profile, Git, internet, upstream terms review and NVIDIA CUDA with native bfloat16. Uses pinned wheels and a separate overlay. |
| `update.cmd` | Source update and base runtime verification | Close the app and preserve tracked edits; it does not install the optional runtimes |

### GEM-X support

Follow the [GEM-X model guide](models/GEMX.md). Use the app's pinned setup
and Model Manager. Review the upstream terms before downloading. Keep
`runtime_exmr/` and `runtime_gemx/` in the same installation. A new setup
checks the wheel hashes and tests CUDA before registering a generation.
`install.cmd` and `update.cmd` keep GEM-X optional.

`mocap/gemx_pipeline.py` publishes a native generation only after the clip,
77 joint transforms, shaped rig and evaluated SOMA surface are complete.
`mocap/gemx_result.py` loads the numeric cache in the GUI without a Torch
dependency on GEM-X. Use native joint names for retargeting and preserve
the original target rest skeleton. `gemx_correct.py` derives support geometry
from native SOMA and uses `gemx_contact_solver.py` in the base CPU runtime.
The GEM-X runtime rebakes the corrected bones and surface and publishes a
separate generation. Keep GEM-X out of SMPL preview preparation.
`gemx_skeleton_export.py` sends the selected native result to the FBX and BVH
workers, using the shaped SOMA bind and Generic / Blender BVH profile.
Its portable Blender export
uses relative paths for the exact surface cache and input video. Test
project reopening and both native and character export when changing this
adapter.

Optional Runtime Setup buttons display instructions/copy a command; they do not
install inside the running app. After updates, rerun the relevant optional setup
when requested by its dependency/runtime notice. Keep verified installed packages
isolated even when Torch versions match. Models are obtained separately.

PromptHMR supporting downloads use `cache/prompthmr/torch/hub/` and
`cache/prompthmr/hf/`, independently of video/result generations. The first empty
cache may need internet access. `prompthmr_cache.py` checks known old project
locations with bounded directory/hash work and imports only the exact verified
DeepLab weights; it preserves old files and does not copy executable Hub code.
Use the printed shared-cache location to diagnose repeated downloads. The 14 base files and two optional Moving Camera files are scoped separately. The Model Manager count excludes supplementary download caches. A reviewed exact
pipeline-hash pair preserves valid existing inference receipts for this cache-only
fix; do not extend that exception to numerical/weight changes. No optional-runtime
rebuild is required. See [supporting downloads](models/PROMPTHMR.md#shared-supporting-model-download-cache).

Model files are in `<installation>/models/` by default; the local `config/local.json` `models_dir` / `model_files` entries (written by Model Manager's **Use in Place**) relocate them or use checked files where they are, see [other folders](USER_GUIDE.md#keep-models-in-another-folder). To reuse a legitimate installation, close both apps, copy the complete models folder and its receipt/license files beside `RGBMoCap.cmd`, and check readiness in Model Manager. Do not copy old `runtime*` folders. SMPL Neutral is required by PromptHMR, although it is not required for ExMR/FlowHMR/GVHMR. Standard SMPL-X Neutral is shared by ExMR/GVHMR/PromptHMR; MAMMA uses its separate locked-head bodies. Exact requirements are defined by `mocap/models.py` and the chosen stages, not by guessed filenames.

### SAM 3.1 person discovery

1. Verify the base installation and ExMR runtime. Review the linked upstream
   terms with the user. Close RGB MoCap, then run `Setup_SAM31.cmd` on Windows
   or `./setup_sam31.sh` in the Linux source checkout. For the Linux AppImage,
   run the AppImage with `--setup-sam31` from a terminal. Setup stages the pinned
   Meta source and SHA-256-pinned wheels in a separate `runtime_sam31/` overlay
   and tests native CUDA bfloat16 attention before
   activation. It uses the base ExMR Python and Torch. Keep the ExMR pins and
   the locked base runtime registry intact.
2. In the app, choose ExMR, FlowHMR or GVHMR, then **SAM 3.1** next to
   **Find person**. Open **Model Manager** and use **Auto Download** on the
   optional SAM 3.1 row. For a manual import use the exact public Comfy-Org
   FP16 safetensors file in the [tracker guide](models/SAM31.md#setup).
   **Download All** skips this optional checkpoint.
3. Test a short video with `person` and no markers, then seek to a clear frame
   in the trim, mark a person and click **Track**. Markers select candidates
   found by text in that frame. Check the desired People rows and run
   **Reconstruct 3D**. Check **Default** separately when troubleshooting
   detector behavior. A setup error and a missing checkpoint require different fixes.

SAM 3.1 uses more VRAM than Default (YOLOv8x + OSNet). Treat the
[measured VRAM peaks](models/SAM31.md#vram-use) as observations from specific
clips. Exact overhead relative to Default remains unmeasured on matching
inputs. Find person and 3D reconstruction run in separate stages.

`mocap/sam31_discovery.py` uses the canonical 30 fps working video, the
shared People publishing code and a discovery identity covering the media,
tracker implementation, checkpoint, prompt and relevant frame/markers. Preserve
the previous People result if a prompt or marker fails. Retain source trim
validation and the nearest valid working frame, including the final frame of a
high-FPS source. Keep the detector's final forward batch within the trim.
Resource Mode can offload tracker state and mask caches to CPU RAM. The SAM 3.1
entry point retries a fast CUDA OOM once in a fresh ExMR process and publishes
only the completed result. Preserve Resource Safety and retain OOM recovery
when its limits are disabled. SAM 3.1 applies to offline discovery for the three
listed models.
MAMMA and PromptHMR keep their reconstruction trackers, GEM-X keeps its primary
person detector, and PEAR keeps its live detector.
See the [recorded checks](VALIDATION.md#sam-31-person-discovery-2026-11-10).

## Standard GUI workflow

The People take selector shows the saved frame range and the readiness of the
selected motion result. Export can select exact take paths across people. Validate
the chosen paths before opening the destination picker and again in the worker.
Whole-scene MAMMA Blender exports and their optional face files resolve the
same take paths. Legacy scene requests specifying only people include all of
those people's takes.

### Face capture

**Track Face** starts the isolated ExMR-Python CPU worker after successful body
processing or when enabled on a prepared project. `face_runtime.py` stages
hash-pinned MediaPipe wheels and model assets into the optional app cache.
`face_capture.npz` is a project-local derived file keyed to the working video
and Person track generations. The project view stores **Track Face** separately
from the archive. Restore the checkbox with its signal blocked so opening a
project can reuse the archive without launching capture. **Save Project As**
copies the archive and preserves the timestamps used to validate it.
Per-person validity determines a deterministic
last-observation hold on the viewport and in exports, including backward seeks.
Keep capture off the GUI thread and never feed it into body inference or contact
correction. MAMMA and PromptHMR locate faces in the primary camera with saved
initial joints and the take camera. The viewport binds expression contours to
the installed mesh surface in the same per-take transform. GEM-X retains its primary person boxes
for new results; legacy single-person projects use CPU face search and tracking.
Face sidecars use the selected export’s FPS and timeline offsets. The default
TXT includes source timing and tracking status. `face_export.py` writes optional
Live Link Face CSV in the recorded iOS ARKit column order with a 30 fps timecode
clock and fractional frames for other export rates. The Unreal importer infers
30 or 60 fps from the largest frame field, so keep the CSV timecode base at 30.
Verify per-person associations, hold behavior, zero placeholders, leading frames
and imported Level Sequence timing. For playback, open the imported sequence
in Sequencer and bind the assembled MetaHuman actor to its Live Link subject.
The user guide lists the controls verified in Unreal Engine 5.7. The export
UI enables **Smooth Face** by default. The worker reuses the PEAR adaptive
filter on each actor's valid capture samples before resolving export frames.
Keep unfiltered captures reusable and use the same filtered values for TXT and
CSV. Missing and held frame behavior follows the existing export rules.

### PromptHMR setup and verification

Follow [the detailed setup/workflow guide](models/PROMPTHMR.md). It is a
separate pinned Python 3.11 runtime, not a MAMMA overlay or part of install.cmd.
Setup_PromptHMR.cmd downloads restricted upstream source, builds SAM2 with
Visual Studio 2022/CUDA Toolkit 12.8 and verifies actual CUDA operations. It
reuses standard registered SMPL-X/SMPL bodies, not MAMMA's locked-head body.
Fixed Camera is the default. Optional Moving Camera uses DroidCalib + Metric3D after a separately verified camera build. Find person is hidden; actors retain one
shared inferred world. The adapter accepts 4–1,800 working frames at 30 fps and
up to eight people. Both resource modes keep one-image batches and full temporal
context; extra offloading is not implemented. Preserve actual memory monitoring.

On the test PC its runtime used 6.32 GiB and specific weights/support data
6.09 GiB, excluding shared bodies and compiler tools. Allow about 18 GiB free
for setup before models/tools. The MAMMA and PromptHMR locks share 147 verified
package download archives (about 3.39 GiB); this saves downloads, not installed
runtime isolation.

1. Verify base `install.cmd` completed, check the documented C++/CUDA tools and
   free space, and close RGB MoCap. Do not install the optional runtime if the
   user only needs ExMR, FlowHMR or GVHMR.
2. Show the provider license requirement and follow the user's permitted-use
   decision. Run `Setup_PromptHMR.cmd` from the app folder; never bypass its
   license review or overwrite another profile's packages.
3. Read `.setup/logs/prompthmr-*.log` if setup fails. Verification must exercise
   SAM2 connected components, torchvision CUDA NMS and native attention; package
   import or a successful cached inference alone does not validate a GPU build.
4. Setup registers a successful environment through
   `.setup/prompthmr-runtime.json`; `.setup/prompthmr-install-report.json` records
   the initial installed size/checks. Never manually mark a failed environment
   Ready. Setup retains a matching verified runtime or prepares a new generation
   before activation. Preserve the previous runtime/models/projects on failure.
5. Reopen the app, select PromptHMR and prepare all 14 scoped model/support/body
   rows in Model Manager. Use the exact hashes; Google Drive can require manual
   download/import. A Missing weight is not a reason to rebuild Python.
6. Test a short fixed-camera video, with People = 2 and Interaction Prompt on
   for two-person interaction. Confirm both people, shared-world playback and
   original/corrected selection before export. No animated fingers are inferred.
7. After a GPU architecture change or runtime dependency notice, rerun the same
   setup and actual CUDA checks. Do not copy arbitrary native modules, edit
   receipts or replace package versions to suppress an error.

Native inference, published actors, corrections and exports have separate
generations/receipts. Never load untrusted cached pickles or independently
reposition actors. Valid native hands use the flat-hand-mean representation conversion. Zero-matrix
hand placeholders from the untracked video path use fixed registered-neutral
hands instead of the invalid 120-degree conversion. Preserve body/root/wrist
motion and the native raw result. Older published finger poses need Reconstruct
3D with a verified inference cache; Prepare Preview alone does not republish them. Optional contact
correction is a subsequent intentional pose change. Preserve other runtimes.

### PromptHMR moving-camera extension

`Setup_PromptHMR.cmd` asks separately whether to prepare camera support. Keep
the base runtime usable when that optional build fails. The camera setup reuses
the verified base Python/Torch; `tools/setup_prompthmr_camera.py` stages pinned
Eigen/LieTorch sources and torch-scatter, restores build-only Torch/Python link
libraries from the verified archives/bootstrap, builds for the actual GPU,
and exercises LieTorch forward/backward, scatter and DroidCalib CUDA operations.
Only then atomically register `.setup/prompthmr-camera-runtime.json`. Preserve
previous registrations, packages, projects and weights on any failure.

`installer/prompthmr_camera_adapters.py` prefixes CUDA kernel names to avoid
MSVC merging CPU templates with GPU launch stubs, uses int64 tensor indices,
and relative compilation source paths to avoid duplicated long object paths.
Do not revert these compatibility fixes or mark imports alone as validation.
Compiler tools and temporary build data are not installed runtime size.

`mocap/prompthmr_camera.py` loads the two SHA-256 checked camera weights,
invokes the pinned masked DroidCalib/Metric3D equations, uses Torch SDPA for
Metric3D attention and one-image depth batches. Keep full temporal context and
the Resource Safety-controlled limits and actual OOM handling. Invalid tracking must fail, not silently switch modes.
The camera module identity/weights are included only for moving inference.
`prompthmr_publish.py` preserves the common world, derives each frame's incam
body transform with the registered pelvis pivot, and commits the native camera
track for overlays/gizmos/Blender export. The common scene/floor anchor remains
constant; never independently rotate/space actors or derive camera motion from
the corrected pose. Older projects without a camera option stay fixed.

### MAMMA setup and verification

MAMMA runs an optional complete multi-view pipeline using a separate pinned
Python 3.11 Windows runtime installed with Setup_MAMMA.cmd. The current app
adapter accepts two or more synchronized fixed-camera views, with one common
trim and thumbnail preview switching. There is no fixed camera-count maximum;
retain Resource Safety control over RAM/Windows allocation-headroom stops and
CUDA allocator budgets, plus actual OOM failure handling in either setting. Do not replace the removed camera-count cap
with another arbitrary count limit. See [MAMMA scope](MAMMA.md).
VGGT-Omega plus GeoCalib is the automatic default; compatible manual calibration is optional.
GeoCalib supplies background gravity before MAMMA 3D fitting, replacing the
initial person's head-to-sole vertical axis. Run it in the pinned Flow/Python 3.10
alignment overlay from `install.cmd`, never inside MAMMA's Python 3.11 process.
Model Manager exposes its shared `pear_geocalib` checkpoint. Manual camera data
bypasses both VGGT-Omega and GeoCalib; do not rotate supplied calibration.
Unreliable gravity must fail clearly rather than silently reverting to a person's
pose. The initial reference-height scale and sole height are still estimates.
Keep support/model/code identities in inference receipts and preserve old results
until the user explicitly reconstructs. Original and
corrected native actors feed the usual contact editing and FBX paths. See
[MAMMA.md](MAMMA.md) for the exact environment, registered assets, scale/world
assumptions and adapter limits. The optional setup downloads official restricted source and builds native
modules locally; Visual Studio 2022 C++ Build Tools and CUDA Toolkit 12.8 are
prerequisites. Model Manager shows its runtime readiness and setup instructions.
Do not claim clean-Windows or other-GPU validation for MAMMA. MAMMA uses SAM 2.1 masks. Keep restricted upstream code, weights and derived bodies private.

For prerequisite checks, automatic setup, independent SAM 2/SDF builds, kernel
verification, registration and rollback, use the
[MAMMA agent setup runbook](MAMMA_AGENT_SETUP.md). Registration alone does not
validate actual native CUDA operations. Do not overwrite a working runtime to
test a new native module, or use cached inference as evidence of a new build.

MAMMA's SAM 2.1 input uses `mocap/mamma_sam_frames.py` to decode the already
prepared video directly, without intermediate RGB PNGs or empty index files.
Keep the same PIL resize, float64 pixel division into FP32 frame storage,
in-place normalization and CPU frame/state offload. Final masks and existing
matching/fitting settings remain unchanged. The adapter is worker-scoped,
restores upstream hooks even on exceptions, and holds only the current camera's
frame-source reference; it owns no new captures or background queues. Other
frame-source types keep the upstream path. Include the helper in the MAMMA code
identity; never reuse a mixed/incomplete generation. Existing completed results
remain readable and are not automatically reprocessed on open. No runtime setup
change is needed. Paired 7/14.2-second mask tests cut time 17–21%, changed RAM by
+23/−3 MiB and retained all masks; these are stage-only results. See
[performance and measurement limits](PERFORMANCE.md#mamma-direct-sam-video-input-2026-11-03).

### Capture, correction and export

1. Select an AI Motion Model and prepare its models, then **File > Open Video** and optionally trim the Video track. Select MAMMA before opening multiple videos. New work defaults to ExMR.
2. For ExMR, FlowHMR and GVHMR with multiple people, use **Find person** and check the desired rows; without it, automatic tracking selects one person and can switch identity on overlap. PromptHMR tracks people jointly during reconstruction; its People setting is a maximum count and Interaction Prompt is optional. MAMMA instead detects and matches people during reconstruction; Find person is hidden and its People setting controls the count. A checked row controls visibility and ExMR/FlowHMR/GVHMR processing targets; the selected row controls the current timeline and contact edits.
3. Choose camera and processing settings, then **Reconstruct 3D**. Processing runs on a 30 fps working clip, without changing the original video. Progress and cancellation are in the app. **Prepare Preview** rebuilds previews from existing motion without rerunning inference.
4. Review motion and contacts. **Enable Correction** selects corrected output; Foot defaults on and Hand off for new work. Flat Ground uses a common floor; Multi Level retains inferred contact heights and does not reconstruct terrain. Inspect automatic contacts; hand contacts represent support, not hanging/object tracking.
5. Edit L/R foot and hand intervals on the timeline, then **Apply Correction**. Pending changes are not applied until then; stale corrected export is blocked. Applying also refreshes linked retargeted characters. Correction and preview do not require fresh AI inference.
6. Open **File > Export...** and select FBX, BVH or Blender. Use **Export scope** to choose the current take, all takes for checked people, or individual takes through **Choose takes…**. A blocked export lists each affected person and frame range with the cause and next action. FBX and BVH save separate files for separate takes. Blender can also combine the selected takes in one file; MAMMA whole-scene export keeps the original timeline and all calibrated cameras. The BVH profile renames joints in the exported copy when Biped is selected. An optional leading reference frame shifts the exported motion by one frame and follows the FBX A/T reference-pose choice. Character exports have their own FBX/VMD dialog.

See [the full workflow and controls](USER_GUIDE.md#start-here). Do not confuse viewport toggles, automatic detection and applied motion corrections. No general scene geometry or collision reconstruction is implied.

Contact correction v9 joins free-foot transitions to the final planned root
and evaluates floor clearance continuously inside the bounded trajectory plan.
Keep stance anchors/timings and source ankle frames fixed; do not smooth final
leg quaternions or add a second IK pass. The native anatomical solver remains
the only publisher of leg rotations. Planning-limit warnings are real residual
notices. Existing saved previews are not automatically replaced; use Apply
Correction to recompute without inference. See the v9 measurements and limits
in [validation](VALIDATION.md#contact-correction-v9-coherent-root-and-swing-transitions-2026-10-30).

### Preview-video export

**File > Export Preview...** saves one MP4 pairing the selected input-camera
video with the actual current 3D viewport over the timeline trim. Preserve the
user's camera/framing, visibility, overlays and Follow Actor choice. PEAR is
excluded. The output uses timeline FPS and selected-source audio when present;
it does not run inference, correction or skeleton export. See
[Preview video](USER_GUIDE.md#preview-video).

`preview_export_ui.py` waits for matching decoded video/3D frame indices and
character matrices, captures the existing framebuffer on the GUI thread and
restores frame/camera/playback afterward. `preview_export.py` composes and encodes
off the GUI thread with a one-frame handoff, bounded FFmpeg logs and existing
FFmpeg. Keep the window/viewport geometry fixed during capture, reject changing
input generations and atomically replace only a finished destination. Cancel
and window close must reap the encoder before destroying its QThread. Keep test
projects, input media and output videos outside the public source package.

### Blender bone streaming

See the [installation steps and controls](USER_GUIDE.md#stream-bones-to-blender). Install `blender_addon/rgbmocap_bone_stream.zip` from Blender's **Add-ons > Install from Disk** menu. The ZIP contains the adjacent Python source as `rgbmocap_bone_stream/__init__.py` and the MIT license. Keep the packaged source and the hash in `installer/source-resources.json` synchronized when the add-on changes.

`mocap/stream_ui.py` samples ready frames for checked offline people, independently of People visibility. Multi-person streams use each take's shared scene transform. Selecting PEAR stops the stream. `mocap/bone_stream.py` sends Blender Z-up rest and pose matrices on a background thread and keeps only the latest live snapshot. The listener binds to `127.0.0.1` on the port selected in the Stream tab, default `49168`. The port is a session setting and **Reset** restores the default when available. Bind the replacement listener before closing an active stream so a port conflict leaves the current connection intact. The receiver must use the same port and reconnect after a successful change. Selection metadata is requested by compatible receivers.

`blender_addon/rgbmocap_bone_stream.py` applies poses on Blender's main thread, within the scene and view layer chosen at Connect. The stream collection stores references to its original armatures, keyed by person and rest skeleton. Keep those references when reusing rigs. Clean checks selection and scene references again at confirmation and skips armatures used by other objects. Bake reads the recorded takes on the sender thread, writes a separate Action in Blender, and detaches only after the action is assigned successfully. The app unchecks that person after Blender acknowledges completion.

### Unreal Live Link body streaming

RGB MoCap Connect 0.5.0 supports MetaHuman and UE5 mannequin targets. The
[Unreal installation and controls guide](UNREAL_LIVE_LINK.md) covers both
connection types. The main README links to the 0.5.0 Win64 receiver ZIP.
The inline
[MetaHuman streaming demo](../README.md#unreal-engine-metahuman-live-streaming)
lives under Sample videos. Use the [plugin guide](../unreal_plugin/RGBMoCapMetaHuman/README.md)
for receiver installation and connection. The Unreal guide also covers the
manual reference FBX and IK Retargeter workflow.

The selected editor target type applies to the next connection. Each
URGBMoCapConnectComponent persists its own type and defaults to MetaHuman
for older maps and Blueprint callers. Resolve the primary Character mesh or
one compatible mannequin mesh. Share validation between the editor and runtime.
Keep source-qualified body and reference sampling unchanged. For mannequins,
read the target rest skeleton, use a shared horizontal ankle offset and drive
the stock hand and foot IK guides. Verify editor save and disconnect restore
the target's original animation and preview tick settings.

Stream target 2 is Unreal Live Link. Target 0 is None and target 1 is Blender.
Keep PEAR excluded.
`Setup_UnrealStream.cmd` / `tools/setup_unreal_stream.py` build the source in
`developer-build/unreal-live-link` against the user's installed UE 5.7+ and
verify commandlet startup before keeping a receipt in `runtime_unreal_stream`.
They do not modify the shared inference registry or distribute engine binaries.

Before running it, verify the normal `install.cmd` runtime exists, close RGB
MoCap and check Unreal plus Visual Studio 2022 **Desktop development with C++**
(MSVC toolchain and Windows SDK). Do not infer compiler readiness merely from
an installed Unreal editor. The prebuilt MetaHuman Connect receiver targets
stock UE 5.7.4; use the receiving project's matching engine for sender setup.
From the app root, run `.\Setup_UnrealStream.cmd` in PowerShell (or
`Setup_UnrealStream.cmd` in CMD). For a custom/multiple-engine installation use
`.\Setup_UnrealStream.cmd --engine "D:\Epic Games\UE_5.7"`; the argument is the
installation root containing `Engine`, not the project or executable. Automatic
discovery only checks the standard `Program Files\Epic Games\UE_5.*` location.

Read the exit status and the **Live Link support built** completion line; the
wrapper's final pause is not proof of success. Setup verifies the actual local
provider handshake before accepting its receipt. On failure, resolve the stated
runtime/compiler/engine error and retry; preserve user projects and previous
receipts, and redact logs before sharing. Reopen the app after success. Rebuild
after provider source or engine changes; this is not a per-session step.
`install.cmd`/`update.cmd` do not run this optional setup, and installing the
receiver ZIP does not replace it. See [Prepare once](UNREAL_LIVE_LINK.md#prepare-once).

`mocap/unreal_stream.py` shares the prepared StreamRig/playhead source contract,
converts off the GUI thread, and retains a bounded latest snapshot. The bridge
uses official ILiveLinkProvider and stock Message Bus reception. Do not replace
the verified cm/parent-local quaternion basis with Blender matrices directly:
rest and animation must use the same handedness/facing conversion. Preserve
multi-person scene transforms, reference/leaf lengths, removal on missing motion,
reference refresh on rest changes, and actual Provider HasConnection status.

Default paused pose refresh is 30 Hz and reference 10 Hz. Switching targets,
entering PEAR or exiting closes only this app's owned bridge. Window close/aboutToQuit
explicitly closes the transport; its non-daemon thread reaps the owned process off
the GUI thread. Do not rely on the thread flag to initiate shutdown; a lost IPC heartbeat
also stops the native bridge. Keep Blender's port rollback, packets and Bake
commands on its own path. Test numeric round trips, real native reception and
GUI selection/seek/scope changes. Receipt/build success alone does not verify
MetaHuman retarget quality. Do not copy test MetaHuman assets into the repository.

The Stream **Export Reference FBX...** button freezes the highlighted person's
current stream rig, independently of its checkbox/connection. With only one
listed person, export selects that row if no row is current. Do not silently
choose among multiple people. Report missing selection, unavailable playhead
motion and export errors with a visible message box as well as status/log text;
checkbox-only selection must not look like a dead Export button. `stream_reference`
is dispatched by `worker_progress.py` in the shared CPU runtime; do not put this
presentation-only operation in the inference fingerprint worker. The reference
writer derives the same Unreal local rest transforms as `rig_schema`, attaches
a procedural weighted mesh to every joint, then uses the installed Blender
writer to produce a standard FBX with no animation/extra leaf bones. Preserve
the verified cm/unit-scale and Z/X bone-axis options in
`mocap/stream_reference_scene.py`. Raw intermediate FBX is private to Blender.
Final files are atomically published; owned staging/requests are cleaned on
success, failure or cancellation. Failed export must retain an existing target.
A reference-only completion must not reload the people scene or change playback
placement. Window/app shutdown must close the stream even when no job is busy.

The final reference display mesh is rebuilt in the imported armature's local
space inside Blender, with weight-1 named groups and a standard Armature modifier.
Do not reuse the intermediate display skin's bind matrices or change the verified
reference/pose basis to compensate for geometry errors. Validate displayed skin
separately from skeleton matrices. Keep Unicode label/name limits consistent with
Unreal FString UTF-16 lengths; truncate labels without splitting a code point,
and reject overlong bone identities rather than truncating them.

### Saved camera display and source export implementation

`mocap/camera_tracks.py` adapts committed camera receipts without inference or
body-based camera refitting. Single-person FlowHMR may have no
`scene_camera.json`; use its validated `flow_world.json` basis/height and native
`flow_camera.npz` without creating a scene receipt or fitting the corrected pose. MAMMA native Z-up calibration is transported to the
app's Y-up world. FlowHMR/ExMR use saved rotation tracks. Flow gizmos additionally display
unscaled VGGT displacement; moving-camera ExMR gizmos use estimated displacement
from ORIGINAL paired world/camera root positions, never corrected poses.
Motion x defaults to 1, with 0 retaining a fixed center. Video overlay projection
remains unchanged when this display scale changes. PromptHMR Moving Camera uses
its native saved per-frame DroidCalib/Metric3D trajectory for world fitting,
overlays and Blender camera animation, without a gizmo-only translation scale.
MAMMA/GVHMR and PromptHMR Fixed Camera remain fixed. Do not interpret
an extrinsic translation vector as the camera center: use `-R.T @ t`.

Single-person ExMR/GVHMR may also have no shared-scene camera receipt.
`mocap/camera_reference.py` reads the original paired global/camera parameters
in a cancellable shared-runtime CPU subprocess, with weights-only memory-mapped
tensor loading. It derives the shape-dependent pelvis pivot from the committed
original joint cache and never fits to corrected bodies. Camera references are
kept in a maximum-eight-entry in-memory loader cache, validated against original
source stamps. Do not import Torch into the editor, write scene/floor receipts,
run inference or change motion to display a missing camera.

`mocap/camera_ui.py` owns one coalescing camera loader and a Camera Preview canvas
sharing the existing source QImage. Its video paint callback also projects the
reconstructed skeleton into the left video view when **Overlays** is on,
independently of the right Bone Overlay/Camera switches. Keep image, camera and actor skeletons paired
at frame delivery, including camera switches and rapid seeks. Do not load camera
files in painting, add a decoder, run Torch in the GUI or alter source camera
receipts, floor or poses when a display checkbox changes. Camera gizmos/path samples are bounded;
missing/stale inputs must be visible. Camera Preview is disabled for PEAR.

FlowHMR/ExMR's default-off **Refine Camera Rotation** is an explicit exception
for derived camera display matrices, not source processing. `camera_background.py`
adapts PR #18's masked first-frame feature tracking, held-out validation, 2-degree
limit and sampled interpolation. Failed/rejected observations retain the saved
rotation, and static-camera results skip the operation. Preserve its acceptance
thresholds rather than forcing a correction when texture or parallax is unsuitable.

`camera_background_worker.py` uses the existing shared runtime in an isolated
CPU-only process; `worker_progress.py` routes the stage before inference/cache
fingerprinting. One coalescing camera loader owns it, cancels/reaps it on toggle,
replacement or close, and never imports Torch into the GUI. Windows progress is
read from a bounded, atomically replaced transient file at no more than 5 Hz plus
phase/final updates. Opaque startup remains active with elapsed time. Temporary
input/output files live under `cache/camera_background/job-*` and are cleaned
when the request ends; stale directories after an abrupt application termination
are disposable with the app closed.

The session refinement cache is limited to four entries and 32 MiB of changed
matrices, validated against input stamps and the camera/core/worker code digest.
Unchanged results do not store another matrix copy. Matrix payloads are limited
to 32 MiB, reports to 1 MiB, captured logs to 512 KiB, fitting samples to 600 and
worker runtime to 180 seconds. Known actors must have usable tracking boxes.
Temporary preparation can use additional bounded memory beyond the cache limit.
No new model/runtime packages, inference signatures, source camera receipts,
source poses, contacts or FBX/BVH animation may change. Checkbox state belongs
only in workspace view metadata (`camera_background`). Test accepted and rejected
cases, exact off restoration, cache invalidation, cancellation, source preservation
and frame/image/camera pairing separately.

`mocap/source_export_dialog.py` supplies source export options.
`biped_bvh.py` adapts PR #18's Character Studio naming and uniform ZXY
conversion for all 52 joints, including fingers. `bvh_profile` propagates through
every person/take export request; reject unknown profiles. Generic output remains
a byte copy when no placement/T-pose preparation is needed. Biped uses bounded
128-frame batches; only placement/T-pose preparation needs a temporary intermediate
file, which is closed on success/error/cancellation. Preserve original offsets,
units, timing, physical keyed rotations and source caches. Uniform Euler order can
change fractional interpolation; do not claim identical between-key motion or
verified 3ds Max behavior from the Blender/numerical checks alone.

`source_export_worker.py` routes BVH saves through the cancellable worker without
changing AI worker/cache fingerprints. Keep selected-result, pending contact and
shared-scene guards, generation checks and atomic destination replacement in
`fbx_export.py`/`export_copy.py`. Unchecked FBX output must remain identical.
**FBX Compatibility (Maya)** is default-off and applies to source FBX, including
GEM-X Native77. The worker prepares the selected take with the existing export
options, then runs `fbx_compatibility_scene.py` in the included Blender runtime.
Reuse the existing rest-pose sampler so the final FBX keeps the original bind
pose and writes animation keys to bones. Verify the source generation again
before publishing the converted file. The worker owns and cleans its staging
folder using the Blender export marker.
Source FBX requests may set `fbx_rest_pose` to `t` or `a`, including Native77.
Apply the pose after any symmetrical-rest geometry changes. Rotate the entire
upper-arm subtree about the shoulder so child offsets and terminal tip offsets
stay fixed in parent space. Write the choice to the bind-pose table and static
arm rotations. Keep the same PreRotation and motion key values so FBX tracks
keep their interpolation between keys. The optional first frame follows the
selected reference. Older requests without the field use the legacy writer
behavior. Validate the bind and keyed/fractional poses after reimport, including
the Blender compatibility pass.

### UE5 Mannequin FBX export implementation

The `ue5_mannequin` FBX profile selects `ue5_target` (`manny` or `quinn`)
and `ue5_in_place`. The default-off `ue5_include_mesh` sends the chosen
stock skin to the Blender worker. The profile uses the same committed take
and generation checks as Generic FBX and supports mapped SMPL and SOMA
Native77 inputs. `mocap/ue5_reference.py` loads the bundled reference and skin
under `third_party/ue5_mannequin`. Keep the Epic notice with the 5.7.4 Third
Person template data. Match the skin and reference to the same template build
and update their byte counts and hashes in `installer/source-resources.json`
and `SOURCE_MANIFEST.json` after regeneration. Follow the
[bundle notice](../third_party/ue5_mannequin/NOTICE.md) for provenance and build
commands. Export needs the character Blender runtime. The Live Link sender
and MetaHuman receiver are separate from this export path. See the
[user workflow](USER_GUIDE.md#ue5-mannequin-fbx) for Unreal import settings.

`mocap/ue5_retarget.py` uses source pose deltas, target lengths, a fixed-length
leg solve and target-oriented IK helper tracks. The stock root carries planar
translation only. Keep one horizontal offset for both ankle targets so the
source stance width scales with the target leg length.
`mocap/ue5_fbx_scene.py` writes the optional skin and animation with Blender in
the owned export staging directory. The rig-only path emits an explicit FBX
`BindPose` record. Preserve the stock rest transforms in both output modes,
including a unit-scale root and centimetre bone offsets. Match the elbow and wrist
directions from source joint positions. Calibrate palm and finger axes from
both rigs' geometry, including the source's registered terminal tip offsets.
Test Manny and Quinn against Epic's existing `SK_Mannequin`. Compare the
evaluated elbow plane, palm and all digit directions to the source alongside
timing, scale and skeleton-asset preservation. With the skin enabled, verify
mesh deformation in Blender and test Legacy and Interchange imports in a fresh
Unreal project with no mannequin assets. Check that each importer creates a
Skeletal Mesh, Skeleton and Animation Sequence with the stock hierarchy.
Compare bind transforms and evaluated poses against the rig-only export.
Use a take whose first motion frame differs from the stock pose. An importer
can use frame 0 as the reference when it rejects the FBX bind pose. Check a
take with varied ankle spacing and verify root travel with `ue5_in_place`
both enabled and disabled. See the
[bind pose and foot spacing checks](VALIDATION.md#ue5-mannequin-bind-pose-and-foot-spacing-2026-11-20)
and [original mesh export checks](VALIDATION.md#ue5-mannequin-fbx-with-optional-mesh-2026-11-20).

### Portable Blender export implementation

Source FBX/BVH requests accept default-off `place_at_origin` and `symmetric_rest`.
`gui.py` attaches a stable `origin_group` for each person, distinct from display
names. Within an export batch, each actor's first exported take determines one
constant XZ offset; later takes retain their relative travel and Y is untouched.
This intentionally discards placement between different people.

`export_options.py` constructs mirrored averaged-length rest offsets and a
vertical central chain. BVH changes exported hierarchy offsets; a rest-only
Generic export preserves MOTION bytes. FBX uses the same canonical reference
geometry with new proper display/rest bases and corresponding terminal markers.
Validate input rigid offsets against the original committed bind, not the new
lengths. Do not materialize a full new animation or rerun inference/contacts.
Changed proportions intentionally change animated endpoints. Both options off
retain the existing bind/output. `blend_export.py` explicitly clears both flags
before its nested FBX export; MAMMA whole-scene requests never propagate them.
Keep source results/receipts immutable, stream bounded BVH blocks, and validate
ordinary Blender imports, profiles, T-pose combinations and independent actors.

`blend_export_ui.py` constructs requests and selects new, collision-free bundle
folders. `blend_export_worker.py` routes `save_blend`/`save_blend_scene` through
`worker_progress.py` before inference fingerprinting. Keep the existing runtime
registry and inference worker untouched. General bundles in `blend_export.py`
reuse `camera_tracks.load` and the isolated original-reference loader; never
re-estimate the camera from a corrected body. Export only the requested take
interval, transport the skeleton once through its existing shared placement, and
use cameras already in that common world without applying the placement twice.
Optional camera tracking, movement scale and background refinement affect only
exported cameras. T-pose shifts both motion and video by one frame.
One-file requests use a single staging bundle and assign take offsets from the
earliest selected source start at the common export frame rate. The Blender
builder offsets each armature action, keys take visibility and binds background
cameras to timeline markers. Reuse identical video files and keep paths relative
to the bundle. When enabled, the leading T-pose occupies one shared frame.

`blend_multicam_export.py` adapts PR #18's full-scene pipeline: current checked
MAMMA people/takes, full selected SMPL-X LBS surfaces streamed one frame at a
time into PC2, source skeleton FBX, saved fixed calibration and working videos.
`blend_multicam_scene.py` builds an overview plus per-camera linked scenes.
Preserve half-open take visibility and the original shared timeline. Body meshes
use caches, not armature deformation. Keep model-derived asset inputs private;
do not distribute body files or model assets with application source. Only the
user-requested animated scene is exported. Registered assets retain their terms.

The builders `blend_export_scene.py`/`blend_multicam_scene.py` run only in the
existing Blender 4.5 runtime, with autoexec disabled and bounded CPU threads.
They install no frame handlers/scripts. Movies and PC2 use relative paths; test
relocation including Unicode folders. Verify imported bone matrices, actual
PC2 evaluated vertices, camera projection (including unequal fx/fy and shifted
principal point), frame timing and object-transform key exclusion.

`blend_support.py` owns exclusive sibling staging directories, bounded video
copying and log reads, free-space checks, progress forwarding, and native-process
termination on errors/cancellation. Remove only exact owned staging paths with
the matching request marker; never remove completed bundles or existing user
folders. GUI cancellation terminates the worker tree and cleans its known staging
directory. Revalidate source/stat/generation/scene receipts before publishing.
MAMMA currently bounds the common trim to 1800 frames/eight people; geometry
baking must not accumulate whole mesh sequences in RAM. No new inference,
runtime dependency, source motion or contact change is part of export.

## Character retargeting and export

- **File > Import Character...** accepts Mixamo-style FBX, VRM and PMX, and adds the character in rest pose below people. Up to eight characters per project are supported.
- Right-click the character and choose **Retarget > [source person]**. Review via timeline; row checkboxes control visibility. Custom bone mapping is not implemented. Unsupported or incomplete mappings must not be silently relabeled.
- The source's applied contact timings are used. **Apply Correction** refreshes linked characters; after other source changes retarget again. Targets that cannot be reached do not cause bone stretching. Different limb lengths can change travel distance, and editing may be required in a 3D tool.
- **Export Character...** offers FBX with or without character mesh, and VMD for PMX characters. VMD is sampled at 30 fps for the same PMX model; facial motion and hair/clothing physics are not generated. Character FBX preserves its imported rest hierarchy and keys body bones, not the armature object. The character viewport uses original-resolution base-color textures with simplified lighting. Texture caches are prepared off the GUI thread, independently of motion; existing characters prepare them on first display.
- A separate Blender runtime is used for character import, rig evaluation and baking/export; the user need not open Blender for the app workflow. Preserve upstream licenses and the bundled Blender environment. See [character retargeting](USER_GUIDE.md#character-retargeting) and [FBX/Blender](USER_GUIDE.md#fbx-and-blender).
- MAMMA retargeting must use `mocap.mamma_body.body_path(project)`, including its actor-specific folded shape data. Never substitute the standard SMPL-X body used by ExMR/GVHMR, or require that extra model for a MAMMA-only workflow.

## Projects, state and protection

**Save Project As** counts shared hardlinked files once and creates any new
links entirely inside the copy. If a destination cannot create a hardlink,
recheck the space budget before falling back to copying. Keep project manifests
independent so staging can rewrite location metadata. Character index writes
preserve unreadable input files before committing a replacement.

- Default projects live under `<installation>/projects/`; **Save Project As** creates and switches to a copy at a user-selected location. Open the saved `project.json` using **File > Open Project**. A project contains model-specific `exmr`, `flowhmr`, `gvhmr`, `mamma` or `prompthmr` result folders; model switching loads existing results when available. For MAMMA, **Save Project As** copies the original camera inputs into the saved project. People, takes, contact edits and imported characters are project data.
- A saved MAMMA take restores camera thumbnails and switching from the prepared clips in the project, including when originals are missing. Keep camera order and the saved take offset. Disable a view when both its prepared clip and original are unavailable. Keep original input paths and metadata for reconstruction. See [MAMMA playback](MAMMA.md#reconstruct-and-export).
- `models/`, `projects/`, `cache/`, `character_scenes/`, `runtime*/` and `.setup/` are local data, not source artifacts. User project folders can also be outside the checkout. Never assume `.gitignore` protects a path that was already tracked or will be force-added. Check `git status` and tracked files before committing; do not upload model weights, video, personal data, credentials, generated motion or caches. Explicitly approved README sample assets are exceptions.
- Preserve source videos, registered weights and project data. Do not delete project subfolders, results or receipts to fix a problem: they can be referenced by saved metadata. Cache cleanup and setup-download cleanup are distinct from deleting saved work; see [storage and cleanup](USER_GUIDE.md#storage-and-cleanup). Obtain authorization for destructive cleanup.
- `mocap/paths.py` defines `ROOT`, `RGB_MOCAP_DATA`, models and projects; normal portable startup sets its own data location. When diagnosing an actual run inspect the effective environment rather than assuming a path.

## Resources and performance

`sidebar_scroll.py` routes ordinary wheel events over closed combo boxes and
number inputs into the right settings scroll area. Do not let scrolling mutate
inference/correction options and trigger model checks or viewport repainting.
Preserve native high-resolution deltas, open popup scrolling, keyboard/text/button
edits and deliberate focused Ctrl+wheel. This is scoped UI input routing, not a
model/cache/precision change or an application-wide wheel filter.

**Resource Safety** defaults on below Resource Mode. Off skips the measured
RAM/commit-headroom stop and sets the PyTorch allocator fraction to 1 (no app
VRAM cap). Keep adaptive batch/path selection, actual OOM backoff/recovery,
the independent stall check, cancellation and generation protection in both
settings. Do not describe this option as disabling every memory check or as a
guarantee that a workload fits. See [behavior and thresholds](PERFORMANCE.md#resource-safety).

It is an installation preference in ignored `config/local.json`, not a project
or inference option. Preserve other local settings when writing it. GUI jobs
snapshot the preference and pass `RGB_MOCAP_RESOURCE_SAFETY=1` or `0` into all
worker runtimes, including isolated launches; only explicit `0` disables it.
PEAR workers inherit the same preference. Never remove actual OOM handling or
change model equations/precision when extending this control. Model result
identity must not include this local checkbox value.

Use **Resource Mode > Automatic** by default. It keeps the fast path when the current GPU budget is sufficient and selects supported Low Memory paths per operation when needed. A supported fast operation that raises CUDA OOM is released and retried once in Low Memory. ExMR Low Memory uses CPU-first body/hand loading and converts Sapiens' existing FP16 layers before GPU transfer; ExMR body/hand batches stay 8. FlowHMR offloads SAM/VGGT blocks; MAMMA offloads VGGT camera blocks while retaining its pinned masks/landmark/fitting settings. GVHMR/add-ons retain adaptive image batches and full-context attention. The saved Automatic selection is not changed. Resource decisions and memory advice appear in the app Console and CMD. Model precision, frame count, resolution and iterations are not reduced. CPU loading/offloading can be slower and use more RAM; it is not a promise every clip will fit. Estimates select a path, never reject a job. With Resource Safety enabled, the desktop checks actual RAM and Windows allocation headroom every 0.5 seconds and stops workers when these become low. Preserve completed stages, genuine model limits and original motion. Do not rerun inference for operations that can use existing results.

PromptHMR retains the same one-image batch/full temporal context in Automatic
and Low Memory; extra block offloading is not implemented. Do not promise a
mode-dependent VRAM saving or an unsupported offload retry for this path. Actual
allocation failures are still reported; proactive limits follow Resource Safety. Its 422-frame/two-person run took 236.2 s, with peak
allocated VRAM 4.61 GiB (7.83 GiB reserved) and worker working set 8.74 GiB; see
[scope and stage times](models/PROMPTHMR.md#recorded-validation).

With Resource Safety enabled, GPU allocator limits share ComfyUI's external-use margin: 600 MiB on Windows, 700 MiB above 15 GiB GPU capacity (400 MiB on other systems). Subtract it from actual free VRAM without an additional percentage cap or fixed FlowHMR/ExMR 3/2 GiB reserve. This margin is separate from inference working space. Keep offloading in the app's validated `memory_mode.py` / `gpu_offload.py` adapters, without importing ComfyUI or changing model precision. Offload is inference-only, synchronous and unpinned; close hooks/models on success and failure. Actual RAM pressure remains a stop condition when Resource Safety is enabled because offloading can increase host memory.

### Measured savings and acceleration

`gpu_offload.load_vggt` uses meta initialization plus `assign=True` only for the
camera-only fast path. Preserve copy loading for Low Memory (including Automatic
fallback/retry) and generic depth loading. The Low Memory assign experiment
increased Windows private allocation despite slightly lower working set. Require
all checkpoint tensors to be materialized before device transfer; do not fill
missing meta tensors with guessed values. Camera outputs must remain unchanged.
This is a loader optimization for FlowHMR/MAMMA VGGT, not a new precision mode or
a global loading policy for the other models. See [its measurements](PERFORMANCE.md#vggt-camera-checkpoint-loading).

Use [PERFORMANCE.md](PERFORMANCE.md) for recorded conditions and current applied
methods, and [older processing records](USER_GUIDE.md#recorded-processing-time-and-vram)
for historical long-clip observations. Do not present old timings as a current
release forecast. On the documented RTX 5090 PC:

| Observation | Result and scope |
| --- | --- |
| ExMR Automatic → Low Memory | Allocated VRAM **5.68 → 3.87 GiB**, RAM working set **11.88 → 11.94 GiB**, **89.01 → 94.78 s**; earlier 60-frame app-mode check. |
| FlowHMR Automatic → Low Memory | Allocated VRAM **6.10 → 2.77 GiB**, RAM **8.42 → 8.75 GiB**, **95.24 → 96.15 s**; earlier 60-frame app-mode check. |
| GVHMR + add-ons, same short mode check | No meaningful whole-job saving: **4.204 → 4.203 GiB** allocated VRAM. Low Memory need not benefit a short job that already fits. |
| MAMMA | Whole-pipeline paired VRAM saving was not recorded. Shared VGGT component offload saved **52%** allocated VRAM but took **3.3×** as long in warm inference; it is not whole-MAMMA offloading. |
| Connected phase workers | One-person / 60-frame worker time fell **44% for ExMR**, **24% for FlowHMR**, with unchanged measured peak VRAM. Excludes discovery, cameras, correction and export. |
| Bounded prefetch / warm feature stages | Flow Fast SAM about **5% less time**; ExMR Sapiens/hand about **25% less**, body with unused readouts omitted **29% less**. Similar VRAM, sometimes slightly more RAM. |

The mode checks predate the latest acceleration. Low Memory saves GPU memory,
not necessarily RAM; individual CPU-loaded models can add several GiB of host
working set. Automatic selects a path per operation, not from a guaranteed
whole-job capacity prediction. FlowHMR's Low Memory SAM batch is one even when
the saved Adaptive Batching box is on; its GPU Hand Crops selection is retained.
The three Fast SAM switches default on for new work, preserve saved off settings,
and only apply with Fast SAM. See [their separate measurements and output differences](PERFORMANCE.md#flowhmr-fast-sam-switches).

FlowHMR additionally uses the reviewed 256-query attention implementation for
every clip length, with FP32, unchanged visibility/RoPE and full sequence context.
Do not substitute temporal chunking or change steps/precision. Network sources
are recorded in `third_party/FlowHMR/RGB_MOCAP_PROVENANCE.json` and motion receipts.
No runtime package changes or additional compilation are needed.

The default-on `sam_hand_stride4` setting controls **4-Frame Hand Sampling**.
It is distinct from the spatial `sam_batch4` key: body features stay per frame,
hands use indices 0,4,8,... plus the last frame, and bounded CPU linear feature
interpolation fills all output frames. Keep the same indices across OOM splits,
one-frame Low Memory batches and people/take jobs. `flow_hand_sampling.py` holds
the schedule/interpolation; `flow_fast_sam.py` also supports body-only sub-batches.
The body decoder provides the first placement hips without requiring hand fusion.
Saved off is respected; an absent setting defaults on for future reconstruction,
without automatically replacing cached motion. Include the option and sampling
code in feature identities; never alias this numerical change to an older cache.
Publish no partially interpolated tokens.
Tests measured 41–49% less warm SAM feature time and 5.43→4.21 GiB allocated VRAM
at batch 16, but body/wrist positions and contacts can change too. See
[the model guide](models/FLOWHMR.md#query-batched-motion-attention-and-sparse-hands).

FlowHMR's optional `hamer` flag defaults **false** and forces `fast_sam` and
`sam_hand_stride4` true. The UI locks those dependent switches while enabled.
SAM publishes per-frame body-derived hand boxes with its feature receipt.
`flow_hamer.py` runs `flow_hamer_entry.py` in the existing **ExMR** runtime, never
imports its incompatible packages into FlowHMR, and reuses shared HaMeR/MANO
assets. It applies ExMR crops/mirroring and Gaussian rotation smoothing to
direct HaMeR finger predictions; this is not ExMR's learned HandMR inference.
HaMeR itself uses every frame, eight hand crops per batch (one with CPU-first
loading in Low Memory), without another detector. Keep invalid-hand fallback,
bounded capture/prefetch, child cancellation and atomic cache receipts intact.
Release the parent GPU model before launching the child. Preserve
`flow_prediction.pt` as the untouched FlowHMR output; optional fingers go to
`flow_hamer_prediction.pt`. `flow_placement.py` selects the source recorded by
the committed generation, not a draft checkbox value. Preserve root/body/wrist,
shape, camera and native contact tensors exactly before subsequent correction.
HaMeR's local matrix predictions map directly to SMPL-H's 15 joints per side;
mirror the left local matrices but do not add/subtract a MANO/SMPL hand mean.
Do not promise better accuracy for every video or ExMR-equivalent temporal stability. See
[HaMeR details and requirements](models/FLOWHMR.md#optional-hamer-fingers).

FlowHMR's `flow_sam_batch.py` selects 1/2/4/8/16 frames from current allocation
headroom, free RAM and resolution after loading. Preserve the legacy
`sam_batch4` boolean key so saved off values stay off. The 16-frame ceiling is
a measured efficiency choice, not an upstream model limit. Recheck memory
between prepared batches; retry an unpublished chunk at a smaller batch only
after releasing its OOM traceback. Keep one-frame failures connected to the
existing scoped Low Memory retry. Do not grow queues, change precision, drop
frames or lose the first 30 placement anchors. Include the helper in feature
cache identities; metadata downloads must not invalidate other AI stages.
The 2026-11-02 tests found 16–23% less warm feature time than batch 4, allocated
VRAM 3.84 → 5.43 GiB and about 0.20 GiB more warm RAM. Fresh short app jobs
benefited much less because loading dominated. Official Lite SAM and the combined
GPU body/hand-crop trial were slower here and remain experimental; body-only
cropping was not isolated. Do not silently replace the active
Fast SAM package. See [the full scope](PERFORMANCE.md#flowhmr-adaptive-batches-2026-11-02).

The selected Fast SAM batch upper bound, GPU Hand Crops state, Low Memory
single-frame choice and OOM batch-retry messages are shown in Console and CMD.
Available-memory checks and a final partial batch can lower the effective size;
cached features do not select a new batch. Keep worker `info` events routed through
the console mirror without replacing processing progress or changing inference.

When reporting new tests, keep inputs, model files, settings and timing scope
fixed. Separate cold worker startup from warm inference, allocated from reserved
VRAM, and working set from private/system RAM. Do not sum speedups from different
baselines or use sampled worker peaks as whole-PC usage/minimum requirements.
Fixed-feature motion was exact in the latest validation; fresh preprocessing
had maximum joint differences below 0.9 mm, with baseline repeats also varying.
The GPU-crop comparison is separate and reached 4.8 mm maximum. Neither measures
accuracy against ground truth. Preserve these distinctions and numerical paths.
Keep private benchmark projects, raw logs, account details, machine paths and
licensed assets out of public documentation; publish only curated measurements.

The app uses separate runtime profiles to maintain compatibility, with bounded batches, previews and background workers. Keep video decoding, large I/O and numerical work off the GUI thread. Playback responsiveness, synchronization and resource cleanup are correctness requirements. If changes affect these, measure frame-time outliers and repeated open/close behavior; do not infer universal performance from a short test.

## Code map and verification

- GUI, people, controls and logging: `mocap/gui.py`, `workspace_ui.py`, `people_ui.py`, `character_ui.py`, `console_log.py`, `tooltips.py`.
- Model selection and loading: `mocap/motion_tools.py`, `models.py`, `model_guide.py`; worker routing and runtimes: `worker.py`, `worker_progress.py`, `runtime_registry.py`, `compute_policy.py`.
- MAMMA inputs/UI: `mocap/mamma_inputs.py`, `mamma_ui.py`; processing: `mamma_pipeline.py`, `mamma_camera.py`, `mamma_publish.py`, `mamma_body.py`; runtime/assets: `mamma_runtime.py`, `mamma_manager.py`, `mamma_models.py`, `installer/mamma-runtime-lock.json`, `tools/setup_mamma.py`. Follow [the native-build runbook](MAMMA_AGENT_SETUP.md) instead of guessing module build commands.
- PromptHMR UI/pipeline: `mocap/prompthmr_ui.py`, `prompthmr_pipeline.py`, `prompthmr_cache.py`, `prompthmr_publish.py`, `prompthmr_scene.py`; runtime/assets: `prompthmr_runtime.py`, `prompthmr_manager.py`, `prompthmr_models.py`, `prompthmr_worker_boot.py`, `installer/prompthmr-runtime-lock.json`, `installer/prompthmr_adapters.py`, `installer/verify_prompthmr.py`, `tools/setup_prompthmr.py`. Follow the separate setup/verification steps above.
- Project storage: `project_storage.py`, `project_workspace.py`, `project_save.py`; contact correction: `correction.py`, `contact_trajectory.py`, `contact_pose_plan.py`, `contact_arm_plan.py`.
- Source mesh / bone fit and preview: `source_animation.py`, `source_surface.py`, `surface_roll.py`, `preview_data.py`, `viewer.py`, `viewport.py`; source export: `fbx_export.py`, `fbx_writer.py`, `scene_export.py`.
- Character pipeline: `mocap/retarget/` plus `character_data.py` and `character_ui.py`; Blender backend and model-specific constraints must be considered separately from source-person export.
- Installation/update: `install.cmd`, `update.cmd`, `tools/git_bootstrap.ps1`, `tools/git_managed_install.py`, `tools/git_update.py`, `installer/runtime-lock.json`. Native correction source/build instructions: [developer-build/motion-correction](../developer-build/motion-correction/README.md).

Inspect [BUILD.md](BUILD.md), [VALIDATION.md](VALIDATION.md) and relevant code before edits. The developer `.venv` workflow is separate from ordinary Git/CMD installation. Run targeted checks for modified behavior; real model inference, real GPU use, Blender/MMD import and clean-install compatibility must only be reported when actually tested. This public checkout may not include private regression tests or licensed assets.

## PEAR live mode

PEAR tracks webcam or video input in real time. Support is **experimental**;
tracking and pose estimates may be unstable;
describe it as an experimental preview, not a production-ready capture workflow.

Close the app and run `install.cmd`. The base installer also downloads
pinned source, pure-Python PyTorch3D utilities and RoMa into `runtime_pear/`,
validates CUDA kernels, and preserves the shared runtime packages. Do not install
a second Torch, upgrade global NumPy/timm/Ultralytics or compile PyTorch3D.
Import the PEAR checkpoint and original registered geometry ZIP using Model
Manager. Standard SMPL-X v1.1 and MAMMA locked-head files are incompatible with
PEAR's SMPL-X 2020/FLAME bundle. Read [PEAR](models/PEAR.md) first.
No separate PEAR setup command is required. `update.cmd` uses the same installation
flow and repairs support when its source or shared-runtime group changes; matching
verified generations are reused. Model weights/body data remain manual or Model
Manager downloads. MAMMA and PromptHMR still use their separate setup commands.

This is a transient single-person preview, not a saved offline project model.
`None` in the webcam selector uses video timeline frames; a named camera selects
the webcam input. Neither loads models until Start Live Tracking is pressed. No recording, Find person, offline contact correction, retargeting or
export is available. Preserve offline projects and camera views when switching.
Keep requests and output mailboxes bounded/latest-only, run native imports and
warmup before the Windows blocking stdin reader, and release only the worker
and device owned by this session. Never kill another camera application.
Registered assets and test captures must remain ignored and unpublished.

PEAR is accepted in the tool selector but rejected as a saved offline model,
including workspace active/member identities. Never create or relabel a project
as PEAR. Source/adaptation pins are in `installer/pear-sources.json`; setup retains
upstream licenses, marks changed files and writes `RGB_MOCAP_PROVENANCE.json` in
the private support generation. Its receipt hashes the support-defining source
bytes: preserve line endings and `.gitattributes`, and rerun installation for a
real adapter change instead of editing receipts. The [PEAR citation](CITATIONS.md#pear)
and [detailed guide](models/PEAR.md) distinguish upstream research from the
application's fixed-length display, filtering and position controls.

PEAR's selected-tool state and running state are separate. Selection, video open,
webcam selection, seeking and settings edits must not create a model worker while
stopped. Start Live Tracking explicitly enables loading/warmup and tracking;
the same button becomes Stop immediately, including during loading. Stop
invalidates pending output, stops playback and closes only this session's worker,
capture and GPU data off the GUI thread. Disable Start until the old worker exits;
never overlap model workers or automatically restart them after Stop/error/leave/reselect.
A new Start loads a fresh worker. While running, input changes reuse its model.
PEAR alone uses the pinned YOLOv8s asset (`pear_yolo`, `models/yolo/yolov8s.pt`).
Keep its size/hash-checked Model Manager download/import and trusted loader.
Do not replace the shared YOLOv8x asset used by offline models. Detector boxes
can alter pose predictions; detection success is not an accuracy measurement.
Idle acknowledgement keeps a deliberately waiting started worker from appearing
stalled. Keep Resource Safety-controlled RAM/VRAM limits and bounded mailboxes while it is running.

Selecting a webcam may create the separate `pear_camera_preview` capture worker.
It imports OpenCV/NumPy, not PEAR or Torch, and uses bounded latest-frame buffers
with a bounded 1280x720 displayed image.
Partial slot assignment is previewable; tracking still requires distinct devices
in every slot. Keep per-device rotation and thumbnail selection active. This
worker alone owns the physical devices and remains connected during Start and
Camera Align. `pear/camera_stream.py` publishes one fixed-size Windows shared
frame per device under a named mutex; model workers use independent RemoteCamera
readers and never reopen those devices. Validate descriptors/device identities,
retain bounded copy/age checks and release reader handles on worker replacement.
Shared raw frames have a 64 MiB per-frame / 512 MiB aggregate allocation limit;
these are not whole-process RAM limits or silent downsampling. No per-frame file
writes are introduced. Keep PEAR and calibration GPU models sequential while
capture and image preview continue. Stop suspends automatic camera
preview until device/rotation/resolution changes or Refresh. None/leave/app close
release preview capture too. Failure stays visible without an automatic restart
loop. Preview does not need model weights or an installed PEAR support overlay.
Video decoding/playback remains available while models load or cameras align.
Freeze alignment sample frame/generation at the click, separately from later
seek epochs. Raw preview rotation is painting-only; completed rotated image/pose
pairs must not be rotated twice. Reset the raw-preview rotation on leaving PEAR.

During PEAR video playback, present each completed image/pose pair together,
accepting monotonically newer results within the current source/seek epoch.
Never require a playing result to equal the newest outstanding request: inference
latency would discard it indefinitely. Preserve the last pair until its replacement.
Manual seeks, loops and pause invalidate old epochs; paused seeks require the exact
request. Keep the request clock separate from the actual displayed frame. The
ordinary decoder is suspended during paired presentation, including exact paused
inference, and resumes after Stop or leaving PEAR.

PEAR live mode now uses `mocap/pear/fixed_rig.py`: fixed-length physical FK pivots,
each bone's own predicted rotation and constant neutral axes. Initial calibration
and **Reset Bones** use a resident CPU EHM model once; ordinary live frames do not
skin a body mesh, regress mesh joints or evaluate animated surface-roll evidence.
Neutral knee/elbow patches and the neutral foot-width basis remain calibration
references only. Never use an animated bend plane or the child rotation to select
incoming upper-limb/foot roll. Fixed FK pivots differ from the previous skinned
landmarks; do not claim identical mesh registration or ground-truth accuracy.
The Bone Overlay checkbox changes drawing only, not inference or calibration.

Calibrate once on the first valid frame of a new input; preserve that rig across
seeks, filter changes and temporary missing detections. Reset Bones is enabled
only with a displayed pose. It pauses video at the displayed frame and sends its
pose ID, not the outstanding playback request. The worker keeps at most 32
parameter-only references (no video/mesh history) and refuses an expired reference
without silently calibrating another pose. Webcam reset also uses that displayed
pose ID. Publish a complete calibration atomically; failure retains the old rig.
Reset must preserve the model process, placement origin and viewport camera.
Frames carry a rig revision; replace bound bone geometry only on a new revision.
Clear references on input/rotation changes, and disable Reset while stopped or
pending. Registered body data are still required for calibration; this does not
remove the geometry download requirement or imply zero startup cost.
Toe terminal tips use each foot's neutral sagittal centreline, not a big-toe
surface point. Preserve terminal length and vertical pitch, native toe pivots,
predicted rotations and other bones. Recompute only this constant calibration
on initial input/Reset Bones; do not aim toes at a world axis each live frame.

PEAR's default-on **World Position** checkbox controls translation only. Off
anchors all three pelvis coordinates to the displayed position (or first valid
pose if disabled before Start). Keep pelvis/bone rotations and relative pose
unchanged. `pear_ui.py` retains one raw worker packet and translates only bounded
preview points/bounds; never mutate that packet, reset filters/rig, submit extra
inference or reload the model for this switch. Use a bounded render key to refresh
paused output immediately. Reset Bones keeps the anchor; a new input or Stop
clears it and releases the cached pose. On restores the raw estimated position,
so a position change is expected. This is pelvis locking, not foot locking.

PEAR's **One Euro Filter** and **Rotation** apply to webcam and video inputs.
All webcam slots belong in one numbered Camera / Device / Rotation table below
the video, with Count/Refresh in that panel and a bounded vertical scroll area.
Do not split camera 1 into the video header. Rotation is independent for each
device; do not restore a shared right-sidebar rotation. In video mode,
the first field edits the selected thumbnail's video, retaining other inputs'
values. Pass ordered per-input `rotations` through ordinary tracking and Camera
Align sampling, invalidate alignment after any angle change, and preserve values
on thumbnail/device refresh. The worker accepts the older scalar `rotation` only
as a compatibility fallback when the ordered list is absent. Rotate
pixels before detection/inference, then display the matching rotated image even
when paused. Source files are never rewritten. Smoothing is a causal adaptive
parameter filter on rotations/shape/face plus a vector-speed root filter; it changes
the live pose intentionally. Rotations use shortest geodesic interpolation with
bounded speed-dependent cutoff, not entrywise matrix averages or vertex blending.
Use actual source timestamps for skipped playback frames. Infer each requested
frame once: do not restore historical pre-roll inference on seeks, loops or gaps.
A discontinuous seek seeds the filter with its own prediction; subsequent streaming
is smoothed. Parameter filtering is batched across rotations/scalars; uniform FP32
pose/face tensors use one packed CPU transfer and one packed GPU upload. Retain the
generic dtype/device fallback, raw parameters and valid rotation matrices.
Filter Strength ranges from 1 to 100; 50 preserves the prior coefficients. Higher
values lower all adaptive cutoffs and slopes together. Strength-only changes keep
filter history, placement origin, identity, camera and resident model.
Reset temporal state on missing people, input/rotation/enable changes, reverse time or
long gaps; preserve the existing placement origin during same-source seeks.
Retain the last matched image/pose while a seek, pause or smoothing toggle computes
its replacement; clear it for a real missing-person result, error or changed input.
Under One Euro Filter, Bone Rotation and World Position are separate default-on
component switches. Requests carry `smooth_rotations` and `smooth_position`;
absent fields retain the previous both-on behavior. The parent `smoothing` flag
gates both. The position-filter switch does not disable world movement or Foot
Lock. Reset only the disabled component; retain the other filter history and
strength. Repeated timestamps from paused control refreshes retain the unaffected
filtered value instead of reseeding it. Actual seeks/gaps still reset normally.
A smoothing-only toggle must not reopen capture, reacquire identity, reframe the
camera or reset the placement origin. It is not an offline full-sequence smoother
or a guarantee of identical live/seek history.
Keep the one resident model, latest-only controls and fixed memory bounds.


### PEAR multiview fixed-camera preview

PEAR now accepts multiple video files through File > Open Video using the MAMMA
input inspector/thumbnail style. The common 30 fps timeline trims every view;
inputs must already be synchronized. Webcam camera-count slots require distinct
devices, with thumbnails selecting the main preview. Start/Stop remains explicit.
Start Live Tracking attempts Camera Align first for one or more cameras. A failure
continues tracking in the selected camera without fusion or aligned foot lock,
with a persistent visible warning and a console message. Missing alignment-only
support/models must not block otherwise-ready PEAR tracking. After alignment,
`pear/multiview.py` fuses rotations on one fixed rig and solves root position from
reprojection. Default-on lightweight Foot Lock uses pre-correction contact evidence, smooth activation/release and fixed-length two-bone leg adjustments. Preserve ankle/toe world rotations and the inferred bend side; cap pelvis shift at 4 cm and allow residual reach error. World Position off prohibits pelvis movement. Reset lock state on missing input, discontinuity or calibration changes; never feed corrected positions into contact detection. No offline correction, recording, retargeting or export is run.

`pear_multi_ui.py` manages transient inputs and calibration. `pear/multi_worker.py`
uses a four-thread CPU pool, bounded capture buffers and GPU batches of at most
four. YOLOv8s stays FP32; PEAR retains its validated FP16 encoder/FP32 decoder.
The graph path has no TensorRT dependency. Camera Align uses a separate
`flow_pear_align` worker over the pinned Flow runtime, sequential VGGT and GeoCalib
GPU stages, and a small GeoCalib/Kornia support overlay from install.cmd/update.cmd.
Model Manager reuses `flow_vggt` and separately obtains SHA-256 checked
`pear_geocalib`. Both weights are required for automatic alignment, including single-view alignment. Missing weights must show an actionable warning while allowing selected-camera tracking. Unaligned placement is approximate; do not claim fused/calibrated output.

Preserve shared runtime packages. `installer/pear-alignment.json` pins the source
and wheels; `tools/setup_pear_alignment.py` installs them under runtime_pear with a
separate inventory receipt. Re-run install.cmd after support changes; never forge
receipts. Initial pose is not an upright constraint. Background geometry supplies
camera orientation/floor; metric scale is inferred from fixed bone lengths.
Changed inputs/rotation/resolution and successful Reset Bones trigger another alignment attempt; failure continues with the selected camera and a warning. Physical camera movement needs an explicit Camera Align. Single-view scale uses inferred metric limb geometry and predicted person depth, so it remains an estimate.
Webcam host-arrival pairing is approximate (80 ms skew bound), not hardware sync.

Keep samples/benchmarks under an ignored/private experiment directory, not Git.
Alignment samples live only in cache/pear/alignment; successful jobs remove images.
Cancel stale alignments and reject mismatched receipts. Preserve offline media,
selection, trims and camera on leaving PEAR. Test paired playback, seeks, thumbnail
switches, cancellation, Stop/reopen and source replacement; actual multiple-webcam
performance requires physical devices and must not be claimed from video simulation.


## International Windows filenames and text output

Use UTF-8 explicitly for application JSON, text logs and subprocess text pipes.
Launch every isolated/development Python worker and setup verifier with
`-X utf8`; `-I` ignores the parent's `PYTHONUTF8` and `PYTHONIOENCODING` settings.
The GUI uses separate incremental UTF-8 decoders for each worker channel and
character job. Preserve decoder state between pipe chunks and flush at EOF;
never decode each arbitrary chunk independently. Keep existing buffer limits.
PEAR's binary IPC remains binary and must not pass through these decoders.
Single and batch workers report the original failure even if writing an error log
fails. Use the Windows path diagnostic for the console and preview messages.
A GUI handler exception stops the pending queue, drains logs and releases the
finished process. Invalid protocol lines are logged as malformed.

Use `mocap.image_io.read_image` / `write_image` for filesystem-backed OpenCV
images on Windows. They retain the existing codec and pixels, while Python
handles Unicode filenames. MAMMA's upstream-script calls and PromptHMR's isolated
worker use the temporary `unicode_image_paths` scope. It restores OpenCV entry
points even on errors. Do not apply this process-wide scope in the GUI or in a
process running concurrent unrelated jobs. Include this helper in affected
inference receipts; never bypass generation validation.

Pass filesystem paths as argument-list elements rather than shell fragments.
Keep delayed expansion disabled in entry-point CMD files so exclamation marks
in installation paths survive. Generated UTF-8 native-build CMD files must
select code page 65001 before reading non-ASCII command lines. Do not change
model versions, regional Windows settings or source motion to repair text.
VMD's specified CP932 fields remain CP932; an international file path does not
make that legacy file format capable of storing every Unicode label.

The 2026-11-02 checks covered English, Western European accents, Japanese,
Cyrillic, Korean, Chinese, combining accents and emoji paths on one Windows PC.
Hostile parent encodings were simulated; different OS languages, clean installs
and fresh CUDA compilation in every such path were not independently tested.
See the [validation record](VALIDATION.md#international-windows-paths-and-output-2026-11-02).


### Unreal face stream contract

`StreamWorkspace.stream_face_at_playhead` reads only a verified FaceArchive.
Resolve each person/take through FaceArchive.resolve and its source-time/FPS
mapping; use held valid samples and send no face before the first success.
Keep person IDs scoped to the body actor identity. Archive errors must not stop
body streaming. Face Overlay and the new-capture Track Face toggle do not hide
already recorded coefficients. Do not perform MediaPipe inference on this path.
The optional face packet has 52 finite values, integer sample/source_frame and
tracked metadata. The bridge publishes a separate Animation-role `_Face` Subject
with the canonical capitalized ARKit names, BodySubject metadata and matching
provider identity, and repeats paused expressions. Remove it when unavailable;
body FaceAvailable metadata distinguishes no sample from a stale connection.
Keep the default body-only packet and Blender protocol unchanged.

RGB MoCap Connect has per-source Face switches for MetaHuman targets. Scope
the switch to MetaHuman connections even when the same source drives a mannequin.
Skip face sampling and mapping entirely for mannequins. For MetaHumans, the
plugin applies the target's additive ARKit mapping curves to the native face preview,
preserves postprocess/RigLogic and restores the prior face class. Do not toggle
the MetaHuman actor's global UseLiveLink/UseARKit construction-script settings:
those can replace Body animation. Validate jaw/blink deformation, person
isolation, no-face/held samples, seeks, toggle/disconnect restoration and editor
save/Undo/reconstruction. Enable and restore the face mesh editor-update/visibility tick flags as well as
its animation class; receiving values or a Streaming status alone do not prove deformation.
The optional sender runtime uses Bridge-<source hash> projects and validates
receipt-relative paths. Keep older active modules intact during setup and roll
back a previous receipt if startup verification fails.


The MetaHuman Face initialization callback can disable editor updates on Body
and its followers. Reapply the connection's owned preview tick settings after
face initialization or restoration, and retain the original recovery baseline.
Validate body transforms and facial deformation across seeks and continuous
Face-enabled playback.

Receiver 0.5.0 uses the 0.4.0 sender protocol. When upgrading from an older
sender, update the app and rerun `Setup_UnrealStream.cmd`. Close Unreal before
replacing the project plugin folder.


### Shared face playback smoothing

`face_smoothing.py` owns the channel profiles and causal valid-sample One Euro
sequence computation; `face_export.py` re-exports smooth_face_values to retain
its existing API and output numerics. Eye/gaze and lip landmark offsets use
responsive profiles; per-point XY vectors share a cutoff to avoid axis skew.
`face_playback.py` prepares a private FaceArchive copy on a single cancellable
QThread with one latest pending request. Replace only its in-memory values and
offsets, then expose read-only arrays. Never rewrite face_capture.npz or modify
FaceArchive's raw-loading/export behavior. The overlay and stream use the same
FaceWorkspace cache; when ready it repaints and invalidates the paused stream
snapshot. Do not send raw values while preparation is pending.
Archive/source manifests identify each cache generation; discard stale replies
after project/file changes. GUI shutdown cancels and waits for this worker.
Resolve smoothed samples through the original held/valid arrays and source-time
mapping. Compute once from capture start so seek order cannot alter smoothing.
Keep export smoothing optional, avoid double filtering, and test raw preservation,
export equality, detection gaps, multiple people, reverse seeks, background
cancellation and actual Unreal-received coefficients.


### Neutral face calibration

`face_neutral.py` stores person IDs and valid capture-frame indices in a separate
versioned face_neutral.json, scoped to FaceArchive.identity. Do not add this file
to the capture fingerprint or mutate captured arrays/files. Save Project As
already copies project-local files with timestamps; no model/runtimes are needed.
Reject held/missing frames, map a selected take through FaceArchive.resolve, and
apply only to the currently selected People row (not every checked person or a
selected imported character). Reset preserves other people's entries.

Calibrate AFTER smoothing so the chosen smoothed frame is exactly neutral:
coefficients = clamp((value - baseline) / max(1 - baseline, 0.1), 0, 1).
The gain cap avoids amplifying near-saturated neutral channels without bound;
decreases below baseline do not invent an inverse ARKit expression. Subtract
the same-frame filtered landmark offsets from the filtered landmark sequence.
For unsmoothed export, derive the baseline from the raw sequence at that same
saved frame instead. Use expression_values on raw FaceArchive inputs only;
do not calibrate an already prepared playback archive twice.

FacePlayback.archive_key includes the neutral sidecar. Set/Reset invalidates
the shared preview/stream cache, and the existing ready callback repaints and
resends a paused frame. Keep preparing off the GUI thread. Do not change body
motion, Live Link subject identity, frame/timecode or the receiver plugin for
neutral calibration. Verify per-person isolation, exact zero at the chosen
frame, raw preservation, valid/held rejection, offset take timing, reset/reload,
and TXT/Live Link Face CSV with Smooth Face both on and off.


### Public Unreal receiver package

Authored receiver source is in unreal_plugin/RGBMoCapMetaHuman. The release ZIP
is built by tools/package_unreal_plugin.py from that source and matching locally
compiled modules. Require the exact supported UE build ID, source-byte identity,
an explicit allowlist, archive CRC/SHA-256 and a clean extracted-package load.
No engine/MetaHuman assets, private diagnostic modules, PDBs, screenshots of
private projects, personal paths, capture arrays or local logs belong in the
source or package. Logical #line filenames keep authored assertion diagnostics
portable without embedding the build directory. Scan both ASCII and UTF-16 DLL
strings and the full new Git diff before publishing.

The bridge now takes its per-launch nonce from RGB_MOCAP_STREAM_TOKEN in the
owned child environment, clears it there and authenticates only the loopback
application hop. Never claim stock UDP Message Bus is authenticated/encrypted.
The receiver evaluates exact source-qualified subjects and requires matching
body/reference RigId plus body/face RigId/MotionFrame metadata. A Face checkbox
is a receiver application setting, not an outgoing privacy control. Target None
and app person checkboxes stop publication. Keep the network boundary explicit.
On source changes rerun Setup_UnrealStream.cmd; it freezes and rechecks source
identity excluding generated build outputs and rolls back a previous receipt if
startup verification fails. Do not publish local receipts or process logs.
