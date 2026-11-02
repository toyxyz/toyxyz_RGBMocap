# RGB MoCap: agent operating and development guide

This guide is for agents helping users operate or modify RGB MoCap. It describes the Git/CMD version. User requests and newer verified behavior take precedence over historical benchmarks. The English GUI is the supported user interface; do not assume a general command-line inference or retargeting API exists. Avoid asking for the user's videos or model weights if the existing local project is sufficient.

## Start and update

- Windows 64-bit, compatible NVIDIA GPU/driver and OpenGL 4.3 are required for the documented inference workflow. Other systems and GPUs were not comprehensively verified; see [validation](VALIDATION.md).
- In a writable checkout, run `install.cmd` once. It sets up separate pinned profiles for ExMR, FlowHMR and GVHMR and their supported add-ons, PEAR support over the shared runtime, Blender 4.5, FFmpeg and the prebuilt native motion-correction module. It does not install only the selected model's runtime or obtain AI weights/body data. Downloading and disk space are required. Models are obtained separately; optional MAMMA uses Setup_MAMMA.cmd and PromptHMR uses Setup_PromptHMR.cmd.
- Run `RGBMoCap.cmd`. Keep its command window open; launch errors, processing messages and app Console output are also sent there, but long messages can be truncated and a saturated output queue can drop messages. Within the GUI, **Help > Check System** checks the system.
- Base worker commands use explicit `-X utf8`, including isolated `-I` launches. Isolated mode ignores `PYTHONIOENCODING`/`PYTHONUTF8` from the parent environment. Any redirected/discarded text log stream must also specify UTF-8. Keep byte-oriented PEAR IPC unchanged.
- Close the app before `update.cmd`. The updater requires a clean tracked worktree and the `main` branch, uses `git pull --ff-only`, then verifies/repairs runtime configuration. It refuses tracked local edits. A plain `git pull` does not perform runtime repair; use `install.cmd` after dependency changes when needed. Do not discard local edits or force an update.
- For setup failure, read the message, check connectivity and free space, then retry `install.cmd`. Preserve models, projects and `.setup` state. Do not repeatedly reinstall or redownload models without evidence.
- Base installation currently estimates at least 32.4 GiB free before models; ExMR/FlowHMR/GVHMR and add-ons were previously measured around 29 GiB of weights. Optional MAMMA and PromptHMR add separate runtimes and model files; see [MAMMA](MAMMA.md) and [PromptHMR](models/PROMPTHMR.md). The five offline model/runtime sets total about **63.22 GiB (67.9 GB)** excluding caches, projects, downloads and compiler tools, counting shared files once. See the [installed-size table](USER_GUIDE.md#installed-size); installed size and temporary setup headroom are different. Measurements are not guarantees.

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

## Models and model files

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
| PromptHMR | Multiple people reconstructed jointly from one video; can improve relative placement and interactions. Overall motion quality was lower than ExMR in our subjective tests. Fixed-camera Windows path, up to eight people; placement can still be wrong. | No finger tracking; fixed neutral fingers. Setup_PromptHMR.cmd; People defaults to 2, Interaction Prompt on. |
| PEAR | Live single-person bone preview from a webcam or the video timeline; transient, with lightweight live foot lock and no recording/export or offline correction. | Fingers included; Start/Stop, fixed lengths/Reset Bones, One Euro filter, Rotation and default-on World Position. Shared support is installed by install.cmd. |

Start with the [model guide index](models/README.md), then [ExMR](models/EXMR.md), [FlowHMR](models/FLOWHMR.md), [GVHMR](models/GVHMR.md), [MAMMA](MAMMA.md) or [PromptHMR](models/PROMPTHMR.md) for exact setup and controls. The [user guide](USER_GUIDE.md#recorded-processing-time-and-vram) includes observed timings, conditions and VRAM; these are historical measurements, not minimum specifications. FlowHMR has a 5,000-frame per-take limit and available memory may require less. MAMMA retains its 4–1,800-frame common trim and eight-person limits; there is no fixed camera-count maximum. Four-column thumbnails are a layout, not admission control. Do not change numerical precision, temporal context or model settings merely to improve a benchmark.

Use **Model Manager** for the selected model and add-ons. Automatic download may fail, especially on Google Drive. Gated models require the user to register, accept upstream terms and obtain files; use **Manual Download...** and **Import...**. Do not handle provider passwords or redistribute licensed weights. Imports validate files; several model receipts tie validity to file size, timestamp and expected hash. If a copied file is not marked ready, import it again rather than assuming it is missing or editing receipts.

### Installation routes

| Command | Scope | Additional prerequisites |
| --- | --- | --- |
| `install.cmd` | Base GUI, ExMR/FlowHMR/GVHMR and supported add-ons, PEAR shared support, Blender, FFmpeg, native correction | See the base installation guide; no PromptHMR/MAMMA native build |
| `Setup_MAMMA.cmd` | Optional independent MAMMA runtime | VS 2022 C++ tools/Windows SDK, CUDA Toolkit 12.8, provider license review |
| `Setup_PromptHMR.cmd` | Optional independent PromptHMR runtime | Same build tools, separate provider license review; MAMMA/SDF not required |
| `update.cmd` | Source update and base runtime verification | Close the app and preserve tracked edits; it does not install the optional runtimes |

Optional Runtime Setup buttons display instructions/copy a command; they do not
install inside the running app. After updates, rerun the relevant optional setup
when requested by its dependency/runtime notice. Keep verified installed packages
isolated even when Torch versions match. Models are obtained separately.

Model files are in `<installation>/models/` by default. To reuse a legitimate installation, close both apps, copy the complete models folder and its receipt/license files beside `RGBMoCap.cmd`, and check readiness in Model Manager. Do not copy old `runtime*` folders. SMPL Neutral is required by PromptHMR, although it is not required for ExMR/FlowHMR/GVHMR. Standard SMPL-X Neutral is shared by ExMR/GVHMR/PromptHMR; MAMMA uses its separate locked-head bodies. Exact requirements are defined by `mocap/models.py` and the chosen stages, not by guessed filenames.

## Standard GUI workflow

### PromptHMR setup and verification

Follow [the detailed setup/workflow guide](models/PROMPTHMR.md). It is a
separate pinned Python 3.11 runtime, not a MAMMA overlay or part of install.cmd.
Setup_PromptHMR.cmd downloads restricted upstream source, builds SAM2 with
Visual Studio 2022/CUDA Toolkit 12.8 and verifies actual CUDA operations. It
reuses standard registered SMPL-X/SMPL bodies, not MAMMA's locked-head body.
Only fixed-camera video is integrated. Find person is hidden; actors retain one
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

### MAMMA setup and verification

MAMMA runs an optional complete multi-view pipeline using a separate pinned
Python 3.11 Windows runtime installed with Setup_MAMMA.cmd. The current app
adapter accepts two or more synchronized fixed-camera views, with one common
trim and thumbnail preview switching. There is no fixed camera-count maximum;
retain existing RAM/Windows allocation-headroom monitoring, CUDA allocator
budgets and OOM failure handling. Do not replace the removed camera-count cap
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
Do not claim clean-Windows or other-GPU validation or SAM 3.1 app support. Keep restricted upstream code, weights and derived bodies private.

For prerequisite checks, automatic setup, independent SAM 2/SDF builds, kernel
verification, registration and rollback, use the
[MAMMA agent setup runbook](MAMMA_AGENT_SETUP.md). Registration alone does not
validate actual native CUDA operations. Do not overwrite a working runtime to
test a new native module, or use cached inference as evidence of a new build.

### Capture, correction and export

1. Select an AI Motion Model and prepare its models, then **File > Open Video** and optionally trim the Video track. Select MAMMA before opening multiple videos. New work defaults to ExMR.
2. For ExMR, FlowHMR and GVHMR with multiple people, use **Find person** and check the desired rows; without it, automatic tracking selects one person and can switch identity on overlap. PromptHMR tracks people jointly during reconstruction; its People setting is a maximum count and Interaction Prompt is optional. MAMMA instead detects and matches people during reconstruction; Find person is hidden and its People setting controls the count. A checked row controls visibility and ExMR/FlowHMR/GVHMR processing targets; the selected row controls the current timeline and contact edits.
3. Choose camera and processing settings, then **Reconstruct 3D**. Processing runs on a 30 fps working clip, without changing the original video. Progress and cancellation are in the app. **Prepare Preview** rebuilds previews from existing motion without rerunning inference.
4. Review motion and contacts. **Enable Correction** selects corrected output; Foot defaults on and Hand off for new work. Flat Ground uses a common floor; Multi Level retains inferred contact heights and does not reconstruct terrain. Inspect automatic contacts; hand contacts represent support, not hanging/object tracking.
5. Edit L/R foot and hand intervals on the timeline, then **Apply Correction**. Pending changes are not applied until then; stale corrected export is blocked. Applying also refreshes linked retargeted characters. Correction and preview do not require fresh AI inference.
6. **File > Export...** opens FBX/BVH/Blender format selection, the BVH-only Generic/Blender (default) or 3ds Max Biped profile, and the default-off **Add T-pose before motion** option. Multiple checked people/takes produce separate named files. Source skeleton exports use the committed 52-bone Hips hierarchy; the Biped profile renames joints in the exported copy. Optional T-pose export adds one reference frame without editing the cached motion or contacts; original motion shifts by one frame. Blender creates portable video/camera bundles; MAMMA additionally offers whole-scene PC2 bodies and every calibrated camera without a T-pose. Character exports keep their separate FBX/VMD dialog.

See [the full workflow and controls](USER_GUIDE.md#start-here). Do not confuse viewport toggles, automatic detection and applied motion corrections. No general scene geometry or collision reconstruction is implied.

Contact correction v9 joins free-foot transitions to the final planned root
and evaluates floor clearance continuously inside the bounded trajectory plan.
Keep stance anchors/timings and source ankle frames fixed; do not smooth final
leg quaternions or add a second IK pass. The native anatomical solver remains
the only publisher of leg rotations. Planning-limit warnings are real residual
notices. Existing saved previews are not automatically replaced; use Apply
Correction to recompute without inference. See the v9 measurements and limits
in [validation](VALIDATION.md#contact-correction-v9-coherent-root-and-swing-transitions-2026-10-30).

### Saved camera display and source export implementation

`mocap/camera_tracks.py` adapts committed camera receipts without inference or
body-based camera refitting. Single-person FlowHMR may have no
`scene_camera.json`; use its validated `flow_world.json` basis/height and native
`flow_camera.npz` without creating a scene receipt or fitting the corrected pose. MAMMA native Z-up calibration is transported to the
app's Y-up world. FlowHMR/ExMR use saved rotation tracks. Flow gizmos additionally display
unscaled VGGT displacement; moving-camera ExMR gizmos use estimated displacement
from ORIGINAL paired world/camera root positions, never corrected poses.
Motion x defaults to 1, with 0 retaining a fixed center. Video overlay projection
remains unchanged when this display scale changes. MAMMA/GVHMR remain fixed. Other models expose their saved reference camera only. Do not interpret
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
The optional leading pose is animation, not a replacement bind pose; validate
both original-frame and fractional-frame suffix poses in Blender after changes.

### Portable Blender export implementation

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

- Default projects live under `<installation>/projects/`; **Save Project As** creates and switches to a copy at a user-selected location. Open the saved `project.json` using **File > Open Project**. A project contains model-specific `exmr`, `flowhmr`, `gvhmr`, `mamma` or `prompthmr` result folders; model switching loads existing results when available. MAMMA saves all input camera videos. People, takes, contact edits and imported characters are project data.
- `models/`, `projects/`, `cache/`, `character_scenes/`, `runtime*/` and `.setup/` are local data, not source artifacts. User project folders can also be outside the checkout. Never assume `.gitignore` protects a path that was already tracked or will be force-added. Check `git status` and tracked files before committing; do not upload model weights, video, personal data, credentials, generated motion or caches. Explicitly approved README sample assets are exceptions.
- Preserve source videos, registered weights and project data. Do not delete project subfolders, results or receipts to fix a problem: they can be referenced by saved metadata. Cache cleanup and setup-download cleanup are distinct from deleting saved work; see [storage and cleanup](USER_GUIDE.md#storage-and-cleanup). Obtain authorization for destructive cleanup.
- `mocap/paths.py` defines `ROOT`, `RGB_MOCAP_DATA`, models and projects; normal portable startup sets its own data location. When diagnosing an actual run inspect the effective environment rather than assuming a path.

## Resources and performance

Use **Resource Mode > Automatic** by default. It keeps the fast path when the current GPU budget is sufficient and selects supported Low Memory paths per operation when needed. A supported fast operation that raises CUDA OOM is released and retried once in Low Memory. ExMR Low Memory uses CPU-first body/hand loading and converts Sapiens' existing FP16 layers before GPU transfer; ExMR body/hand batches stay 8. FlowHMR offloads SAM/VGGT blocks; MAMMA offloads VGGT camera blocks while retaining its pinned masks/landmark/fitting settings. GVHMR/add-ons retain adaptive image batches and full-context attention. The saved Automatic selection is not changed. Resource decisions and memory advice appear in the app Console and CMD. Model precision, frame count, resolution and iterations are not reduced. CPU loading/offloading can be slower and use more RAM; it is not a promise every clip will fit. Estimates select a path, never reject a job. The desktop still checks actual RAM and Windows allocation headroom every 0.5 seconds and stops workers when these become low. Preserve completed stages, genuine model limits and original motion. Do not rerun inference for operations that can use existing results.

PromptHMR retains the same one-image batch/full temporal context in Automatic
and Low Memory; extra block offloading is not implemented. Do not promise a
mode-dependent VRAM saving or an unsupported offload retry for this path. Actual
memory guards still apply. Its 422-frame/two-person run took 236.2 s, with peak
allocated VRAM 4.61 GiB (7.83 GiB reserved) and worker working set 8.74 GiB; see
[scope and stage times](models/PROMPTHMR.md#recorded-validation).

GPU allocator limits share ComfyUI's external-use margin: 600 MiB on Windows, 700 MiB above 15 GiB GPU capacity (400 MiB on other systems). Subtract it from actual free VRAM without an additional percentage cap or fixed FlowHMR/ExMR 3/2 GiB reserve. This margin is separate from inference working space. Keep offloading in the app's validated `memory_mode.py` / `gpu_offload.py` adapters, without importing ComfyUI or changing model precision. Offload is inference-only, synchronous and unpinned; close hooks/models on success and failure. Actual RAM pressure remains a stop condition because offloading can increase host memory.

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
The two Fast SAM switches default on for new work, preserve saved off settings,
and only apply with Fast SAM. See [their separate measurements and output differences](PERFORMANCE.md#flowhmr-fast-sam-switches).

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
- PromptHMR UI/pipeline: `mocap/prompthmr_ui.py`, `prompthmr_pipeline.py`, `prompthmr_publish.py`, `prompthmr_scene.py`; runtime/assets: `prompthmr_runtime.py`, `prompthmr_manager.py`, `prompthmr_models.py`, `prompthmr_worker_boot.py`, `installer/prompthmr-runtime-lock.json`, `installer/prompthmr_adapters.py`, `installer/verify_prompthmr.py`, `tools/setup_prompthmr.py`. Follow the separate setup/verification steps above.
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
stalled. Keep RAM/VRAM monitoring and bounded mailboxes while it is running.

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
