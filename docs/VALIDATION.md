# Source package validation

## Resource Safety control (2026-11-03)

The default-on checkbox below Resource Mode controls the existing desktop
low-RAM/Windows commit-headroom stop and the PyTorch VRAM allocator cap. Off
retains adaptive batching/path selection, supported actual-OOM recovery,
cancellation, generation protection and the independent no-output/no-CPU stall
check. No runtime dependencies, model equations or precision were changed.

Six private policy tests covered default and corrupt preferences, preserving
unrelated local settings, unchanged enabled thresholds/routing, disabled RAM
queries, low/critical memory, injected OOM exception backoff, PID identity and
stall handling. Offscreen GUI checks covered default-on layout, visible
unchecked state, persistence after reopening, six model selections, disabled
controls during work and both flag values reaching isolated workers and their
desktop guards. PEAR session checks covered inherited flags and process closure.
These are control/lifecycle checks, not playback throughput measurements.

A small real CUDA allocation probe passed in the installed shared, FlowHMR,
ExMR, MAMMA and PromptHMR profiles: a synthetic 4 MiB allocator allowance
rejected a 16 MiB allocation with safety on; safety off allowed it. This did not
exhaust physical RAM/VRAM. Both installed Torch 2.7.1 and 2.8.0 builds were used.
An actual legacy fixed-camera PromptHMR receipt remained reusable without new
video preparation or inference after the logging-only pipeline change.

A subsequent manual ExMR run completed with Resource Safety off and Automatic
fast paths: 2,586 working frames at 30 fps (about 86.2 seconds), total 299.22
seconds including tracking, correction and preview/export preparation. No OOM,
Low Memory fallback or safety stop appeared in that log. The keypoint receipt
recorded 5.68 GiB peak allocated VRAM; a whole-run peak RAM measurement was not
recorded. This is not a paired on/off performance or output-accuracy benchmark.
FlowHMR in the same log reused cached results, so its time is not comparable.

Checks used the existing Windows test PC. No clean-Windows or other-GPU test
was performed, and disabling safety does not guarantee success under real
memory exhaustion. Test inputs, scripts, logs and local preferences stay
outside the public source package. See [behavior and thresholds](PERFORMANCE.md#resource-safety).

## PromptHMR shared download cache (2026-11-02)

Torch Hub/Hugging Face download locations now belong to the app's shared
`cache/prompthmr/` tree instead of each inference generation. Known older
DeepLab checkpoints are imported only after full SHA-256 verification; directory
enumeration and legacy hashing are bounded. Old project files are preserved.
The runtime, model versions, batch size and numerical processing are unchanged.
An exact reviewed pipeline hash pair retains existing valid inference-cache
identities, while future source edits invalidate normally.

Private tests covered two different eight-frame clips, including one and two
people. Original and updated tracking, pose and world/post-optimization stages
completed with network/download attempts blocked after priming the upstream Hub
repository source cache. Both updated clips used one shared checkpoint, with no
per-generation weight downloads. A separate empty-cache test exercised real
torchvision/Torch Hub weight loading through a local copy of the official
checkpoint bytes: first use requested one download, the second video requested
none. This was not a live external-server availability test.

Tracking outputs, camera-space motion, body rotations and shape were exact in
the two independent comparisons. Fresh 2D keypoints varied by at most 0.000184
pixels and final world translations by at most 4.64 mm; these end-to-end runs
are not bitwise identical. Replaying identical camera-motion inputs with matched
NumPy/Python/OpenCV RNG in the test produced exact world/post-optimization output
before and after the fix. Production random behavior was not changed. Peak
allocated VRAM was unchanged per corresponding stage; the two updated worker
runs took about 42–43 seconds each. These short runs establish neither a quality
ranking nor a general speed improvement.

Additional checks passed for cross-volume copy fallback, Unicode cache paths,
invalid-cache rejection, preserving original files, and reusing an actual old
complete inference receipt without FFmpeg preparation or AI inference. Models,
input videos and test artifacts remain outside the public source package.

## Side-by-side preview video export (2026-11-02)

File > Export Preview was checked using private copies of an ExMR project with
an animated PMX character and a two-person MAMMA project with a non-primary
camera selected. Twelve repeated 30-frame exports per case retained the chosen
view and restored the original frame/camera. Saved H.264 MP4s decoded to exactly
30 frames at 30 fps (one second); the selected source's audio was included when
available. First/last encoded frames matched the submitted video/framebuffer
pair within normal lossy encoding differences (mean channel error below 3/255).
The test panels were 1404 × 684 and 1954 × 684; this is not a universal speed or
resolution guarantee. Rendering still requires GUI-thread framebuffer capture.

Cancellation preserved an existing destination and removed partial output.
Playback resumed after export when previously running. Closing the app during
export cancelled and reaped FFmpeg before the window/workers closed. Additional
checks covered one frame at 30000/1001 fps, an externally changed destination,
incomplete frame delivery, Unicode output names, input-file overwrite rejection
and PEAR exclusion. Repeated-export testing caught Python 3.10 DummyThread
registration through joining a Python log reader from a Qt thread; completion
now uses an event instead. Handles and thread counts stabilized in the bounded
repeat test; it does not establish leak freedom for hours of use or other PCs.
No source inference/contact math or installed dependencies were changed. Test
scripts, source media, copied projects and rendered videos remain outside the
public source package.

## MAMMA background gravity before fitting (2026-11-02)

Automatic calibration now uses VGGT-Omega camera placement with GeoCalib
background gravity before MAMMA 3D fitting. GeoCalib executes in the existing
Flow/Python 3.10 support overlay in a separate process; the optional MAMMA
Python 3.11 environment is unchanged. Manual calibration bypasses both models.
The camera-balanced gravity estimator rejects uncertain/conflicting samples;
there is no silent person-direction fallback. Its acceptance thresholds have
limited scene coverage, not universal calibration guarantees.

On the documented test PC, the integrated camera path was exercised on an
outdoor four-camera, 450-frame sequence. It loaded both camera models afresh,
reused the existing masks/landmarks and completed in 47.59 seconds. Up-axis
difference from the dataset calibration was 0.75 degrees, versus approximately
11.09 degrees in the prior person-direction result. Full 450-frame MAMMA 3D
fitting then completed in 120.09 seconds with finite output. Peak PyTorch
allocated VRAM was 1.57 GiB and peak process RSS 2.19 GiB during fitting.
These are camera/fitting measurements, not end-to-end reconstruction timings
or total desktop GPU memory. Reference-height scale and initial sole height
remain estimates; a known camera calibration is not ground-truth human motion.

A second indoor four-camera, 422-frame camera-path check completed in 46.34
seconds, with a 0.76-degree up-axis difference from its dataset calibration.
Three cameras supported the accepted gravity consensus; the remaining camera's
uncertain observations were rejected. This was a camera-path check, not a second
complete fitting or end-to-end reconstruction benchmark.

Targeted checks cover manual-camera extrinsic preservation and pixel resizing,
Unicode camera IDs, vertical-looking reference cameras, conflicting gravity,
invalid background/duplicate observations, model scoping and isolated worker
routing. Existing runtime pins and other models' inference paths were retained.
Tests and generated data remain outside the publishable source checkout.

The Git-based distribution was tested on the Windows PC listed in the
[user guide](USER_GUIDE.md#install). It has not been tested on a clean Windows
installation or on other GPUs or drivers.

## Recorded checks

- A separate local Git checkout installed all six pinned runtime profiles using
  `install.cmd`. Blender and FFmpeg were downloaded and hash checked; previously
  verified package downloads were reused. Installation took approximately 10 min
  32 sec on the tested PC. CMD launch and launch-error reporting were checked.
- ExMR, FlowHMR and GVHMR each processed a one-second video, including inference,
  correction, rig generation and FBX export. Thirteen worker fingerprint inputs
  were checked. This does not establish long-video performance on other systems.
- The included correction module and a developer rebuild produced identical
  root positions and local rotation values on the tested clips: 30 frames for
  each model and 210 additional frames for ExMR and FlowHMR, with automatic,
  manual and disabled foot contact. Other motions and platforms were not covered.
- One Mixamo FBX, one VRM and one PMX character were imported and retargeted.
  Mesh-inclusive and animation-only FBX files were reimported into Blender; bone
  animation and mesh options were present without armature-object transform keys.
  A PMX VMD was reimported with Blender MMD Tools. Visual equivalence in the
  standalone MMD application was not established by that check.
- A local Git update retained test model and project data and relaunched the GUI;
  runtime verification took approximately 1 min 55 sec. Twenty automated tests
  covered simulated download errors, unsafe archive paths, update guards and
  interrupted replacement. Actual power loss and disk exhaustion were not tested.
- The public source snapshot was checked for forbidden generated/model payloads,
  and its manifest hashes matched. Pattern checks found no typical private-path or
  credential formats in scanned text; such scans cannot prove absence of secrets.

## Latest published-source audit (2026-10-27)

The audit started from Git commit `1daf49c` and used the existing installed
runtimes and legitimately acquired local model files. All test scripts and
outputs were kept outside the public checkout.

- 150 targeted regression checks passed for project saving, model selection,
  character handling, export options, runtime guards and update transactions.
- Each of the three AI models processed 30 new frames through correction and FBX
  export. Blender imported all three 52-bone source rigs with `Hips` as root;
  sampled integer and fractional animation frames contained finite transforms.
- Mixamo FBX, VRM and PMX imports, retargeting and six character FBX exports
  passed. Reimported FBX rest-joint positions matched the imported character's
  original rest pose within 0.1 mm; no armature-object transform keys were found.
  The PMX VMD also loaded with Blender MMD Tools.
- The source checker was corrected to accept the approved README samples and
  root agent guide, ignore untracked local model data, and still reject model
  files accidentally tracked by Git. Archive and Git-checkout cases were tested.
- Local documentation links resolved. The source manifest was refreshed after
  the small documentation and checker fixes.

This audit did not repeat fresh installation, long-clip performance testing,
interactive playback, standalone MMD playback or testing on other GPUs.

## Character preview and long-path fixes (2026-10-27)

- Final preview windows no longer reload themselves. Loader checks covered
  1, 2, 10, 111, 112 and 400 frames, exact frame retrieval, bounded windows and
  worker shutdown. An isolated five-second character viewport check loaded once
  and performed no redraws after warm-up. This is an idle-loop check, not a
  playback FPS or whole-desktop performance guarantee.
- A 474-frame PMX animation was baked and reopened under a 295-character data
  path. All 215 animation channels retained exactly the same key values as the
  short-path baseline; no armature-object animation keys were added.
- Animation-only FBX and PMX VMD exports succeeded at 289-character destination
  paths. Blender reimport checks found finite sampled FBX poses and no object
  transform keys. Short- and long-path VMD outputs were byte-identical.
- Test scripts, character assets and generated outputs remained outside the
  public checkout. These checks did not repeat standalone MMD playback or test
  other GPUs, network shares or every possible Windows path.

## Original-resolution character textures (2026-10-27)

- Mixamo FBX, VRM and PMX samples retained their original image dimensions and UVs.
  A pre-existing PMX asset prepared its texture cache without reimport or retargeting.
  Sampled GPU-deformed vertex buffers were unchanged, including a 474-frame take.
- Isolated 1920 x 1080 GPU checks ran 600 draws per sample, excluding the first 50
  from timing statistics. Original-texture median / p95 draw times were
  0.108 / 0.264 ms (FBX), 0.331 / 0.536 ms (VRM), 0.466 / 0.697 ms (PMX rest),
  and 0.411 / 0.617 ms (PMX animation). The largest recorded draw was 2.50 ms.
  These are isolated renderer measurements, not complete application frame times.
- Twelve renderer load/close cycles kept the handle count constant. Process RAM
  rose about 13 MiB during warm-up and stabilized in the last three cycles.
  Three loader playback/seek cycles had no missing animation windows, constant
  handles and less than 0.1 MiB additional RAM. Timer lateness reached 15.7 ms;
  these checks do not establish video/audio synchronization or universal FPS.
  Cancelling background texture preparation shut the loader down in 0.107 seconds.
- Original images are cached once per character on CPU and uploaded once per
  loaded renderer, with mipmaps for distant viewing. Motion buffers remain bounded.
  Full PBR materials and general translucent-material sorting are not implemented.
  Tests and generated character assets remained outside the public checkout.

## Runtime memory-pressure policy (2026-10-28)

- Removed predicted RAM/VRAM admission failures from shared workers, FlowHMR and
  ExMR. FlowHMR's 5,000-frame limit, adaptive batches, FP32 temporal context,
  allocator reserves and the existing GVHMR full-key query-split path remain.
  Estimates can still select a computation path; they no longer reject it.
- A real 12-frame 1920 x 1080 input completed moving-camera VGGT, Fast SAM and
  FlowHMR generation while worker policies were given 24 GiB available RAM.
  This simulated the reported availability; the physical test PC had more RAM.
  VGGT peak process RAM was 8.89 GiB and peak Torch CUDA allocation 6.32 GiB.
  This does not establish a minimum RAM requirement or behavior on other GPUs.
- Qt timer/worker-tree integration tests simulated low available RAM, low Windows
  allocation headroom and critical pressure. Workers and their children stopped
  in 0.65–1.14 seconds; stage errors were saved and prior committed test data
  survived. One low sample followed by recovery did not stop a job. The largest
  observed 16 ms UI-timer gap was 31.8 ms in these short tests.
- Injected CPU MemoryError and CUDA allocation failures returned explicit failed
  events and retained existing motion files. A real FlowHMR retry completed and
  cleared its pending marker. Actual system RAM was never exhausted deliberately.
- Exact policy-only file-hash aliases preserve preceding generic/GVHMR signatures
  and all three ExMR stage identities. Later byte changes invalidate them normally.
  ExMR identity checks used synthetic input stamps and installed model statistics;
  full ExMR and GVHMR inference were not repeated. GVHMR allocation/retry control
  flow was exercised with a test model and simulated allocator observations.
- Tests and generated inputs/results remain outside the source checkout. Runtime
  monitoring cannot guarantee intervention before every sudden large allocation.

## ComfyUI-style external VRAM margin (2026-10-28)

- Shared, FlowHMR and ExMR CUDA allocator budgets now subtract 600 MiB from
  current free VRAM on Windows, or 700 MiB above 15 GiB GPU capacity, matching
  [ComfyUI's external-use margin](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_management.py).
  The old FlowHMR/ExMR fixed 3/2 GiB deductions and extra 95/90% caps are removed.
  ComfyUI's separate 0.8 GiB inference planning allowance is not another allocator
  deduction: weights, activations and temporary buffers share the allowed budget.
  This margin alone does not implement offloading or guarantee free memory against
  other applications or non-PyTorch allocations.
- Boundary checks covered simulated 4/8/12/15/16/24/32 GiB capacities, the exact
  15 GiB transition, exhausted free memory and all four allocator call sites.
  Existing batch backoff and GVHMR dense-OOM/full-key retry checks passed.
- On the RTX 5090 test PC, all three isolated runtime profiles applied a 700 MiB
  margin, performed CUDA allocations, raised a real allocator OOM under a tiny
  test cap and successfully allocated again after restoring the normal budget.
  The test deliberately limited the allocator rather than exhausting the GPU.
- A private 12-frame 1080 x 1920 clip completed FlowHMR camera/Fast SAM/generation,
  GVHMR observations/generation and ExMR keypoints/features/generation using the
  installed model weights. Other GPUs and long clips were not tested here.
  FlowHMR generation from identical cached features produced identical motion
  tensors with the old and new allocator caps.
- Existing generic/GVHMR cache fingerprints and all three ExMR stage identities
  were retained. Test code, inputs, generated motion and logs remain outside the
  public checkout under the private validation artifacts directory.

## Memory-policy follow-up review (2026-10-28)

- Fixed GVHMR's dense-OOM retry report to compute `fits` from observed capacity
  rather than always reporting a shortfall. The retry still preserves FP32 and
  the full key/temporal context and does not reject work using that estimate.
- Made worker cancellation idempotent while its Windows termination helper is
  active. Late helper callbacks check that the worker still exists and only clear
  their own helper reference. Four callback/worker-lifetime combinations, each
  with five duplicate requests, passed without Qt callback exceptions.
- ExMR now replaces a previous failed stage status after successful computation
  or validated cache reuse. Its explicit force flag also bypasses the cache.
  All three cached ExMR stages repaired failed status without altering tensors;
  a forced 12-frame generation was separately checked for a non-cached completion.
- Each model's actual worker entry reported injected CPU allocation failures and
  real CUDA allocator OOMs with an error, exit code 1 and unchanged prior motion.
  Tiny allocator limits triggered CUDA failures without exhausting system VRAM.
  Subsequent retries completed using existing preprocessing data.
- Four Qt timer/worker-tree tests simulated low RAM, low Windows allocation
  headroom and critical pressure. Stops completed in 0.66-1.17 seconds, preserved
  committed results and left no test child worker running. The largest observed
  16 ms UI heartbeat gap was 29.12 ms; a transient low sample did not stop work.
- Exact RAM/VRAM boundaries, Windows commit observations, traceback cleanup,
  exhausted-batch errors, existing inference cache identities and source checks
  passed. This is a single-PC regression check, not a low-memory hardware test or
  a guarantee against sudden allocations or competing GPU applications.

MAMMA multi-view input setup (2026-10-29):

- Five real camera videos opened through the multi-select picker and produced a
  four-column thumbnail grid with a second row. Ten preview-camera switches kept
  the frame, trim, 3D camera and one persistent decoder; sampled switch latency
  was approximately 0.03–0.21 seconds on cached local files. The short sequence
  showed no net thread/handle increase; this is not a long-run leak guarantee.
- Real 25 fps / 60 fps videos shared a 30 fps timeline limited by the shorter
  input. At timeline frame 45 (zero-based, 1.5 seconds), native requests were 38
  and 90. Playback and resume after switching were exercised in the Qt event
  loop; this was not a visible full-app GPU FPS measurement or a VFR test.
- The actual Save Project As worker copied every input, restored common trim,
  active camera, manual calibration and correction selection, and saved a second
  portable project copy. Reopening in another tool and returning to MAMMA also
  recovered the saved inputs. Eight rapid replacement requests coalesced to the
  latest set, and saving was blocked during input preparation.
- JSON and MAMMA YAML camera imports were checked, including camera-to-world
  quaternion conversion against matching world-to-camera matrices. Invalid
  rotations/intrinsics and missing files retained previous valid inputs/settings.
  Existing three-model controls and single-video decoding were checked.
- These initial checks covered input setup only. Later native-pipeline checks
  below extend their scope; they do not validate fresh-user installation.
  Original videos and motion data were not modified. Private test files remain
  outside the public checkout.

MAMMA native app pipeline (2026-10-29, same tested PC):

- The GUI Reconstruct 3D queue completed SAM 2.1 segmentation, exact-ROI FP32
  landmarks, VGGT cameras, full FP32 MAMMA fitting, native actor conversion,
  shared-floor correction and source export preparation. Two synchronized views,
  two people and 30 frames (one second) took 299.50 seconds including worker
  startup and native preparation. A verified repeat reused inference in 14.48
  seconds and retained the raw fitted file's timestamp and size.
- Recorded PyTorch peak allocated/reserved VRAM (GiB) was 2.532/2.846 for masks,
  0.585/0.645 for landmarks, 6.097/6.121 for cameras and 0.315/0.352 for fitting.
  The largest process peak working set was 8.928 GiB in camera estimation.
  These are process/allocator observations, not whole-PC or driver VRAM totals,
  minimum requirements, or measurements for a long clip.
- Exact sixteen-beta conversion matched original fitted meshes within 0.001 mm.
  Original and corrected previews for two different body templates rendered
  together on the real GPU. Readback against the CPU skin reference at three
  frames differed by at most 0.002 mm. Twenty-four visibility changes retained
  constant sampled handle/thread/GPU-buffer counts; sampled RSS changed about
  0.04 MiB after warm-up. This is not a long-run leak guarantee.
- Camera switching and six original/corrected transitions preserved the selected
  frame and 3D camera. In a three-second playback check, the GUI event-loop gap
  p95 was 9.57 ms and maximum 15.64 ms. These are event-loop gaps, not rendered
  frame times or a universal FPS claim. Original GPU images were inspected.
- Save Project As/reopen retained both actors and both working camera previews
  with the original project temporarily offline. Manual lock edits blocked stale
  export; Apply Correction refreshed both actors in 15.85 seconds without
  changing either original source tensor. Incomplete publication, changed input
  receipts and invalid saved settings were rejected. Removing a required cached
  landmark rejected the cache and allocated a separate attempt. Missing
  project-private body data was regenerated from the unchanged raw fit in
  16.31 seconds without rerunning inference.
- Four real source FBX exports (two people, original/corrected) imported through
  ordinary Blender 4.5 settings with 52 Hips-root bones and no armature-object
  keys. Maximum imported bone-head difference was 0.0034 mm. Retargeted-character
  output and manual-camera end-to-end inference were not tested in this pass.
- Model Manager imports were registered locally with expected SHA-256 receipts.
  No registered weights, external MAMMA engine, body-derived assets, videos or
  private tests were added to the publishable checkout. See [MAMMA](MAMMA.md)
  for its separate-environment limitation; this is not a clean-install or
  other-GPU validation, nor a validation of the 60-second adapter limit.

Managed CMD startup after MAMMA routing changes (2026-10-29):

- Reviewed unchanged dependency pins and refreshed the installer lock's registry
  hash. The standard managed installation worker verified and reused all six
  installed profiles without runtime downloads or replacement. It refreshed the
  installed verification state; startup checks were not bypassed.
- Actual RGBMoCap.cmd startup and its child-worker environment check passed.
  A MAMMA project then opened through that same launcher and played two people
  for three seconds without render errors or observed video/mesh frame mismatch.
- Source composition validation now rejects a runtime registry hash that differs
  from the installer lock. A private fixture confirmed both valid acceptance and
  mismatched-lock rejection; private files remain outside the checkout.

These checks describe the recorded examples, not a guarantee for every video,
character or computer. Model weights and registered body data remain separate and
subject to their providers' terms. Review the applicable third-party notices and
corresponding source/build requirements before redistributing binaries.

MAMMA supported optional runtime and UI (2026-10-29, same Windows/RTX 5090 PC):

- A new isolated Python 3.11.17 / torch 2.7.1+cu128 runtime was installed from
  SHA-256 pinned package/source archives, without copying the research venv.
  SAM 2 and SDF were compiled locally. 176 package versions and 41,138 runtime
  files were checked; deliberately pruned debug/developer files were excluded.
  Actual runtime size: 6.19 GiB, excluding model weights and compiler tools.
  Re-running setup reused the verified runtime without downloading/rebuilding.
- CUDA SAM connected components and SDF distance, index width, winding and point
  gradients passed. SDF squared-distance max error: 1.21e-8; gradient error: 0.
  Restricted upstream code/native binaries remain outside this repository.
- Fresh two-view, two-person, 30-frame processing completed in 297.55 s through
  correction/preview. Peak PyTorch allocated/reserved memory was 6.10/6.12 GiB;
  Windows process peak working set was 8.93 GiB. These are different measures
  from whole-board VRAM or total system use. On the same working inputs, all
  compared landmarks, visibility/contact predictions, fitted poses, betas and
  translations matched the previous research engine exactly (maximum 0).
- Original native vertices matched fitted MAMMA vertices within 0.001 mm.
  Four source FBX files imported in Blender 4.5 with 52 Hips-root bones and no
  armature-object animation; max sampled bone-head error was 0.0034 mm.
- Height is numeric-only with a separate metre label; People has a separate
  label, and MAMMA hides Find person. Nine model assets, runtime readiness/setup
  instructions, other models' button restoration and the model guide were checked.
- Fixed asynchronous thumbnail restoration capturing a temporary default 3D
  camera. Fresh projects without a saved view framed both actors; saved camera
  target/distance were preserved on reopen. One earlier direct-GUI reopen timed
  out; subsequent fresh/saved-camera checks and CMD playback passed. This does
  not establish long-duration reliability or performance on every project.
- Actual RGBMoCap.cmd playback (4 s): 29.74 fps, frame interval p95 34.78 ms,
  max 35.65 ms, video/3D and mesh/video frame differences 0; two actors and name
  labels were visibly present, render errors absent, paused skin dispatches 0.
- Setup failure injection retained the old pointer and model/project sentinels
  and removed its failed staging directory. Unsafe archive aliases were rejected;
  generated batch paths with Korean text, percent signs and shell punctuation
  were checked. This is not a full non-ASCII compiler-install test.
- User's separate manual three-view/413-frame log completed in 712.08 s without
  fatal traceback or OOM. No clean-Windows, other-GPU, 60-second/eight-person
  capacity, or newly retargeted-character test is claimed in this pass.

MAMMA camera-count restriction removal (2026-10-30, same Windows PC):

- Removed the GUI/worker four-view maximum and the input/calibration parser's
  separate 64-camera ceiling. Reconstruction still requires at least two views;
  duplicate names, calibration/size checks and bounded metadata files remain.
  Existing clip/person limits were not changed.
- Input, calibration, request and saved-media validation accepted 2, 4, 5, 8, 16,
  32 and 65 camera entries. Empty/duplicate inputs, a single-view reconstruction
  and mismatched camera calibration were rejected.
- The actual preparation worker decoded/resampled all 5, 16 and 65 test videos,
  retaining 12 frames per view. Latest observed preparation times were 1.35,
  2.93 and 9.88 seconds. These were tiny synthetic 320x180 clips with repeated
  content for input-count regression, not multi-camera reconstruction benchmarks.
- Actual GUI widgets loaded 5, 16 and 65 thumbnails, selected the final camera
  and retained the shared trim and timeline frame. The 65-view GUI request queued
  all six reconstruction stages. Save Project As and actual GUI reopening
  retained all 65 media references, calibration and the selected last camera.
  No 3D render/playback performance claim is made from this hidden-window test.
- Shared system-RAM/Windows allocation-headroom checks and CUDA allocator budgets
  were unchanged. Low-memory observations triggered the existing guard. Injected
  CUDA OutOfMemoryError and CPU MemoryError produced failed events and nonzero
  worker results while preserving the existing project manifest. This was fault
  injection, not deliberate exhaustion of system/GPU memory.
- No new 5/16/32/65-view SAM, VGGT or complete MAMMA fitting run was performed.
  Removing a fixed input-count ceiling is not a measured memory-capacity guarantee.
  Private test scripts, synthetic media and logs remain outside the source checkout.

Pre-publication regression audit (2026-10-30, same Windows 11 / RTX 5090 PC):

- A separate source/runtime copy and local-only Git remote exercised the
  pre-MAMMA-to-current update, `install.cmd` verification/reuse and actual
  `RGBMoCap.cmd` startup before the optional MAMMA runtime was registered.
  Model/project/cache sentinels survived. Tracked local edits blocked updating;
  the application lock blocked a second launch. This reused installed pinned
  runtime files; it was not a clean-Windows or fresh-download installation.
- `Setup_MAMMA.cmd` accepted the license response, performed full package/native
  CUDA verification and retained the verified runtime. Declining left it intact.
  Injected setup failure retained the old pointer and user-data sentinels;
  archive escapes were rejected. Fresh native compilation was not repeated in
  this audit; its earlier build evidence remains recorded above.
- Fresh ExMR, FlowHMR and GVHMR processing each used the same 60-frame real-video
  segment. FlowHMR included VGGT and Fast SAM; GVHMR included HTD, FootMR and
  HandFlow. Reconstruction, correction, preview preparation and source FBX
  saves completed. MAMMA separately processed two synchronized views, 30 frames
  and two people through new SAM 2.1 masks, landmarks, VGGT, fitting and publication.
- Seven source FBX files were actually imported in Blender 4.5. All retained
  52 Hips-root bones and bone-only animation. Maximum sampled bone-head error
  was below 0.008 mm. Character imports covered Mixamo FBX, VRM and a 797-bone
  PMX; all four source models retargeted successfully. Eight character FBX
  reimports covered mesh/motion-only options, original rest transforms and
  sampled poses, with no armature-object or hair/clothing animation channels.
  PMX VMD exports passed finite-channel/frame/structure checks; native MMD
  playback was not repeated in this audit.
- Fixed MAMMA preparation/publication failures incorrectly returning process
  exit code zero. Both now emit failure and exit nonzero. Fixed retargeting's
  MAMMA body selection; a real PMX retarget also passed while the separate
  standard SMPL-X file was unavailable in the private test copy.
- Real GUI checks covered all four scoped model managers, original/corrected
  switching without frame/camera changes, manual contact editing, stale export
  blocking, Apply Correction without inference, Save Project As/reopen and
  MAMMA camera switching/private-body relocation. Eight project replacements
  were sampled; this short test does not prove leak freedom. Cached switching
  among three models in one project, GVHMR result variants, GUI character
  import/retarget, linked-character refresh after correction and selected-row
  character removal also completed.
- Four-second actual CMD playback samples for each model measured 29.74 fps,
  frame-interval p95 34.64–35.21 ms and maximum 35.28–36.74 ms. Video/3D and
  mesh/video frame differences were zero; paused skin dispatches were zero.
  These short clips on one PC are not long-video or cross-GPU guarantees.
- A separate 797-bone PMX viewport check with two MAMMA source actors measured
  29.73 fps over four seconds (p95 36.44 ms, maximum 37.87 ms), with no character
  renderer error. Two-second rest-pose and paused-animation observations each
  recorded zero loader-change events and zero frame swaps. This specifically
  checks the short/final-window repaint regression, not long-duration stability.
- Test scripts, licensed inputs, runtime copies, logs and generated motion were
  kept outside the publishable checkout. Main, user, per-model, build and agent
  guides were checked against current controls, file requirements and limits.

### Person-selection responsiveness (2026-10-30)

A two-person MAMMA project's independent copy was switched six times before
and after the GUI change. With profiling disabled, median synchronous selection
time fell from 803 ms to 173 ms; selected-preview/video readiness fell from
1,117 ms to 534 ms. The common scene, source frame and world camera were retained.
An unchanged active camera was reopened zero times instead of twice per switch.
The remaining selected-person preview loading still runs in the background.

Nested control updates now validate readiness once at their synchronous outer
boundary. Shared-floor checks reuse one read-only snapshot inside an individual
GUI validation call, never across calls or threads. Export and later callbacks
still check disk afresh. Tests confirmed that pending publication, changed floor
receipts and unapplied manual contact edits block export after a preceding valid
check. Manual contact editing, actor switching and Apply Correction completed
without replacing inference. The live non-default camera choice was retained.
Existing ExMR, FlowHMR and GVHMR cached original/corrected views also rendered.
Video reuse checks file identity, FPS and timeline offset; changed inputs and
decode failures reopen normally. These are short same-PC measurements, not
long-recording or other-hardware guarantees. Test data remains outside source.

Four-second CMD playback samples across the four models measured 29.74–29.82 fps,
with frame-interval p95 34.71–35.04 ms and maxima 35.86–36.73 ms. Video/3D and
mesh/video frame differences were zero. GVHMR finished one queued frame just
after Stop; a separate settled two-second pause recorded zero skin dispatches.

### Automatic / Low Memory routing (2026-10-30)

Fresh independent runs on the tested Windows/RTX 5090 PC covered both modes:
ExMR and FlowHMR each processed 60 frames through correction and export; GVHMR
also ran HTD, FootMR and HandFlow. MAMMA processed two synchronized views,
two people and 30 frames through fresh masks, landmarks, VGGT, fitting and
correction. Existing separate pinned runtimes were reused, not reinstalled.
FlowHMR's Original SAM (Fast SAM off) also completed both modes on an independent
eight-frame feature-extraction fixture.

Automatic selected fast paths with spare GPU memory. Allocator-budget tests
selected Low Memory and completed ExMR preprocessing at 3.9 GiB, body/hand
features at 4.5 GiB, FlowHMR camera at 3.5 GiB, and SAM features/motion at 3 GiB.
A real fast-loading CUDA OOM was released and retried once successfully in
Low Memory. ExMR features at 4.1 GiB still failed safely with memory advice;
the batch remained 8. These are caps on a 5090, not validation of real 3–4 GiB
GPUs. Models and full-sequence context can still exceed available memory.

Automatic versus Low Memory body-joint differences in the three single-video
samples were at most 0.84 mm before correction and 0.53 mm after correction;
separate inference runs are not bit-identical. MAMMA camera matrices and both
actors' sampled exported joints/rotations were identical in this comparison.
Ten corrected-source FBX files imported in Blender 4.5 with 52 Hips-root bones,
bone-only animation and sampled positions within 0.2 mm of committed frames.
Seven policy/retry/hook-lifecycle checks and actual Console/CMD mirroring passed;
the saved Automatic selection was retained. A short hook test does not prove
long-run leak freedom. Playback/rendering code was unchanged in this revision.

ExMR Sapiens peak PyTorch allocation decreased from 5.68 to 3.10 GiB; combined
body/hand feature-stage peak decreased from 5.15 to 3.87 GiB. These exclude
driver/display allocations. Low Memory can increase elapsed time and host RAM;
single-pass cold-start timings are not speed benchmarks. Precision, frame count,
resolution, temporal context and fitting iterations were retained. RAM/Windows
allocation-headroom monitoring remains a stop condition. Test scripts, weights,
logs, motion and temporary fixtures remain outside the publishable checkout.

### FlowHMR Fast SAM acceleration controls (2026-10-30)

The app now saves two independent, default-on switches for bounded four-frame
batching and GPU hand crops. Four combinations completed a real 60-frame input
with 30 valid pelvis anchors each. Against both switches off, final joint-position
differences were 0.21 mm at the 95th percentile / 1.52 mm maximum for batching,
and 1.23 mm / 2.45 mm with both switches on. These measure output disagreement,
not ground-truth accuracy. The separate earlier short-clip speed tests remain
the basis of the tooltip estimates; this integration check is not a new paired
speed benchmark.

Peak PyTorch allocation was 3.44 GiB without batching and 3.84 GiB with batching.
A seven-frame Low Memory run used one-frame batches with GPU crops and peaked
at 1.92 GiB. Original SAM completed three frames with acceleration flags ignored.
One-, two- and three-frame partial batches completed with finite outputs. CUDA
calls are not claimed bit-identical. Precision, crop resolution and the 20-step
FlowHMR generator were retained. No new packages or runtime installation were
needed; no forced physical OOM test was performed in this integration check.

Qt checks covered defaults, all four saved option combinations, busy-state
disablement, Fast SAM off, and all four model pages at short/tall window heights.
The settings use consistent top-down spacing and scroll when needed; hidden
model pages no longer reserve the tallest page's height. Test files and logs
remain outside the app checkout.


## Sequential multi-person model reuse (2026-10-30)

The app now groups ExMR, FlowHMR and GVHMR actor/take jobs by supported stage.
One model serves actors sequentially, with explicit actor paths and separate
receipts. GVHMR runs its observation and core phases in one worker, closing each
model scope before the next phase. Optional HTD, FootMR and HandFlow keep their
existing separate workers. Single-actor jobs keep their previous queue.

App-path measurements on the documented Windows RTX 5090 PC used two tracked
people with 60 and 45 frames at 30 fps and static cameras. Timings include feature
and core worker startup, excluding discovery, camera estimation, correction,
export and optional add-ons. These are one-trial observations, not universal
speedups; import and filesystem warm-up vary.

| Model | Separate actor workers | Stage reuse | Peak PyTorch allocated VRAM, before → after |
| --- | ---: | ---: | ---: |
| FlowHMR | 108.69 s | 65.34 s | 3.84 → 3.84 GiB |
| ExMR | 161.08 s | 106.51 s | 5.68 → 5.68 GiB |
| GVHMR | 56.02 s | 22.97 s | 2.82 → 2.82 GiB |

All six fixed-input actor/model comparisons were tensor-exact. With fresh image
preprocessing, maximum joint differences against the preceding experiment were
0.344 mm for FlowHMR, 0.242 mm for ExMR and zero for GVHMR; upstream preprocessing
roundoff is not a claim of improved or degraded accuracy.

Sixteen unit checks covered model lifetime, replacement, error cleanup, allocator
budgeting, OOM retry, queue ordering, bounded requests and path/runtime validation.
Real workers checked cache hits without model construction, two-frame Low Memory
feature extraction for FlowHMR/ExMR, Original SAM, corrupt ExMR hand-feature
recovery, and correction/derived export for all three models. The app writer also
saved three binary FBX files; Blender reimport was not repeated for this change.
A controlled real CUDA allocator OOM in a separate 1%-capped test process recovered
once in Low Memory and ended with zero allocated tensor bytes. This is not a
full-model OOM stress test on another GPU.

Actual GUI QProcess runs checked repeated cached queues and cancellation during
actor B, followed by reuse of completed actor A and reconstruction of B. A 20 ms
GUI timer had a recorded worst delay of 45 ms during that cancellation/retry test;
this measures offscreen event-loop responsiveness, not viewport FPS. Repeat checks
were short and do not establish universal leak freedom. GVHMR retained a stable
32 MiB PyTorch allocation between phases and its worker then exited. No new
runtime packages or changes to the pinned runtime registry were needed. Private
fixtures, model links and test scripts remain outside the application checkout.

## Pipeline and CPU preparation follow-up (2026-10-30)

A separate Windows RTX 5090 experiment retained the pinned runtimes, native
precision, batch sizes, resolution and temporal context. Two trials reversed
execution order; stages released their model before the next stage loaded.
AI worker times include startup and saved results but exclude discovery,
camera estimation, correction and export.

| Model / people | Separate phase workers, two trials | Connected phases, two trials | Peak allocated VRAM |
| --- | ---: | ---: | ---: |
| FlowHMR / 1 | 58.65, 53.33 s | 41.12, 44.18 s | 3.84 GiB in both |
| ExMR / 1 | 84.83, 82.88 s | 46.32, 46.99 s | 5.68 GiB in both |
| ExMR / 2 | 94.90, 90.76 s | 52.78, 53.81 s | 5.68 GiB in both |

Inputs were 60 frames for one actor and 60+45 frames for two actors, at 30 fps
and 720x1280. FlowHMR's two-person connected-phase benefit was smaller than
trial variability, so its existing multi-person phase grouping was retained.
Short-clip startup gains do not predict the same percentage on long footage.

Warm per-stage tests used those clips plus a separate 120-frame 1080x1920 clip,
each processed twice: 450 frames per variant. CPU preparation ahead by at most
two batches reduced total measured Sapiens time from 11.53 to 8.63 s, ExMR body
features from 18.57 to 13.75 s, and hand features from 7.50 to 5.63 s. Flow Fast
SAM changed from 42.96 to 40.79 s. Peak CUDA allocation was unchanged and sampled
RAM growth was small. ExMR final body-readout omission alone took 16.63 s; with
preparation overlap it took 13.15 s. The analogous hand-readout omission had no
clear benefit and was not adopted. These stage gains cannot be added together
to claim a whole-job speedup.

Sapiens and hand features were tensor-exact; ExMR body features differed by up
to 4.8e-6, versus 3.9e-6 on baseline repeats. Optimized final motion differed by
at most 0.83 mm / 0.306 degrees for FlowHMR and 0.88 mm / 0.068 degrees for ExMR.
Baseline repeats also differed, by up to 0.77 mm / 0.217 degrees and 0.69 mm /
0.047 degrees respectively. This is an output-preservation check without ground
truth, not a claim of improved or degraded tracking accuracy. Fixed feature
inputs produced tensor-exact motion when phases were connected.

Automatic CPU preparation is limited to one producer and two prepared batches,
with an additional 256 MiB prepared-storage estimate. It is disabled in Low
Memory or with insufficient CPU/RAM headroom. Existing OOM recovery and actual
system-memory monitoring remain active. No new runtime packages were added.

The integrated app code passed 25 focused policy/lifetime checks and five
tensor-exact grouped-versus-independent motion comparisons on fixed features.
Real one-/two-person worker runs verified model release and cached reruns without
loading a model. Offscreen GUI QProcess checks covered automatic single-person
phase joining, cancellation in the second actor's body extraction and retry
using completed preprocessing. A 20 ms GUI timer had a maximum observed delay
of 44.92 ms during that cancellation/retry run; this is not a viewport FPS test.
Eight-frame real clips checked Low Memory for FlowHMR/ExMR and FlowHMR Original
SAM. A damaged private ExMR hand-feature cache recovered without recomputing
valid body features. Correction and binary FBX saving were checked for both
models; Blender reimport was not repeated for this optimization.

## Contact correction v9: coherent root and swing transitions (2026-10-30)

Cached-source regression covered ten sequences / 10,121 actor frames: two
MAMMA actors, ExMR slow motion and stairs, FlowHMR long motion and backflip,
GVHMR dance, and pushups from all three single-video models. No AI inference
was rerun. Contact timings and stance ankle targets were retained. This is a
correction comparison without motion-capture ground truth.

The planner now joins swing targets to the final root-shifted free trajectory,
fades its extra target adjustment at the free boundary, and evaluates a smooth
floor-clearance constraint inside the root search. Native anatomical IK still
publishes leg rotations once. Existing 1,024-frame cores, four-second halos,
2 cm planning budgets, 60-evaluation cap, source ankle frames and hand solver
remain. Jacobian column scaling improves numerical conditioning. An exhausted
planning window is reported explicitly; it is not a convergence claim.

For the first MAMMA actor, pooled knee short-period RMS fell from 0.935 to
0.332 degrees (64.5%); the second fell from 0.390 to 0.329 degrees (15.6%).
The identified free-leg join's added bend step fell from +14.22 to -0.41
degrees. Short-period RMS subtracts a nine-frame degree-two Savitzky-Golay
trace; it is a diagnostic, not ground-truth error. Source-relative knee
second-difference RMS decreased on all ten sequences. Individual metrics do
not all improve: backflip total short-period RMS increased by about 2%, and
the stairs' worst added step increased by about 0.23 degrees. Residual source
noise, incompatible/unreachable pins and rapid intended movement remain.

MAMMA full-contact skinned sole height error stayed within 1.36 mm; across the
other tested foot contacts it stayed within 1.14 mm. Stance ankle positions
changed by less than 0.001 mm, fixed lengths and source ankle orientation were
retained, and torso world rotations were preserved. Hand timings and arm-only
behavior were checked; existing unreachable-palm residuals remain explicitly
reported. The backflip retained 103 contact-free flight poses exactly. Manual
intervals, disabled feet, empty timings, one-frame bypass and native window
sizes 64 versus 1,024 were checked. Two-, three- and seven-frame contact clips
remained finite. An intentionally exhausted planning cap produced its warning.

CPU correction-only measurements, excluding body sampling, hands, inference,
rendering and export: the MAMMA pair changed from 1.42 to 1.07 s (sum of three-run
medians); the long FlowHMR case increased from 1.34 to 1.53 s. Other long cases
were faster in these runs. Sampled process peak working set stayed below
300 MiB for these numeric-input tests; this is not whole-app RAM usage. No GPU
allocation or additional dependency is introduced by this change.

A separate copied MAMMA project completed the installed correction, preview
and binary FBX workers. Ordinary Blender 4.5 import checked all 422 frames,
52 bones, rest transforms, terminal tips and sampled half frames: maximum
joint error 0.0044 mm and fractional rotation error 0.0028 degrees against the
committed export. This was not a new playback-cadence or clean-install test.
User projects were not recalculated or replaced; Apply Correction is required
to obtain the new result. Private fixtures and media are outside the checkout.

## Final pre-publication review (2026-10-30)

Rechecked the 25 queue/prefetch/model-lifetime tests, four-model settings layouts,
Fast SAM option persistence and a real GUI connected-worker cache rerun. Current
correction code also passed the short-clip, contact/flight and installed-versus-
isolated checks above. Two-actor application-worker smoke tests completed for
FlowHMR, ExMR and GVHMR using private projects; these included fresh ExMR/GVHMR
processing and a FlowHMR cache hit. GVHMR retained 32 MiB of cuBLAS workspace
after releasing its model: a separate diagnostic cleared that library workspace
and measured zero remaining allocation. This is not a model leak or a guarantee
of zero CUDA context memory. No new runtime packages or clean-PC tests were added.
Source syntax, manifest hashes, local documentation links and publishable file
composition were checked; private tests, assets and logs remain outside the repo.

## Automatic VGGT meta/assign loading (2026-10-30)

Only the camera-only fast loader now uses meta initialization and direct
checkpoint assignment. Low Memory and its Automatic retry retain copy loading
and the existing block offloader; generic depth loading keeps its previous path.
No packages, model precision, sampled frames or camera equations changed.

The preceding independent component study ran 14 configurations across the Flow
and MAMMA Python runtimes. Camera pose encodings, intrinsics and extrinsics were
exact; six real load/infer/close cycles and one controlled failure cleared model
references and hooks. See [the measured scope](PERFORMANCE.md#vggt-camera-checkpoint-loading).

App integration used copied private data, not user projects. FlowHMR's actual
camera worker processed a 60-frame moving-camera clip with old/new fast loading,
Low Memory and one deliberately injected CUDA transfer OOM. All numerical camera
arrays were exact. The OOM test observed meta/assign on the first attempt, disposal
of that model, then non-meta copy loading on the Low Memory retry. MAMMA's separate
runtime processed its camera/capture worker from cached masks/landmarks for a
30-frame, two-view clip. Old fast, new fast and Low Memory produced exact world
cameras, estimated scale and calibration YAML. No new mask/landmark/body fitting,
full-motion quality, Blender or clean-PC test is claimed for this loader change.

Those seven camera-worker runs passed. Two additional checks confirmed a FlowHMR
camera cache hit loads no VGGT model and Static Camera still skips VGGT with zero
recorded GPU allocation. Five focused loader checks covered fast/Low Memory
selection, unchanged generic depth routing, missing checkpoint rejection and
unmaterialized-meta rejection. Test files, raw logs and assets stay outside the repo.


## PromptHMR integration (2026-10-30)

- The optional pinned Windows runtime was built and verified on the documented
  RTX 5090 PC: SAM2 connected components, torchvision CUDA NMS and native SDPA
  passed. Installed package hashes and a complete Pipeline import were checked.
  Base installation verification retained all six existing profiles. No clean-PC
  installation or other-GPU test was performed.
- A fresh 422-frame, two-person fixed-camera video completed tracking, camera,
  keypoint/image/video inference, native world post-optimization, app correction
  and preview/bone preparation in 236.2 s. Peak worker working set was 8.74 GiB;
  peak allocated/reserved CUDA memory was 4.61/7.83 GiB. See the
  [PromptHMR guide](models/PROMPTHMR.md#recorded-validation) for timing scope.
- Flat-hand-mean representation conversion matched native mesh/joint samples
  exactly for both actors at three sampled frames. This is a coordinate/data
  convention check, not ground-truth reconstruction accuracy.
- Both actors' original and corrected FBXs imported into Blender with 52 bones,
  Hips root, finite bone-only animation and distinct rest transforms. Save As,
  reopening and full-clip Mixamo character retargeting/baking passed.
- A short native OpenGL playback check recorded 31 ms median and 47 ms p95/max
  frame-swap intervals, without a sampled video/timeline frame mismatch. Three
  repeated project opens retained 784 handles and 51 threads, with about 534 MiB
  working set. This is not universal FPS or proof of leak freedom.
- Fresh 60-frame inference/export smoke tests passed for ExMR, FlowHMR and GVHMR.
  Model switching and scoped Model Manager rows were checked for all five tools.
  These checks do not cover every add-on, long sequence or difficult occlusion.
- Test programs, source videos, model weights and outputs were kept outside the
  public source file list. Restricted upstream PromptHMR source is downloaded
  locally by setup, rather than bundled into the source distribution.


## PromptHMR untracked hand placeholders (2026-10-31)

The pinned native video path writes zero 3×3 matrices for all 30 untracked finger
joints. They are not rotation matrices; subsequent axis-angle conversion produced
a fixed 120-degree rotation at every joint. Earlier native-mesh equivalence checks
preserved that invalid posture and did not establish a valid hand default.

Publication now uses the registered relaxed hand mean for this all-zero missing
hand case, while keeping raw native inference files intact. Valid native hand
rotations keep their prior conversion; inconsistent invalid matrices are rejected.
Receipts record the hand policy. Older published results show a refresh notice.

Private copies of two two-person clips (32 and 85 frames) were republished from
cached inference. Original and corrected body/root/shape tensors were exact;
the first 22 body/wrist joint positions were unchanged. Four original/corrected
FBXs imported into Blender with 52 bones and 30 finger bones. Local finger
rotation variation across the 32 frames was zero. Source projects/raw caches were
not changed. The runtime and AI inference calculations were not modified.


## ExMR public API and bounded toe IK (2026-10-31)

The integrated upstream source is v0.1.5, commit
`57e4824b9c532b52ed1c8b12c44000ad41ef4c4c`, retaining the documented Windows,
Low Memory, bounded prefetch and stage-scoped reuse adaptations. Public
`hand_boxes`, `model_inputs`, `ExMR.from_pretrained` and direct model forward
replace private pipeline calls. Upstream toe IK now limits its target offset to
0.03 m per axis. Existing weights, runtime package pins and other model backends
are unchanged.

Only those numerical changes were adopted. The app keeps its existing world
frame through an explicit local `normalize_world=False` option (generic upstream
default remains enabled). The upstream Boolean foot-contact detector is not
used by app correction. Shared contact code and manual timing precedence remain
unchanged. Source/API/solver hashes invalidate outdated ExMR stage receipts;
opening saved projects does not initiate reconstruction.

- Seven cached-feature actor segments from five real scenes (997 actor frames)
  produced tensor-exact camera/world parameters and static-confidence logits
  against the independently verified toe-only reference. Scenes included two
  dancers, burpees, stairs, moving-camera motion and two-person interaction.
- Fresh 60- and 45-frame actors passed keypoint, body-feature, hand-feature and
  motion stages in Automatic and Low Memory. Model instances were reused
  sequentially within each phase and released on session exit. Observed stage
  sums were 46.69 and 53.00 seconds; these single runs are not a controlled
  speedup benchmark or universal performance promise.
- Original/corrected meshes, paired shared cameras and shared-floor correction
  completed. Eight actual app FBX exports imported in Blender 4.5 with 52 bones,
  Hips root, finite bone-only keys and rest geometry distinct from animation.
- The new SimpleVO JPEG-input path differed substantially on a moving-camera
  sample. A small explicit frame-reader/count adapter retains the old canonical
  video pixels without extracting another JPEG sequence. A 267-frame Unicode-path
  test reproduced old rotations within 0.0000014 degrees; repeated calls reused
  the cache without rewriting it. Neither estimate was scored against camera
  ground truth.
- Updated stale inference receipts recomputed successfully; an immediate repeat
  hit the cache without rewriting motion. Changing the solver hash invalidated
  the stage identity. Manual intervals remained authoritative when automatic
  confidence was zero.

The separate contact-policy comparison did not justify replacing shared contact
detection. On the stairs segment, maximum frame-to-frame change of *added knee
flexion* decreased from 3.553 to 1.983 degrees with the new toe IK and current
contact policy, but rose to 4.114 degrees with the official Boolean timings used
directly. This is not total knee jitter or error against motion ground truth.
The body-height-only gate excluded just two frames across the tested segments;
the existing large burpee transition remained. No blanket contact-quality claim
is made. No clean Windows installation or second GPU was tested, and the other
four models were not fully reinferred for this ExMR-only integration. Private
test programs, footage, registered bodies and outputs remain outside the public
source distribution.


## Official ExMR checkpoint metadata (2026-10-31)

Model Manager includes `ExMR configuration` (`models/exmr/config.json`, 180
bytes), pinned to the same Hugging Face revision as the existing ExMR weights.
Download All and the row's Auto Download use the official HTTPS endpoint and
SHA-256 validation. Manual Download and Import use that same revision. It is
checkpoint metadata; it does not alter model architecture or app settings.

The actual download passed size/hash checks in an isolated data folder. A second
call performed no network request or file rewrite. Invalid manual imports were
rejected while preserving the valid file; a valid import succeeded. Offscreen
Qt checks confirmed the row only appears for ExMR, changes to Ready after
validation and leaves other model scopes unchanged. The development app also
downloaded and verified its own copy. Test artifacts and model files remain
outside the public source manifest. This metadata-only registration does not
require a runtime reinstall or a new checkpoint.

## PEAR live integration (2026-10-31)

The app live mode was checked on the documented Windows/RTX 5090 test PC using
the pinned shared runtime plus the small PEAR support overlay. Six fixed-crop
frames matched the earlier validated fast PEAR experiment exactly: maximum joint
and vertex difference was zero. This is equivalence to that FP16-encoder fast
path, not a claim of identity to every upstream precision/configuration.

Video timeline requests and native GPU mesh rendering were exercised. Readback
matched submitted vertex positions exactly and normals were finite. After
warmup, sampled automatic-detector frames typically took about 23–28 ms, with a
35 ms outlier; peak PyTorch
allocated VRAM was approximately 1.93 GiB. Startup varied around 18–34 seconds.
A 10 ms GUI heartbeat timer usually ran at 16 ms, with startup outliers around
0.55–0.8 seconds. These short tests do not establish universal frame rates.

DroidCam appeared by name. Camera images, a 90-degree rotation, switching back
to video and releasing the worker/device were checked. No person was visible
in that camera scene: a missing-person notice and cleared pose were verified,
not live human pose accuracy. No video or motion recording was created.

Three full worker open/close cycles settled bursts of 1,000 latest-frame requests
and exited in approximately 0.77–0.80 seconds. The supervising test process kept
218 handles and about 32 MiB RSS after each cycle; this is not a long-duration
GUI/driver leak guarantee. Twenty UI switch pairs across the five offline models
and PEAR preserved their selected offline result and camera target. Scoped model
managers and guides opened for all six tools. Existing offline inference/correction
math and pinned runtime packages were not changed or rerun for this UI integration.

Source/setup receipts, malformed live packets, bounded mailbox behavior, syntax
and public-source composition were checked. Models, user video frames and private
benchmark scripts remain outside the public source manifest.

### PEAR video playback follow-up

A continuous-playback regression was reproduced: each new request cleared the
mesh, and the exact-request test rejected almost every completed playing result.
The original seven-second test showed a pose in only 0.9% of visibility samples.
Playback now presents the completed video image and pose together, retains the
last pair while processing, and rejects results from earlier seek/source epochs.
Paused inspection still requires the exact requested frame. The ordinary video
decoder is suspended during paired display; bounded forward skips avoid repeated
codec seeks. Twenty-four decoded test frames had zero pixel differences against
individual seeks. Inference inputs, precision and model calculations are unchanged.

On the same test PC, a seven-second 30 fps source produced 195 completed pose
presentations (about 27.9 fps), with 100% pose visibility and zero image/pose frame
mismatches. Presentation intervals were 31 ms median, 52 ms p95 and 78 ms maximum;
request-to-display delay was 62 ms median, 78 ms p95 and 94 ms maximum. Peak PyTorch
allocated VRAM remained 1.93 GiB. These are short local observations, not guaranteed
rates. The preview image transfer is bounded to 1280 by 720; source pixels used
for inference and original video files are unchanged.

Additional tests covered 2x playback, manual seeking during playback, repeated
half-second loops, pause/exact seek, and deliberately slowed presentation polling.
Every presented image/pose pair matched. Loop and slow-poll outliers reached about
157 ms and 250 ms respectively; an exact seek can briefly wait for its first pose.
GUI heartbeat p95 was 16 ms, with a 188 ms maximum including startup. Observed GUI
RSS ranged roughly 522–545 MiB and settled around 534 MiB after leaving; handles
stayed around 1,495 after initial warmup. Worker shutdown took about 0.8–1.1 seconds.
These short runs do not prove long-duration leak freedom. Twenty model-switch
pairs also preserved offline state. Packet checks covered late/duplicate results,
old epochs, missing people and recovery, and paused exact-frame acceptance.

### PEAR bone appearance follow-up

PEAR now shares the offline octahedron generator and BoneRenderer, including flat
faces, joint caps and outlines. Its 52 body/finger display bones use native PEAR
landmarks; jaw/eye control bones are excluded. This does not fit an export rig or
change inference. Terminal head/toe/finger points use the pinned EHM surface-landmark
layout, checked against its registered metadata. Six cached poses retained their
input joints exactly; maximum display endpoint error was below 0.00000002 metres.
GPU readback matched the unmodified skin exactly and bone normals were finite.
Bone Overlay toggling preserved the pose. CPU frame preparation was about 0.81 ms
median / 0.88 ms p95; the bounded bone buffers occupied about 1.07 MiB. Short live
playback tests showed 24–25 completed presentations per second with uninterrupted
pose visibility and exact image/pose pairing on this PC. These observations are
not a guaranteed frame rate or a controlled renderer-only speed comparison.

### PEAR resident model lifetime

PEAR now preloads/warmups when selected, including with no input, and keeps one
model worker until the tool is left or closed. Switching inputs and Restart Live
Tracking release/reopen captures and reset tracking without rebuilding models.
Recoverable input errors retain the worker; fatal model/resource failures still
release it. Idle acknowledgements keep intentional waiting distinct from stalls,
and RAM/commit-pressure supervision remains active.

Two local lifecycle runs kept one worker PID through preload, two video sources,
six repeated video replacements, camera-open failure and recovery, playback and
tool exit. Initial preload took about 34 seconds; warm video replacement to first
pose took roughly 0.19–0.30 seconds, and input restart about 0.16 seconds. Worker
RSS was about 2,344 MiB while ready/idle and 2,513–2,519 MiB across the last repeated
video switches. Handles settled at 1,171 during those switches. Exit released the
worker in about one second. Short playback checks retained 100% pose visibility;
GUI heartbeat p95 was 16 ms, with startup maxima up to 578 ms. These short tests
do not establish long-term leak freedom. An active physical webcam stream was
not available during this follow-up; camera-open failure recovery was checked.
Packet checks also confirmed that an obsolete recoverable camera error cannot
unload models for the newly selected source. Tests/captures stay outside source.

### PEAR live limb roll

The endpoint-only preview basis was underdetermined around each shaft and could
spin near its reference-axis singularity. PEAR now uses its own-joint EHM/FK
rotations with fixed registered-neutral surface calibration, reusing the shared
front/side surface evidence. A source-relative soft correction is bounded to five
degrees, fades opposed/weak hints, and uses only a bounded past 0.10-second circular
history while streaming. This is a lightweight causal adaptation, not the offline
centred/paired iterative solver. Original inferred rotations, joints and skin are
not smoothed or refitted by this display calculation. Surface sampling and roll
run in the worker even with Bone Overlay hidden; painting uploads finished frames.
Neutral calibration is prepared once on CPU.

In a 180-frame / six-second real clip, the largest inter-frame change of **added
display roll relative to the inferred source** fell from 179.04 to 0.60 degrees;
the p95 fell from 59.49 to 0.22 degrees. These are display-error measurements, not
the total physical rotation of a limb or a claim that the underlying inference has
no jitter. Maximum display endpoint error was below 0.00000003 metres. Seek-order,
input-reset, full-body rotation covariance and independence from lower-joint roll
were checked. Twenty-six earlier sampled poses also passed the fidelity checks.
Six fixed-crop inference regressions retained exactly identical joints and vertices
to the earlier fast-path outputs. GPU mesh readback was unchanged and bone normals
were finite; a 180-frame before/after display video was rendered privately.

Worker roll calculation took about 0.56 ms median / 0.60 ms p95 on the test PC.
A short actual app playback showed about 25.6 completed presentations per second,
100% pose visibility and zero image/pose frame mismatches. Presentation intervals
were 32 ms median, 47 ms p95 and 63 ms maximum. Peak allocated VRAM remained about
1.93 GiB. GUI heartbeat p95 was 16 ms, with a 360 ms maximum including startup.
These short observations are not universal speed or long-duration leak guarantees.

### PEAR thigh neutral-roll calibration follow-up

The previous added-roll continuity checks did not establish alignment to the mesh.
Separate proximal and knee patches gave the thigh and shin different constant
neutral references. Both segments now use the same knee material patch; the arm
pair likewise shares the elbow patch. Each bone retains its own inferred rotation.
This changes display bases only, without modifying model outputs, joints or skin.

Removing shaft-direction differences by parallel transport, neutral left/right
thigh-to-shin roll disagreement fell from -39.05/+13.27 degrees to +1.77/-0.28.
In 180 continuous frames, absolute leg surface-facing residual p95 fell from 42.01
to 4.51 degrees (maximum 44.43 to 6.63). Across 26 separate sampled poses, p95 fell
from 44.13 to 5.73 degrees. These are material-proxy alignment measurements, not
ground-truth anatomical accuracy. Arm surface residual p95 also improved: 9.35 to
7.50 degrees in the continuous clip and 26.37 to 14.50 in the separate poses.

The largest inter-frame change of added display roll in the continuous clip was
0.79 degrees, versus 0.60 before; fixing static alignment is distinct from reducing
every temporal metric. Unmodified inputs, proper rotations, stateless seeks, input
reset and independence from lower-joint rotations passed. Maximum endpoint error
was below 0.00000003 metres. GPU mesh readback was identical; bone normals remained
finite in the 180-frame comparison render. A separate CPU check measured display
orientation work at 0.70 ms median / 0.94 ms p95. No model inference or additional
GPU allocation is needed for this patch mapping change. Tests and captures remain
outside the source tree.

A fresh seven-second app playback retained 100% pose visibility, zero image/pose
frame mismatches and about 25.7 completed presentations per second. Peak allocated
VRAM was 1.93 GiB; GUI heartbeat p95 was 16 ms (359 ms maximum including startup).
Presentation intervals were 32 ms median, 62 ms p95 and 93 ms maximum.
Closing released the worker in about 0.94 seconds. This short check does not test
long-term resource growth or an active physical webcam.

### PEAR foot/toe display basis follow-up

The generic shortest-arc zero-roll frame left a constant foot-width tilt when
combined with the registered ankle-to-toe pitch/yaw. Foot and toe display bases
now use the registered-neutral foot direction and body up, with consistent lateral
sign. This calibration is constant; each bone still follows its own inferred FK.
It does not move endpoints, change inference/skin, use toe rotation for incoming
ankle roll, or add frame-by-frame filtering or GPU work.

Validation used separate ankle/toe material regions (51/96 left, 53/96 right
vertices with more than 85% weight to the respective joint). A best-fit material
rotation transports the same neutral foot-width reference for comparison. In 180
continuous frames, aggregate width-direction error p95 fell from 26.29 to 3.66
degrees (maximum 28.76 to 4.52); in 26 separate poses, p95 fell from 26.70 to 3.71.
These measure the chosen neutral reference's agreement with deforming material,
not anatomical ground truth. A forefoot span alone is unsuitable for ankle-roll
validation because it also contains the toe joint's animation.

Non-foot display frames remained bit-identical to the preceding thigh-roll fix.
Endpoint error stayed below 0.00000003 metres, original arrays were unchanged,
and proper frames, seek/reset behavior, rigid rotation covariance and independent
ankle/toe rotations passed. Continuous physical foot-frame step p95 remained about
8.89 degrees (8.88 before), maximum 18.09 degrees; this is inferred motion, not
newly added roll. GPU mesh readback was exact and normals finite in a separate
180-frame comparison render. Private scripts, model data and video captures remain
outside the public source tree.

A fresh seven-second app playback showed 25.1 completed presentations per second,
100% pose visibility and zero image/pose frame mismatches. Presentation intervals
were 32 ms median, 62.3 ms p95 and 125 ms maximum; GUI heartbeat p95 was 16 ms,
maximum 359 ms including startup. Peak allocated VRAM was 1.93 GiB. Worker exit
took about 0.97 seconds. This short run is not a universal speed or leak guarantee;
no active physical webcam was tested in this follow-up.

### PEAR common input rotation and smoothing (2026-11-01)

**Smooth Motion** and **Rotation** now apply to video and webcam. Rotated input
pixels are used for detection/inference and the matching image is returned even
for a paused video seek. Source files are unchanged. Video playback uses source
timestamps across small forward skips instead of resetting its filter at each
skip. Paused seeks/loops/new epochs prime up to eight past parameter frames
(six at 30 fps), skipping geometry for hidden context frames. Obsolete context
requests are cancelled between inferences; no future frames or growing cache.

The causal adaptive rotation filter uses shortest geodesic interpolation, velocity
vectors in one coordinate frame, a 1.2 Hz base cutoff and 8 Hz upper cutoff.
Motion adaptation uses slope 0.65; shape uses 0.35 Hz / 0.02. Root translation
uses vector speed, 1.2 Hz / slope 4 and the same upper cutoff. This is an original
live adaptation of the [One Euro filter principle](https://gery.casiez.net/publications/CHI2012-casiez.pdf),
not an offline motion/contact solve. Smoothing intentionally changes the live pose.

In one 150-frame real clip, rotation-acceleration RMS changed from 109.48 to 86.65
rad/s² versus the preceding smoother (about 21% lower); root-relative joint
acceleration RMS changed from 31.15 to 25.66 m/s² (about 18% lower). Actual motion
also contributes to these measures; they are not ground-truth reconstruction
errors. The shared process compared raw, preceding and revised parameter geometry;
peak allocated VRAM was 1.93 GiB and process RSS about 2.47 GiB.

Controlled synthetic 1.5-degree rotation noise reduced stationary RMS error from
1.11 to 0.91 degrees and residual acceleration about 32%. The tradeoff is lag:
the slow-motion signal's best frame delay increased from about 67 to 100 ms;
fast-motion unshifted RMS error increased from 9.27 to 12.34 degrees. This is
stronger stabilization, not a claim of equal responsiveness or universally more
accurate motion. The checkbox disables parameter/root smoothing for raw inspection.

A real worker roundtrip verified repeated smoothed seeks were bit-identical,
unfiltered results returned identically after toggling, 90-degree image pixels
matched the rotated/resized decoded source exactly, and forward skipped playback
used zero extra context frames. Warm six-frame-context seeks took roughly 0.20–0.22
seconds; subsequent sampled playback requests roughly 0.05 seconds including pipe
polling. One resident model PID was retained; worker close took about 1.06 seconds.
Turning an upright test image sideways lost person detection; pixel rotation
support does not guarantee recognition of every input orientation. Pure checks
covered valid rotation matrices, time resets, bounded context, root coordinate
invariance and expanded arbitrary-angle rotation. Private tests and captures are
kept outside the public checkout; physical webcam capture was not retested here.

Actual app playback for seven seconds retained 100% pose visibility with zero
image/pose frame mismatches at about 25.7 completed presentations/s. Intervals were
32 ms median, 62 ms p95 and 63 ms maximum. GUI heartbeat p95 was 16 ms, maximum
359 ms including startup. Peak allocated VRAM remained 1.93 GiB, and worker exit
took about 0.89 seconds. These short observations do not prove universal FPS or
long-term leak freedom.

Eight repeated input reopens in the updated worker retained one model PID.
RSS settled near 2,577 MiB (first reopen about 2,575 MiB), with 1,160 handles
throughout. Reopens including six-frame seek context took about 0.22–0.28 seconds.
The repeated run closed its worker in about 0.95 seconds. This bounded sample
showed no sustained resource growth; it is not a long-duration leak guarantee.

### PEAR support included in base installation (2026-11-01)

`install.cmd` now prepares PEAR's small source/utility overlay after the base
runtime transaction commits. `update.cmd` reaches the same managed installation
flow. PEAR reuses the pinned shared Python/Torch packages; model weights and
registered geometry remain separate. The separate PEAR setup CMD was removed.

The support receipt binds its source/adapters and installed shared-runtime group.
A matching, intact receipt is retained without download/extraction or another CUDA
verification process. A changed base, changed support source or damaged inventory
stages and verifies a new support generation before replacing the pointer. The
installer keeps its retry marker until this succeeds; failed support preparation
does not roll back or corrupt the committed base packages or erase user data.

Eight isolated preparation checks covered initial receipt creation, idempotent
reuse without downloads, base-change invalidation, verification/download failure
preserving the old pointer, damaged-source recovery, application-lock refusal and
the integration call's ordering. A separate simulated full installer flow confirmed
the helper runs under the already-held lock after base commit, failure retains the
retry marker, retry succeeds, and model/project sentinels survive both attempts.
These use small mocked archives/processes for fault injection, not a clean-PC test.

The actual Windows `install.cmd` flow then completed on the test PC: all six base
runtime groups passed package/import verification and were retained; cached pinned
PEAR archives prepared 88 support files (about 0.87 MiB) and passed actual CUDA
transform/NMS checks. The shared-group receipt matched and the retry marker cleared.
A subsequent support preparation retained the same generation without download or
another verification process. This used an existing installation and cached
archives; it is not a clean-Windows or fresh-network installation test.
Six fixed-crop real PEAR inference cases through the newly prepared support path
matched the earlier fast-path references exactly (zero joint/vertex difference).
This checks the support path and package reuse, not webcam hardware or every pose.

### PEAR lightweight One Euro path (2026-11-01)

This revision supersedes the earlier same-day video pre-roll design. The existing
One Euro equations were not the main latency source: a smoothed seek or large
frame gap reconstructed six hidden past frames, and setting changes cleared the
last visible pair. Each request now infers exactly one frame. A seek seeds the
filter from that pose; continuous playback/webcam samples use their real time
delta. No future samples, hidden inference or history cache are used. The last
matched image/pose stays visible during replacement. A smoothing-only toggle
retains capture, person association, placement origin and camera framing.

The same filter coefficients/geodesic equations are retained. CPU parameters are
processed in two bounded batches (rotations/scalars), and the normal FP32 pose/face
path packs its tiny tensors into one transfer in each direction. A generic mixed
dtype/device fallback is retained. In 360 parameter samples, batched results
matched the preceding filter exactly, including face axis-angle conversions.
Median CPU filter time fell from 0.243 to 0.104 ms (p95 0.114 ms). On real CUDA
pose/face tensors, filtering plus transfers fell from 0.648 to 0.256 ms median
(p95 0.851 to 0.399 ms); rebuilt joints and vertices were bit-identical. This is
filter/transfer overhead, not the whole model's inference time.

Paired worker runs on one real video measured smoothed random-seek processing at
195.49 ms median before versus 58.67 ms after (response including polling about
217 versus 77 ms). Large skipped-frame requests fell from 171.12 to 57.92 ms.
Ordinary short skipped playback stayed around 30 ms; the gain is chiefly removing
hidden reconstruction, not multiplying steady-state model FPS. Isolated seeks now
show their own prediction rather than the old pre-rolled filtered pose. Sequential
filtering has identical parameters, but seek/gap histories intentionally differ.
Worker allocated VRAM remained about 1.93 GiB; end-of-run RSS was about 2,537 MiB
before and 2,532 MiB after. Single short runs are not universal performance or
long-duration leak guarantees. Tests remain private and source videos unchanged.

An actual seven-second app run toggled smoothing off and back on during playback.
Both toggles retained the displayed pose and camera state. Pose visibility stayed
100%, with zero image/pose mismatches, zero context frames and one inference per
reported frame. Observed presentation rate was about 26.9/s; intervals were 32 ms
median, 57.5 ms p95 and 156 ms maximum. The maximum includes setting changes;
continuous visibility is not a claim of perfectly uniform cadence. GUI heartbeat
p95 was 16 ms, maximum 359 ms including startup. Allocated VRAM remained 1.93 GiB,
and closing released the worker in about 0.97 seconds. Physical webcam capture
was not retested in this follow-up.

Eight input replacements retained one resident worker; RSS stayed around
2,532–2,533 MiB and handles remained 1,160. Replacement-to-pose times were about
0.125–0.141 seconds including capture reopening, with no temporal pre-roll. Repeat
seeks remained deterministic, smoothing-off results returned identically, rotated
pixels still matched the decoded source and the worker exited in about 0.97 s.


## PEAR adjustable One Euro strength (2026-11-01)

The live checkbox is now One Euro Filter with a short tooltip. Filter Strength
ranges from 1 to 100, default 50. Higher settings lower adaptive cutoffs/slopes
for pose, face, shape and root placement; this reduces jitter and increases lag.
Strength-only edits preserve filter history and do not reload the resident model.

At default 50, 360 parameter samples still matched the previous filter exactly
(maximum difference zero). Synthetic alternating-noise and step tests confirmed
monotonic smoothing/response changes at 1/25/50/75/100, valid rotations, unchanged
input parameters and exact seek reseeding. A seven-second real-video GUI run
changed strength five times and retained pose/camera state each time: 100% pose
visibility, zero image/pose mismatches and one inference per displayed result.
Observed presentation rate was 26.4/s; pose intervals were 32 ms median, 60 ms p95
and 78 ms maximum. Allocated VRAM remained 1.93 GiB.
These short local tests are not universal FPS or long-duration stability claims.
Physical webcam capture was not retested; the same filter path serves both inputs.
Tests and source videos remain outside the public repository.


## PEAR explicit Start / Stop (2026-11-01)

Selecting PEAR no longer loads or warms models. The right-side Start Live Tracking
button explicitly starts the worker and becomes Stop during loading and tracking.
Stop invalidates pending output, stops playback, clears the live mesh and releases
only this worker/camera. Start is disabled while shutdown finishes. Video/webcam
changes while running still reuse the model; input or filter edits while stopped
cannot start it. Guides supersede the earlier automatic-preload workflow.

A real-video GUI lifecycle test cancelled one loading worker, then completed two
Start/play/Stop cycles with fresh workers. Model selection, video open, stopped
webcam selection/settings and leave/reselect created no worker. Loading cancellation
completed in about 2.06 seconds; both ready workers exited in about 0.92 seconds.
Fresh load/warmup was about 35.8 and 36.5 seconds on the test PC. All sampled video
images matched their pose frame. The repeat lifecycle run used a minimized GUI;
the preceding visible run also verified video playback and the Stop button.
Ten packet-order/seek/missing-result regression checks passed. GUI heartbeat p95
was 16 ms, maximum 1,609 ms across setup, lifecycle and model switches; this does
not establish uniform rendering cadence or long-duration leak freedom. Physical
webcam capture was not retested. Private tests and user videos were not published.


## PEAR YOLOv8s detector (2026-11-01)

PEAR alone now defaults to official YOLOv8s v8.3.0 (22,588,772 bytes), scoped as
`pear_yolo` in Model Manager with direct official download and SHA-256 checked
imports/loading. Offline models retain their existing YOLOv8x asset and behavior.
No runtime package, detector resolution (640), threshold (0.5), crop ratio (1.25),
PEAR precision, full mesh, filter or bone-roll calculation changed.

Separate warmed worker tests processed 26 sampled frames from two real videos
and 90 sequential smoothed frames. Both variants detected the person in all
116 requests. Frame processing median was 27.29 ms with x versus 24.80 ms with s
(about 9.1% less time); p95 was 32.24 versus 32.07 ms. Detector medians were
11.26 versus 9.03 ms. These exclude decoding, GUI transfer and drawing; they are
not guaranteed viewport FPS. Peak allocated GPU memory was 1,932 versus 1,664 MiB;
end RSS was 2,174 versus 2,144 MiB. Short runs are not leak or hardware guarantees.

Detected boxes averaged 0.972 IoU against the previous detector. Pose rotations
differed by 1.08 degrees mean, 3.89 degrees p95 and 26.50 degrees maximum over
body/hand joints. Root-relative joint differences averaged 11.2 mm; wrist/ankle
p95 was 36.6 mm, maximum 121.1 mm. These are differences, not ground-truth errors
or evidence of equal accuracy. With the same x boxes supplied to the production
s path, all 1,392 stored pose/box/camera/joint arrays were bit-identical, isolating
the changes to detection/cropping. Five model-asset checks covered scoped readiness,
verified import, rejection of a tampered replacement, preservation of the previous
model/receipt and invalidation after an installed-file change. Tests and local
source videos remain outside the public checkout.

An actual seven-second visible app run with the integrated s detector presented
189 image/pose pairs (about 27/s), retained the mesh throughout and had no sampled
frame mismatches. Pose intervals were 32 ms median, 47 ms p95 and
110 ms maximum. Allocated worker VRAM peaked around 1.64 GiB.
The worker closed in about 0.84 seconds. This is not a synchronized before/after
viewport-FPS benchmark. Webcam capture and longer-term identity stability were
not retested in this follow-up.

## PEAR fixed-length bones and forward toe tips (2026-11-01)

The live worker uses fixed-length physical FK pivots and predicted rotations.
A resident CPU EHM model calibrates anatomy on first valid input and Reset Bones;
ordinary frames do not evaluate body skinning or mesh-regressed joints. A bounded
32-entry parameter-only history supports reset from the displayed pose ID. Expired
references warn and retain the rig. Seeks/filter edits retain calibrated lengths.
There is no new video or motion archive.

Three private clip replays covered 539 frames. CPU calibration matched the
earlier GPU-based calibration within 0.00024 mm. Per-segment length variation
was below 0.00018 mm (FP32 roundoff). Invalid NaN calibration retained the old rig.
A real worker verified first-frame calibration, fixed lengths, expired-reference
rejection, exact-reference reset and new-input calibration without reloading.

In a visible five-second app test, 150 image/pose pairs matched, approximately
30/s. Presentation intervals were 31 ms median, 56 ms p95 and 78 ms maximum.
Worker processing was 21.43 ms median, 23.80 ms p95 and 160.31 ms maximum.
Six consecutive resets paused on the displayed frame, preserved the model process
and viewport camera, and updated bound geometry in 0.109–0.141 seconds. Hiding
bones or changing filter strength did not recalibrate. Stop released the worker
in about 0.94 seconds. Startup took about 34 seconds; peak allocated worker VRAM
was 1,541 MiB. GUI handles remained at 1,691 across six resets; worker working set
changed from about 2.4057 to 2.4061 GB and GUI RSS from 587.8 to 588.6 MB.
Timer intervals including startup were 16 ms median, 31 ms p95 and 344 ms maximum.
Short runs do not establish universal FPS, leak freedom or webcam/other-GPU behavior.

The subsequent toe-tip change centres each neutral terminal vector on its foot's
forward sagittal plane instead of the big-toe surface landmark. Length and vertical
pitch are retained; native pivots and predicted rotations are unchanged. Cached
three-clip replay and OpenGL rendering verify the new constant display reference.
Other models, source correction and exports are unaffected.

## PEAR World Position (2026-11-01)

The default-on World Position option retains estimated placement. Off translates
the finished preview uniformly to the currently displayed pelvis XYZ, including
height; when stopped, the first valid pose supplies the anchor. Rotations and
relative pose continue. Only the latest raw packet and one anchor are retained.
Input changes and Stop clear them; Reset Bones retains the anchor. Toggling does
not submit inference, restart models, reset filtering or recalibrate bones.

Three cached clips (539 frames) verified exactly fixed pelvis coordinates,
unchanged source arrays and bit-identical rotation/facing arrays. Relative pose
differences stayed below 0.00012 mm from FP32 translation roundoff. The translation
step took approximately 0.011 ms median, with 0.0125–0.0230 ms p95 across clips.

A visible app test covered 182 paired video/pose presentations across two-second
off/on/off phases. Both off phases had zero root range on all three axes; on
restored the unmodified worker packet. Presentation intervals were 31 ms median,
47 ms p95 and 63 ms maximum. Worker processing was 22.54 ms median, 27.00 ms p95
and 178.72 ms maximum. GUI timer intervals including loading were 16 ms median,
16 ms p95 and 343 ms maximum. Ten paused toggles retained the request/pose ID,
camera settings and bounded render keys. Reset Bones preserved the fixed anchor;
Stop released the cached packet and anchor. These are short video checks, not a
webcam benchmark or a guarantee of sustained FPS or leak freedom.

## PEAR publication review and offline regression (2026-11-01)

Verified the PEAR paper citation against the pinned official README, updated
installation/model/agent guides, current live timings, storage and cleanup
instructions, and kept offline resource modes distinct from live preview.
Setup now marks adapted upstream files and records their pinned provenance.
Saved-workspace validation rejects PEAR as an active or member offline model,
including malformed files, without rewriting them. Retired identities remain
rejected. The live selector still accepts PEAR.

All 80 existing offline tool/processing-option combinations match the preceding
published revision. Existing model paths, URLs, hashes and stage requirements
are unchanged. Twenty GUI switch pairs and all six scoped model managers passed.
Four copied workspaces covered ExMR, FlowHMR, GVHMR, MAMMA and PromptHMR cached
results: loading, seek/playback, PEAR entry/exit, camera/settings restoration and
selected export availability passed. Original project file inventories and
timestamps were unchanged. Five source FBX exports were imported into Blender
4.5 with the normal importer settings: each retained a 52-bone Hips hierarchy,
finite poses and bone-only animation curves. This check does not rerun full AI
reconstruction or retargeting for every model.

PEAR support was freshly staged from the verified cached upstream archives with
real shared-runtime CUDA checks, then reused without reconstruction or package
upgrades. Isolated installer fault injection covered download/verification
failure, damaged source, base changes, lock enforcement, preserved prior pointers,
retained retry markers and successful retry. Model import checks rejected a
tampered detector while preserving the previous registered model and receipt.
The current live app passed default/position/filter controls, paired presentation,
Reset Bones, no extra inference on paused position toggles and worker shutdown.
Five repeated fixed-frame renderer transitions and framebuffer inspection did
not reproduce persistent display corruption. A transient composite window grab
during playback is not a framebuffer-quality measurement.

Publication checks cover source syntax, local documentation links, personal-path
and credential candidates, and excluded runtime/model/project/test payloads.
Private tests and derived files remain outside the source checkout. These are
tests on the existing Windows PC, not a fresh OS, alternate GPU or new physical
webcam certification. Short playback checks do not establish long-term leak freedom.


## PEAR fixed-camera multiview preview (2026-11-01)

Added multiple synchronized video inputs with a common 30 fps trim, four-column
preview thumbnails, distinct webcam assignment slots, and explicit Camera Align.
The latter runs person-masked VGGT camera/depth and GeoCalib gravity sequentially
in a separate Flow-profile worker, then restarts PEAR to avoid GPU overlap.
Scale is inferred from fixed bone lengths; background depth estimates the floor.
No upright-pose constraint or foot lock is added. These are estimates, not measured
calibration. More physical webcams and other GPUs remain unverified.

The production multiview path uses FP32 YOLOv8s, exact BGR crops, four bounded CPU
threads and at most four GPU rows, with the existing FP16 PEAR encoder/FP32 head.
Fixed-size padded rows avoid graph recapture during occlusion; they are excluded
from output/fusion. No TensorRT package or engine is required. Setup staged the
pinned GeoCalib/Kornia overlay and verified imports on the existing Windows Flow
runtime. SHA-256 checked model import, optional manager rows and existing base
PEAR startup were exercised; this is not a clean-PC installation test.

On the previously documented RTX 5090 test PC, a 12-frame/four-view port check
against cached single-view PEAR predictions measured root-relative joint
position differences of 0.755 mm p95 / 2.871 mm maximum and local rotation
0.101 degrees p95 / 0.408 degrees maximum. Detector-box differences were at most
0.0292 pixels. This compares predictions, not ground-truth accuracy. Warm batched
prediction/crop time was 21.37 ms median / 22.04 ms p95, excluding decode, fusion,
IPC and GUI; peak PyTorch allocated memory was 1.524 GiB in that run.

A native GUI run completed alignment and tracking restart, displayed 168 paired
frames in eight seconds, switched the preview without reloading the resident
model, invalidated calibration on rotation change, and exercised cancel/Stop.
During that playback sample the GUI timer had 15.90 ms median, 23.25 ms p95 and
141.46 ms maximum spacing. Startup and calibration were excluded from this timer
sample. One cold alignment-plus-tracking-restart took 81.3 seconds; VGGT/GeoCalib
work alone was approximately 26 seconds in a separate run, with respective peak
allocated CUDA memory around 6.54/0.59 GiB. Scope and startup costs differ.

A separate paced video-backed simulation exercised four camera capture slots,
one continuing command and temporary loss of a view. It delivered 203 paired
sets in 8.016 seconds; processing time was 34.42 ms mean, 40.19 ms p95 and 46.40 ms
maximum after warmup. Three/four available views remained explicit. Sampled peak
worker working set was about 5.65 GiB; the worker exited normally. This is not
physical-webcam FPS, synchronization accuracy or camera-driver validation.

Five repeated four-decoder open/seek/close cycles retained 555 handles after each
cycle. Partial camera-open failure cleanup, bounded parameter archives, malformed
calibration/path rejection, thumbnail/slot validation and selected-pose bone-tail
errors (below 0.000001 m) passed. These short tests do not prove universal leak
freedom. Cached ExMR, FlowHMR, GVHMR, MAMMA and PromptHMR projects retained their
project/result, frame, trim and viewport camera after opening PEAR multi-inputs
and returning. No new offline AI-inference/export equivalence claim is made.

Reset Bones followed by 4-to-2-to-1-to-4 input switching retained the resident
worker and accepted the new rig baseline. Stop/restart after a reset also
passed without replaying an expired displayed-pose reset.

### Independent input rotation controls

Rotation now sits beside each webcam selector. Native and offscreen GUI checks
verified independent angles, preservation by device identity after refresh and
slot reordering/resizing, selected-video angle restoration, and camera-alignment
invalidation after editing any input angle. Thumbnail selection retained valid
alignment; no tracking was automatically started by editing controls.

Four real video decoders and four simulated webcam slots applied 0/90/180/-90
degrees independently, with exact pixel agreement against the existing rotation
operation. Both Camera Align sample times retained each input's angle and rotated
dimensions. A real CUDA worker displayed all four rotated inputs, returned to a
single input and exited normally while retaining one resident model process.
These are input-routing checks, not physical multi-webcam driver certification
or a new pose-accuracy/performance benchmark.

### Webcam image preview before tracking

Webcam selection now opens a separate capture-only worker before Start, without
loading PEAR, its weights or Torch. A native GUI test with video-backed simulated
cameras verified partial four-slot assignment, all four thumbnail images,
per-device rotation and preview switching without reopening the capture process,
capture shutdown before model handoff, Stop remaining stopped, None releasing
devices, offline return and a visible failure without automatic retry loops.
Worker logs confirmed Torch was not imported. Sampled preview working set was
about 191 MiB; a three-second GUI timer sample measured 16 ms median, 31 ms p95
and 141 ms maximum spacing. Five separate 30-frame open/close cycles retained
219 parent-process handles and 27 threads after each close; delivered packet
spacing ranged up to 94 ms in that simulation. These short tests are not physical
webcam FPS or long-term leak certification. The screenshot's NDI Webcam was not
present in the subsequent device probe, so its driver/feed was not verified.

The webcam controls were subsequently consolidated below the video into one
Camera / Device / Rotation table, with Count and Refresh in the same panel.
Native layout checks covered 1, 2, 4, 8 and 32 slots and shrinking back to one:
columns/row centres aligned, one-camera selection remained visible, horizontal
scrolling stayed unnecessary and longer lists scrolled within a 220-pixel table.
The camera-only preview, thumbnail, rotation, Start/Stop handoff and failure-path
simulation passed again after moving the controls. Capture/inference math and
worker ownership were unchanged by this layout edit.

### Continuous capture through Start and Camera Align

This supersedes the earlier capture-worker handoff. The capture-only process
now remains the sole owner of webcams during model loading and alignment.
Independent model readers copy its latest raw BGR frames through fixed Windows
shared-memory allocations protected by named mutexes; no per-frame disk cache
is introduced. Stop/None/leave still release capture. Device/resolution changes
may reopen it; an inference or calibration restart does not.

A native two-view GUI test used deterministic initial dataset frames repeatedly
published by simulated camera producers, plus a changing corner marker. Actual
PEAR inference, VGGT/GeoCalib alignment and the following PEAR restart succeeded.
The capture PID and shared-frame names remained unchanged: two device opens,
zero device closes during Start/alignment, and zero blank-image observations.
Image updates continued through loading, sample preparation, GPU shutdown and
alignment. Calibration completed in about 36 seconds and the first aligned pose
after restart arrived around 73 seconds after the click in that run. These are
workflow checks, not physical-camera timing, pose accuracy or universal FPS.

Five repeated shared-frame reader open/close cycles retained 224 parent handles;
frames remained complete under concurrent publishing, and a closed owner was
rejected. A native real-video decoder test with delayed mocked model/alignment
workers displayed 62 distinct frames during loading and 91 during the alignment
phase, retained play/pause/seek and fixed the alignment sample at the click's
frame/epoch. Its GUI timer measured 16 ms median, 31 ms p95 and 203 ms maximum
spacing in that short test. Raw-preview rotation and reset on leaving PEAR passed.
An injected sample failure exited alignment, retained its error notice and kept
the same capture process displaying frames (29 image updates in one second).
Physical multi-webcam drivers and long-duration behavior remain unverified.


## PEAR automatic alignment and causal live Foot Lock (2026-11-01)

This revision supersedes the earlier optional/manual-only Camera Align and
no-live-contact behavior. Start aligns one or more fixed views before tracking.
VGGT and GeoCalib are required; their model-manager readiness is checked.
Single-view scale is estimated from inferred body geometry and person depth,
not a triangulation baseline. Failed alignment keeps an actionable error.

Validation used private tests outside the repository and preserved source videos:

- Real PEAR + VGGT + GeoCalib completed automatic Start for one and two simulated
  camera streams. Capture process/shared buffers were retained and no blank
  preview samples were observed. These were dataset frames through the capture
  protocol, not physical webcams.
- Four real synchronized video files aligned and played in the app. The eight-
  second observation contained 180 paired results. Worker median/p95/max were
  39.56/44.72/692.92 ms, including first-frame overhead. Tracking peak CUDA
  allocation/RSS were 1.57/2.32 GiB. Calibration VGGT peak allocation was 6.53 GiB;
  tracking and alignment GPU models run sequentially.
- A final 450-frame cached four-view replay reduced locked-frame mean ankle
  displacement from 3.46 to 1.95 mm; knee-angle second-difference p95 stayed
  4.02 degrees. Single-view replay had only seven contact frames, so its contact
  improvement is not broad evidence. Original pose noise remains.
- Fixed leg/toe lengths, ankle/toe world rotations, finite orthogonal rotations,
  input-array immutability, root-fixed behavior, seek/disabled reset and invalid
  single-camera depth rejection passed. A 240-frame near-straight perturbation
  case had 0.17 mm maximum knee step after extension resistance was added.
- Real video decoding continued during mocked alignment/loading, including seeks,
  input rotation and a frozen alignment sample time. GUI timer p95 was 31 ms;
  its maximum was 422 ms including startup/transition work. This does not prove
  stutter-free playback on other systems.
- Intentional sampling failure exited alignment and kept its error visible while
  camera preview continued. Stop during alignment and Reset Bones triggering
  realignment passed UI contract checks.
- Existing ExMR, PromptHMR, FlowHMR, GVHMR and MAMMA cached projects retained
  selection, timeline trim, displayed result and viewport camera after entering
  and leaving PEAR. Their AI inference was not rerun for this isolated change.

The live constraint is deliberately conservative: it allows reach/floor residuals
instead of stretching bones or forcing nearly straight knees to bend. World
Position off prohibits pelvis translation. It does not replace offline contact
editing, solve bad calibration or guarantee removal of inferred motion jitter.


## PEAR independent One Euro components (2026-11-02)

One Euro Filter now exposes default-on Bone Rotation and World Position
checkboxes. The parent switch gates both; their selections are retained while
it is off. World-position filtering is separate from the world-movement switch
and from lightweight Foot Lock. Runtime flags default to both on when absent.
Changing a component does not change source epoch, camera calibration, resident
model, capture identity or the other component's filter history.

A cached replay tested all four component combinations for 120 frames each with
one, two and four views. Both-on results exactly matched the previous solver
and temporal filter reconstructed and verified against the prior source-manifest
hashes (maximum numeric difference zero). Both-off matched the parent-disabled
path. Position-filter selection left rotation outputs and root-relative pose
unchanged. Fixed lengths, valid rotation matrices, missing-input resets,
re-enable behavior and same-timestamp history retention passed. No AI inference
was rerun; these are filter/fusion regression checks, not a new inference or
webcam throughput benchmark.

Native UI checks passed for component defaults, request forwarding, master-off
selection retention, Filter Strength availability, no model reload/realignment,
and layout at 1100/1500-pixel window widths. The five offline models' cached
project restoration checks also passed. Test scripts and cached data remain
outside the source distribution. Source inference, calibration, foot-lock math
and installed dependencies are unchanged by this component split.


## FlowHMR Unicode progress output fix (2026-11-02)

Fast SAM loading, single-frame and four-frame wrappers now explicitly open
UTF-8 discard streams. Base managed/development worker commands select Python
UTF-8 mode; `-I` otherwise ignores parent UTF-8 environment variables. Standard
worker stdout/stderr already had UTF-8 reconfiguration, so the independently
opened locale-encoded discard stream was the critical remaining failure path.

Private regression checks reproduced all three prior discard-stream failures
with simulated cp1252 defaults and box-drawing output. All corrected paths
passed, including collector cleanup. Actual managed interpreter probes verified
UTF-8 mode despite hostile cp1252 environment variables for shared, Flow, Hand,
HTD and ExMR profiles; their runtime boots passed. Binary pipe bytes were retained.
Development command construction was checked separately.

Real CUDA Fast SAM extraction passed for one frame and four-frame batches with
CPU/GPU hand crops. Same-input repeat controls showed existing token variation:
single-frame patched/control maximum difference was below 1e-6; batch differences
were within the observed baseline repeat maxima (0.01623 with CPU hand crops and
0.00750 with GPU crops). This was a feature-extraction check, not a full motion
quality or timing benchmark. SAM syntax trees differ only in the log encoding
keywords; model operations, precision and input settings are unchanged.

The full feature cache identity matched before/after on a private cached project.
Two complete-byte source aliases preserve valid preprocessing caches for this
logging-only change; unknown subsequent source edits still invalidate them.
Runtime package/profile pins remain unchanged; the reviewed registry hash and
source-lock identity were updated for the new worker launch flags. Tests, images,
models and cached project data remain outside the source distribution.

The normal managed installer reused all six verified base profiles, refreshed
the source/launch receipts and passed PEAR CUDA support verification. The actual
launch-source admission checks passed with GUI main stubbed, and the Flow feature
identity remained unchanged after installation. No inference dependency versions
or model files were replaced.


## International Windows paths and output (2026-11-02)

The encoding audit expanded beyond Fast SAM to optional MAMMA/PromptHMR workers,
scene-camera preparation, setup/verifier commands, character jobs, JSON metadata
and chunked GUI process output. Python launches explicitly select UTF-8 even
under isolated mode. CMD entry points disable delayed expansion; PromptHMR's
generated UTF-8 native-build script now selects code page 65001, as MAMMA's did.
Dependency/package pins and optional runtime identities remain unchanged.

Seven independent path cases covered ASCII English, Western European accents,
Japanese, Russian Cyrillic, Korean, Chinese and combining-accent/emoji names with
spaces, brackets, ampersand, exclamation mark and percent sign. Each passed real
FFmpeg probing/preparation/trimming, PyAV/OpenCV video decoding, UTF-8 metadata,
NumPy archive/mapping reads and project Save As/reopen checks. Fifty interpreter
probes passed with hostile parent encodings (CP1252, CP932, CP1251, CP949, CP936
and ASCII), including separate optional-runtime probes. Byte-fragmented GUI
protocol and character-progress messages retained complete Unicode values and
EOF tails. Model import/hash rejection and Torch tensor file round trips passed.

The old native OpenCV image path API failed in twelve non-ASCII RGB/mask cases
on this PC. Python byte I/O plus the same OpenCV codecs passed all cases and
preserved PNG bytes/pixels, uint16 masks and RGBA data. Upstream image entry
points are bridged only within isolated MAMMA/PromptHMR scopes and restored on
errors. A 1080p PNG read check measured 9.31 ms median, 10.57 ms p95 and 11.32 ms
maximum versus 10.87 ms native ASCII-path median in this short run. Fifty reads
showed zero net handle growth and about 0.004 MiB retained RSS growth after
warm-up. This is not a universal speed or long-session leak guarantee.

Real Blender FBX/VRM/PMX imports passed in ASCII and mixed-language output
folders, with identical compared numerical canonical rig arrays. The existing
PMX motion exported to mixed-language FBX paths both with and without mesh, and
to VMD. Reimport/parsing passed: FBX retained its Unicode armature name and
bone-only animation channels; VMD retained its specified CP932 data encoding.
Twenty-one entry-point CMD path-dispatch tests passed under inherited delayed
expansion, using private standard-library-only runtime copies and stubbed app,
installer and updater bodies. These do not constitute clean installation or
remote-update tests. Restricted data and original user projects were not changed
or included in source; private fixtures, logs and generated files remain outside
the distribution.

The contact solver source, numerical inference settings, body data and source
motion were retained. New image adapters participate in MAMMA/PromptHMR cache
identities, so new inference validates the changed connector instead of silently
reusing a receipt for different code. Existing committed results are preserved.
These checks used one Windows PC without changing its system locale. They do
not certify every Windows language edition, GPU, new native build or third-party
file format. No full-model inference benchmark was repeated for this text/I/O
change.

## PEAR alignment fallback and single-view scale (2026-11-02)

Camera Align is attempted before tracking; failures now retain a visible warning
and continue with the selected camera, without calibrated fusion or aligned foot
lock. GUI checks covered missing alignment prerequisites, sample/solve/launch
failures, explicit retry, successful calibration and Stop. Actual PEAR workers
returned finite poses from an unaligned single video and from selected view 2 of
a two-slot input. The latter reused one source video to exercise dispatch, not
independent-view reconstruction quality. The worker exited after testing.

In a real single-view alignment, the old whole-image confidence percentile
discarded all 24 person-depth samples despite passing reprojection checks
(95th-percentile errors about 11 pixels, limit 38.4 pixels). Ranking confidence
within valid person samples retained 18, with a relative scale spread of 4.7%.
The complete VGGT/GeoCalib/floor run then succeeded; estimated floor residual
p95 was 15.9 mm. This is internal consistency, not ground-truth camera accuracy.
Missing depth, nonfinite/constant confidence, invalid observations and inconsistent
scale remained rejected. Multi-view calibration/fusion math was unchanged.
No clean-install, physical-webcam or sustained-playback benchmark was repeated
for this change. Private samples and test code remain outside the distribution.

The final publish check separated PEAR's core required assets (checkpoint,
geometry and YOLOv8s) from optional Camera Align weights. Missing VGGT/GeoCalib
now reaches the warning/fallback path rather than failing the core-model check.
GUI checks used the real stage dependency lookup with only model readiness
simulated: absent alignment weights allowed tracking, an absent PEAR checkpoint
still blocked Start, and all five Model Manager rows remained available with
alignment rows marked optional. Console/CMD routing, alignment failures, retry
and Stop regression checks were repeated. This does not make alignment optional
for calibrated fusion or live foot locking, and is not a new inference benchmark.


## Saved camera inspection and source export options (2026-11-02)

Selected PR #18 ideas were adapted locally, without merging its other changes.
Original inference, contact correction, runtime profiles and model calculations
were not changed.

- The default-off Camera display and Camera Preview were exercised with four
  real MAMMA calibrations. Real FlowHMR (235 frames) and ExMR (32 frames) saved
  rotation tracks loaded. Nonidentity rotation/translation fixtures checked
  camera-center and image-corner projection conventions. Camera toggles retained
  the existing viewer eye/basis. Camera painting performed no camera metadata I/O.
- A four-second, 120-request decoded-video test shared its source image with the
  camera canvas: 121 deliveries including initial presentation, zero mixed frame
  pairs. Delivery interval median/p95/maximum was 31.64/47.11/48.00 ms. Rapid seeks
  retained the latest request; camera switching retained a complete frame pair.
  This short offscreen test is not a general playback or GPU FPS guarantee.
- Sixty temporary camera-loader start/cancel/close cycles had 459-460 process
  handles. RSS warmed from 183.61 to 191.21 MiB and stabilized in later samples.
  Loader/decoder threads stopped at close. This does not establish indefinite
  leak freedom or replace long interactive testing.
- Unchecked FBX writer output matched the previous writer byte for byte for a
  real three-frame fixture. Blender 4.5 imports checked integer and quarter-frame
  suffix samples with and without the leading T-pose: maximum position/matrix
  differences were zero. The FBX and BVH reference-pose relative joint positions
  differed by less than 0.001 mm. The actual BVH worker saved both option states;
  unchecked BVH matched the committed source bytes. Original frame count/FPS and
  channels were retained after the optional prefix, and unavailable source
  export remained blocked before opening options.

Camera display is calibration inspection, not camera re-estimation. Flow VO
translation is unscaled and defaults off; ExMR has rotation only; other offline
models may expose only a reference camera. Distorted calibration uses a pinhole
overlay with a warning. No fresh AI inference or clean-Windows installation was
claimed for these display/export-only changes. Private inputs, test scripts and
outputs remain outside the public checkout.


### Left video overlay follow-up (2026-11-02)

The first integration exposed projected bones only in the right Camera Preview.
The left video's existing Overlays switch now also paints the reconstructed
skeleton through the saved camera. A real four-camera/two-person MAMMA fixture
checked automatic camera loading in the ordinary Perspective view, both actors,
Overlays off/on, independence from the right Bone Overlay/Camera switches,
latest seek pairs and immediate visibility changes while paused. It shares the
same decoded image and frozen camera/pose pair; no second decoder was created.


### Single-person FlowHMR camera compatibility (2026-11-02)

Single-person FlowHMR projects do not necessarily publish the multi-person
scene_camera.json receipt. Camera display now reuses the validated native camera
track and committed placement basis/height in memory. A real 2,267-frame single
project loaded in 0.020 seconds; sampled camera centers retained the saved
height and the basis matched the existing shared-camera calculation. The
corrected-source video overlay loaded successfully. Original project file sizes
and timestamps stayed unchanged; no inference or new scene receipt was needed.
Missing camera inputs are reported as a concise notice instead of a raw Windows
file/path error. Existing four-camera MAMMA and saved FlowHMR/ExMR camera tests
remain separate regressions.


### Single-person ExMR/GVHMR camera compatibility (2026-11-02)

Both workflows can omit the shared-scene camera receipt. An isolated CPU reader
now derives the display reference from the original paired global/camera motion
and registered pelvis pivot represented by the committed original joints. It
reads only the first joint from the NPZ stream and memory-maps tensor storage;
no body model load, inference, correction or project-file writes are needed.

A real single-person ExMR project loaded 267 camera rotations, including about
45 degrees of relative rotation. Sampled source times matched the saved SimpleVO
rotation track. A single-person GVHMR project loaded its reference camera.
Both displayed corrected-source video overlays and camera-gizmo data, passed
seek checks, and rejected deliberately stale in-memory reference stamps.
Original project file sizes/timestamps were unchanged. First reference loads
were 2.20/2.19 seconds (ExMR/GVHMR); reused loads were 4.84/3.66 ms on the test PC.
The editor did not import Torch. Cancelling a starting reference process and
closing its loader took 60 ms and left no reference child process running.
These are short local checks, not universal latency or indefinite leak claims.


### Approximate camera movement visualization (2026-11-02)

Camera gizmos now use estimated movement by default (Motion x = 1), separately
from the saved video-overlay projection. FlowHMR uses its existing VGGT relative
camera-center displacement. Moving-camera ExMR derives displacement from the
original paired world/camera roots and saved camera rotations in the isolated
CPU reader; it does not use corrected motion or rerun visual odometry/inference.
ExMR shared scenes use the scene's fixed anchor take, not the currently selected
person. Incomplete anchor coverage falls back to rotation only with a notice.

A real 267-frame moving-camera ExMR sample produced a nonzero estimated path
(maximum displacement 13.61 inferred metres, not ground truth); sampled times
and camera-center sign/scale checks passed. A private static-policy fixture
retained a fixed camera center. A real 2,267-frame Flow cache had a small nonzero
VGGT displacement (maximum 0.000912 relative units). These values establish
that the stored estimates were displayed, not trajectory accuracy.
All four MAMMA cameras and the current GVHMR reference remained fixed. A known
nonidentity camera fixture validated displacement direction and scaling.
GUI tests confirmed that changing Motion x leaves the video-overlay camera
matrix unchanged, and that the movement control is hidden for fixed references.
The existing single-person overlay/seek, stale-reference, CPU cancellation and
original-project preservation checks also passed. No new model/runtime or
inference is required. Monocular scale and body-motion prediction errors remain.

### Leading reference frame placement

The optional leading T-pose follows PR #18: it uses the original first motion
frame's root position, including shared-scene placement; it does not recenter
the animation. The default remains off. Blender 4.5 imports were checked for a
real source sample, a nonidentity scene transform and a one-frame clip. The
reference root matched the first original root exactly. Original suffix poses
(including quarter-frame samples) differed by less than 0.001 mm in position
and 0.000001 in matrix entries; rest matrices were unchanged, and no armature
object transform channels were added. Unchecked FBX output remained byte-identical
to the previous writer. BVH suffix channels retained the original motion.

### BVH export profiles (2026-11-02)

Added a BVH-only Generic/Blender (default) or 3ds Max Biped selector, adapted
from PR #18. The Biped copy uses Character Studio names and uniform ZXY channels;
source inference, correction, previews and existing files are unchanged.

- Full existing MAMMA/FlowHMR/ExMR/GVHMR samples (493/2267/267/267 frames)
  retained 52 joints, hierarchy offsets, terminal offsets, root translations and
  timing. Maximum keyed local rotation difference was below 2e-11 radians.
  Conversion took approximately 0.15-1.23 seconds on the test PC; this excludes
  application startup and destination I/O and is not a universal speed claim.
- Blender 4.5 imports at sampled keyed frames differed by at most 0.012 mm in
  heads/tails and 0.000029 in rotation-matrix entries. The numerical double-precision
  conversion is tighter than Blender's single-precision animation storage.
- Sampled original-motion half-frames differed by up to 5.62 mm and 1.18 degrees
  because the Euler encoding changes. The synthetic interval from the optional
  T-pose to original frame one showed larger interpolation differences (104.73 mm,
  18.63 degrees). No keyed pose was altered. Prefer Generic/Blender when preserving
  the existing channel interpolation is needed.
- Verified all-person request propagation, profile visibility/default, shared
  placement with leading T-pose, gimbal/branch-cut sequences crossing bounded
  blocks, and the cancellable BVH save entry. Default Generic output remained
  byte-identical. Unknown profiles, invalid data and cancellation retained existing
  destinations and removed staged partial files; original source stamps stayed fixed.
- 3ds Max was not installed. Actual Biped loading/retargeting remains unverified;
  Blender import is not a substitute for that application test.

### Optional background camera rotation refinement (2026-11-02)

Added default-off FlowHMR/ExMR controls for PR #18-derived camera display
post-processing. Existing source tracking, poses, contacts and animation export
remain unchanged; Static Camera is skipped. No AI inference or additional runtime
installation is introduced.

- An isolated textured-scene fixture with a known 1.1-degree rotation accepted
  15/15 samples and ended with approximately 0.0042-degree rotation error. Camera
  centers and input matrices were unchanged. This is an implementation fixture,
  not a real-scene accuracy benchmark.
- On an existing 267-frame moving-camera clip, the adaptation matched the PR
  implementation exactly: all 45 samples were rejected and the original track
  was retained. The GUI reported 0/45; off restored the original matrices exactly,
  and re-enabling reused the validated session result.
- Verified dependency invalidation, static skip, Unicode working paths, malformed
  legacy view preferences, optional-worker failure fallback, temporary-file cleanup
  and termination after observed fit progress (6/106 samples; approximately 0.013s
  from cancellation to child reaping on this PC).
- During the real-clip toggle test, 120 distinct frames were delivered with median
  33.1ms, p95 37.4ms and maximum 38.7ms intervals. Image/camera/pose mixed pairs: zero.
  Frame and orbit/pan state were retained; original project stamps were unchanged.
  A separate short UI heartbeat test had a maximum gap around 12.1ms. These short
  checks do not guarantee universal playback speed or long-run leak freedom.
- Existing ExMR/GVHMR camera overlay and seek checks and the 2267-frame FlowHMR
  saved-camera regression passed with refinement off. Console output and the
  command-window mirror included the refinement result and cache reuse.

### Portable Blender and MAMMA scene export (2026-11-02)

Added `.blend` to source export with PR #18-derived portable video/camera
bundles and a MAMMA-only whole-scene option. Camera tracking, approximate
movement scale and optional background refinement affect cameras only. Whole
MAMMA scenes retain the original common timeline and reject inserted T-poses.

- Exported an existing 174-frame ExMR take, a 510-frame FlowHMR take with a
  leading T-pose and saved moving camera, a 267-frame GVHMR original and a
  PromptHMR original in its shared scene. All worker saves completed; existing
  source files were retained. No model inference or clean installation was run.
- Exported a corrected MAMMA scene with two people, four cameras and 493 frames.
  The existing working videos, full body surfaces and reference skeletons were
  used. Moved the ExMR/FlowHMR/MAMMA bundle folders into accented/Korean paths
  and reopened them in Blender 4.5. Relative movie and PC2 references resolved.
- At original frames 0, 80 and the last frame, imported skeleton head errors
  were at most 0.0048 mm; general bundle bone matrices matched a direct FBX
  import within 1e-6. MAMMA evaluated PC2 surfaces matched the independently
  evaluated selected cache at those frames. Armature object transforms had no
  animation keys; whole-scene visibility keys are separate timeline controls.
- All four MAMMA camera matrices and render aspects were checked. Pinhole
  projection differences were below 0.0046 px in the tested points. A separate
  unequal-focal-length/off-center fixture was below 0.00015 px. These verify
  serialization/projection, not absolute camera-estimation accuracy.
- On this PC, the initial worker builds took about 7.4s ExMR, 5.1s FlowHMR
  and 6.6s for the MAMMA scene; sampled worker-tree peak RSS was about
  441/257/373 MiB respectively. These short-case observations are not general
  speed or memory guarantees. Output size grows with videos and PC2 frames.
- Checked existing-destination retention, failed-build cleanup, stale selection
  rejection and cancellation while a real Blender child was active. The child
  was reaped and owned staging directories removed; source project stamps were
  unchanged. GUI option scoping, default FBX options, MAMMA T-pose exclusion
  and compact dialog text layout passed. Existing FBX golden bytes, BVH
  original-frame suffix and camera/video sharing checks still passed.

### Export/camera release review (2026-11-02)

The final review additionally tested a delayed MAMMA take (visible on frames
5 through 84, hidden at 4 and 85), with a person named identically to a camera.
Camera-view scenes now keep direct camera-object references instead of looking
them up by potentially colliding Blender names. Half-open visibility and PC2
offsets passed; sampled surface error was zero and the reference-skeleton head
error stayed below 0.0039 mm. Export controls/help and every offline model guide
were updated to FBX/BVH/Blender terminology.

The 120-frame decoded-playback regression found no mixed frame/camera pairs;
observed intervals were median 31.5 ms, p95 47.0 ms and maximum 49.9 ms. Sixty
loader open/close cycles settled at about 192 MiB RSS after initialization,
with 467 to 468 handles observed. These bounded tests are not a long-duration
leak guarantee or a universal frame-rate claim. The known-rotation background
fixture, original PR equivalence, off restoration, cache invalidation and
worker cancellation checks also passed. Runtime locks, installer scripts,
AI worker and source-correction implementation stayed unchanged.

### Optional PromptHMR moving-camera support (2026-11-02)

- Built the pinned LieTorch/DroidCalib Windows CUDA extensions over the existing
  PromptHMR Python 3.11 / Torch 2.7.1 CUDA 12.8 runtime. Real LieTorch
  forward/backward, scatter and DroidCalib CUDA operations passed on RTX 5090
  (compute capability 12.0). Setup also retained the verified extension on retry.
  Base runtime pins and installed packages did not change. The installed
  extension measured 19,547,213 bytes; its two managed weights total
  1,664,022,188 bytes. Compiler/download/build caches are excluded.
- The 80-frame, 30 fps, single-person moving-camera test completed all five
  normal worker stages, including fresh tracking and correction/export
  preparation, in 130.3 seconds of wall time. Phase times were 1.4s preparation,
  33.2s tracking, 57.2s camera/keypoints/motion, 31.0s world optimization and 7.4s
  publication/correction. Peak allocated/reserved Torch VRAM was 6.73/6.93 GiB;
  sampled worker-tree RSS peaked at 7.42 GiB. These are one-PC short-clip
  observations, not minimum memory or long-clip guarantees.
- The complete 80-frame camera trajectory loaded both with and without a shared
  scene context. Published incam translations and rotation matrices matched
  direct per-frame world-to-camera conversion within 1.2e-7 m and 1.9e-7
  respectively, including the registered shape-dependent pelvis pivot.
  Original and corrected sources remained exportable. Blender 4.5 reopened
  the portable bundle with relative synchronized video and 52 body/finger bones;
  sampled camera matrix errors stayed below 1.2e-7. Armature objects had no
  animation keys. These are serialization/coordinate checks, not ground-truth
  camera-accuracy measurements.
- A separate eight-frame, two-person fixed-camera regression reran image/video
  inference and world fitting before/after. Body/root camera-space motion stayed
  exact; independent GPU runs differed by at most 0.000123 px in saved 2D
  keypoints and about 3.0 mm in native final world translation. Replaying
  identical camera-motion intermediates produced exact world outputs across
  121 compared arrays in the overall regression. No new physical correction
  or precision reduction was introduced. An older fixed-camera inference
  receipt was reused without FFmpeg or inference; moving identities retain
  their actual camera code/module/model fingerprints.
- App controls restored Moving Camera, loaded its 80-frame saved track and
  showed 16/16 ready models. Switching to Fixed Camera showed 14/14 and a
  reconstruction notice without processing or replacing the current motion.
  Legacy options restored Fixed Camera. Corrupted extension files, stale or
  escaped camera paths and a simulated failed repair were rejected; prior
  registrations and the base runtime remained intact.

No clean Windows installation, other-GPU verification, long-video stability or
absolute SLAM scale/accuracy claim follows from these checks. Tracking loss,
zoom/cuts, poor texture and scale drift remain possible.

## FlowHMR query and hand-sampling integration (2026-11-03)

Validation used the installed Windows runtime on the test PC, with private
projects outside the source checkout. No user project, video or model weight
was published or replaced. This initial sampling/query check preceded the
optional HaMeR integration described below.

- A 327-frame app-backend comparison completed with sparse hands on and off,
  including camera receipt handling, SAM features, query-batched generation and
  native world placement. Every stage reused its verified cache on the second
  call. Changing only hand sampling invalidated feature identity while leaving
  the camera identity unchanged.
- A 117-case CPU FP32 attention check covered local/global masks, padding and
  fully masked rows. Maximum difference from original math attention was
  4.77e-7. Actual app features generated a 327-frame motion with a maximum
  source-joint difference of 0.00034 mm against original attention.
- Sampling/interpolation checks covered 98 frame-length/batch combinations,
  including one frame, final endpoints, maximum context and simulated batch
  allocation backoff. Body slots and sampled anchors were preserved. Actual
  SAM calls checked CPU/GPU crops, batches without a hand sample and body-pelvis
  placement before/after hand fusion; placement differences stayed below
  0.0004 mm. Every-frame output was within repeated baseline GPU variation,
  rather than bitwise identical across separate feature calls.
- App controls defaulted on, restored saved off values and supplied the on
  default for missing legacy settings. Switching tools/toggles did not start
  inference. The three controls and explanations were inspected using the
  application's theme in an offscreen Qt widget render.
- A separate nine-frame app project completed CPU hand crops with one-frame
  batches, explicit Low Memory block offloading, query generation, native
  correction, preview generation and FBX saving. Cancelling feature extraction
  left the previous tokens intact and marked the stage pending; a retry committed
  a complete generation and removed the marker. This was a controlled
  cancellation/backoff check, not a physically exhausted-GPU test.

Timing and quality differences are in [performance measurements](PERFORMANCE.md#flowhmr-query-batches-and-four-frame-hands-2026-11-03).
These checks do not establish other-GPU performance, universal motion accuracy
or absence of long-session resource leaks.

## Optional FlowHMR HaMeR integration (2026-11-03)

Private nine- and 327-frame app-backend projects completed per-frame SAM hand
box capture, four-frame SAM conditioning, query motion generation and the
independent ExMR-runtime HaMeR pass. The direct HaMeR finger output uses ExMR
crops/rotation smoothing, not its learned temporal HandMR network. Body/root,
wrist rotations, shape and native foot contacts were bit-exact to the same
FlowHMR prediction before hand replacement. The original FlowHMR tensor remained
separate. A nine-frame run completed correction, preview and FBX saving. Turning
HaMeR off and reconstructing restored the original SAM-conditioned prediction.
The nine-frame paired contact-correction check also retained identical corrected
body/wrist/root/shape/contact tensors; only fingers differed in that test.

- A nine-frame Low Memory run exercised SAM's one-frame CPU-crop/offload path
  and HaMeR's one-hand-crop CPU-first loader. Both finished and reused verified
  feature/motion caches on repeat calls. A separate 327-frame Automatic run
  used body batches of 16 and hand batches of eight. HaMeR child resource/time
  measurements are recorded in [performance](PERFORMANCE.md#optional-flowhmr-hamer-fingers-2026-11-03).
- Controlled OOM splits retained exactly one ordered hand-box row per source
  frame. Invalid-hand fallback retained original fingers exactly. Cancelling
  the actual child before model loading preserved the previous completed hand
  cache, left no partial output and reaped the process. A malformed hand cache
  entered recovery. These do not establish physically exhausted-GPU behavior
  or long-running child-process leak freedom.
- Themed offscreen Qt checks verified default off, missing legacy flag defaults,
  saved on/off restoration, automatic Fast SAM/four-frame activation and locking,
  additional Model Manager rows and no inference on toggles. Draft checkbox
  changes retained the committed source until reconstruction.

No new runtime package, model weight or test video was distributed. Other GPUs,
clean installation and universal accuracy/stability were not established.

## Right-sidebar wheel routing (2026-11-03)

A native Windows Qt/OpenGL test (150% scaling, 773×476 logical 3D viewport,
no loaded actor or playback) sent 72 wheel events per target. Previously,
hovering the preset combo changed its value 23 times and the numeric pelvis
field 72 times, consuming the wheel without scrolling. These triggered the same
number of model-readiness checks and 3D paints. After routing to the panel, both
targets scrolled with no value changes, readiness checks or 3D paints. Measured
event p95 changed 7.42→1.72 ms for the combo and 7.94→1.03 ms for the number
field; maxima changed 8.97→3.85 ms and 9.34→1.32 ms. These are synthetic wheel
event timings, not overall FPS or a heavy-scene playback benchmark.

Additional Qt checks preserved pixel-delta behavior, open popups, focused
Ctrl+wheel, text/keyboard edits, frame/camera and idle processing state. Six
offscreen create/close cycles retained 419 handles and 34 threads; the final
repeat's RSS ranged 81.56–84.43 MiB. This short trend does not establish long-session leak freedom.
Inference, correction, preview math and installed packages were unchanged.

## MAMMA direct SAM input (2026-11-03)

The app now reads SAM 2.1 frames through the prepared-video reader instead of
saving and rereading intermediate lossless RGB PNGs. The adapter creates no
empty frame-index files. PIL resize, float64 division into FP32 storage,
normalization, CPU video/state offload, prompts, matching, propagation and final
mask cleanup are unchanged. Final person-mask PNGs are still saved normally.
The helper is included in inference code identities. Existing committed results
are not rewritten on open; explicit reconstruction uses a new generation.

Independent four-camera, two-person 1080p tests used the same video prefix at
210/426 frames (7/14.2 seconds, 30 fps). All 5,088 paired person-mask images
were pixel-identical, with no memory-protection stop or OOM. Mask-stage time
fell 16.9%/21.2%; peak RAM differences were +23/−3 MiB, with the same allocated
and reserved CUDA peaks within each pair. These were fresh-process, single-run
measurements on RTX 5090 / 96 GB RAM, not minute-scale or other-GPU guarantees.
See [performance](PERFORMANCE.md#mamma-direct-sam-video-input-2026-11-03).

The integrated app path was then exercised on a separate dynamic 90-frame,
four-view/two-person 1080p segment through masks, FP32 landmarks and 3D fitting.
Against the earlier PNG control with identical prepared inputs and calibration,
all 720 person-mask images, landmark/contact arrays, final meshes and joints
were bit-identical. The new mask worker took 91.48 s with 4.731 GiB maximum RAM
working set and 2.536 GiB allocated / 2.861 GiB reserved CUDA memory. This is
the mask stage only; fresh camera estimation, app correction, GUI playback and
exports were not rerun for this input-loading change.

Controlled checks covered sequential reads, the same normalized tensor as PNG,
Unicode path registration, empty-source rejection, unregistered/image-source
fallback, and restoration on success, decode/OOM/cancellation exceptions and
repeated app-wrapper failures. The adapter owns no additional readers, queues
or threads. The updated Model Guide rendered in offscreen Qt. The installed
MAMMA engine's 368 checked source files were unchanged. Tests/inputs remained
outside the public checkout; no runtime package or model weight was added.
