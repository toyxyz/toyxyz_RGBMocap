# PEAR: live pose preview

**Experimental support. Tracking and pose estimates may be unstable.**

[Original PEAR project](https://github.com/Pixel-Talk/PEAR) · [Official checkpoint](https://huggingface.co/BestWJH/PEAR_models)

[Paper and BibTeX](../CITATIONS.md#pear) · [Upstream source license](https://github.com/Pixel-Talk/PEAR/blob/8e915f8c3dfae4063d66763e6296b2b0b02eb303/LICENSE.txt)

PEAR tracks webcam or video input in real time and displays fixed-length body and
finger bones for one visible person. For video input, seek or play the timeline
to track its current frames. Multiple fixed views of the same
person can be aligned and fused; single-view preview remains available. Body mesh and face geometry are
not calculated for each live frame. It is a live preview tool. It does not use
Find person, record motion, run the offline Motion Correction pipeline, retarget characters or export
animation in this integration. Use an offline reconstruction model for those tasks.

Selecting PEAR only shows its controls. Click **Start Live Tracking** to load and
warm the models, attempt Camera Align for one or more cameras, then track the selected webcam or video. Show the person and visible floor during alignment. If alignment fails or its support/models are missing, tracking continues in the selected camera view with a visible warning. Unaligned views are not fused, and aligned Foot Lock is unavailable. Camera Align can be retried later. The button becomes
**Stop**, including during loading. Stop ends tracking and releases the worker,
models and camera; wait for shutdown before starting again.
Selecting a webcam opens a lightweight image preview before Start, without PEAR
or Torch loading. After Stop, reselect a camera or click Refresh to reopen image
preview; selecting None for all slots releases the cameras.

While tracking is started, opening another video or switching webcams replaces
the input and automatically recalibrates the changed camera configuration. Changing physical cameras requires calibration; ordinary seeks and thumbnail changes reuse the loaded tracking model.
Camera Align temporarily releases PEAR while its separate calibration worker runs,
then reloads tracking to avoid overlapping their GPU allocations.
The separate camera-capture worker remains connected throughout Start, model
loading and Camera Align. It supplies bounded raw frames to tracking through
Windows shared memory, so those operations do not reopen the webcams. Image
preview continues while pose processing is busy; ordinary tracking displays the
completed image/pose pair. Stop, None, leaving PEAR or closing the app release
capture. Changing device assignment or requested resolution can reopen capture.

## Setup

1. Close RGB MoCap and run `install.cmd` for the base app. It also downloads pinned
   PEAR source, RoMa and the required
   pure-Python PyTorch3D utilities. It checks CUDA operations and reuses the existing
   shared Python/Torch runtime without upgrading its packages. No CUDA Toolkit,
   C++ compiler, separate Torch installation or native PyTorch3D build is needed.
2. Open `RGBMoCap.cmd`, select **PEAR · Live (Experimental)**, and open **Model Manager**.
3. Download/import `pear_model.pt` and **PEAR Person Detection / YOLOv8s**
   (`yolov8s.pt`). Model Manager verifies the supported official files.
4. Prepare the VGGT and GeoCalib rows for automatic camera alignment. VGGT reuses the FlowHMR/MAMMA model file.
5. In the geometry row, use **Manual Download**. Follow the upstream Preparation
   instructions, review the SMPL-X/FLAME provider terms, and obtain the original
   `assets.zip`. Import that ZIP without editing it. Import validates its SHA-256
   and extracts only required files. The source ZIP is preserved.

PEAR requires **SMPL-X 2020 and FLAME 2020** geometry. The standard SMPL-X v1.1
used by other models and MAMMA's locked-head model cannot replace these files.
Registered geometry is not included in the repository or downloaded automatically
by setup. The PEAR source license does not replace body-data or weight licenses.

The checkpoint is approximately **2.50 GiB**. Selected geometry occupies about
**0.32 GiB** logically (about **0.27 GiB** when duplicate files can be hardlinked).
The original PEAR source/utilities were about **0.87 MiB** on the test PC.
Camera alignment support adds about **11.35 MiB** of GeoCalib/Kornia files, over
the existing Flow runtime, without upgrading shared packages. GeoCalib weights
are about **110.7 MiB**; VGGT reuses the existing FlowHMR/MAMMA weight file (about **4.26 GiB** if not already present). Both are required for alignment, including one camera; unaligned tracking can run without them. The original
geometry ZIP download is about **1.48 GiB** and needs separate temporary space.
PEAR also uses the official YOLOv8s detector (about 21.5 MiB), downloaded or
imported through its Model Manager. The larger YOLOv8x used by offline models
is unchanged. These are additional to the base installation, not a new runtime.

## Video and webcam controls

- **File > Open Video**: choose one or multiple synchronized camera videos.
  With multiple files, the timeline uses their common duration at 30 fps; a trim
  applies to all views. Thumbnails use up to four columns, adding rows as needed.
  Click a thumbnail to change only the main camera preview. Leave **Webcam** at **None — use video**.
  Click **Start Live Tracking**, then seek or play to process the requested frame. Requests are
  coalesced to the latest frame, so scrubbing does not create an inference queue.
  During playback, the completed video frame and its pose are displayed together.
  The header shows the displayed frame and delay; the timeline remains the playback
  request clock. Slower inference skips frames rather than hiding every late pose.
  Pause or seek for the exact selected frame. Live video images are bounded to
  1280 × 720 for transfer. Rotation is applied before inference and this transfer
  resize; the original video file is unchanged.
- **Webcams**, below the video: **Count** and Refresh sit above one table with
  **Camera / Device / Rotation** columns. All slots, including camera 1, appear
  here in numerical order; longer lists scroll vertically. **Count** sets 1–32
  assignment slots. Assign a different
  connected device to every slot. Set all slots to None to return to video inputs.
  Thumbnails select the displayed webcam. Assigned devices show image previews
  even while other slots are still None; fill every slot before starting tracking.
  **Start Live Tracking** leaves the capture worker connected; the model reads
  its shared frames instead of opening the devices again. During tracking the
  displayed image and pose belong to the
  same processed frame. Old camera frames are dropped to bound latency.
- **Refresh**: rescan connected cameras. Close other apps using the device if it
  cannot open. Choose **None** to release it and return to the video.
- **Rotation**, beside each webcam selector on the left: clockwise degrees for
  that device only, applied before display, detection and camera alignment.
  Each selected webcam retains its own value when devices are refreshed or moved
  between slots within the session. For videos, select a thumbnail and edit the
  first Rotation field; each video retains its own value. `90` turns right;
  `-90` turns left. Arbitrary angles expand the image to avoid clipping.
  The source file is unchanged. Changing a rotation invalidates Camera Align.
- **One Euro Filter**: applies to both webcam and video. Adaptive filtering stabilizes
  quiet poses, follows moving poses faster, and caps its response to noisy speed
  estimates. Rotations remain valid rotations; the fixed rig follows the smoothed
  pose. Root placement is also filtered. This intentionally changes the live
  pose and adds some motion-dependent lag; it is not contact correction.
  This lightweight One Euro path processes each requested frame once, using actual
  source time when frames are skipped. There is no extra inference, look-ahead or
  recording-length cache. A seek seeds the filter with that frame's prediction;
  subsequent playback is smoothed. The last matching image/pose pair stays visible
  while a replacement is computed. Turn smoothing off for unfiltered predicted
  rotations; fixed bone lengths remain. Switching it does not reload models or reframe the view.
- Under **One Euro Filter**, **Bone Rotation** smooths body/finger rotations and
  **World Position** smooths estimated root translation. Both default on and can
  be selected independently. The parent switch disables both while retaining
  their selections. Filter Strength is shared. Changing a component does not
  reload models or run Camera Align, and preserves the other filter history.
  This position-filter checkbox is separate from the top-level **World Position**
  movement switch. Foot Lock remains independently enabled/disabled.
- **Filter Strength**: 1–100, default 50 (the previous filter setting). Higher values
  reduce jitter but add delay. Applies to pose and root placement; filtered shape
  parameters are used only when bone lengths are calibrated.
  Adjusting it while playing retains filter history and the loaded model.
- **World Position**: on by default, retaining estimated pelvis movement. Off
  fixes all three position coordinates, including height, at the currently
  displayed pelvis position. If turned off before tracking, the first valid pose
  supplies that position. Pelvis rotation and joint motion continue normally.
  Switching it does not reload models or reset bone lengths; a paused frame may be processed again.
  Turning it back on restores estimated position; this may move the character.
- **Foot Lock**: on by default and active only after successful Camera Align. Low, slow feet can hold an estimated ground point. A small causal leg correction and bounded pelvis adjustment ease in and out; fixed bone lengths and ankle/toe world rotations are preserved. Turn it off for uncorrected live placement. With World Position off, pelvis translation stays fixed; unreachable feet are allowed to miss the floor instead of stretching legs or forcing the pose. Incorrect floor estimates, fast jitter and missed contacts can still cause sliding; this is not a full offline correction solver.
- **Start Live Tracking / Stop**: attempt alignment and begin tracking, or stop and
  release the models and webcam. Stop also cancels model loading. Selecting a new
  input or changing settings while stopped does not start pose tracking. Webcam
  selection, rotation/resolution changes or Refresh can reopen image preview.
- **Reset Bones**: recalculate bone lengths and neutral axes from the pose currently
  displayed. Video playback pauses on that frame. For a webcam, its displayed
  pose parameters are used rather than an unrelated newer camera frame. Automatic realignment temporarily releases and reloads the tracking model; the viewport camera is preserved. Choose a frame where the
  person is clearly visible. If the displayed pose has expired, try again; the
  existing rig is retained. After a successful reset, Camera Align runs again for the new body scale. Changing input calibrates a new rig automatically.
- **Bone Overlay** uses the same octahedra, joint caps and outlines as the other
  models. Its 52 body/finger bones form a fixed-length FK rig; jaw and eye control bones
  are hidden. This is a live display, not a committed offline export skeleton.
  Each bone follows its own inferred rotation with a constant neutral calibration.
  Lengths remain fixed while playing, seeking or changing filter strength. There
  is no animated bend-plane pole or live mesh-surface roll correction.
  Original inferred twisting can remain.
  Foot and toe roll use a fixed neutral foot-plane reference and each bone's own
  inferred rotation; they do not use the toe rotation to orient the incoming foot.
  Toe tips extend along each foot's neutral forward centreline, rather than toward
  its big-toe landmark. This constant display reference preserves toe length,
  neutral pitch and predicted foot/toe rotations.

The largest detected person is selected initially and followed while visible.
There is no manual person selection in this live mode. Occlusion, leaving the
image and multiple people can cause loss or reacquisition of identity.

## Multiple fixed cameras and Camera Align

1. Run `install.cmd` after updating; its pinned GeoCalib/Kornia support requires
   no extra Torch installation, TensorRT, CUDA compiler or separate PEAR setup CMD.
   GeoCalib code uses Apache-2.0 and its checkpoint uses CC BY 4.0; see
   [citation and licenses](../CITATIONS.md#geocalib-pear-camera-align).
2. In PEAR **Model Manager**, prepare **VGGT** and **GeoCalib**.
   Both are required for automatic alignment, including a single camera.
3. Open synchronized views of the same performance, or assign every webcam slot.
   PEAR still tracks **one person**, not multiple identities. Use views showing
   that same person and overlapping background. Video starts must already match;
   this feature does not automatically synchronize audio/video.
4. Put the playhead at a clearly visible moment with background floor, then click
   **Start Live Tracking**. It runs Camera Align automatically; use the separate
   **Camera Align** button to recalibrate while tracking. For videos it samples that time and up to two seconds later;
   leave at least two frames after the playhead. Webcam calibration samples two
   recent frame sets about a second apart. Keep cameras stationary.
5. Wait for preparation, VGGT, GeoCalib and tracking restart. **Cancel Camera Align**
   cancels the operation; **Stop** also cancels tracking and releases workers.
   Webcam preview remains live. Opened videos can play, pause and seek during
   model loading and alignment. Alignment samples use the playhead position at
   the time of the click, independently of later playback/seeks. Background
   alignment is still a fixed-camera assumption; playback does not recalibrate
   moving cameras. Rotation remains visible while ordinary video decoding runs.

Start attempts alignment first and continues with the selected camera if it fails. This fallback uses camera-relative placement, so floor tilt and world scale may be inaccurate. A single aligned view uses background gravity/depth and an inferred body-to-depth scale, not multi-view triangulation. Alignment does not assume that the person starts upright. Person depth confidence is compared within the person samples rather than against easier background surfaces; reprojection, minimum sample count and scale-consistency checks remain. No live pose is published from the sampling stage. After successful alignment, a causal rotation
blend, analytic visibility/foreshortening weights and reprojection-based root
placement combine available views on one fixed-length rig. These weights are
heuristics, not measured confidence. Alignment uses GeoCalib gravity and background floor geometry, with actor masks for floor/gravity. Multiview camera estimation masks the actor; single-view VGGT retains the person to estimate body-to-depth scale;
it does not require an upright initial pose. Scale and floor are estimates, not
calibrated measurements. Lightweight live Foot Lock can reduce planted-foot movement; it cannot repair incorrect calibration or guarantee exact sole contact.

Changing inputs, device assignment, input rotation, requested webcam resolution
invalidates alignment and automatically aligns again while tracking. Reset Bones also recalibrates before continuing. Moving physical cameras cannot be detected
reliably: click Camera Align again. A thumbnail change retains alignment. Current
alignment is transient and is not a saved offline project. World Position and
One Euro controls also apply to the fused pose. Camera alignment does not fix
wrong identity matches, bad crops or severe occlusion.

Webcam frames are paired by host arrival time with an 80 ms skew limit, using
bounded latest-frame buffers. This is approximate software synchronization, not
hardware synchronization or latency calibration. A delayed/disconnected device
produces a waiting/error notice; frames do not accumulate indefinitely. Four
views are the measured optimization case; more slots increase processing and
calibration cost and may not fit available memory.

## Placement and performance

### Alignment and setup failures

Alignment is optional for tracking. A failed alignment, missing alignment support
or missing VGGT/GeoCalib files produces a warning in the sidebar, Console and CMD,
then starts camera-relative tracking from the selected view. Other views remain
selectable, but their poses are not fused and aligned Foot Lock is inactive.
Use **Camera Align** to retry after improving the view or preparing its models.
The warning clears when a new alignment attempt starts. Ordinary pose tracking
still requires PEAR's own installed support, checkpoint, detector and geometry.

If the app reports that PEAR support needs installation, close it and rerun
`install.cmd`. A support-script update can require refreshing the verified support
generation even when model files are already ready. Existing verified model
files and projects are retained; do not edit setup receipts manually.

### Placement and measured speed

Pelvis movement is enabled by default through **World Position**. Turning it off
holds pelvis translation in place; it does not lock feet to the ground.
Camera Align estimates a static world frame; it is not measured camera calibration or camera-motion tracking. Foot Lock only acts near that estimated floor. Its contact evidence comes from pre-correction motion. Correction rotations and lock transitions are smoothed causally; no future frames or iterative physics are used. Inferred pose jitter can remain.

The single-view fast path uses SDPA, an FP16 encoder, FP32 decoder, encoder CUDA Graphs,
fixed-length CPU FK and YOLOv8s person detection. A reusable CPU body model
calibrates neutral anatomy on the first valid frame and on Reset Bones; body
skinning is not evaluated on each subsequent frame. It detects every processed
frame at 640 pixels.
Changing the detector can change its box and therefore the inferred pose; this
is a speed tradeoff, not a claim of identical motion or equal accuracy. Model
loading and initial warmup take longer than subsequent frames. Performance depends
on the GPU, camera driver, resolution and scene; live preview can skip frames.

On the test PC (RTX 5090 32 GB, Ryzen 9 9950X, 96 GB RAM), a five-second visible
video test of the current bone path presented 150 paired frames, about 30/s.
Worker processing was 21.43 ms median and 23.80 ms p95, with a 160.31 ms maximum;
peak PyTorch allocated GPU memory was about 1.50 GiB. Initial loading took about
34 seconds. A separate warmed paired test measured 19.03 ms with mesh computation
versus 13.84 ms with fixed-length FK (27% less processing time). These measure
different scopes; neither promises webcam throughput or minimum VRAM. See
[validation and limits](../VALIDATION.md#pear-fixed-length-bones-and-forward-toe-tips-2026-11-01).
An earlier full-mesh test used about 1.93 GiB. The integration
keeps video decoding, camera capture, calibration and inference off the GUI thread, with bounded
mailboxes and no automatic video/pose recording. Available RAM and CUDA allocation
failures are reported rather than repeatedly retrying a failing worker.
The offline Resource Mode selector does not apply to PEAR. Its live worker uses
the validated fast configuration. Available-memory stopping and CUDA allocation
limits follow the local [Resource Safety](../PERFORMANCE.md#resource-safety)
preference selected in an offline model; it has no block-offload or automatic Low Memory retry.

## Files and development

`runtime_pear/` contains pinned support source/packages; `.setup/pear-runtime.json`
is the setup receipt. `models/pear/` contains checkpoint and imported geometry.
Recent worker diagnostics are in `cache/pear/sessions/`; no motion archive is
created there. Camera Align temporarily writes sampled images/observations under
`cache/pear/alignment/`; successful jobs remove sampled images and retain small
camera/rig receipts. Failed or cancelled jobs are disposable caches; at most four recent job folders
are retained, including the current calibration. Close PEAR
before clearing these folders. Camera alignment support has its own
`.setup/pear-alignment.json` receipt and `installer/pear-alignment.json` source pins. Close the app before deleting support files. Rerun
`install.cmd` after support-source changes; `update.cmd` also prepares/verifies
this support. Matching verified files are reused. Preserve model files and projects.

`mocap/pear_ui.py` owns the transient GUI mode. `mocap/pear/worker.py` owns the model
and capture device. `session.py`/`wire.py` implement bounded non-pickle IPC.
`installer/pear_adapters.py` records the pinned upstream import adaptations. Keep
the base runtime read-only; do not upgrade NumPy, timm or Ultralytics globally.

Pinned source/weight revisions and hashes are recorded in `installer/pear-sources.json`
and `mocap/pear_models.py`; the geometry inventory is in `installer/pear-assets.json`.
The One Euro filter follows the [original method](https://gery.casiez.net/1euro/),
with adaptive rotation and placement filtering in the live worker. The optional
pelvis-position control also draws on the [PoseCap](https://github.com/CorridorTech/PoseCap)
workflow; RGB MoCap implements its own preview controls and fixed-position anchor.

## Live Foot Lock validation (2026-11-01)

The default-on live lock is separate from the offline Kimodo correction path.
It has two constant-size contact states, velocity/height hysteresis, inertialized
lock/release targets, continuous pelvis handoff and source-relative correction
filtering. Nearly straight legs retain the source bend instead of turning tiny
floor errors into crouching. It preserves lengths and ankle/toe world frames;
reach failures remain residuals, with a visible notice for larger errors.
One Euro filtering remains responsible for reducing inferred pose jitter.

On the test PC, a 450-frame four-view cached replay reduced mean ankle movement
between consecutive locked frames from 3.46 to 1.95 mm. The knee-angle second
frame difference p95 remained 4.02 degrees: the source itself still contained
jitter, and this is not a ground-truth accuracy measurement. A separate near-
extension perturbation test limited maximum knee movement to 0.17 mm/frame.
The lightweight solver used roughly 0.2–0.3 ms median and 0.3–0.4 ms p95.

A real four-video app test completed automatic alignment and presented 180
paired frames in an eight-second observation. Worker time was 39.56 ms median,
44.72 ms p95 and 692.92 ms maximum including the first post-alignment frame.
Tracking peak PyTorch allocation was 1.57 GiB and worker RSS about 2.32 GiB.
Calibration is a separate startup cost: the four-view VGGT stage used about
6.53 GiB peak allocation, and PEAR is released before it runs. These short tests
are not universal FPS, physical multi-webcam synchronization or leak guarantees.

The transition design uses standard causal foot-lock/analytic IK ideas discussed
in [Inverse Kinematics and Foot Locking](https://theorangeduck.com/page/inverse-kinematics-foot-locking).
RGB MoCap implements its own lightweight preview solver; no additional model,
physics simulation, future-frame window or TensorRT installation is used.
