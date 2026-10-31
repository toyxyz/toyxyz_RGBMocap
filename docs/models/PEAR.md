# PEAR: live pose preview

**Experimental support. Tracking and pose estimates may be unstable.**

[Original PEAR project](https://github.com/Pixel-Talk/PEAR) · [Official checkpoint](https://huggingface.co/BestWJH/PEAR_models)

[Paper and BibTeX](../CITATIONS.md#pear) · [Upstream source license](https://github.com/Pixel-Talk/PEAR/blob/8e915f8c3dfae4063d66763e6296b2b0b02eb303/LICENSE.txt)

PEAR displays fixed-length body and finger bones for one visible person, from a
webcam or the current frame of an opened video. Body mesh and face geometry are
not calculated for each live frame. It is a live preview tool. It does not use
Find person, record motion, run contact correction, retarget characters or export
animation in this integration. Use an offline reconstruction model for those tasks.

Selecting PEAR only shows its controls. Click **Start Live Tracking** to load and
warm the models, then track the selected webcam or video. The button becomes
**Stop**, including during loading. Stop ends tracking and releases the worker,
models and camera; wait for shutdown before starting again.

While tracking is started, opening another video or switching webcams replaces
only the input. Models remain loaded until Stop, leaving PEAR or app close.

## Setup

1. Close RGB MoCap and run `install.cmd` for the base app. It also downloads pinned
   PEAR source, RoMa and the required
   pure-Python PyTorch3D utilities. It checks CUDA operations and reuses the existing
   shared Python/Torch runtime without upgrading its packages. No CUDA Toolkit,
   C++ compiler, separate Torch installation or native PyTorch3D build is needed.
2. Open `RGBMoCap.cmd`, select **PEAR · Live (Experimental)**, and open **Model Manager**.
3. Download/import `pear_model.pt` and **PEAR Person Detection / YOLOv8s**
   (`yolov8s.pt`). Model Manager verifies the supported official files.
4. In the geometry row, use **Manual Download**. Follow the upstream Preparation
   instructions, review the SMPL-X/FLAME provider terms, and obtain the original
   `assets.zip`. Import that ZIP without editing it. Import validates its SHA-256
   and extracts only required files. The source ZIP is preserved.

PEAR requires **SMPL-X 2020 and FLAME 2020** geometry. The standard SMPL-X v1.1
used by other models and MAMMA's locked-head model cannot replace these files.
Registered geometry is not included in the repository or downloaded automatically
by setup. The PEAR source license does not replace body-data or weight licenses.

The checkpoint is approximately **2.50 GiB**. Selected geometry occupies about
**0.32 GiB** logically (about **0.27 GiB** when duplicate files can be hardlinked).
The installed support source/utilities were about **0.87 MiB** on the test PC. The original
geometry ZIP download is about **1.48 GiB** and needs separate temporary space.
PEAR also uses the official YOLOv8s detector (about 21.5 MiB), downloaded or
imported through its Model Manager. The larger YOLOv8x used by offline models
is unchanged. These are additional to the base installation, not a new runtime.

## Video and webcam controls

- **File > Open Video**: choose a video. Leave **Webcam** at **None — use video**.
  Click **Start Live Tracking**, then seek or play to process the requested frame. Requests are
  coalesced to the latest frame, so scrubbing does not create an inference queue.
  During playback, the completed video frame and its pose are displayed together.
  The header shows the displayed frame and delay; the timeline remains the playback
  request clock. Slower inference skips frames rather than hiding every late pose.
  Pause or seek for the exact selected frame. Live video images are bounded to
  1280 × 720 for transfer. Rotation is applied before inference and this transfer
  resize; the original video file is unchanged.
- **Webcam**, above the video: select a connected device. Camera capture and PEAR
  tracking begin after **Start Live Tracking**. The displayed image and pose belong to the
  same processed frame. Old camera frames are dropped to bound latency.
- **Refresh**: rescan connected cameras. Close other apps using the device if it
  cannot open. Choose **None** to release it and return to the video.
- **Rotation**: clockwise degrees applied to video or webcam images before display
  and inference. `90` turns right; `-90` turns left. Arbitrary angles expand the
  image to avoid clipping. The source file is unchanged.
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
- **Filter Strength**: 1–100, default 50 (the previous filter setting). Higher values
  reduce jitter but add delay. Applies to pose and root placement; filtered shape
  parameters are used only when bone lengths are calibrated.
  Adjusting it while playing retains filter history and the loaded model.
- **World Position**: on by default, retaining estimated pelvis movement. Off
  fixes all three position coordinates, including height, at the currently
  displayed pelvis position. If turned off before tracking, the first valid pose
  supplies that position. Pelvis rotation and joint motion continue normally.
  Switching it does not reload models, rerun inference or reset bone lengths.
  Turning it back on restores estimated position; this may move the character.
- **Start Live Tracking / Stop**: load the models and begin tracking, or stop and
  release the models and webcam. Stop also cancels model loading. Selecting a new
  input or changing settings while stopped does not start tracking.
- **Reset Bones**: recalculate bone lengths and neutral axes from the pose currently
  displayed. Video playback pauses on that frame. For a webcam, its displayed
  pose parameters are used rather than an unrelated newer camera frame. Models
  remain loaded and the viewport camera is preserved. Choose a frame where the
  person is clearly visible. If the displayed pose has expired, try again; the
  existing rig is retained. Changing input calibrates a new rig automatically.
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

## Placement and performance

Pelvis movement is enabled by default through **World Position**. Turning it off
holds pelvis translation in place; it does not lock feet to the ground.
Placement uses the crop camera with an assumed fixed
full-image camera and one initial reference height. It is **not calibrated world
motion**, camera-motion tracking or foot locking. Feet may float or slide.

The fast path uses SDPA, an FP16 encoder, FP32 decoder, encoder CUDA Graphs,
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
the validated fast configuration with available-memory monitoring and CUDA
allocation limits; it has no block-offload or automatic Low Memory retry.

## Files and development

`runtime_pear/` contains pinned support source/packages; `.setup/pear-runtime.json`
is the setup receipt. `models/pear/` contains checkpoint and imported geometry.
Recent worker diagnostics are in `cache/pear/sessions/`; no captured video or motion
archive is created there. Close the app before deleting support files. Rerun
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
