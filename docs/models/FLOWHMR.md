# FlowHMR

[Original project](https://github.com/flowhmr/flowhmr) · [All model guides](README.md)

FlowHMR reconstructs SMPL-H body and finger motion from one video. In this
project's subjective tests it was slower than ExMR but gave the best overall
motion quality. This is not a universal accuracy ranking.

## Prepare files

Run the normal `install.cmd`, then select **FlowHMR > Model Manager**.
FlowHMR does not require `Setup_MAMMA.cmd`, `Setup_PromptHMR.cmd` or their
C++/CUDA native-build prerequisites. Key assets are:

| Asset | Location under the app's `models/` folder |
| --- | --- |
| Supported FlowHMR latest checkpoint and configuration | `flowhmr/flowhmr_latest.ckpt`, `flowhmr/config.yml` |
| Official checkpoint metadata (public automatic download) | `flowhmr/config.json` |
| SAM 3D Body model, configuration and MHR | `flowhmr/sam/model.ckpt`, `flowhmr/sam/model_config.yaml`, `flowhmr/sam/assets/mhr_model.pt` |
| AMASS SMPL-H neutral body, 16 shape components | `body_models/smplh/neutral/model.npz` |
| VGGT-Omega 512, for camera estimation | `flowhmr/vggt_omega_1b_512.pt` |

Model Manager also lists shared tracking dependencies. On the MANO/SMPL-H
provider's site, follow the manager's instructions for the **Extended SMPL+H**
model. A SMPL-X file cannot replace it. Gated models require the provider's
registration/access steps; use **Manual Download... > Import...** for the exact
file. The current checkpoint is `flowhmr_latest`, not an arbitrary base model.
See [model preparation](../USER_GUIDE.md#prepare-models).

Downloading the FlowHMR checkpoint also fetches the official 25-byte
`config.json` from the same pinned Hugging Face revision, with SHA-256 checking.
Existing installations can obtain it using its separate Model Manager row.
It identifies the checkpoint for Hugging Face download accounting; it does not
replace `config.yml`, change inference settings or invalidate existing motion.

## Controls

| Control | Default for new work | Meaning |
| --- | --- | --- |
| **Fast SAM** | On | Speed up feature extraction using the same supported weights. Small numerical differences can affect motion or automatic contact timings. |
| **Fast SAM Acceleration > Adaptive Batching** | On | Automatic selects 1, 2, 4, 8 or 16 frames from current GPU allocation headroom, free RAM and image size. Short tests of 4 to 16 frames reduced warm feature time by 16–23%; allocated VRAM rose from 3.84 to 5.43 GiB. Off and Low Memory use one frame. Small numerical differences are possible. |
| **Fast SAM Acceleration > GPU Hand Crops** | On | Prepare hand crops on the GPU. Tests on top of 4-frame batching measured about 8–11% less feature time, with unchanged allocated VRAM at that batch size. Image interpolation differs from CPU crops, so the resulting motion can differ slightly. |
| **Fast SAM Acceleration > 4-Frame Hand Sampling** | On | Infer hand features every four frames and linearly interpolate the intermediate features. Reduces processing time and VRAM, but finger accuracy may decrease. Body/wrist motion and automatic contacts can also change. Off infers hands every frame. |
| **HaMeR Finger Refinement** | Off | Improve finger tracking accuracy with optional every-frame HaMeR inference, ExMR hand crops and Gaussian rotation smoothing. Accuracy varies by video. Enables and locks Fast SAM and 4-Frame Hand Sampling while selected. Slower; jitter and inaccurate fingers remain possible. Requires the shared HaMeR and MANO files. |
| **Camera / Sampling** | Moving / Unknown - Auto | Automatically choose camera-estimation sampling when the camera moves or its behavior is uncertain. |
| **Camera / Sampling > Static Camera** | Optional | Skip camera estimation for a fixed camera without zooming. This path does not require the VGGT weight. |
| **Camera / Sampling > Every ... Frames** | Optional | Analyze the camera every 1, 2, 5, 10 or 30 frames. Denser camera sampling uses more memory. It does not reduce motion frame count. |
| **Resource Mode** | Automatic | Use the fast GPU path when memory is sufficient; otherwise offload SAM/VGGT transformer blocks to CPU. Low Memory forces offloading. This can be slower and use more RAM; precision and motion frame count stay unchanged. |

These options apply on **Reconstruct 3D**. Fast SAM is an extraction option,
not MAMMA's SAM 2.1 masks, PromptHMR's SAM2 tiny tracker or the standalone
SAM 3.1 experiments. Those checkpoints are not interchangeable.

The three acceleration switches work independently while HaMeR is off and are saved with the project.
They are used only while **Fast SAM** is on. They do not lower model precision,
reduce image resolution or skip motion frames. In the short validation clips,
GPU crops changed final joint positions by about 1 mm at the 95th percentile,
with differences up to 5 mm; this measures disagreement with CPU crops, not
accuracy against ground truth. Speed figures describe feature extraction after
warm-up, not the whole reconstruction, and vary with the PC and video. CUDA
memory pressure can retry with Low Memory and a smaller batch.

4-Frame Hand Sampling keeps every body and output motion frame. Its interval is
independent of Adaptive Batching: it also works with one-frame Low Memory batches
and either CPU or GPU hand crops. The final frame is sampled as an endpoint.
Feature interpolation uses up to three future frames; this is an offline option.
Saved off values are retained. New work and projects without this setting default
on for their next reconstruction; opening a cached result does not rerun inference.

### Query-batched motion attention and sparse hands

The motion generator uses **256-query attention batches for all clip lengths**,
retaining FP32, the original attention visibility, RoPE and full temporal context.
This is not a shorter temporal window. No extra package, TensorRT or native build
is required. On the test PC, a real 4,806-frame sequence's warm generator time
changed **25.45 → 7.97 s** and peak allocated VRAM **17.41 → 3.06 GiB**.
Worker RAM RSS increased **4.37 → 5.04 GiB**. For that long sequence, time fell
**68.7% (3.19× throughput)** and allocated VRAM fell **82.4%**. Short clips can
be slightly slower; these are warm generator-only measurements, not whole-job
timings or requirements. See the [before/after table and motion differences](../PERFORMANCE.md#flowhmr-query-batches-and-four-frame-hands-2026-11-03).

Independent 60/120/240-frame tests of four-frame hand sampling reduced warm SAM
feature time by **41–49%**, with allocated VRAM **5.43 → 4.21 GiB** at body batch
16. A separate 327-frame upper-body dance gave **20.55 → 11.33 s** for features;
feature-plus-generator stage sums were **21.38 → 12.11 s**. These sums exclude
initial loading, decoding, tracking, camera estimation, correction and export.
That clip's body/wrist joint disagreement averaged **22 mm**, with a maximum
**84 mm**. There is no ground-truth accuracy claim; reduced hand conditioning
can affect more than fingers. Disable the option when preserving per-frame hand
evidence matters more than speed/memory. HaMeR is an optional separate pass below.

### Optional HaMeR fingers

**HaMeR Finger Refinement** aims to improve finger tracking accuracy with a
separate hand model. Results vary by video; jitter or inaccurate fingers can
remain. It defaults off, including projects without a saved
`hamer` option. Enabling it automatically turns on Fast SAM and four-frame SAM
hand sampling; those two switches stay checked while HaMeR is on. HaMeR itself
still infers hands **every frame**. It uses hand boxes from each frame's SAM body
decoder, not interpolated boxes or another detector.

Prepare **Hand Features / HaMeR**, **HaMeR mean parameters** and **MANO Right** in
FlowHMR's Model Manager after enabling the option. These files are shared with
ExMR/HandFlow; existing verified files are reused. The normal `install.cmd`
already installs the independent ExMR runtime used for this pass. No separate
HaMeR setup or new runtime is added. MANO still requires its provider's access
and license steps; do not substitute or redistribute registered body data.

The pass uses ExMR's 2× hand crops, left-hand mirroring, FP16 network execution
and the same Gaussian rotation filter (sigma 1 frame, up to four future frames).
It does **not** run ExMR's learned temporal HandMR/body network, so it does not
promise ExMR-level quality. Only the 30 local finger rotations are replaced;
FlowHMR's body, wrist rotations, trajectory, shape and native foot contacts are
retained. Invalid or out-of-view hand crops retain FlowHMR's fingers. Filtering
does not bridge invalid-crop gaps. The untouched FlowHMR prediction is preserved
separately from the optional hand result.

This adds model loading and another full video read, and can be substantially
slower and require more VRAM than FlowHMR without HaMeR. Automatic runs at most eight hand crops per batch; Low Memory uses one
and CPU-first loading. FlowHMR's GPU weights are released before the separate
HaMeR worker starts. Console/CMD report the pass, hand batch, valid hand counts,
time and measured worker peaks. Existing OOM recovery and Resource Safety apply.
Use **Reconstruct 3D** after changing the option. Toggling alone does not replace
an already committed preview; disabling and reconstructing restores SAM hands.

Console and CMD report query size, hand interval, sampled frame count and the
selected spatial batch. Changing hand sampling requires **Reconstruct 3D** and
invalidates the affected feature/motion caches; it does not replace the currently
displayed cached result merely by toggling the checkbox.

Adaptive Batching preserves the previous batching checkbox's saved on/off value.
Memory is checked again between prepared batches. CUDA allocation failures retry
only unpublished frames with a smaller batch after releasing the failed call;
if one frame still fails, the existing Automatic-to-Low-Memory retry applies.
RAM/prefetch limits, full resolution, original precision, every motion frame and
the first 30 placement-anchor frames are retained. The 16-frame ceiling favors
measured efficiency: 32 added only about 4% over 16 on the longer short clip but
used another 2.13 GiB of allocated VRAM. These are one-PC observations.

## Capture and export

1. Select FlowHMR and **File > Open Video**, then trim the **Video** track.
2. Use **Find person** if you need explicit selection or several people. Check
   processing targets; select a row to inspect its timeline and contacts.
   Choose **Default** or use [SAM 3.1](SAM31.md) on Windows for text prompts
   and click selection.
3. Prepare the required files and choose the camera option, then **Reconstruct 3D**.
4. Review contact correction, edit timings and **Apply Correction** if needed.
   Original motion is retained; applying contact changes does not rerun FlowHMR.
5. Use **File > Export...** (FBX, BVH or Blender) for the displayed applied motion, or
   [retarget an imported character](../USER_GUIDE.md#character-retargeting).

The working clip is 30 fps; the source video is preserved. The app supports up
to **5,000 frames per take** (about 166.7 seconds at 30 fps), but available memory
may require a shorter take. Results are kept under the project's `flowhmr/`
folder. **Save Project As** copies saved work; switching models keeps their
results separate.

## Limitations and resources

Single-person reconstruction connects feature extraction and motion generation
in one worker, releasing the feature model before loading the motion model.
Multiple people retain stage-scoped model reuse. Automatic may prepare up to
two CPU image batches ahead when memory is available; Low Memory disables this
overlap. This does not change the Fast SAM switches, resolution or precision.

### Measured effects

The 2026-11-02 paired checks used 60- and 120-frame 1080p landscape/portrait
clips on the test PC below. Warm feature-only time changed **4.71 → 3.98 s**
and **9.86 → 7.63 s** for 4 → 16 frames; peak allocated VRAM was
**3.84 → 5.43 GiB**, reserved VRAM **4.34 → 6.81 GiB**, and warm worker RAM
increased by about **0.20 GiB**. Fresh app feature jobs, including initialization,
took **34.80 → 34.88 s** and **39.77 → 38.74 s**: short whole jobs benefit much
less than warm extraction. Final joint disagreements were **0.15 / 0.44 mm at
the 95th percentile**, with maxima **0.59 / 1.63 mm**. These are differences
from the previous implementation, not ground-truth accuracy. Low Memory's
peak allocated VRAM remained **1.91 GiB** in its paired 60-frame check.

The newer official Lite SAM and OpenCV-style GPU body/hand crops were also
tested as separate candidates. The crop trial changed body and hand crops together;
body-only cropping was not isolated. Both candidates were slower than the existing
optimized Fast SAM path on this PC, so they are not substituted into the app. No new package,
model, TensorRT engine or compilation step is required. See
[the adaptive-batch review](../PERFORMANCE.md#flowhmr-adaptive-batches-2026-11-02).

Automatic's VGGT camera-only loader skips temporary initial weights using meta
initialization and direct checkpoint assignment. A paired camera-component test
reduced loading from **7.24 to 2.34 s**, with identical camera outputs and peak
allocated VRAM; peak RAM working set fell **8.88 to 5.40 GiB**. Low Memory and
Automatic fallback retain the previous loader, avoiding the experimental increase
in Windows private allocation. This does not change SAM or motion inference.
See [loading measurements and scope](../PERFORMANCE.md#vggt-camera-checkpoint-loading).

Measured on Windows with RTX 5090 32 GB, Ryzen 9 9950X and 96 GB RAM.
The rows use different stages/baselines; do not combine their speedups.

| Comparison | Measured change |
| --- | --- |
| Automatic → Low Memory, 60 frames including cameras/correction/preview preparation | Allocated VRAM **6.10 → 2.77 GiB** (55% less); RAM working set **8.42 → 8.75 GiB**; **95.24 → 96.15 s**. This mode check predates the latest acceleration. |
| 4-frame batching, warm SAM features | About **2.0–2.4× faster**; allocated VRAM **3.44 → 3.84 GiB**, reserved VRAM **3.74 → 4.45 GiB**; RAM about **6.47 GiB** in both. |
| GPU hand crops on top of 4-frame batching | Two clips: **6.10 → 5.44 s** and **10.87 → 9.81 s**, about 10% less feature time; a repeated alternating test found 8%. Allocated VRAM remained **3.84 GiB**. |
| Connected phases, one person / 60 frames | **55.99 → 42.65 s** (24% less); allocated VRAM **3.84 GiB** in both. Includes worker startup/saving, not discovery, cameras, correction or export. |
| Bounded CPU prefetch, warm SAM features / 450 frames | **42.96 → 40.79 s** (5% less); VRAM unchanged at **3.84 GiB**; RAM **2.45 → 2.49 GiB**. Disabled in Low Memory. |

Offloading can noticeably slow individual stages despite similar whole-job
times in a single short mode check. The shared VGGT component's warm inference
took about **3.3× as long** with offloading while its peak VRAM fell by 52%.
For model reuse across people, exact test scopes and separate output-difference
checks, see [resource modes and measured optimizations](../PERFORMANCE.md).

Camera estimation, occlusion and image features can introduce pose, trajectory
or finger errors. A fixed-camera option cannot compensate for a camera that
actually moves. Contact correction reduces supported-foot/palm sliding but does
not repair bad camera geometry or identity swaps.

The app retains allocation-error reporting. RAM/Windows allocation-headroom
stopping and CUDA allocator budgets follow [Resource Safety](../PERFORMANCE.md#resource-safety), default on. Camera analysis and long motion context
can still exceed available memory. Historical [time/VRAM records](../USER_GUIDE.md#recorded-processing-time-and-vram)
include camera estimation and predate Fast SAM; they are not predictions or
minimum specifications. See [validation](../VALIDATION.md) and
[upstream licenses](../../THIRD_PARTY_NOTICES.md).

## Windows console encoding

The app Console and the CMD window report the selected Fast SAM batch size, GPU
hand-crop setting and Low Memory single-frame selection. CUDA OOM retries report
the reduced batch size. Available memory can further reduce the effective batch;
the selected size is an upper bound, and a final partial batch may be smaller.
Cached features skip extraction, so they do not select a new SAM batch.

A `UnicodeEncodeError` mentioning `cp1252` and SAM timing/box-drawing output
is a log-encoding failure. The corrected version starts base workers in explicit
UTF-8 mode and uses UTF-8 for discarded Fast SAM logs, including batched
batches. Windows language/console code-page changes are not required. Close the
app and use `update.cmd` to install a version containing the fix; for an already
updated source checkout, `install.cmd` refreshes runtime verification. Matching
package profiles and model files are retained. Valid feature caches remain usable.

## Acknowledgments

Special thanks to [Scar1etCrystal](https://github.com/Scar1etCrystal) for their valuable help with FlowHMR optimization in RGB MoCap.

## Citation

[FlowHMR paper](https://arxiv.org/abs/2610.03691) · [Research citations and BibTeX](../CITATIONS.md#flowhmr)


### Camera display

The viewport Camera switch shows stored camera rotation and the estimated VGGT
movement path. Displacement is not metrically calibrated; it is a rough visual
guide. Motion x adjusts the displayed displacement scale (default 1; 0 fixes the
camera center). This controls camera gizmos/path only. Video overlays retain
their saved calibrated projection, and animation/correction/export are unchanged.


### Optional background camera rotation refinement

**Refine Camera Rotation** is a default-off checkbox below the camera setting.
It uses PR #18's background-feature method to validate corrections of up to
2 degrees to the saved moving-camera rotation. It masks known actors and keeps
unreliable samples unchanged; static-camera results are skipped. Fast movement,
parallax or sparse/blurred backgrounds may prevent any correction from being accepted.

This is CPU post-processing for video bone overlays, Camera Preview and camera
gizmos. It runs when a camera display is needed and does not rerun motion inference
or modify original tracking files, camera centers, motion/contact results or
FBX/BVH animation. Off restores the original saved camera path. Workspace view
settings preserve the checkbox; verified session results can be reused. No extra
runtime/models are needed. Progress, accepted counts and time are reported in the
app/Console/CMD. See [the camera preview guide](../USER_GUIDE.md#saved-cameras-and-camera-preview).
