# ExMR

[Original project](https://github.com/twehrbein/ExMR) · [All model guides](README.md)

ExMR reconstructs SMPL-X body and finger motion from one video. It is the default
model for new work. In this project's subjective tests it was generally fast
with good motion quality; results depend on the footage and settings.

For a qualitative two-person contact example, see the
[PromptHMR and ExMR interaction comparison](PROMPTHMR.md#single-video-interaction-example).

## Prepare files

Run the normal `install.cmd`; no separate ExMR setup command is needed.
`Setup_MAMMA.cmd` and `Setup_PromptHMR.cmd` are optional for other workflows;
ExMR does not require their native builds or system CUDA Toolkit.
Select **ExMR > Model Manager** and prepare the listed files. Key assets are:

| Asset | Location under the app's `models/` folder |
| --- | --- |
| ExMR weights | `exmr/exmr.safetensors` |
| Official ExMR checkpoint metadata | `exmr/config.json` |
| Sapiens2 pose weights | `exmr/sapiens2_1b_pose.safetensors` |
| SMPL-X v1.1 neutral body | `body_models/smplx/SMPLX_NEUTRAL.npz` |
| MANO right hand | `body_models/mano/MANO_RIGHT.pkl` |
| Shared SAM 3D Body model, config and MHR | `flowhmr/sam/model.ckpt`, `flowhmr/sam/model_config.yaml`, `flowhmr/sam/assets/mhr_model.pt` |
| HaMeR and hand mean | `handflow/hamer.ckpt`, `handflow/mano_mean_params.npz` |

Model Manager also lists shared tracking dependencies. Use its exact supported
files and import checks; do not substitute arbitrary checkpoint versions.
**Download All** includes the small public `config.json` (180 bytes), or use
**Auto Download** in the **ExMR configuration** row. The metadata and weights
are pinned to the same official Hugging Face revision and verified by SHA-256.
Manual Download and Import are also available. This is checkpoint metadata,
not a file for changing the app's camera, smoothing or correction settings.
Registered SMPL-X/MANO and gated SAM 3D Body files may require manual download.
The shared `flowhmr` and `handflow` folder names do not mean you must run those
models or enable GVHMR's HandFlow add-on. ExMR uses these assets directly.
MAMMA's locked-head SMPL-X bodies are different files and cannot replace ExMR's
standard body. PromptHMR can reuse the same standard SMPL-X Neutral file,
while its runtime and other weights stay separate. See [model download and reuse](../USER_GUIDE.md#prepare-models).

## Controls

| Control | Default for new work | Meaning |
| --- | --- | --- |
| **Camera** | Static Camera | Use for a camera that stays still without zooming. Choose Moving Camera / SimpleVO when it moves. ExMR does not use FlowHMR's VGGT camera stage. |
| **Smooth Motion** | On | Reduce small frame-to-frame changes; off preserves the unsmoothed result. |
| **Resource Mode** | Automatic | Use fast loading when GPU memory is sufficient; otherwise use CPU-first body/hand loading and prepare Sapiens' existing FP16 layers before GPU transfer. Low Memory forces these paths. Body/hand batches remain 8. CPU loading can be slower and use more RAM. |

Camera and smoothing changes take effect on **Reconstruct 3D**, not on
**Apply Correction**. Existing projects retain saved settings.

## Capture and export

1. Select ExMR and use **File > Open Video**. Trim the **Video** track as needed.
2. For explicit identity selection or several people, run **Find person** and
   check who to process. The selected row controls preview and contact editing.
   Choose **Default** or use [SAM 3.1](SAM31.md) on Windows for text prompts
   and click selection.
3. Prepare models, choose camera settings and run **Reconstruct 3D**.
4. Review the original and corrected motion. Edit foot/palm support intervals
   and **Apply Correction** when needed. This reuses inference results.
5. Use **File > Export...** (FBX, BVH or Blender) for the displayed applied source motion, or
   [import and retarget a character](../USER_GUIDE.md#character-retargeting).

Processing uses a 30 fps working clip and preserves the source video. Saved
results are under the project's `exmr/` folder. Switching models loads that
model's own available results; it does not convert one model's inference into
another's. **Save Project As** copies saved work to the selected location.

## Limitations and resources

Reconstruction reuses one model at a time across selected people and connects
its processing phases in one worker. With enough CPU/RAM headroom, Automatic
prepares up to two image batches ahead; Low Memory disables this overlap.
The body feature extractor omits unused final pose readouts while retaining
intermediate feedback. Precision, image resolution and motion context are unchanged.

### Measured effects

Measurements below used an RTX 5090 32 GB, Ryzen 9 9950X and 96 GB RAM on
Windows. They use different baselines and must not be added together:

| Comparison | Measured change |
| --- | --- |
| Automatic → Low Memory, 60 frames including correction/preview preparation | Allocated VRAM **5.68 → 3.87 GiB** (32% less); RAM working set **11.88 → 11.94 GiB**; **89.01 → 94.78 s**. This mode check predates the latest acceleration. |
| Connected phases, one person / 60 frames | Worker time **83.86 → 46.66 s** (44% less); peak allocated VRAM **5.68 GiB** in both. Includes worker startup/saving, not discovery, correction or export. |
| Connected phases, two people / 60+45 frames | **92.83 → 53.29 s** (43% less); peak allocated VRAM unchanged at **5.68 GiB**. |
| CPU preparation overlap, warm feature stages | About **25% less time** for Sapiens and hand features; **29% less** for body features with unused final readouts omitted. Measured VRAM was unchanged; RAM was similar, with small increases in some stages. |

CPU-first loading can increase RAM substantially: the isolated hand-model test
rose from **1.54 to 5.79 GiB** working set. It is not a host-memory-saving feature.
Fixed-feature motion comparisons were exact; fresh optimized preprocessing
produced maximum joint differences of **0.88 mm**, with baseline repeats also
differing by up to 0.69 mm. These are output differences, not ground-truth accuracy.
See [full conditions, loading trade-offs and measurements](../PERFORMANCE.md).

Occlusion, motion blur, poor visibility and identity changes can produce wrong
poses or fingers. Automatic contact detection can also miss intentional support;
edit its timings rather than expecting correction to recover missing evidence.
Contact correction is distinct from Smooth Motion and preserves original results.

RAM/Windows allocation-headroom stopping and the shared CUDA allocator budget
follow [Resource Safety](../PERFORMANCE.md#resource-safety), default on. Allocation failure stops the job with an error; shorten footage
or free resources if necessary. Historical [processing/VRAM measurements](../USER_GUIDE.md#recorded-processing-time-and-vram)
are individual runs, not minimum specifications. See [validation](../VALIDATION.md)
for the tested PC and scope, and [licenses](../../THIRD_PARTY_NOTICES.md) for
upstream restrictions.

## Integrated upstream update

The app uses ExMR v0.1.5's public input helpers and model API. Its toe IK target
is limited to a 3 cm offset per axis from the network's target, reducing excessive
ankle changes in the tested motions. Existing weights and the pinned ExMR runtime
are reused. The app keeps its existing scene frame, camera sampling and shared
contact-correction policy; the upstream Boolean foot-contact detector does not
replace the app's automatic or manually edited lock timings.

New reconstruction uses the updated solver and refreshes its result receipts.
Opening an existing project does not automatically rerun inference or rewrite
its saved motion. See [validation](../VALIDATION.md#exmr-public-api-and-bounded-toe-ik-2026-10-31)
for the measured scope and limitations.

## Citation

[ExMR paper](https://arxiv.org/abs/2610.26213) · [Research citations and BibTeX](../CITATIONS.md#exmr)

Please also credit FootMR and the original GVHMR work.

```bibtex
@Article{wehrbein26exmr,
    author    = {Wehrbein, Tom and Rosenhahn, Bodo},
    title     = {{ExMR}: Extremity Motion Refinement for Monocular Human Motion Capture},
    journal   = {arXiv preprint arXiv:2610.26213},
    year      = {2026},
}
```


### Camera display

The viewport Camera switch can show estimated movement for Moving Camera mode.
Camera rotation comes from the stored SimpleVO track; approximate displacement
is derived from the ORIGINAL paired world/camera body-root predictions, without
rerunning inference. Body prediction errors can appear as camera movement.
Static Camera retains a fixed position. Motion x controls the gizmo/path distance
scale (default 1; 0 fixes the center), independently of the video overlay and
exported animation. A full-video reference take is required for displacement;
otherwise a notice identifies rotation-only display.


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
