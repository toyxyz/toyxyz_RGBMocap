# GVHMR

[Original project](https://github.com/zju3dv/GVHMR) · [All model guides](README.md)

GVHMR reconstructs SMPL-X body motion from one video. It does not infer fingers
on its own; the app's optional HandFlow stage adds them. In this project's
subjective tests overall quality was lower than ExMR/FlowHMR, but some unusual
poses, including backflips, were better. Test your footage rather than treating
that observation as a universal ranking.

## Prepare files

Run the normal `install.cmd`, then select **GVHMR > Model Manager**. Required
rows depend on enabled add-ons; disabling an add-on skips its processing.
GVHMR does not require `Setup_MAMMA.cmd`, `Setup_PromptHMR.cmd` or their
C++/CUDA native-build prerequisites.

| Stage | Key files under the app's `models/` folder |
| --- | --- |
| GVHMR body reconstruction | `gvhmr/gvhmr_siga24_release.ckpt`, `hmr2/epoch=10-step=25000.ckpt`, `vitpose/vitpose-h-multi-coco.pth` |
| Registered body | `body_models/smplx/SMPLX_NEUTRAL.npz` (SMPL-X v1.1 neutral) |
| HTD observations | `pvanet.pt` |
| FootMR | `footmr/footmr_checkpoint.ckpt`, `vitpose/vitpose-h-wholebody.pth` |
| HandFlow | HandFlow/HaMeR weights and configuration in `handflow/`, plus `body_models/mano/MANO_RIGHT.pkl` |

Model Manager lists the exact shared tracking and add-on requirements. Use the
supported versions and import checks. Standard SMPL-X v1.1 and MANO may require
provider registration; MAMMA's locked-head bodies cannot replace this body.
PromptHMR reuses this standard SMPL-X Neutral file, while its runtime and
other weights remain separate.
Automatic downloads, particularly Google Drive files, can fail; use the row's
**Manual Download...** and **Import...** actions. See [model preparation](../USER_GUIDE.md#prepare-models).

## Controls and result selection

| Control | Default for new work | Meaning |
| --- | --- | --- |
| **HTD Refine** | On | Refine body motion using additional 2D observations. Runs before FootMR when both are enabled. |
| **FootMR** | On | Refine ankle/foot orientation. This is not contact locking. |
| **HandFlow** | On | Add finger motion while retaining the selected body and wrist motion. Requires its extra models. |
| **Analyze 2D** | Separate action | Prepare refinement observations. Reconstruct 3D includes this work when needed. |
| **Resource Mode** | Automatic | Select supported batch/attention behavior from available memory. Low Memory is separately selectable. |

Saved original, HTD, FootMR and HTD+FootMR results remain distinct. The checkboxes
also select the corresponding cached preview/export; they do not trigger fresh
inference by themselves. HandFlow and contact correction remain associated with
the selected base result. If a requested variant is missing or stale, reconstruct
it; the app reports it as unavailable rather than substituting another variant.
Existing projects retain their saved selections.

## Capture and export

1. Select GVHMR, open one video and trim its **Video** track.
2. Optionally run **Find person** and check the people to process. The selected
   row is the timeline/contact-edit target, independent of other visible people.
3. Prepare models for enabled add-ons and run **Reconstruct 3D**.
4. Review the selected result and correction. Foot contacts and palm support
   are edited on the timeline; **Apply Correction** reuses inferred motion.
5. **File > Export...** (FBX, BVH or Blender) follows the displayed applied result, including its
   enabled add-ons. [Character retargeting](../USER_GUIDE.md#character-retargeting)
   uses that source motion and applied contact timings.

Processing uses a 30 fps working clip without modifying the source video.
Results are saved under the project's `gvhmr/` folder. Other models keep their
own results; **Save Project As** copies saved work to the chosen location.

## Limitations and resources

FootMR does not guarantee ground contact; use Motion Correction for that.
HandFlow can still make finger mistakes, and neither add-on fixes a wrong person
or an unobserved pose. Correction retains original motion for comparison.

RAM/Windows allocation-headroom stopping and CUDA allocator limits follow
[Resource Safety](../PERFORMANCE.md#resource-safety), default on. Adaptive supported
GPU work and actual allocation-error reporting stay active in either setting. Automatic keeps the fast path when GPU
memory is sufficient and selects existing Low Memory image processing when
needed. Full temporal context is preserved; dense attention remains enabled
when it fits. Resource decisions and memory advice appear in Console and CMD.

### Measured effects

On Windows with RTX 5090 32 GB, Ryzen 9 9950X and 96 GB RAM:

- A short 60-frame mode check with HTD, FootMR and HandFlow showed **no meaningful
  whole-job VRAM saving**: Automatic **4.204 GiB**, Low Memory **4.203 GiB**.
  RAM working set was **6.87 → 6.85 GiB**. Times were **127.53 → 113.28 s**, but
  single-pass startup/cache variation does not establish that Low Memory is faster.
- A separate two-person, 60+45-frame test of feature/core processing **without
  add-ons** reduced **56.02 → 22.97 s** (59% less time) through model reuse.
  Allocated VRAM stayed about **2.82 GiB**. This was one timing trial, excluding
  discovery, camera estimation, correction and export.

Low Memory does not force slower query splitting when dense attention already
fits. Full-sequence motion context still consumes memory, so these short tests
do not establish long-video capacity. See [shared measurements and limitations](../PERFORMANCE.md).

Long-sequence context can still fail
to fit; no option guarantees a particular duration on every PC. Historical
[time/VRAM measurements](../USER_GUIDE.md#recorded-processing-time-and-vram)
include HTD, FootMR and HandFlow and are individual observations, not minimum
specifications. See [validation](../VALIDATION.md) and
[licenses](../../THIRD_PARTY_NOTICES.md).

## Citation

[GVHMR paper](https://arxiv.org/abs/2409.06662) · [Research citations and BibTeX](../CITATIONS.md#gvhmr)
