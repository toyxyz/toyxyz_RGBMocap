# AI model guides

Choose a model in **AI Motion Model** before preparing its files or opening
videos. ExMR is the default for new work; reopening a project restores its
saved selection. Only implemented tools appear in the list.

| Model | Input and body | Fingers | Detailed guide |
| --- | --- | --- | --- |
| ExMR | One video; SMPL-X | Included | [ExMR](EXMR.md) |
| FlowHMR | One video; SMPL-H | Included | [FlowHMR](FLOWHMR.md) |
| GVHMR | One video; SMPL-X | Optional HandFlow | [GVHMR](GVHMR.md) |
| MAMMA | Two or more synchronized fixed-camera videos; SMPL-X | Included | [MAMMA](../MAMMA.md) |
| PromptHMR | One fixed-camera video; jointly reconstructed people; SMPL-X | Not generated | [PromptHMR](PROMPTHMR.md) |
| PEAR | Experimental live preview of one person from a webcam or video timeline; may be unstable | Included | [PEAR](PEAR.md) |

One-video input does not mean one-person output. ExMR, FlowHMR and GVHMR can
process checked people independently after **Find person**. Without discovery,
automatic tracking selects one person and may switch identity during overlap.
MAMMA detects and matches people across views during reconstruction; its
**Find person** button is hidden.
PromptHMR also tracks people during reconstruction. It defaults to two people
and Interaction Prompt on; see its guide for setup and fixed-camera limits.
For manual MAMMA cameras, see the [calibration format and JSON example](../MAMMA.md#manual-cameras).
PEAR support is prepared by `install.cmd` and `update.cmd` over the shared runtime;
its weights/body data are obtained separately. It is a live preview without motion
recording, contact correction, retargeting or export.

In this project's subjective tests, ExMR was generally fast with good quality;
FlowHMR was slower with the best overall quality; GVHMR was less consistent
overall but sometimes better for unusual poses such as backflips. These are
not benchmark rankings. PromptHMR could improve relative placement and
interactions for multiple people in one video, but its overall motion quality
was lower than ExMR in our subjective tests, and it has no finger tracking.
MAMMA is a separate multi-view workflow rather than another single-video option.

## Installation and shared workflow

1. Run `install.cmd` in the app folder. It prepares ExMR, FlowHMR and GVHMR
   runtimes and their supported add-ons, PEAR's shared support, Blender, FFmpeg and native
   motion correction. It does **not** download AI weights or registered body data.
2. For MAMMA, additionally follow [its runtime setup](../MAMMA.md#install-the-runtime).
   `Setup_MAMMA.cmd` is separate from the base installation and update commands.
   PromptHMR independently uses [Setup_PromptHMR.cmd](PROMPTHMR.md#install-the-separate-runtime)
   and does not require MAMMA/SDF. Both optional setups require the documented
   C++/CUDA build tools and provider license review. Close the app before setup;
   reopen it before preparing weights. `update.cmd` does not install either
   optional runtime; rerun its own setup if the app requests it after an update.
3. Run `RGBMoCap.cmd`, select the model and use **Model Manager**. Follow each
   row's exact download/import instructions. Some files require registration
   and license acceptance; Google Drive downloads may need manual import.
4. Open footage, trim, reconstruct and review. Contact correction defaults on;
   Foot defaults on and Hand off. Edit contact timings and **Apply Correction**
   without rerunning AI inference. Export the displayed applied motion as FBX.

For PEAR, use **Start Live Tracking** instead of reconstruction. It displays
live bones only; offline contact, project-save, retarget and export steps do not
apply. See its guide for filtering, World Position and Reset Bones.

Model-specific settings affect reconstruction; they are separate from contact
correction and display-only viewport controls. Original and corrected results
are preserved. One project can hold separate results for each model. MAMMA saves
all of its camera videos with **Save Project As**.

Use **Resource Mode > Automatic** initially. Actual RAM/Windows allocation
headroom monitoring, CUDA allocator budgets and allocation-error reporting remain
active. More frames, pixels, people or views can increase work; no setting
guarantees that every clip fits. Automatic selects supported Low Memory paths
when the GPU budget is low; a supported operation can recover once after CUDA
OOM. Low Memory uses CPU loading/offloading and existing bounded image batches,
which can be slower and need more RAM. It does not reduce model precision,
motion frame count or MAMMA view count/iterations. Decisions appear in Console
and the CMD window. PromptHMR keeps its validated one-image batch in both modes;
additional block offloading is not implemented. Low Memory does not guarantee a
smaller PromptHMR footprint.
See [resources](../USER_GUIDE.md#model-and-processing-settings)
and [measured validation scope](../VALIDATION.md).

Low Memory primarily saves **VRAM**, not system RAM. Short paired mode checks
measured ExMR at **5.68 → 3.87 GiB** and FlowHMR at **6.10 → 2.77 GiB** allocated
VRAM; GVHMR's short test had no meaningful whole-job reduction. MAMMA's paired
whole-pipeline VRAM saving was not recorded; only its shared VGGT offload
component has a controlled comparison. See [measured savings and speedups](../PERFORMANCE.md)
for RAM, timing, test scope and numerical differences. These figures are not
minimum requirements or predictions for every clip.

See the [user guide](../USER_GUIDE.md) for shared controls, contacts, retargeting,
FBX/VMD export, storage and cleanup. For agent-assisted setup and diagnostics,
use [the agent guide](../AGENT_GUIDE.md). Model code, weights and body data retain
their own [licenses](../../THIRD_PARTY_NOTICES.md).
