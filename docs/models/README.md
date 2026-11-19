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
| PromptHMR | One video; fixed or optional moving camera; jointly reconstructed people; SMPL-X | Not generated | [PromptHMR](PROMPTHMR.md) |
| GEM-X | One fixed-camera video; one primary person; SOMA Native77 | Included | [GEM-X](GEMX.md) |
| PEAR | Experimental real-time tracking from webcam or video input; one person, one or multiple fixed views; may be unstable | Included | [PEAR](PEAR.md) |

One-video input does not mean one-person output. ExMR, FlowHMR and GVHMR can
process checked people independently after **Find person**. Without discovery,
automatic tracking selects one person and may switch identity during overlap.
On Windows, **Find person** also offers [optional SAM 3.1 tracking](SAM31.md)
with text prompts and click selection for those three models. SAM 3.1 uses
more VRAM than the Default tracker. See [measured VRAM](SAM31.md#vram-use).
MAMMA detects and matches people across views during reconstruction; its
**Find person** button is hidden.
PromptHMR also tracks people during reconstruction. It defaults to two people
and Interaction Prompt on; see its guide for setup and fixed-camera limits.
For manual MAMMA cameras, see the [calibration format and JSON example](../MAMMA.md#manual-cameras).
Automatic MAMMA cameras combine VGGT-Omega placement with GeoCalib background
vertical direction before 3D fitting; both checkpoints appear in its Model Manager.
GeoCalib reuses the base Flow support overlay rather than MAMMA's separate runtime.
PEAR support is prepared by `install.cmd` and `update.cmd` over the shared runtime;
its weights/body data are obtained separately. It is a live preview without motion
recording, offline contact correction, retargeting or export. Camera Align is
attempted first; if it fails, tracking continues from the selected camera with a
warning, without camera fusion or aligned live foot lock.

In this project's subjective tests, ExMR was generally fast with good quality;
FlowHMR was slower with the best overall quality; GVHMR was less consistent
overall but sometimes better for unusual poses such as backflips. These are
not benchmark rankings. PromptHMR could improve relative placement and
interactions for multiple people in one video, but its overall motion quality
was lower than ExMR in our subjective tests, and it has no finger tracking.
GEM-X processed clips quickly, with relatively low overall motion quality in those tests.
MAMMA uses synchronized views in a separate multi-view workflow.

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
   GEM-X uses [Setup_GEMX.cmd](GEMX.md#setup), which reuses the ExMR Python
   base and needs no local native build. The optional SAM 3.1 **Find person**
   tracker for ExMR, FlowHMR and GVHMR uses [Setup_SAM31.cmd](SAM31.md#setup)
   and a separate Model Manager download.
3. Run `RGBMoCap.cmd`, select the model and use **Model Manager**. Follow each
   row's exact download/import instructions. Some files require registration
   and license acceptance; Google Drive downloads may need manual import.
4. Open footage, trim, reconstruct and review. Contact correction defaults on;
   Foot defaults on and Hand off. Edit contact timings and **Apply Correction**
   without rerunning AI inference. **File > Export...** offers FBX, BVH or a portable Blender project for the displayed applied motion. MAMMA also offers checked bodies and all cameras in one Blender scene; see [export options](../USER_GUIDE.md#fbx-and-blender).

For PEAR, use **Start Live Tracking** instead of reconstruction. It displays
live bones only; offline contact, project-save, retarget and export steps do not
apply. Multiple synchronized files/webcam slots and Camera Align are described
in its guide, along with filtering, World Position and Reset Bones.

GEM-X uses its native SOMA result for preview, foot and hand correction,
and character retargeting. **File > Export...** offers Native77 FBX and BVH for
the skeleton and a Blender bundle with the evaluated surface and source video.

Model-specific settings affect reconstruction; they are separate from contact
correction and display-only viewport controls. Original and corrected results
are preserved. One project can hold separate results for each model. MAMMA saves
all of its camera videos with **Save Project As**.

Use **Resource Mode > Automatic** initially. **Resource Safety**, default on,
controls low-RAM/Windows headroom stopping and CUDA allocator limits. Off bypasses
these limits; adaptive batching and actual allocation-error handling stay active.
See [Resource Safety](../PERFORMANCE.md#resource-safety). More frames, pixels, people or views can increase work; no setting
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
