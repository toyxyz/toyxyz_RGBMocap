# Resource modes and measured optimizations

These are measurements on one Windows PC: RTX 5090 32 GB, Ryzen 9 9950X,
96 GB RAM. They are observations, not minimum hardware requirements or promised
speedups. No clean-Windows or other-GPU certification is implied. Source videos,
private projects and benchmark scripts are not included in this repository.

## Automatic or Low Memory?

| Mode | Behavior | Trade-off |
| --- | --- | --- |
| **Automatic** — default | Uses the fast supported path when the current GPU budget is sufficient. Selects the supported Low Memory path per operation when needed; supported fast operations can release their failed allocation and retry once after CUDA OOM. | Usually the best starting point. The saved selection stays Automatic even when one operation falls back. |
| **Low Memory** | Requests CPU-first loading, block offloading or bounded image work, depending on the model. Disables the new CPU batch prefetch. FlowHMR SAM uses one frame per batch. | Reduces some VRAM peaks, but transfers/loading can take longer and system RAM can increase. It does not guarantee that a model or long sequence fits. |

Neither mode deliberately lowers the existing model-native precision, input
resolution, motion frame count or temporal context. MAMMA keeps the selected
views and fitting iterations. Existing native mixed precision remains in use;
this is not a claim that every model layer is FP32.

**Low Memory is primarily a VRAM-saving mode, not a system-RAM-saving mode.**
With **Resource Safety** enabled (the default), actual RAM and Windows allocation
headroom are monitored in both modes. Work stops with an error when the measured
headroom becomes low. Decisions and memory advice appear in both the app Console
and CMD window. See the independent safety control below.

### Resource Safety

The **Resource Safety** checkbox sits below Resource Mode and defaults **on**.

- **On:** retain the existing low-RAM/Windows commit-headroom stop and PyTorch
  VRAM allocation budget. RAM reserve stays 1–4 GiB according to system size;
  Windows commit headroom below 1 GiB also triggers the monitor. Ordinary low
  samples must repeat; critical headroom below 256 MiB stops immediately.
  The desktop checks every 0.5 seconds. The VRAM budget subtracts the existing
  600 MiB Windows external-use margin, or 700 MiB for GPUs over 15 GiB, from
  currently free VRAM. These thresholds have not been increased.
- **Off:** skip that RAM/commit stop and remove the app's PyTorch allocator cap.
  This does not create additional memory. Actual allocation failures or system
  slowdowns can still occur; other applications and non-PyTorch allocations are
  outside the allocator cap in either setting.

In both settings, Automatic still measures resources to select batches and
supported Low Memory paths. Actual CUDA OOM backoff/recovery, model limits,
cancellation, incomplete-result protection and the separate 180-second
no-output/no-CPU stall check remain. Disabling safety does not change inference
precision or intentionally force larger batches. It is not a model speed preset.

The preference is local to this installation, saved in ignored
`config/local.json`, independently of projects and model selections. Console
and CMD show the setting for queued work. Changing it does not rerun inference;
running jobs retain their starting setting. PEAR inherits the same preference
for its live and alignment workers; its offline resource controls remain hidden.

### What changes for each model

| Model | Low Memory behavior |
| --- | --- |
| ExMR | Load body/hand checkpoints on CPU before GPU transfer; prepare Sapiens' existing FP16 layers on CPU. Body/hand batches remain 8. |
| FlowHMR | Offload SAM/VGGT transformer blocks to CPU and transfer them for inference. SAM uses one frame per batch; the saved GPU Hand Crops selection is retained. |
| GVHMR and add-ons | Use existing adaptive image batches and bounded processing. Preserve full-context FP32 attention; use the validated query split only when dense attention does not fit or after its supported OOM recovery. |
| MAMMA | Offload the VGGT camera transformer blocks. Masks, FP32 landmarks and 3D fitting keep their current settings; this is not full-MAMMA model offloading. |
| PromptHMR | Retains one-image batches and full temporal context; additional block offloading is not implemented. |

PEAR is a separate live tool: the offline Resource Mode control is hidden. Its
fixed-length bone path uses FP16 encoding/FP32 decoding, one requested frame per
prediction and bounded latest-frame presentation. Its RAM monitor and CUDA
allocation budget follow Resource Safety; it has no automatic offload/retry path. See the
[PEAR timings and limitations](models/PEAR.md#placement-and-performance).

## Measured memory savings

### Paired application runs

The following earlier mode-validation runs used 60 frames at 30 fps per
single-video model, including correction and preview/export preparation.
GVHMR included HTD, FootMR and HandFlow; FlowHMR included camera estimation.
They **predate the latest batching/prefetch and connected-worker changes**,
so their times are historical mode checks, not the current release's speed.
Both modes used the same clip/settings for each model.

| Model | Peak VRAM: Automatic → Low Memory | VRAM reduction | Peak RAM: Automatic → Low Memory | Time: Automatic → Low Memory |
| --- | ---: | ---: | ---: | ---: |
| ExMR | 5.68 → 3.87 GiB | 1.81 GiB / 32% | 11.88 → 11.94 GiB | 89.01 → 94.78 s |
| FlowHMR | 6.10 → 2.77 GiB | 3.33 GiB / 55% | 8.42 → 8.75 GiB | 95.24 → 96.15 s |
| GVHMR + add-ons | 4.204 → 4.203 GiB | No meaningful reduction on this short clip | 6.87 → 6.85 GiB | 127.53 → 113.28 s |
| MAMMA, 2 views / 2 people / 30 frames | Not recorded in this paired run | Not established for the whole pipeline | 8.15 → 8.12 GiB | 241.61 → 222.00 s |

VRAM means maximum recorded **PyTorch allocated** memory across workers, not
reserved memory or whole-board usage. RAM is sampled process-tree working set;
shared pages can be counted more than once. The table does not report total
system memory. Startup, kernel initialization and filesystem caching affect
these single-pass times; lower times in a Low Memory row do not establish that
Low Memory is faster. Stage-specific savings can disappear from the whole-job
peak when another stage uses more memory.

### Loading/offload trade-offs measured separately

These controlled component tests help explain where memory is saved. They are
not additional whole-application savings to add to the preceding table.

| Component / change | Peak allocated VRAM, whole component run | Peak process RAM working set | Measured time scope |
| --- | ---: | ---: | --- |
| ExMR Sapiens: prepare existing FP16 layers before GPU transfer | 5.68 → 3.10 GiB | 11.91 → 11.96 GiB | Loading: 12.24 → 12.63 s; warm inference: 1.47 → 1.45 s |
| ExMR body: CPU-first checkpoint loading | 4.91 → 3.77 GiB | 5.32 → 6.84 GiB | Loading: 4.28 → 9.97 s |
| ExMR hand: CPU-first checkpoint loading | 5.14 → 3.87 GiB | 1.54 → 5.79 GiB | Loading: 2.46 → 3.21 s |
| Shared VGGT camera component: block offload | 6.47 → 3.14 GiB | 8.75 → 8.85 GiB | Warm inference: 0.225 → 0.751 s |

ExMR used a 60-frame clip. VGGT used nine images from three cameras, at its
512 input setting; its shared component is used by FlowHMR and MAMMA. Offload
cut that component's peak by about **3.33 GiB / 52%**, but warm inference took
about **3.3 times as long**. This does not establish a 52% reduction for MAMMA's
full fitting pipeline. For that pipeline, see the separate
[MAMMA processing records](MAMMA.md#recorded-processing-time-and-memory).

## Applied speed improvements

The following methods are enabled by the current app. Their measurements use
different stages and baselines; **do not add or multiply the speedups** to
estimate a complete job. No extra packages, quantization, TensorRT engine or
user compilation is needed for these optimizations. MAMMA's optional runtime
still has its separately documented installation/build requirements.

### VGGT camera checkpoint loading

The Automatic camera-only fast path uses `meta` initialization and
`load_state_dict(assign=True)` to avoid disposable initial weights and copying
over them. FlowHMR and MAMMA share this loader. Low Memory, including an Automatic
fallback/retry, retains the previous copy-based loader and block offload. Generic
depth loading is unchanged. There is no new setting or runtime dependency.

Paired component tests used nine images from three cameras, balanced 512
preprocessing (9x3x432x592 tensor), on the test PC above. Each Flow runtime
configuration ran in two separate processes; times below are means and memory
is the largest recorded value. Image decoding is outside the reported timers.

| Automatic VGGT camera component | Previous loading | Meta + assign |
| --- | ---: | ---: |
| Instrumented model loading | 7.24 s | 2.34 s |
| Peak process RAM working set | 8.88 GiB | 5.40 GiB |
| Peak Windows process private allocation | 14.19 GiB | 14.19 GiB |
| Peak PyTorch allocated VRAM, including inference | 6.47 GiB | 6.47 GiB |

Loading timers include measurement-time cleanup/synchronization; filesystem
caches were not cleared. The MAMMA Python runtime separately completed a
one-pair check, 7.10 to 2.41 s loading. All 14 experimental runs retained exact
camera pose encodings, extrinsics and intrinsics, with unchanged FP32 parameter
storage and native inference autocast. These are camera-component observations,
not whole-motion speedups or a guarantee of improved system allocation headroom.

Meta/assign was deliberately not adopted for Low Memory: experimental Windows
private allocation increased from 11.04 to 13.86 GiB, while peak working set
changed only from 8.88 to 8.46 GiB. Some reading also moved to first inference.
`assign=True` without meta initialization did not provide the combined benefit.

### Model reuse and connected workers

One resident model serves actors sequentially, then is released before the next
phase. This avoids repeatedly loading the same weights without making a larger
multi-person GPU batch. A first two-person test used 60+45 frames, static cameras,
and feature/core stages only, with one timing trial:

| Model | Independent actor workers → model reuse | Time reduction | Peak allocated VRAM before → after |
| --- | ---: | ---: | ---: |
| FlowHMR | 108.69 → 65.34 s | 40% | 3.84 → 3.84 GiB |
| ExMR | 161.08 → 106.51 s | 34% | 5.68 → 5.68 GiB |
| GVHMR, without add-ons in this timing | 56.02 → 22.97 s | 59% | About 2.82 GiB in both |

A later two-trial comparison connected phases after phase-scoped reuse was
already available. Values below are means; worker startup and result saving
are included, while discovery, camera estimation, correction and export are not.

| Model / people | Separate phase workers → connected phases | Time reduction | Peak allocated VRAM | Peak RAM working set before → after |
| --- | ---: | ---: | ---: | ---: |
| FlowHMR / 1 / 60 frames | 55.99 → 42.65 s | 24% | 3.84 GiB in both | 6.75 → 6.81 GiB |
| ExMR / 1 / 60 frames | 83.86 → 46.66 s | 44% | 5.68 GiB in both | 11.92 → 11.74 GiB |
| ExMR / 2 / 60+45 frames | 92.83 → 53.29 s | 43% | 5.68 GiB in both | 11.77 → 11.81 GiB |

ExMR now connects keypoint, body-feature, hand-feature and motion phases.
FlowHMR connects features and motion for one actor; its two-person connection
gain was smaller than trial variability, so the existing multi-person phase
grouping remains. GVHMR already connects its observation/core phases. Short-clip
startup gains do not predict the same percentages on long footage.

### Bounded CPU preparation and unused body readouts

Automatic prepares at most two CPU batches ahead when CPU/RAM headroom permits;
Low Memory disables it. There is one producer thread and an additional 256 MiB
limit on estimated prepared storage, in addition to the batch being consumed.
This is not a whole-process RAM cap. It overlaps reading/cropping with GPU work, without
writing an extra disk cache. Actual RAM monitoring remains active.

Warm tests processed 60-, 45- and 120-frame inputs twice: **450 frames per
variant**. The first two inputs were 720x1280; the third was 1080x1920, at 30 fps.
Times exclude model loading/warm-up, and RAM is sampled process working set.

| Stage / applied optimization | Time before → after | Reduction | Peak VRAM before → after | Peak RAM before → after |
| --- | ---: | ---: | ---: | ---: |
| Flow Fast SAM / CPU prefetch | 42.96 → 40.79 s | 5% | 3.84 → 3.84 GiB | 2.45 → 2.49 GiB |
| ExMR Sapiens / CPU prefetch | 11.53 → 8.63 s | 25% | 3.10 → 3.10 GiB | 4.07 → 4.10 GiB |
| ExMR body / prefetch + unused final pose-readout omission | 18.57 → 13.15 s | 29% | 3.80 → 3.80 GiB | 5.43 → 5.42 GiB |
| ExMR hand / CPU prefetch | 7.50 → 5.63 s | 25% | 3.87 → 3.87 GiB | 1.67 → 1.67 GiB |

Body-readout omission alone reduced 18.57 to 16.63 s, about 10%. Intermediate
pose feedback and keypoint-token updates remain. Hand-readout omission had no
clear benefit and was not adopted. Small RAM decreases in this table are sample
variation, not a RAM-saving feature.

Fixed-feature motion comparisons were tensor-exact. With fresh optimized image
preprocessing, maximum joint differences were 0.83 mm for FlowHMR and 0.88 mm
for ExMR; baseline repeats also differed, up to 0.77 and 0.69 mm. These are
output disagreements, not accuracy against ground-truth capture.

### FlowHMR Fast SAM switches

Both switches default on for new work; saved off settings remain off. They work
only with Fast SAM and can be controlled independently. The following records
describe the earlier 4-frame implementation; the current Adaptive Batching
policy and its additional measurements follow below:

- **4-Frame Batching:** warm feature extraction was about **2.0–2.4 times faster**
  than one-frame processing. Peak allocated VRAM rose **3.44 → 3.84 GiB** and
  reserved VRAM **3.74 → 4.45 GiB**; RAM working set was about 6.47 GiB in both.
  Low Memory uses one-frame batches.
- **GPU Hand Crops:** on top of batching, separate short clips took
  **6.10 → 5.44 s** and **10.87 → 9.81 s**, roughly 10% less feature time,
  with unchanged 3.84 GiB allocated VRAM and about 6.47 GiB RAM. An alternating
  repeated test found a smaller 8% median time reduction, showing variability.

These are warm feature-stage results, not whole-job speedups. GPU interpolation
differs from CPU OpenCV cropping; combined output differences reached about
1.1–1.3 mm at the 95th percentile and 4.8 mm maximum joint displacement in the
tested clips. This is a separate comparison from the sub-millimetre prefetch
test above. Inspect your motion/contact timings rather than assuming bit identity.

### FlowHMR adaptive batches (2026-11-02)

**Adaptive Batching** replaces the 4-Frame Batching label while retaining its
saved boolean. Automatic checks free VRAM, reusable allocator cache, the process
allocation cap, free RAM and image dimensions after model loading. It selects
1/2/4/8/16 frames and rechecks headroom between prepared batches. The existing
Windows GPU reserve is retained, not subtracted twice. Low Memory and a disabled
batching checkbox always use one frame. Decoder precision, crop resolution,
temporal context, frame count and the first 30 placement anchors are unchanged.

On the same RTX 5090/Windows test PC, two 1080p landscape/portrait clips gave:

| Measurement | Previous batch 4 | Batch 16 |
| --- | ---: | ---: |
| Warm features, 60 frames | 4.71 s | 3.98 s |
| Warm features, 120 frames | 9.86 s | 7.63 s |
| Peak allocated VRAM | 3.84 GiB | 5.43 GiB |
| Peak reserved VRAM | 4.34 GiB | 6.81 GiB |
| Warm RAM working set, 60 frames | 2.54 GiB | 2.74 GiB |
| Warm RAM working set, 120 frames | 2.89 GiB | 3.08 GiB |

Warm timers exclude initialization and decoding; they are the second complete
sequence pass after warmup, not full-job acceleration guarantees. Fresh app
feature jobs including initialization were 34.80 → 34.88 s and
39.77 → 38.74 s; their RAM peaks remained about 6.5 GiB. At batch 32, allocated
VRAM reached 7.56 GiB and the 120-frame warm pass took 7.32 s, only about 4%
less than 16. The current ceiling therefore favors 16's measured efficiency.
Other GPUs, drivers, inputs and concurrent applications can differ.

Across the actual app paths, final joint disagreements against the previous
batch-4 output were p95/max **0.15/0.59 mm** and **0.44/1.63 mm**. Placement
offset changes were below 0.08 mm. Existing repeat extractions also had small
numerical differences; these are output disagreements, not ground-truth error.
Low Memory remained one frame and measured 1.91 GiB allocated in both versions.
Cache reuse, sequential actor/model reuse and interrupted-job recovery were
exercised. A controlled 4.4 GiB allocator cap triggered 16 → 8 → 4 retries
without dropping frames or publishing a partial feature file. This is a local
allocation-failure test, not a clean-PC or physically exhausted-GPU test.

The [FlowHMR team's performance proposal](https://github.com/toyxyz/toyxyz_RGBMocap/issues/17)
and [upstream measurements](https://github.com/flowhmr/flowhmr/blob/main/docs/performance.md)
also motivated separate Lite SAM and OpenCV-style GPU body/hand crop tests.
At batch 16, official Lite GPU extraction took 4.11/8.69 s on the two clips,
versus Fast SAM's 3.98/7.63 s. Its full-output placement path was also slower.
The exact-coordinate crop candidate over the existing Fast path took 4.50 s
on the 60-frame clip. Neither candidate is enabled; Fast SAM remains the app
backend. Body-only cropping was not isolated in that combined crop test.
Compilation, TensorRT, model weights and installed dependencies are
unchanged. The upstream A100 report compares against an unaccelerated original
baseline, so its 5–10× claim must not be applied to this app's previous Fast path.

### FlowHMR query batches and four-frame hands (2026-11-03)

The app now uses the validated official 256-query attention implementation for
all FlowHMR motion lengths. FP32, full key visibility, RoPE, temporal context,
checkpoint and 20 ODE steps are retained. The before/after measurements below
use identical features on the test PC (RTX 5090 32 GB, Ryzen 9 9950X, 96 GB RAM):

| Motion length (30 fps) | Original time | Query-256 time | Original / query peak allocated VRAM | Original / query worker RAM (RSS) |
| --- | ---: | ---: | ---: | ---: |
| 60 frames / 2.0 s | 0.47 s | 0.55 s | 1.83 / 1.83 GiB | 4.97 / 4.97 GiB |
| 474 frames / 15.8 s | 0.77 s | 0.90 s | 2.04 / 1.93 GiB | 4.48 / 4.99 GiB |
| 999 frames / 33.3 s | 2.16 s | 1.82 s | 2.64 / 2.07 GiB | 5.03 / 5.01 GiB |
| 4,806 frames / 160.2 s | 25.45 s | 7.97 s | 17.41 / 3.06 GiB | 4.37 / 5.04 GiB |

At 4,806 frames, this is **3.19× throughput (68.7% less time)** and
**82.4% less allocated VRAM**, while worker RSS increases about **0.67 GiB**.
Short sequences can be slightly slower. These are warm motion-generator timings
reported as medians of three runs per case, excluding model load, warmup, SAM extraction,
camera estimation, correction and export. Allocated VRAM is the PyTorch allocation
peak, not total GPU/desktop usage; RSS is the worker working set, not all processes.
These are measured differences, not minimum requirements or whole-job speedups.
The source-motion comparison found joint differences below **0.001 mm**; a
separate downstream correction comparison reached **0.674 mm / 0.202°** maximum
disagreement with unchanged contact timing. Floating-point results are not
guaranteed bit-identical.

**4-Frame Hand Sampling** is a separate, default-on Fast SAM option. Body
features and output motion still use every frame. Hand features are sampled
every four frames plus the final endpoint, then interpolated on CPU. It works
with one-frame Low Memory batches and CPU or GPU hand crops. It is independent
of the number of spatial frames processed together. Disable it to retain
every-frame hand evidence. This option does not itself run HaMeR or add a runtime dependency.

On the test PC, a 327-frame 1080×1920 clip's fresh app feature jobs took
**54.05 s off / 43.21 s on**, including model loading and decoding. Allocated
VRAM peaks were **5.43 / 4.21 GiB**. Earlier warm extraction measurements were
**20.55 / 11.33 s**; these are different timing scopes. Neither is a universal
whole-job speedup. The app integration comparison retained all 327 frames.
Query versus original attention, using identical features, differed by at most
**0.00034 mm** across source joints. Sparse versus every-frame hands differed
by **22.26 mm mean / 85.24 mm maximum** across body/wrist joints in this
upper-body clip; the all-joint maximum was **167.85 mm**. These are disagreements,
not measured accuracy losses against ground truth. The hands condition the
whole motion model, so changes are not confined to fingers or contact timing.

See [FlowHMR controls and limits](models/FLOWHMR.md#controls) and the
[integration checks](VALIDATION.md#flowhmr-query-and-hand-sampling-integration-2026-11-03).

### Optional FlowHMR HaMeR fingers (2026-11-03)

The separate default-off HaMeR option uses the existing ExMR runtime and shared
HaMeR/MANO weights. It infers every frame with ExMR crop preparation, FP16 network
execution and Gaussian rotation smoothing. It is not ExMR's learned HandMR
temporal inference. It forces four-frame SAM hand conditioning on, but does not
reduce HaMeR's own temporal sampling. It replaces fingers only and may still
produce jitter or inaccurate poses. FlowHMR body/wrists/root/shape/native contacts
remain exact before downstream correction.

A 327-frame app check measured **26.67 s** for the fresh HaMeR child pass,
including its startup/model load and video read. Peak child allocated VRAM was
**5.14 GiB**, with peak child RAM working set **1.75 GiB**. These are child-worker
figures, not combined parent/desktop RAM peaks or universal requirements. The
parent releases its GPU model before the child starts. Warm timings from earlier
standalone FP32 crop experiments have a different scope and precision.

### MAMMA direct SAM video input (2026-11-03)

MAMMA now skips the intermediate lossless RGB PNG write/read before SAM 2.1.
It uses the existing video reader and the same PIL resize, FP32 tensor and
normalization. YOLO/CLIP matching, mask prompts, propagation, tiny-component
cleanup, landmark inference, cameras, fitting and correction are unchanged.
Final person masks remain available for downstream stages. This optimization
is used in both resource modes and needs no new runtime or weights.

Independent paired measurements on the RTX 5090 / 96 GB RAM test PC used four
1920×1080 videos, two people, 30 fps, and the same source prefix at both lengths.
Each variant ran in a fresh process, once per duration, with CPU frame/state
offload unchanged. Time covers mask-model setup, input preparation, matching,
propagation, mask saving and cleanup; camera/landmark/3D fitting are excluded.

| Duration | Time: PNG / direct | Peak RAM working set: PNG / direct | Peak allocated VRAM: PNG / direct |
| --- | ---: | ---: | ---: |
| 7 s / 210 frames | 221.79 / 184.40 s | 6.569 / 6.592 GiB | 2.536 / 2.536 GiB |
| 14.2 s / 426 frames | 398.88 / 314.52 s | 10.053 / 10.050 GiB | 2.536 / 2.536 GiB |

Mask-stage time fell **16.9% / 21.2%** (1.20× / 1.27× throughput).
RAM changed **+23 / −3 MiB**, with identical allocated/reserved CUDA peaks
within each pair. Windows private-commit peaks were 10.661 / 10.670 GiB for
7 seconds and 13.989 / 13.981 GiB for 14.2 seconds. Private commit is not
physical RAM; working set is the OS process peak, including short transients.

All **5,088 paired person-mask images** were pixel-identical. PNG input payloads
of 2.57 / 5.23 GiB were avoided; final masks and other necessary outputs still
write to disk. Process logical writes fell 3.89→1.32 / 7.88→2.65 GiB; these are
Windows process I/O counters, not physical SSD writes. The SAM CPU tensor alone
uses about 12 MiB per frame at its 1024-square setting in either path, so long
clips can still need substantial RAM. These tests do not establish minute-scale
memory use, universal speedups or less RAM for every input.

An earlier separate four-view, three-second comparison also retained identical
masks, landmarks, final mesh and joints. The app-path regression and remaining
scope are recorded in [validation](VALIDATION.md#mamma-direct-sam-input-2026-11-03).

### Other MAMMA optimizations

The current MAMMA adapter uses exact-coordinate ROI preprocessing, CPU checkpoint
loading, a mask-based landmark path without an unnecessary detector, fewer
unused video/debug outputs and reduced fitting diagnostics. Fixed-camera VGGT
uses up to three sampled times per camera at its 512 input setting and omits
unneeded depth output. Completed inference is reused when identities/receipts
match; changing only contact correction does not run MAMMA again.

Earlier four-camera, two-person, 30-frame FP32 tests measured landmark processing
**104.43 → 57.18 s** (45% less time) with exact ROI/CPU loading, and mask
processing **129.02 → 92.39 s** (28% less time) after omitting unused outputs.
Reduced fitting logs did not show a reliable fitting-stage speedup. These are
stage measurements, not a prediction for a complete capture.

MAMMA uses SAM 2.1 and FP32 landmarks/fitting in the app. SAM 3.1, quantization,
additional TF32, reduced fitting iterations and full fitting-model offload
remain outside the supported app path. The latest ExMR/FlowHMR prefetch and
connected-worker changes were not added to MAMMA.

For remaining scope and actual checks, see [validation](VALIDATION.md).
