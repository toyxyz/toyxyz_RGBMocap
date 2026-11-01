# MAMMA runtime and native build: agent guide

Use this runbook when helping a Windows user install, repair or independently
build the optional MAMMA environment. Read [AGENT_GUIDE.md](AGENT_GUIDE.md) first.
For user-facing input/model instructions, see [MAMMA.md](MAMMA.md).

The preferred route is **Setup_MAMMA.cmd**. It already downloads, builds, tests
and registers the environment. Do not ask ordinary users to reproduce the manual
build below when the supported installer can do the work.

## 1. Scope and authoritative files

| File | Responsibility |
| --- | --- |
| `Setup_MAMMA.cmd` | Human-readable prerequisite/license prompt, then optional setup |
| `tools/setup_mamma.py` | Pinned downloads, compiler discovery, isolated staging, native build, verification and atomic activation |
| `installer/mamma-runtime-lock.json` | Exact Python, package artifacts, source commits, sizes and SHA-256 values |
| `installer/mamma_adapters.py` | Windows/validated engine changes and SDF MSVC fixes |
| `installer/mamma_helpers/` | Original exact-ROI/FP32 landmark helpers copied into the engine |
| `installer/verify_mamma.py` | Full locked-package check, actual CUDA kernel tests and installed-file verification |
| `mocap/mamma_runtime.py` | Runtime registration, activation checks and engine fingerprint |
| `tools/configure_mamma.py` | Registration of an independently prepared environment |
| `mocap/mamma_models.py` | Supported model paths, hashes and provider instructions |

Read the current files before using copied commands: the lock is authoritative,
not the version examples in this document. This guide does not install packages
into the GUI, shared, FlowHMR, HandFlow, HTD, ExMR or Blender runtimes.

Current optional pins include Python **3.11.17**, PyTorch **2.7.1+cu128**,
torchvision **0.22.1+cu128**, NumPy **1.26.4**, SciPy **1.17.1**, smplx
**0.1.28** and timm **1.0.30**. All remaining dependencies are also locked.
Do not substitute the latest pip versions or merge the model runtimes.

## 2. Before changing anything

1. Identify the actual app checkout containing `RGBMoCap.cmd`. Do not assume the
   user's working directory, a historic test folder or a different installation
   is the active app. Run commands from that checkout.
2. Save work and close RGB MoCap and other installers/updaters for this checkout.
   The installer uses an OS file lock in `.setup/app.lock`. A file remaining on
   disk is normal; deleting it is not a way to release a live process's lock.
3. Read `git status --short`. Preserve local source edits and user data. Do not
   reset the repository, delete runtimes or copy another installation over it.
4. If `.setup/mamma-runtime.json` exists, inspect its `version`, `python`,
   `engine`, `managed_id` and `compute_capability` when present. These are local
   absolute paths. Keep configuration/log copies private; redact usernames and
   personal paths from public reports.
5. Back up the existing configuration to a private diagnostics directory outside
   the publishable source. Preserve its referenced environment in place. A
   backup of the JSON alone cannot restore a deleted runtime.
6. Check the available disk space and GPU/driver. Do not start competing model
   inference while compiling or running GPU validation.

Model files, registered body data, projects, contact edits, character data and
source videos are not build inputs to delete. Never include them in a commit.

## 3. Prerequisites and GPU target

- Complete the ordinary `install.cmd` installation first. The optional setup
  uses `runtime/python.exe` and the verified `.setup/bootstrap-git/uv.exe`.
- Use native 64-bit Windows with an NVIDIA CUDA-capable GPU and a driver
  compatible with the pinned CUDA 12.8 PyTorch stack. No CPU/AMD fallback is
  implemented for this MAMMA pipeline.
- Install **Visual Studio 2022 Build Tools**, **Desktop development with C++**,
  the x64 compiler and a Windows SDK. The current discovery uses `vswhere` for
  Visual Studio version 17.x; a different Visual Studio major version is not a
  drop-in replacement.
- Install **CUDA Toolkit 12.8**, including `nvcc.exe`. The CUDA version shown by
  `nvidia-smi` describes driver capability; it does not prove the toolkit is
  installed. Setup checks `CUDA_PATH_V12_8`, or the standard CUDA `v12.8` folder.
- Review the provider licenses. The setup prompt requires acceptance; the
  noninteractive `--accept-licenses` switch is an explicit acknowledgment, not a
  switch to bypass those terms. Use it only when the user has already authorized
  installation under the reviewed terms. Never handle provider passwords.

Typical GeForce targets are RTX 30 = 8.6, RTX 40 = 8.9 and RTX 50 = 12.0.
These examples are not a GPU support promise. Query the actual target interpreter:

Before a MAMMA environment exists, use `nvidia-smi` for the initial GPU/driver
inventory. Run the PyTorch probe below only after automatic setup or preparation
of the complete private environment; do not switch another model's runtime to
make the probe work.

```powershell
# PowerShell, from the app checkout. Replace the interpreter path below.
$mammaPython = 'D:\MAMMA-build\runtime\python.exe'
& $mammaPython -I -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name()); print(torch.cuda.get_device_capability())"
if ($LASTEXITCODE -ne 0) { throw 'GPU probe failed; stop before building.' }
```

The installer sets `TORCH_CUDA_ARCH_LIST` to the installed GPU's capability.
Different GPU models can share a target architecture, but that alone does not
prove binary compatibility: the complete Python/PyTorch/OS/dependency stack must
also match. Multi-architecture builds are not the current installer behavior.
Do not promise a 5090-only test validates a 4090, 3090 or every 50-series GPU.

Measured optional runtime: **6.19 GiB**. Allow about **18 GiB free during setup**,
excluding separately installed compiler tools, models and projects. Downloaded
package/source archives are about **3.57 GiB**, excluding Python and weights.
The ten models total about **7.80 GiB**, or **3.53 GiB additional** if shared VGGT is
already available. GeoCalib is also shared with PEAR (about 0.108 GiB); count it
once. Automatic camera gravity uses the base Flow/Python 3.10 alignment overlay
installed by `install.cmd`, in a separate child process before 3D fitting. Do not
inject its cp310 packages into MAMMA's Python 3.11 runtime. Manual camera data
bypasses both automatic camera models. Confirm space on the app drive, not just a different drive.

## 4. Supported automatic installation

In PowerShell at the app root:

```powershell
.\Setup_MAMMA.cmd
if ($LASTEXITCODE -ne 0) { throw 'MAMMA setup failed; inspect the setup logs.' }
```

The CMD presents the license prompt. In Command Prompt use `Setup_MAMMA.cmd`
without the PowerShell `.\` prefix. Keep the terminal open until completion.
No manual Python activation or package installation is needed for this route.

Expected setup sequence:

1. Acquire the app/installer lock; derive runtime identity from the optional
   lock, adapter and two helper files.
2. If the existing managed generation has the same identity, verify it again
   and retain it when valid. Do not rebuild simply because setup emits no
   download progress.
3. Locate Visual Studio 2022 and CUDA 12.8; check free space.
4. Stage a separate full Python installation under
   `.setup/mamma-staging/<generation>/`.
5. Download locked packages and upstream source archives into
   `.setup/mamma-downloads/`, checking sizes and SHA-256. Install the exact
   artifacts without resolving alternative dependencies. Build the few locked
   source packages as specified by the installer.
6. Extract the pinned MAMMA, SAM 2 and SDF sources. Apply the Windows adapters to
   fresh source exactly once, copy the landmark helpers, and place MAMMA in
   the staged `engine/` directory.
7. Generate `build-native.cmd`, initialize `vcvars64.bat`, select the actual
   GPU capability, and build SAM 2 and SDF against the staged Python/PyTorch.
8. Prune the installer-defined unneeded build artifacts; run the full verifier.
9. Move the verified generation to `runtime_mamma/<generation>/` and verify it
   again from its final location before registering it.
10. Write the readiness/installation reports and atomically replace
    `.setup/mamma-runtime.json`. Other model runtimes and user data are retained.

Opaque phases report periodic activity with elapsed time, not a fabricated
percentage. Logs include `.setup/logs/mamma-python.log`,
`mamma-packages.log`, `mamma-native.log` and `mamma-verify.log`.
Source-package steps have their own `mamma-<package>.log` files.

Reopen the app and select **MAMMA > Model Manager**. The managed runtime should
show **Ready**. Model readiness is separate; follow [the model checklist](MAMMA.md#prepare-model-files).
The runtime readiness label alone does not test the complete inference pipeline.

## 5. Independently built environment: advanced route

Use this route only to register an already prepared, independently verified
environment or to develop native changes. There is no GUI `.pyd` import action.
Build in a **new private directory**, never on top of the active generation.

### Prepare the complete environment and source

1. Prepare a separate Python environment with **all** packages and versions in
   `installer/mamma-runtime-lock.json`. The supported automatic setup is the
   reference for fetching/installing these artifacts. The native build commands
   below do not create that full environment or install missing dependencies.
   Do not use a bare Python environment with only torch installed.
2. Acquire fresh source from the exact lock entries `sources.mamma`,
   `sources.sam2` and `sources.sdf`. Check each archive's bytes/SHA-256 before
   extraction; retain upstream licenses. Do not fetch a moving `main` branch.
   If writing a preparation helper, use the installer's validated download and
   `extract_source` functions rather than an unchecked ZIP extraction routine.
3. Apply `adapt_engine(engine_path, helpers_path)` and `adapt_sdf(sdf_path)` from
   `installer/mamma_adapters.py`. `helpers_path` is the current checkout's
   `installer/mamma_helpers`. SAM 2 uses the pinned source without this SDF patch.
   These adapters assume **fresh upstream source**; do not apply them twice.

The SDF adapter addresses MSVC alignment, Windows integer width (`long` versus
`int64_t`), the PyTorch scalar-type API and preprocessor/dispatch compatibility.
The engine adapter handles Windows FFmpeg/EGL/multiprocessing behavior,
CPU checkpoint loading, exact-ROI helpers and validated fitting diagnostics.
Read the actual patch before changing it. Do not silently remove algorithms,
switch precision or reduce fitting iterations to make a build pass.

Example adapter invocation, after obtaining fresh pinned source and setting the
real paths in PowerShell:

```powershell
$mammaApp = (Resolve-Path '.').Path
$mammaEngine = 'D:\MAMMA-build\engine'
$mammaSdfSource = 'D:\MAMMA-build\sources\sdf'
& "$mammaApp\runtime\python.exe" -B -c "from installer.mamma_adapters import adapt_engine,adapt_sdf; import sys; adapt_engine(sys.argv[1],sys.argv[2]); adapt_sdf(sys.argv[3])" $mammaEngine "$mammaApp\installer\mamma_helpers" $mammaSdfSource
if ($LASTEXITCODE -ne 0) { throw 'Adapter failed; do not continue with partially patched source.' }
```

### Compile the two native packages

Use a **Visual Studio 2022 x64 Native Tools Command Prompt**, then start
PowerShell inside it so the compiler environment is inherited. Alternatively,
use a generated CMD that calls the discovered `vcvars64.bat`, as the installer
does. `cl.exe` in an unrelated terminal is not enough evidence of a complete SDK.

The following PowerShell example assumes the fully prepared private Python and
fresh adapted sources above. Adjust the example paths. Fail after either build;
do not proceed because only the second command succeeded.

```powershell
$mammaApp = (Resolve-Path '.').Path
$mammaPython = 'D:\MAMMA-build\runtime\python.exe'
$mammaSamSource = 'D:\MAMMA-build\sources\sam2'
$mammaSdfSource = 'D:\MAMMA-build\sources\sdf'
$mammaSite = (& $mammaPython -I -c "import sysconfig; print(sysconfig.get_path('purelib'))").Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot locate the target site-packages.' }
$mammaArch = (& $mammaPython -I -c "import torch; assert torch.cuda.is_available(); print('.'.join(map(str,torch.cuda.get_device_capability())))").Trim()
if ($LASTEXITCODE -ne 0 -or $mammaArch -notmatch '^\d+\.\d+$') { throw 'Cannot determine the GPU architecture.' }

$env:CUDA_HOME = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:TORCH_CUDA_ARCH_LIST = $mammaArch
$env:MAX_JOBS = '2'
$env:DISTUTILS_USE_SDK = '1'
$env:MSSdk = '1'
$env:FORCE_CUDA = '1'
$env:SAM2_BUILD_ALLOW_ERRORS = '0'
$env:SAM2_BUILD_CUDA = '1'

foreach ($mammaSource in @($mammaSamSource, $mammaSdfSource)) {
    & "$mammaApp\.setup\bootstrap-git\uv.exe" pip install --python $mammaPython --target $mammaSite --no-deps --no-build-isolation --no-index --offline --no-cache --link-mode copy $mammaSource
    if ($LASTEXITCODE -ne 0) { throw "Native build failed: $mammaSource" }
}
```

All build dependencies must already exist for the offline commands. This limits
compiler concurrency to two jobs, as in the installer, instead of using every
CPU core and exhausting RAM. Capture the output in a private log. Never enable
SAM 2's allow-errors fallback and report a successful CUDA build afterward.
Do not force unsupported compilers or copy unrelated DLLs into Windows folders.

If a candidate already contains older SAM/SDF packages, create a fresh candidate
or reinstall those distributions cleanly within that disposable candidate first.
Do not leave stale `.pyd` files beside a new build or overwrite an active runtime.

### Verify before registration

From the app root, with the private interpreter:

```powershell
& $mammaPython -I -X utf8 -B installer\verify_mamma.py --lock installer\mamma-runtime-lock.json --output 'D:\MAMMA-build\verification.json'
if ($LASTEXITCODE -ne 0) { throw 'Verification failed; do not register this environment.' }
```

The output directory must exist. This command needs no model weights. It checks:

- Exact Python and every locked distribution version.
- Real SDF CUDA BVH distances, closest triangle indices and point gradients.
- Inside/outside winding calculations on a known mesh.
- SAM 2 CUDA connected-component areas on a known mask.
- Distribution RECORD SHA-256 hashes, excluding the explicitly skipped
  developer/debug artifacts in the verifier.

Current numeric limits: squared-distance/gradient error no more than `2e-6`;
winding error no more than `1e-4`; a known four-pixel component must have area
four. Do not loosen limits to accept a bad build. Import success alone does not
detect wrong index width, wrong kernels or unsupported GPU code.

Save GPU capability, driver/toolkit/compiler versions, source commits, build
arguments, native binary hashes and the verification JSON privately. If moving
the candidate to a final location, repeat verification at that location.

### Register and check the app

Close the app first and preserve the previous configuration as described above.
Only after the full verifier succeeds:

```powershell
& .\runtime\python.exe -B tools\configure_mamma.py --python $mammaPython --engine $mammaEngine --reviewed-mamma-license
if ($LASTEXITCODE -ne 0) { throw 'Registration failed; preserve the previous runtime.' }
```

This writes a **version 2** external configuration. Model Manager says
**Configured separately**, which is expected; the automatic installer writes a
**version 3** managed configuration instead. The command requires the adapted
engine and helper files, checks selected package versions/imports and CUDA
availability, then records the engine source fingerprint. It **does not run the
full native verifier** and does not replace that step.

External configurations do not have all the managed runtime identity and GPU
architecture checks. Their owner must revalidate after GPU, dependency or native
module changes. Do not edit JSON to fake version 3, `managed_id`, verification
results or package versions. Do not relocate the configured paths afterward.

Native binaries are not fully represented in the engine-source fingerprint.
An existing inference cache can therefore conceal a manual native change, even
after registering a different interpreter path with identical package versions.
For a changed-module inference test, use a **new separate test project** and
confirm that stages actually run, rather than deleting the user's cached work.

## 6. End-to-end acceptance and evidence

Use legitimately obtained local videos/models, in a separate private project.
Do not upload registered weights or copy them into the public test source.

1. Reopen `RGBMoCap.cmd`; inspect runtime readiness and required model rows.
2. Select MAMMA and two synchronized fixed-camera videos, initially trimming to
   about one second (at least four frames), with clearly visible people.
3. Use VGGT, or valid manual calibration when deliberately testing that path.
   Run Reconstruct 3D and record the actual masks, landmarks, cameras, fitting
   and final correction/preview stages. A cache hit is not a new inference test.
4. Confirm the expected people, finite poses and camera/video alignment. Review
   original and corrected motion; inspect a few contact boundaries.
5. Verify save/reopen, shared trim and thumbnail switching. If claiming export
   compatibility, export the selected result and actually import it in Blender.
   Retargeting requires its own test; do not infer it from a successful body fit.
6. Record per-stage wall time, peak allocated/reserved PyTorch VRAM and process
   RAM when available. Distinguish these from whole-GPU/whole-system usage.
   Keep diagnostic logs and generated artifacts outside publishable files.

The app has no fixed camera-count maximum, but still requires at least two
views, 4–1,800 common frames and no more than eight people. Start small for
runtime acceptance, then measure the user's intended workload. Do not restore
the old four-view/64-entry ceilings or replace them with an arbitrary GPU-based
camera cap. Keep the existing RAM/Windows allocation-headroom guard, CUDA
allocator budget and allocation-error reporting. The 65-input synthetic
preparation/save/restore tests do not validate 65-view masks, VGGT or fitting.

Only report checks actually performed. Current evidence is one Windows 11 /
RTX 5090 PC; it is not a clean-Windows, other-GPU or full 60-second validation.
See [VALIDATION.md](VALIDATION.md) for recorded measurements.

## 7. Failure diagnosis, retry and rollback

| Symptom | Check and action |
| --- | --- |
| App/installer lock error | Close the relevant app/setup process normally. Do not delete a live lock or terminate unrelated processes. |
| `nvcc`/Visual Studio not found | Check the full CUDA 12.8 toolkit and VS 2022 x64 C++ workload/SDK; the driver and another VS major version are insufficient. |
| Package/source hash mismatch | Stop. Inspect the URL, partial download and exact lock entry. Do not disable hash checks or accept a different checkpoint/source. |
| Adapter cannot find expected text/API | Source is wrong, already patched or changed. Start from a fresh exact pinned source copy; do not partially improvise the patch. |
| Compiler fails or runs out of RAM | Inspect `mamma-native.log`; keep the two-job bound and close competing work. Check the exact compiler/toolkit, not model files. |
| DLL load failure / native import failure | Confirm target Python/torch/CUDA and x64 architecture, package metadata and dependent DLLs. Rebuild the candidate; do not copy a random `.pyd`. |
| No kernel image / invalid device function | Verify actual GPU capability and build architecture; rebuild and run CUDA tests on the target GPU. |
| SDF/component verification fails | Reject the candidate. Check MSVC index-width patches, source versions and actual native module loaded. Do not relax numeric limits. |
| Engine fingerprint mismatch | Engine code changed since registration. Revalidate a new candidate, then register it; do not edit fingerprints manually. |
| Model Manager Ready but missing model | Runtime and weights are separate. Follow the supported hash-checked model import instructions. |

Automatic setup publishes its pointer only after final verification. On failure,
the previous active configuration is retained and the owned failed staging area
is cleaned. Verified downloads remain for retry. Save the log, fix the actual
prerequisite and rerun `Setup_MAMMA.cmd`; do not remove all of `.setup`.

For a failed **manual** replacement, close the app, verify that the previous
environment still exists, and restore only its backed-up
`.setup/mamma-runtime.json`. Do not restore a pointer to a removed folder.
Keep failed candidates/logs privately until diagnosed; remove a candidate only
after checking its resolved absolute path and confirming it is not active.

Running `Setup_MAMMA.cmd` later may select/register a managed generation again;
it is not a promise to maintain or preserve custom changes inside an external
environment. Treat a custom build and the automatic environment as separate.

## 8. Publication and maintenance rules

- Commit original setup/adapters/docs only. Restricted MAMMA/SDF source, native
  builds, private environments, weights, body data, logs and videos stay local.
  Preserve all upstream license notices; do not assume a local build authorizes
  redistribution of binaries or multi-architecture wheels.
- Review `git status` and `git ls-files`; ignore rules cannot protect a file
  already tracked or force-added. Keep private experiments outside the checkout.
- A new source/dependency/native policy needs lock/adapters review and new kernel
  plus inference evidence. Do not merge it into another model runtime to save space.
- If setup, adapters or locks change, refresh the source manifest and relevant
  installation/validation docs. A docs-only change does not require recompilation.
- Give the user a short result: what was installed, actual validation performed,
  runtime readiness, storage used, limitations and any remaining action. Never
  publish provider credentials or claim untested GPU support.
