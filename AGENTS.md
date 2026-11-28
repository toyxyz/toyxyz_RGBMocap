# Agent guide for RGB MoCap

Read [docs/AGENT_GUIDE.md](docs/AGENT_GUIDE.md) before operating or changing this project. It covers installation, model selection, GUI workflows, retargeting, projects, resources, data protection and the main code paths. Consult the linked user and developer guides for detail.

For model-specific files, defaults and workflow, use [docs/models/README.md](docs/models/README.md).

For MAMMA runtime installation, native SAM 2/SDF builds or repair, follow
[docs/MAMMA_AGENT_SETUP.md](docs/MAMMA_AGENT_SETUP.md). It distinguishes automatic
setup from custom builds and requires actual CUDA verification before registering
a manually built environment. For optional PromptHMR setup, its SAM2 build and
the optional DroidCalib/LieTorch moving-camera build, follow
[docs/models/PROMPTHMR.md](docs/models/PROMPTHMR.md).

- This is the Windows Git/CMD distribution. Preserve the separate pinned runtimes and model-specific behavior. `install.cmd` covers ExMR/FlowHMR/GVHMR and the small shared-runtime camera support overlay used by MAMMA gravity; MAMMA, PromptHMR and GEM-X use separate setup commands. ViDiHand2 setup fetches pinned source with Git and reuses FlowHMR. See the [installation routes](docs/AGENT_GUIDE.md#installation-routes) before repairing or changing any of these environments.
- For Windows setup and update failures, follow [installation troubleshooting](docs/USER_GUIDE.md#installation-verification-and-repair), [long path recovery](docs/USER_GUIDE.md#windows-path-length-during-installation) or [ViDiHand2 source setup](docs/FIRST_PERSON_ARMS.md#local-runtime-setup) as applicable. Use the platform branch's own instructions for [Linux](https://github.com/toyxyz/toyxyz_RGBMocap/blob/linux_support/docs/LINUX.md) and [macOS](https://github.com/toyxyz/toyxyz_RGBMocap/blob/macos_support/docs/MACOS.md).
- GEM-X uses its own verified inference packages over ExMR Python. Keep Native77 data separate from SMPL, including contact correction and FBX/BVH export. Follow [the GEM-X guide](docs/models/GEMX.md).
- For real-time tracking, use [RGB MoCap Live](https://github.com/toyxyz/rgbmocap_live). See [real-time tracking and updates](docs/AGENT_GUIDE.md#real-time-tracking-and-updates) for legacy data protection and MAMMA's GeoCalib dependency.
- Treat local `models/`, `projects/`, `cache/`, `character_scenes/`, runtime folders, external project folders and registered model data as user-owned. Do not delete, overwrite, upload or migrate them without appropriate user authorization. Ignore rules are not backups.
- Preserve inference and selected-result behavior when working on preview, contacts or export. Review the relevant code and validation records before changing numerical processing. For Manny/Quinn FBX, follow the [UE5 Mannequin export guide](docs/AGENT_GUIDE.md#ue5-mannequin-fbx-export-implementation).
- Resource Safety defaults on below Resource Mode. Its local `config/local.json` preference controls measured low-RAM/commit stopping and the PyTorch VRAM allocator cap. Off must retain adaptive routing, actual OOM recovery, stall detection and generation protection. Do not save it in projects or inference identities; see `docs/PERFORMANCE.md#resource-safety`.
- Follow the user's request and perform checks appropriate to the change. Do not claim an inference, import, export, GPU, playback or clean-install test that was not actually run.
