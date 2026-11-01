# Agent guide for RGB MoCap

Read [docs/AGENT_GUIDE.md](docs/AGENT_GUIDE.md) before operating or changing this project. It covers installation, model selection, GUI workflows, retargeting, projects, resources, data protection and the main code paths. Consult the linked user and developer guides for detail.

For model-specific files, defaults and workflow, use [docs/models/README.md](docs/models/README.md).

For MAMMA runtime installation, native SAM 2/SDF builds or repair, follow
[docs/MAMMA_AGENT_SETUP.md](docs/MAMMA_AGENT_SETUP.md). It distinguishes automatic
setup from custom builds and requires actual CUDA verification before registering
a manually built environment. For optional PromptHMR setup and its separate SAM2
build, follow [docs/models/PROMPTHMR.md](docs/models/PROMPTHMR.md).

- This is the Windows Git/CMD distribution. Preserve the separate pinned runtimes and model-specific behavior. `install.cmd` covers ExMR/FlowHMR/GVHMR and the small shared-runtime camera support overlay used by PEAR and MAMMA gravity; MAMMA and PromptHMR remain optional through their separate setup commands. Do not merge their packages or assume `update.cmd` installs them.
- PEAR is a transient single-person bone preview. Selecting it must not start models or modify an offline project. Start/Stop controls the resident worker; see `docs/models/PEAR.md` for fixed bones, Reset Bones, One Euro filtering and World Position. Keep inference and capture off the GUI thread, use bounded latest-frame queues and preserve offline state on exit.
- Treat local `models/`, `projects/`, `cache/`, `character_scenes/`, runtime folders, external project folders and registered model data as user-owned. Do not delete, overwrite, upload or migrate them without appropriate user authorization. Ignore rules are not backups.
- Preserve inference and selected-result behavior when working on preview, contacts or export. Review the relevant code and validation records before changing numerical processing.
- Follow the user's request and perform checks appropriate to the change. Do not claim an inference, import, export, GPU, playback or clean-install test that was not actually run.
