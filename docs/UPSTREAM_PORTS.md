# Selected upstream ports

## VGGT camera checkpoint loading (2026-10-30)

Adapted [toyxyz commit c0d5f30](https://github.com/toyxyz/toyxyz_RGBMocap/commit/c0d5f30b89481bd3b6979f773ffbfe350ada04f3)
to the existing FlowHMR camera loader. Automatic camera-only loading initializes
on meta and assigns checkpoint tensors directly. Low Memory and generic depth
loading retain copy loading. Missing weights and remaining meta tensors fail
before device transfer. Cache keys distinguish the loading policy.

This is a selective port of that commit, not all intervening upstream releases.
It adds no MAMMA support, block offloading, packages or Git remote. Upstream
performance measurements do not establish performance on this installation.
The modified camera adapter hash invalidates old camera-stage fingerprints on
the next processing request; saved results remain on disk.

Local focused checks used a small Torch module and checkpoint to verify exact
assigned values, policy-specific caching, Low Memory and generic-depth routing,
missing-weight rejection and nonpersistent meta-buffer rejection. No real VGGT
inference or speed benchmark was run locally for this port.


## MAMMA multi-camera integration (2026-11-01)

Selectively integrated upstream e530eba9d5 plus current MAMMA camera/pipeline/UI
code and its memory_mode, gpu_offload and model_session dependencies from
ab80ced732. No upstream remote was added to this checkout. ExMR API changes,
PEAR and PromptHMR were not included. Custom source export, camera preview,
background refinement, application branding and README samples were retained.
The optional MAMMA engine and weights remain separate, provider-licensed data.


## Full upstream integration (2026-11-01)

Integrated all upstream source changes through ab80ced73287eb30330b52cce7cf521877c66826,
including PromptHMR, experimental PEAR, ExMR public API/toe IK, batch processing,
resource routing and contact improvements. Earlier selective-port exclusions above
are historical. Custom export, camera, README media and attribution are retained.
Optional runtime/model readiness is separate from source integration.
