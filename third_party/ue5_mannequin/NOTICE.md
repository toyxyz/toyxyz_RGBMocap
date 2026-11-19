# Epic Games UE5 mannequin asset notice

Manny and Quinn mesh and rig data are derived from Epic Games' Unreal Engine
5.7.4 Third Person template. The [Unreal Engine EULA](https://www.unrealengine.com/eula/unreal)
governs the Epic content. Sections 1 and 5(b) define installation-template
Examples and permit distribution of Examples, including modified Examples.
Epic retains its rights in the template assets. The repository's MIT license
covers the original RGB MoCap code and documentation.

Unreal® is a trademark or registered trademark of Epic Games, Inc. in the
United States of America and elsewhere.
Unreal® Engine, Copyright 1998 – 2026, Epic Games, Inc. All rights reserved.

## Source and conversion

The installed source is under
`Templates/TemplateResources/High/Characters/Content/Mannequins/Meshes`.
The table records the SHA-256 of each installed source file used to produce
this bundle.

| Bundle file | Installed source | Source SHA-256 |
| --- | --- | --- |
| `manny.npz` | `SKM_Manny_Simple.uasset` | `be9f011191ee1887b97cd13bce09886a6066c2fd62c619991070ec142184e1bf` |
| `quinn.npz` | `SKM_Quinn_Simple.uasset` | `1ef45d92507a76d74a5e06737f94cad40eb7d5f5769dfbb63f7f74e784a178c6` |
| `reference.json` | Both skeletal meshes above | See both hashes above |

The NPZ files contain base mesh triangles, UV coordinates, normals and skin
weights. The JSON records reference bone names, parents and transforms.
The extraction uses only these two installation-template meshes. Materials,
textures, animations, Blueprint assets, physics, LOD variants and Unreal
binaries are excluded. The export preserves an Epic source and terms notice
in the FBX custom properties. Those properties describe the Epic-derived
parts of the FBX. The rights in captured motion remain with their respective
owners.

To reproduce the files, generate the profiles with
[`read_ue5_mannequin.py`](../../tools/read_ue5_mannequin.py) inside a temporary
Unreal project containing the matching installed template assets. Export both
skeletal meshes at base LOD with Unreal's Skeletal Mesh FBX exporter. Run
[`build_ue5_mannequin_bundle.py`](../../tools/build_ue5_mannequin_bundle.py)
in Blender with the profiles and the two FBXs. The builder checks bone names,
parents, reference joint positions and skin weights against the expected data.
