"""The official ViDiHand2 checkpoint pair and the shared right-hand MANO model."""

HUB = "https://huggingface.co/WANG-XIXI/ViDiHand"
REVISION = "5117c7a34939cd6a2af0da12859237a38ede4276"

CHECKPOINTS = {
    "vidihand2_vdm": "vidihand_v2/vdm.safetensors",
    "vidihand2_projector": "vidihand_v2/projector.safetensors",
    "mano_right": "mano/MANO_RIGHT.pkl",
}
KEYS = tuple(CHECKPOINTS)


def asset(key, name, size, sha256):
    remote = CHECKPOINTS[key]
    filename = remote.rsplit("/", 1)[-1]
    return dict(key=key, name=name, relative="vidihand2/" + filename,
                url=f"{HUB}/tree/{REVISION}/vidihand_v2", minimum_bytes=size,
                download_url=f"{HUB}/resolve/{REVISION}/{remote}", expected_sha256=sha256)


ASSETS = (
    asset("vidihand2_vdm", "ViDiHand2 / Video Diffusion Backbone and VAE", 6074049720,
          "8767174a11122c669912c523bc2245d285065b83cc7cc7d88478515f7c788e3b"),
    asset("vidihand2_projector", "ViDiHand2 / Hand Decoder (Projector)", 41311156,
          "535c0f93b3f3c5b341f77537e7557c8763aee7e7bc0ec8ad7728a32f094ec534"),
)
