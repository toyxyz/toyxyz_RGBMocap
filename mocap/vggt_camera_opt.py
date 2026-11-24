"""Lower-memory VGGT-Omega inference for the camera-only workflows."""
from functools import lru_cache
from types import MethodType


DEFAULT_ENABLED = True
LABEL = 'VGGT-Ω OPT'
HELP = ('Use less GPU memory for VGGT-Omega camera estimation. Uses PyTorch SDPA and BF16 '
        'on supported GPUs. Turn off to use the standard path. Applies on Reconstruct 3D.')
FRAME_BATCH = 8


def enabled(options):
    value = options.get('vggt_opt', DEFAULT_ENABLED)
    if type(value) is not bool:
        raise ValueError('Invalid VGGT-Ω OPT setting; expected on or off.')
    return value


def available(device='cuda'):
    import torch

    target = torch.device(device)
    if target.type != 'cuda' or not torch.cuda.is_available():
        return False
    with torch.cuda.device(target):
        return torch.cuda.is_bf16_supported(including_emulation=False)


def select(requested, samples, emit):
    optimize = requested and available()
    if optimize:
        emit('info', message='VGGT-Ω OPT: BF16 weights and PyTorch SDPA.')
        return True, 4. + .065 * samples
    if requested:
        emit('info', message='VGGT-Ω OPT requires native CUDA BF16. Using standard VGGT-Omega.')
    return False, 6. + .1 * samples


def configure(model):
    """Prepare CPU weights before upload or installation of block offload."""
    import torch
    from torch import nn

    if model.dense_head is not None or model.camera_head is None or not model.autocast:
        raise ValueError('VGGT-Ω OPT requires the camera-only model with autocast.')
    aggregator = model.aggregator
    for layer in aggregator.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            layer.to(dtype=torch.bfloat16)
    model.requires_grad_(False)
    patch_rope = aggregator.patch_embed.rope_embed
    patch_rope.forward = lru_cache(maxsize=1)(patch_rope.forward)
    aggregator.forward = MethodType(_camera_tokens, aggregator)


def _camera_tokens(self, images):
    import torch
    from vggt_omega.models.aggregator import slice_expand_and_flatten

    batch, frames, channels, height, width = images.shape
    pixels = images.reshape(batch * frames, channels, height, width)
    mean = self._resnet_mean.reshape(1, channels, 1, 1)
    std = self._resnet_std.reshape(1, channels, 1, 1)
    patches = torch.cat([self.patch_embed((part - mean) / std)['x_norm_patchtokens']
                         for part in pixels.split(FRAME_BATCH)])
    tokens = torch.cat((slice_expand_and_flatten(self.camera_token, batch, frames),
                        slice_expand_and_flatten(self.register_token, batch, frames), patches), dim=1)
    del patches
    count, width_per_token = tokens.shape[-2:]
    rope = self.rope_embed(H=height // self.patch_size, W=width // self.patch_size)
    start = self.patch_token_start
    for index, block in enumerate(self.frame_blocks):
        local = torch.empty_like(tokens)
        for offset in range(0, len(tokens), FRAME_BATCH):
            span = slice(offset, offset + FRAME_BATCH)
            local[span] = self._run_block(block, tokens[span], rope)
        kind = self.inter_frame_attention_types[index]
        if kind == 'register':
            separated = local.reshape(batch, frames, count, width_per_token)
            registers = separated[:, :, :start].reshape(batch, frames * start, width_per_token)
            registers = self._run_block(self.inter_frame_blocks[index], registers, None)
            tokens = torch.cat((registers.reshape(batch, frames, start, width_per_token),
                                separated[:, :, start:]), dim=2)
            del registers, separated
        else:
            tokens = self._run_inter_frame_attention_block(
                local, batch, frames, count, width_per_token, index, kind)
        tokens = tokens.reshape(batch * frames, count, width_per_token)

    # The stock camera head only reads these tokens from the final layer.
    local = local[:, :start].reshape(batch, frames, start, width_per_token)
    global_tokens = tokens[:, :start].reshape(batch, frames, start, width_per_token)
    return [torch.cat((local.bfloat16(), global_tokens.bfloat16()), dim=-1)], start
