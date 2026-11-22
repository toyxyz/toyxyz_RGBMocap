"""RIFE 4.9 inference adapted from Fannovel16/ComfyUI-Frame-Interpolation.

Retains the pinned upstream model's parameter names and FP32 inference math.
See LICENSE and ../SOURCES.json.
"""
import torch
from torch import nn
from torch.nn import functional as F


def conv(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 2, 1), nn.LeakyReLU(0.2, True))


class ResConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, value):
        return self.relu(self.conv(value) * self.beta + value)


class IFBlock(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.conv0 = nn.Sequential(conv(in_channels, channels // 2), conv(channels // 2, channels))
        self.convblock = nn.Sequential(*(ResConv(channels) for _ in range(8)))
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(channels, 24, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, value, flow, scale):
        value = F.interpolate(value, scale_factor=1 / scale, mode='bilinear', align_corners=False)
        if flow is not None:
            smaller = F.interpolate(flow, scale_factor=1 / scale, mode='bilinear', align_corners=False)
            value = torch.cat((value, smaller / scale), 1)
        prediction = self.lastconv(self.convblock(self.conv0(value)))
        prediction = F.interpolate(prediction, scale_factor=scale, mode='bilinear', align_corners=False)
        return prediction[:, :4] * scale, prediction[:, 4:5]


class IFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(15, 192)
        self.block1 = IFBlock(20, 128)
        self.block2 = IFBlock(20, 96)
        self.block3 = IFBlock(20, 64)
        self.encode = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ConvTranspose2d(16, 4, 4, 2, 1))
        self._grid = None

    def warp(self, value, flow):
        _, _, height, width = flow.shape
        key = (height, width, flow.device, flow.dtype)
        if self._grid is None or self._grid[0] != key:
            x = torch.linspace(-1., 1., width, device=flow.device).view(1, width).expand(height, width)
            y = torch.linspace(-1., 1., height, device=flow.device).view(height, 1).expand(height, width)
            self._grid = key, torch.stack((x, y), 0).unsqueeze(0)
        scaled = torch.cat((flow[:, :1] / ((width - 1) / 2), flow[:, 1:] / ((height - 1) / 2)), 1)
        grid = (self._grid[1] + scaled).permute(0, 2, 3, 1).to(value.dtype)
        return F.grid_sample(value, grid, mode='bilinear', padding_mode='border', align_corners=True)

    def forward(self, left, right, phase):
        height, width = left.shape[-2:]
        padding = (0, -width % 64, 0, -height % 64)
        left = F.pad(left.clamp(0, 1), padding)
        right = F.pad(right.clamp(0, 1), padding)
        joined = torch.cat((left, right), 1)
        left, right = joined[:, :3], joined[:, 3:]
        phase = (joined[:, :1].clone() * 0 + 1) * phase
        f0, f1 = self.encode(left), self.encode(right)
        flow, mask = self.block0(torch.cat((left, right, f0, f1, phase), 1), None, 8)
        for block, scale in ((self.block1, 4), (self.block2, 2), (self.block3, 1)):
            features = (self.warp(left, flow[:, :2]), self.warp(right, flow[:, 2:]),
                        self.warp(f0, flow[:, :2]), self.warp(f1, flow[:, 2:]), phase, mask)
            correction, mask = block(torch.cat(features, 1), flow, scale)
            flow = flow + correction
        mask = mask.sigmoid()
        result = self.warp(left, flow[:, :2]) * mask + self.warp(right, flow[:, 2:]) * (1 - mask)
        return result[:, :, :height, :width]
