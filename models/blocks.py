"""
Reusable blocks matching Fig. 13's boxes:
 - SelfAttentionBlock  (signal encoder, §3.2.1)
 - CrossAttentionBlock (mask/pose decoders, §3.2.2)
 - ResNetBottleneckConv / ConvUpsampleBlock ("Conv Layer -> Up Sample", §3.2.2, ref [16])

The paper specifies the *topology* of these blocks (see the figure and
the prose cited in each docstring) but not layer widths, block counts,
or head counts; those are read from config.py and are explicitly NOT
sourced from the paper.
"""

from __future__ import annotations

import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.GELU(),
            nn.Linear(dim * hidden_mult, dim),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttentionBlock(nn.Module):
    """
    §3.2.1: "we employ self-attention blocks ... modify the
    self-attention block using a bottleneck design. Preceding and
    following the self-attention block, we integrate two fully
    connected layers for dimension reduction and subsequent expansion
    ... employ parameter-free identity shortcuts to connect the
    inputs and outputs of the module. With a reduction scale of X ...
    the module's parameter amount is also reduced by a factor of X."

    Implements: identity_shortcut( FC_reduce -> MHA -> Add&Norm ->
    FeedForward -> Add&Norm -> FC_expand )(x), matching Fig. 13's
    "Multi-Head Attention / Add & Norm / Feed Forward / Add & Norm"
    stack inside "N x Self-Attention Block", wrapped by the
    dimension-reduction/expansion FC pair and identity shortcut
    described in the text.
    """

    def __init__(self, dim: int, reduction: int, n_heads: int):
        super().__init__()
        bottleneck_dim = dim // reduction
        self.fc_reduce = nn.Linear(dim, bottleneck_dim)
        self.mha = nn.MultiheadAttention(bottleneck_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(bottleneck_dim)
        self.ff = FeedForward(bottleneck_dim)
        self.norm2 = nn.LayerNorm(bottleneck_dim)
        self.fc_expand = nn.Linear(bottleneck_dim, dim)

    def forward(self, x):  # x: (B, T, dim)
        identity = x
        z = self.fc_reduce(x)
        attn_out, _ = self.mha(z, z, z)
        z = self.norm1(z + attn_out)
        z = self.norm2(z + self.ff(z))
        z = self.fc_expand(z)
        return identity + z  # parameter-free identity shortcut


class CrossAttentionBlock(nn.Module):
    """
    §3.2.2 + Fig. 13 decoder stack: "we initialize h x w trainable
    pixel embeddings and employ cross-attention to compute the
    attention weights of each pixel embedding towards each sensing
    signal feature." The figure shows, per block: Multi-Head
    Attention -> Add & Norm -> Multi-Head Attention -> Add & Norm ->
    Feed Forward -> Add & Norm, i.e. a self-attention pass over the
    pixel embeddings followed by a cross-attention pass against the
    signal-encoder feature.
    """

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.self_mha = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.cross_mha = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, pixel_embed, signal_feature):
        # pixel_embed: (B, h*w, dim); signal_feature: (B, 1, dim)
        x = self.norm1(pixel_embed + self.self_mha(pixel_embed, pixel_embed, pixel_embed)[0])
        x = self.norm2(x + self.cross_mha(x, signal_feature, signal_feature)[0])
        x = self.norm3(x + self.ff(x))
        return x


class ResNetBottleneckConv(nn.Module):
    """
    §3.2.2: "our convolutional layer uses a bottleneck residual block
    from ResNet [16], which utilizes 1x1 convolutions to reduce
    feature channels and consequently decrease the module's
    parameters." Standard He et al. 2016 bottleneck: 1x1 reduce ->
    3x3 -> 1x1 expand, with a residual shortcut.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = channels // reduction
        self.block = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class ConvUpsampleBlock(nn.Module):
    """One "Conv Layer -> Up Sample" pair from Fig. 13's decoder stack."""

    def __init__(self, channels: int, scale_factor: int = 2):
        super().__init__()
        self.conv = ResNetBottleneckConv(channels)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False)

    def forward(self, x):
        return self.upsample(self.conv(x))