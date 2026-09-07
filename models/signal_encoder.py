"""
Signal encoder: left column of Fig. 13 ("Signal Processing" /
"Signal Encoder"). Bottom-to-top per the figure: Conv Layer -> Max
Pool -> [N x Self-Attention Block] -> Conv Layer -> Max Pool -> Linear
Layer, taking the raw per-antenna sensing signal as input and
producing the "C x 1 x 1" feature vector referenced in §3.2.2.
"""

from __future__ import annotations

import config
from models.blocks import SelfAttentionBlock
import torch.nn as nn


class SignalEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = config.SIGNAL_IN_CHANNELS,
        conv_channels: int = config.CONV_CHANNELS,
        embed_dim: int = config.EMBED_DIM_C,
        num_blocks: int = config.NUM_SELF_ATTN_BLOCKS,
        reduction: int = config.BOTTLENECK_REDUCTION_X,
        n_heads: int = config.ATTN_HEADS,
    ):
        super().__init__()

        # "we first apply convolution layers to extract temporal
        # features" (§3.2.1) -- first Conv Layer -> Max Pool pair (Fig. 13)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.input_proj = nn.Linear(conv_channels, embed_dim)

        # "N x Self-Attention Block" (Fig. 13)
        self.self_attn_blocks = nn.ModuleList(
            [SelfAttentionBlock(embed_dim, reduction, n_heads) for _ in range(num_blocks)]
        )

        # Second Conv Layer -> Max Pool pair (Fig. 13, after the self-attention stack)
        self.conv2 = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )
        # Collapses the time axis -> the "C x 1 x 1" feature (§3.2.2)
        self.pool2 = nn.AdaptiveMaxPool1d(1)

        # "Linear Layer" (Fig. 13, top of the signal encoder)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):  # x: (B, in_channels, T)
        x = self.pool1(self.conv1(x))  # (B, conv_channels, T/2)
        x = x.transpose(1, 2)           # (B, T/2, conv_channels)
        x = self.input_proj(x)          # (B, T/2, embed_dim)
        for block in self.self_attn_blocks:
            x = block(x)
        x = x.transpose(1, 2)            # (B, embed_dim, T/2)
        x = self.conv2(x)
        x = self.pool2(x).squeeze(-1)     # (B, embed_dim) == "C x 1 x 1"
        x = self.linear(x)
        return x  # (B, embed_dim)