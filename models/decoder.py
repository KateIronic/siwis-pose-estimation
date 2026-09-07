"""
Decoder head: right-hand columns of Fig. 13 ("Mask Segmentation" and
"Pose Estimation"). Both heads share the identical topology described
in §3.2.2; they are separate parameter instances because they are
supervised by different labels (mask vs. keypoint -- the two
"Supervise" arrows in Fig. 13) and emit a different number of output
channels (1 mask channel vs. keypoint heatmap + tag channels).
"""

from __future__ import annotations

import config
from models.blocks import ConvUpsampleBlock, CrossAttentionBlock
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = config.EMBED_DIM_C,
        pixel_h: int = config.PIXEL_EMBED_H,
        pixel_w: int = config.PIXEL_EMBED_W,
        out_h: int = config.HEATMAP_H,
        out_w: int = config.HEATMAP_W,
        out_channels: int = 1,
        num_blocks: int = config.NUM_CROSS_ATTN_BLOCKS,
        n_heads: int = config.ATTN_HEADS,
    ):
        super().__init__()
        self.pixel_h, self.pixel_w = pixel_h, pixel_w

        # "we initialize h x w trainable pixel embeddings" (§3.2.2)
        self.pixel_embedding = nn.Parameter(torch.randn(1, pixel_h * pixel_w, embed_dim) * 0.02)

        # "N x Cross-Attention Block" (Fig. 13)
        self.cross_attn_blocks = nn.ModuleList(
            [CrossAttentionBlock(embed_dim, n_heads) for _ in range(num_blocks)]
        )

        # "Conv Layer -> Up Sample -> Conv Layer -> Up Sample" (Fig. 13),
        # taking the feature map from h x w to h' x w' (= out_h x out_w),
        # "aligning with the dimensions of the ground truth heatmap
        # features" (§3.2.2). Two upsample stages, matching the
        # figure's two Conv+UpSample pairs.
        stage_scale_h = round((out_h / pixel_h) ** 0.5)
        self.up1 = ConvUpsampleBlock(embed_dim, scale_factor=stage_scale_h)
        self.up2 = ConvUpsampleBlock(embed_dim, scale_factor=stage_scale_h)
        self.out_conv = nn.Conv2d(embed_dim, out_channels, kernel_size=1)
        self.out_h, self.out_w = out_h, out_w

    def forward(self, signal_feature):  # signal_feature: (B, embed_dim)
        B = signal_feature.shape[0]
        pixel_embed = self.pixel_embedding.expand(B, -1, -1)  # (B, h*w, dim)
        signal_feature = signal_feature.unsqueeze(1)            # (B, 1, dim)

        x = pixel_embed
        for block in self.cross_attn_blocks:
            x = block(x, signal_feature)

        x = x.transpose(1, 2).reshape(B, -1, self.pixel_h, self.pixel_w)  # (B, dim, h, w)
        x = self.up1(x)
        x = self.up2(x)
        x = self.out_conv(x)

        # exact target resolution (48 x 64, §4.1.2), in case the
        # integer upsample factors don't land on it precisely
        if x.shape[-2:] != (self.out_h, self.out_w):
            x = F.interpolate(x, size=(self.out_h, self.out_w), mode="bilinear", align_corners=False)
        return x