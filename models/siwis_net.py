"""
Full dual-branch DNN (Fig. 13, signal-processing side): one shared
SignalEncoder feeding two DecoderHead instances, one for mask
segmentation and one for pose estimation, matching the two parallel
decoder columns and "Supervise" arrows in the figure.
"""

from __future__ import annotations

import config
from models.decoder import DecoderHead
from models.signal_encoder import SignalEncoder
import torch.nn as nn


class SiWiSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.signal_encoder = SignalEncoder()
        self.mask_head = DecoderHead(out_channels=config.NUM_MASK_CHANNELS)
        # +1 output channel per keypoint reserved for the associative-
        # embedding "tag" used by L_group (§3.2.3, ref [33]); see
        # losses/losses.py for how the tag channels are consumed.
        self.pose_head = DecoderHead(out_channels=config.NUM_KEYPOINTS * 2)

    def forward(self, signal):
        feat = self.signal_encoder(signal)   # (B, embed_dim)
        mask_pred = self.mask_head(feat)       # (B, 1, H, W)
        pose_out = self.pose_head(feat)         # (B, 2*K, H, W)

        num_kp = config.NUM_KEYPOINTS
        heatmaps = pose_out[:, :num_kp]          # (B, K, H, W)
        tags = pose_out[:, num_kp:]               # (B, K, H, W), associative-embedding tags

        return {"mask": mask_pred, "pose_heatmaps": heatmaps, "pose_tags": tags}