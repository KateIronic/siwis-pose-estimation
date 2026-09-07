"""
PyTorch Dataset that assembles SiWiS training samples:
 - input: a windowed signal-feature sequence S_k (Eqn. 11)
 - targets: the mask heatmap (Mask R-CNN) and keypoint heatmaps
   (HRNet) of the video frame whose timestamp ends the window,
   "aligning them by their timestamps" (§3.1).

Split policy: "we allocated the first 80% of the data for training
and the remaining 20% for testing" after sorting by timestamp (§4.2).
"""

from __future__ import annotations

import bisect

import numpy as np
import torch
from torch.utils.data import Dataset

import config


class SiWiSDataset(Dataset):
    def __init__(
        self,
        feature_seq: np.ndarray,
        frame_timestamps: np.ndarray,
        mask_labels: np.ndarray,
        keypoint_heatmaps: np.ndarray,
        keypoint_coords: np.ndarray,
        keypoint_visible: np.ndarray,
        signal_duration_sec: float = config.SIGNAL_DURATION_SEC,
    ):
        """
        feature_seq: (K_resampled, 1 + num_streams) real float64, from
            csi_pcap_loader.build_feature_sequence -- column 0 is t_k,
            columns 1.. are per-recovered-TX-chain amplitude, already
            resampled onto a uniform time grid.
        frame_timestamps: (num_frames,)
        mask_labels: (num_frames, H, W) float32 in [0, 1]
        keypoint_heatmaps: (num_frames, K, H, W) float32, identity-free
            union-over-people target (bottom-up, §3.2.3)
        keypoint_coords: (num_frames, MAX_PEOPLE, K, 2) int, heatmap-
            space (row, col) per joint, used only for L_group
        keypoint_visible: (num_frames, MAX_PEOPLE, K) bool
        """
        self.t_k = feature_seq[:, 0]
        self.y_k = feature_seq[:, 1:]  # real amplitude, (K_resampled, num_streams)
        self.frame_timestamps = frame_timestamps
        self.mask_labels = mask_labels
        self.keypoint_heatmaps = keypoint_heatmaps
        self.keypoint_coords = keypoint_coords
        self.keypoint_visible = keypoint_visible
        self.window_sec = signal_duration_sec

    def __len__(self):
        return len(self.frame_timestamps)

    def _window_indices(self, t_end: float):
        t_start = t_end - self.window_sec
        i0 = bisect.bisect_left(self.t_k, t_start)
        i1 = bisect.bisect_right(self.t_k, t_end)
        return i0, i1

    def __getitem__(self, idx: int):
        t_end = self.frame_timestamps[idx]
        i0, i1 = self._window_indices(t_end)
        window = self.y_k[i0:i1]  # (T, num_streams) real amplitude

        if window.shape[0] == 0:
            window = np.zeros((1, self.y_k.shape[1]), dtype=np.float64)

        # [.]_ac : remove the DC (mean) component across the window,
        # isolating the moving-reflector term (Eqn. 10) -- applied here
        # to amplitude, since phase is dropped upstream (see
        # data/csi_pcap_loader.py point 6).
        window = window - window.mean(axis=0, keepdims=True)

        # (T, num_streams) -> (num_streams, T) for Conv1d(channels, length)
        feat = window.T.astype(np.float32)

        # Pad/truncate to a fixed sample count for batching. With
        # resampling in csi_pcap_loader.py, this is now a deterministic
        # target derived from SIGNAL_DURATION_SEC * RESAMPLE_RATE_HZ
        # (config.SIGNAL_SEQ_LEN_SAMPLES), not a guessed packet count --
        # this only pads short edge windows near the start of a capture.
        T = feat.shape[1]
        target_len = config.SIGNAL_SEQ_LEN_SAMPLES
        if T >= target_len:
            feat = feat[:, -target_len:]
        else:
            pad = np.zeros((feat.shape[0], target_len - T), dtype=np.float32)
            feat = np.concatenate([pad, feat], axis=1)

        return {
            "signal": torch.from_numpy(feat),
            "mask_label": torch.from_numpy(self.mask_labels[idx]).unsqueeze(0).float(),
            "keypoint_label": torch.from_numpy(self.keypoint_heatmaps[idx]).float(),
            "keypoint_coords": torch.from_numpy(self.keypoint_coords[idx]).long(),
            "keypoint_visible": torch.from_numpy(self.keypoint_visible[idx]).bool(),
        }


def chronological_split(dataset_len: int, train_frac: float = config.TRAIN_FRACTION):
    """§4.2: samples sorted by timestamp, first 80% train / last 20% test."""
    split = int(dataset_len * train_frac)
    return list(range(0, split)), list(range(split, dataset_len))