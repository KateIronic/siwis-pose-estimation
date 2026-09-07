"""
Offline vision-processing step (Fig. 13, bottom "Vision Processing"
box): runs Mask R-CNN and HRNet over a directory of timestamped video
frames and saves the label arrays consumed by data/dataset.py and
train.py.

Expected input: a directory of frames named "<timestamp_seconds>.png"
(or .jpg), where timestamp_seconds is on the SAME clock as the WiFi
capture's packet timestamps -- required for "aligning them by their
timestamps" (§3.1).

Output (written to --out_dir):
  frame_timestamps.npy   (num_frames,)
  mask_labels.npy         (num_frames, HEATMAP_H, HEATMAP_W) float32 in [0,1]
  keypoint_heatmaps.npy   (num_frames, NUM_KEYPOINTS, HEATMAP_H, HEATMAP_W)
                          identity-free, union-over-people heatmaps
                          (the bottom-up target consumed alongside the
                          associative-embedding tags, §3.2.3, ref [33])
  keypoint_coords.npy     (num_frames, MAX_PEOPLE, NUM_KEYPOINTS, 2) int64,
                          heatmap-space (row, col) per joint, used only
                          to compute L_group (needs per-person identity,
                          which a merged heatmap discards)
  keypoint_visible.npy    (num_frames, MAX_PEOPLE, NUM_KEYPOINTS) bool
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from PIL import Image

import config
from data.vision_labels import KeypointLabelExtractor, MaskLabelExtractor, keypoints_to_heatmap


def _boxes_from_mask(mask: np.ndarray, min_area: int = 500) -> np.ndarray:
    """Connected-component boxes from the Mask R-CNN union mask, used as HRNet's top-down input."""
    from scipy import ndimage

    labeled, n = ndimage.label(mask > 0.5)
    boxes = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) < min_area:
            continue
        boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
    return np.array(boxes, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--hrnet_config", required=True)
    p.add_argument("--hrnet_checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    mask_extractor = MaskLabelExtractor(device=args.device)
    kp_extractor = KeypointLabelExtractor(args.hrnet_config, args.hrnet_checkpoint, device=args.device)

    frame_paths = glob.glob(os.path.join(args.frames_dir, "*"))
    frame_paths.sort(key=lambda fp: float(os.path.splitext(os.path.basename(fp))[0]))
    timestamps = [float(os.path.splitext(os.path.basename(fp))[0]) for fp in frame_paths]

    mask_labels, kp_heatmaps, kp_coords, kp_visible = [], [], [], []

    for fp in frame_paths:
        frame = np.array(Image.open(fp).convert("RGB"))
        h, w = frame.shape[:2]

        mask = mask_extractor.extract(frame)
        mask_resized = np.array(
            Image.fromarray(mask).resize((config.HEATMAP_W, config.HEATMAP_H))
        )
        mask_labels.append(mask_resized)

        boxes = _boxes_from_mask(mask)
        if len(boxes) == 0:
            kp_heatmaps.append(
                np.zeros((config.NUM_KEYPOINTS, config.HEATMAP_H, config.HEATMAP_W), dtype=np.float32)
            )
            kp_coords.append(np.zeros((config.MAX_PEOPLE, config.NUM_KEYPOINTS, 2), dtype=np.int64))
            kp_visible.append(np.zeros((config.MAX_PEOPLE, config.NUM_KEYPOINTS), dtype=bool))
            continue

        people_kp = kp_extractor.extract(frame, boxes)  # (N, K, 3): x, y, score

        person_heatmaps = []
        coords = np.zeros((config.MAX_PEOPLE, config.NUM_KEYPOINTS, 2), dtype=np.int64)
        visible = np.zeros((config.MAX_PEOPLE, config.NUM_KEYPOINTS), dtype=bool)
        sx, sy = config.HEATMAP_W / w, config.HEATMAP_H / h

        for n, kp in enumerate(people_kp[: config.MAX_PEOPLE]):
            hm = keypoints_to_heatmap(kp[:, :2], config.HEATMAP_H, config.HEATMAP_W, h, w)
            person_heatmaps.append(hm)
            coords[n, :, 0] = np.clip((kp[:, 1] * sy).astype(np.int64), 0, config.HEATMAP_H - 1)
            coords[n, :, 1] = np.clip((kp[:, 0] * sx).astype(np.int64), 0, config.HEATMAP_W - 1)
            visible[n] = kp[:, 2] > 0.3  # HRNet keypoint confidence threshold

        union_heatmap = np.max(np.stack(person_heatmaps), axis=0)  # identity-free, bottom-up target
        kp_heatmaps.append(union_heatmap)
        kp_coords.append(coords)
        kp_visible.append(visible)

    np.save(os.path.join(args.out_dir, "frame_timestamps.npy"), np.array(timestamps))
    np.save(
        os.path.join(args.out_dir, "mask_labels.npy"),
        np.stack(mask_labels).astype(np.float32),
    )
    np.save(os.path.join(args.out_dir, "keypoint_heatmaps.npy"), np.stack(kp_heatmaps))
    np.save(os.path.join(args.out_dir, "keypoint_coords.npy"), np.stack(kp_coords))
    np.save(os.path.join(args.out_dir, "keypoint_visible.npy"), np.stack(kp_visible))
    print(f"Saved labels for {len(frame_paths)} frames to {args.out_dir}")


if __name__ == "__main__":
    main()