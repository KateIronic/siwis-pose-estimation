"""
Visual sanity check of infer.py's output: renders predicted mask/pose
heatmaps side-by-side with the RGB frame and the vision-branch ground
truth (from data/build_labels.py) -- similar in spirit to Fig. 18 in
the paper -- plus a literal keypoint-skeleton overlay and a per-joint
heatmap-channel grid.

This is NOT a metric (see train.py's missing test-set evaluation loop
for that). It exists purely so a checkpoint's output can actually be
looked at, since infer.py only saves raw NumPy arrays.

Outputs go to <out_dir>/<tag>/ so a "before" run (e.g. --tag epoch2)
and an "after" run (e.g. --tag full_trained) land in separate,
directly comparable folders instead of overwriting each other.

Usage:
    python visualize_predictions.py --predictions predictions_smoketest.npz \
        --labels_dir data/labels_poses01 --frames_dir data/raw_frames/poses01 \
        --tag epoch2_baseline --n_samples 6
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Standard COCO 17-keypoint order and skeleton connectivity (as used by
# pycocotools / HRNet, matching data/vision_labels.py's HRNet output).
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]


def heatmaps_to_keypoints(heatmaps: np.ndarray, frame_h: int, frame_w: int):
    """
    heatmaps: (K, H, W). Returns (pts, conf01): pts is (K, 2) in frame
    pixel space (x, y), conf01 is each channel's peak value rescaled to
    [0, 1] *within that channel* (min-max over its own H x W map) -- a
    relative, not absolute, confidence, since predicted heatmaps aren't
    bounded to any fixed range (see module note in the grid plot).
    """
    K, H, W = heatmaps.shape
    sy, sx = frame_h / H, frame_w / W
    pts = np.zeros((K, 2))
    conf01 = np.zeros(K)
    for k in range(K):
        hm = heatmaps[k]
        row, col = np.unravel_index(np.argmax(hm), hm.shape)
        pts[k] = [col * sx, row * sy]
        lo, hi = hm.min(), hm.max()
        conf01[k] = (hm[row, col] - lo) / (hi - lo) if hi > lo else 0.0
    return pts, conf01


def draw_skeleton(ax, pts, conf01, color, label):
    ax.scatter(pts[:, 0], pts[:, 1], s=25, c=[color], alpha=np.clip(conf01, 0.15, 1.0), label=label)
    for a, b in COCO_SKELETON:
        alpha = min(conf01[a], conf01[b])
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], c=color,
                linewidth=1.5, alpha=max(alpha, 0.1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, help=".npz saved by infer.py")
    p.add_argument("--labels_dir", required=True, help="output directory of data/build_labels.py")
    p.add_argument("--frames_dir", required=True, help="RGB frames directory, e.g. data/raw_frames/poses01")
    p.add_argument("--out_dir", default="predictions_viz")
    p.add_argument("--tag", required=True, help="e.g. 'epoch2_baseline' or 'full_trained' -- keeps runs comparable")
    p.add_argument("--n_samples", type=int, default=6)
    args = p.parse_args()

    out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)

    pred = np.load(args.predictions)
    pred_t = pred["timestamps"]
    pred_masks = pred["masks"]          # (N, 1, H, W), already sigmoid'd by infer.py
    pred_poses = pred["pose_heatmaps"]  # (N, K, H, W), raw (unbounded) regression output

    if len(pred_t) == 0:
        raise SystemExit("No predictions in this .npz -- infer.py produced zero windows "
                          "(check the pcap covers enough time for a full SIGNAL_DURATION_SEC window).")

    gt_t = np.load(os.path.join(args.labels_dir, "frame_timestamps.npy"))
    gt_masks = np.load(os.path.join(args.labels_dir, "mask_labels.npy"))
    gt_poses = np.load(os.path.join(args.labels_dir, "keypoint_heatmaps.npy"))

    frame_paths = {
        float(os.path.splitext(os.path.basename(fp))[0]): fp
        for fp in glob.glob(os.path.join(args.frames_dir, "*"))
    }
    frame_ts_sorted = np.array(sorted(frame_paths.keys()))

    n = min(args.n_samples, len(pred_t))
    idxs = np.linspace(0, len(pred_t) - 1, n).astype(int)

    # ---- 1) mask + max-projection-pose grid (RGB | pred mask | GT mask | pred pose | GT pose) ----
    col_titles = ["RGB frame", "Predicted mask", "GT mask",
                  "Predicted pose (max over joints)", "GT pose (max over joints)"]
    fig, axes = plt.subplots(n, 5, figsize=(15, 3 * n), squeeze=False)
    rgb_cache = {}
    gt_idx_cache = {}
    for row, i in enumerate(idxs):
        t = pred_t[i]
        gi = int(np.argmin(np.abs(gt_t - t)))
        fi = int(np.argmin(np.abs(frame_ts_sorted - t)))
        rgb = np.array(Image.open(frame_paths[frame_ts_sorted[fi]]).convert("RGB"))
        rgb_cache[i] = rgb
        gt_idx_cache[i] = gi

        cells = [
            (rgb, None, None),
            (pred_masks[i, 0], 0, 1),
            (gt_masks[gi], 0, 1),
            (pred_poses[i].max(axis=0), None, None),
            (gt_poses[gi].max(axis=0), 0, 1),
        ]
        for col, (img, vmin, vmax) in enumerate(cells):
            ax = axes[row, col]
            ax.imshow(img) if img.ndim == 3 else ax.imshow(img, cmap="viridis", vmin=vmin, vmax=vmax)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(f"t={t:.2f}s", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    grid_path = os.path.join(out_dir, "mask_pose_grid.png")
    plt.savefig(grid_path, dpi=120)
    plt.close(fig)

    # ---- 2) literal keypoint-skeleton overlay: predicted (red) vs GT (green) on the RGB frame ----
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5), squeeze=False)
    heat_h, heat_w = pred_poses.shape[-2:]
    for col, i in enumerate(idxs):
        rgb = rgb_cache[i]
        frame_h, frame_w = rgb.shape[:2]
        gi = gt_idx_cache[i]
        pred_pts, pred_conf = heatmaps_to_keypoints(pred_poses[i], frame_h, frame_w)
        gt_pts, gt_conf = heatmaps_to_keypoints(gt_poses[gi], frame_h, frame_w)

        ax = axes[0, col]
        ax.imshow(rgb)
        draw_skeleton(ax, gt_pts, gt_conf, "lime", "GT")
        draw_skeleton(ax, pred_pts, pred_conf, "red", "Predicted")
        ax.set_title(f"t={pred_t[i]:.2f}s", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    skel_path = os.path.join(out_dir, "skeleton_overlay.png")
    plt.savefig(skel_path, dpi=120)
    plt.close(fig)

    # ---- 3) per-joint heatmap channel grid, predicted vs GT, for the middle sampled frame ----
    mid = idxs[n // 2]
    gi = gt_idx_cache[mid]
    fig, axes = plt.subplots(4, 17, figsize=(34, 8))
    for k in range(17):
        axes[0, k].imshow(pred_poses[mid, k], cmap="viridis")
        axes[1, k].imshow(gt_poses[gi, k], cmap="viridis", vmin=0, vmax=1)
        axes[0, k].set_title(COCO_KEYPOINT_NAMES[k], fontsize=7, rotation=45, ha="left")
        for r in (0, 1):
            axes[r, k].set_xticks([]); axes[r, k].set_yticks([])
    axes[0, 0].set_ylabel("Predicted", fontsize=9)
    axes[1, 0].set_ylabel("GT", fontsize=9)
    for k in range(17):
        axes[2, k].axis("off")
        axes[3, k].axis("off")
    plt.tight_layout()
    channels_path = os.path.join(out_dir, "keypoint_channels.png")
    plt.savefig(channels_path, dpi=110)
    plt.close(fig)

    print(f"Saved to {out_dir}/:")
    print(f"  mask_pose_grid.png     ({n} samples)")
    print(f"  skeleton_overlay.png   ({n} samples, predicted=red, GT=green)")
    print(f"  keypoint_channels.png  (all 17 joint channels, frame t={pred_t[mid]:.2f}s)")


if __name__ == "__main__":
    main()
