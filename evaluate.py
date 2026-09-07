"""
Evaluation metrics for a trained SiWiSNet checkpoint (§4.3):
 - Mask segmentation: IoU-based mAP / AP@a (Eqn. 15, §4.3.1)
 - Pose estimation: PCK@a for single-person data (Eqn. 17, §4.3.2)

Runs on the SAME chronological test split train.py holds out (last 20%
by timestamp, §4.2), so these numbers reflect held-out performance, not
training-set fit.

Scope notes -- read before trusting these numbers:
 - Mask mAP is exactly Eqn. 15: per-frame binary IoU against a
   threshold a, averaged over a in {0.50, 0.55, ..., 0.95}.
 - Pose metric is the paper's *single-person* PCK (Eqn. 17, the same
   metric used for the single-person case as [49,60,77]), NOT the
   multi-person OKS-based mAP (Eqn. 16). OKS-mAP needs distinct people
   decoded out of the heatmaps+tags via the associative-embedding
   grouping algorithm (Newell et al. [33]); this codebase only ever
   computes L_group as a scalar training loss, it never decodes it back
   into discrete people at inference time. PCK is used here because it
   matches this project's actual single-person captures -- multi-person
   OKS-mAP would need that decoding step built first.
 - The paper's transcribed Eqn. 17 normalization term (sqrt(rs^2+lh^2),
   with rs/lh described in the text as keypoint *positions*) is
   dimensionally inconsistent as written -- most likely lossy OCR/
   typesetting of the source PDF, not a resolvable ambiguity. This
   implementation uses the standard interpretation from the
   single-person pose literature: normalize by the Euclidean distance
   between the right-shoulder and left-hip keypoints (a torso-size
   reference), i.e. ||rs - lh||_2, not a literal sqrt(rs^2 + lh^2).
   Flagged here rather than silently guessing.
 - Assumes person-slot 0 in keypoint_coords/keypoint_visible is "the"
   tracked person -- correct for this project's single-person captures,
   not a real multi-person evaluation.

Run:
  python evaluate.py --pcap poses.pcap --src_mac 8c:86:dd:b7:46:73 \
      --labels_dir data/labels_poses01 --checkpoint checkpoints/siwis_full_trained.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.csi_pcap_loader import build_feature_sequence
from data.dataset import SiWiSDataset, chronological_split
from models.siwis_net import SiWiSNet

RIGHT_SHOULDER, LEFT_HIP = 6, 11  # COCO keypoint indices


def mask_map(pred_masks: np.ndarray, gt_masks: np.ndarray):
    """pred_masks, gt_masks: (N, H, W). Returns (mAP, {threshold: AP@threshold})."""
    pred_bin = pred_masks >= 0.5
    gt_bin = gt_masks >= 0.5
    inter = (pred_bin & gt_bin).sum(axis=(1, 2)).astype(np.float64)
    union = (pred_bin | gt_bin).sum(axis=(1, 2)).astype(np.float64)
    iou = np.where(union > 0, inter / np.maximum(union, 1), 1.0)  # both-empty -> IoU=1

    thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    ap = {a: float((iou >= a).mean()) for a in thresholds}
    m_ap = float(np.mean(list(ap.values())))
    return m_ap, ap, iou


def argmax_keypoints(heatmaps: np.ndarray):
    """heatmaps: (N, K, H, W) -> (N, K, 2) [row, col] of each channel's argmax."""
    N, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(N, K, -1)
    idx = flat.argmax(axis=-1)
    rows, cols = np.unravel_index(idx, (H, W))
    return np.stack([rows, cols], axis=-1).astype(np.float64)


def pose_pck(pred_heatmaps: np.ndarray, gt_coords: np.ndarray, gt_visible: np.ndarray):
    """
    pred_heatmaps: (N, K, H, W). gt_coords: (N, K, 2) [row, col], person 0.
    gt_visible: (N, K) bool, person 0. Returns ({threshold: PCK@threshold}, n_frames_used).
    """
    pred_pts = argmax_keypoints(pred_heatmaps)
    gt_pts = gt_coords.astype(np.float64)

    thresholds = [0.1, 0.2, 0.3, 0.5]
    hits = {a: [] for a in thresholds}
    n_used = 0
    for i in range(len(gt_pts)):
        if not (gt_visible[i, RIGHT_SHOULDER] and gt_visible[i, LEFT_HIP]):
            continue
        norm = np.linalg.norm(gt_pts[i, RIGHT_SHOULDER] - gt_pts[i, LEFT_HIP])
        if norm < 1e-6:
            continue
        n_used += 1
        dists = np.linalg.norm(pred_pts[i] - gt_pts[i], axis=-1) / norm
        vis = gt_visible[i]
        for a in thresholds:
            correct = (dists <= a) & vis
            hits[a].append(correct.sum() / max(vis.sum(), 1))

    pck = {a: (float(np.mean(v)) if v else float("nan")) for a, v in hits.items()}
    return pck, n_used


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pcap", required=True)
    p.add_argument("--bw", type=int, default=80)
    p.add_argument("--chip", default="43455c0")
    p.add_argument("--variant", default="legacy", choices=["legacy", "pull46", "pull256"])
    p.add_argument("--src_mac", default=None)
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()

    feature_seq = build_feature_sequence(
        args.pcap, chip=args.chip, bw=args.bw, variant=args.variant, src_mac=args.src_mac
    )
    frame_timestamps = np.load(f"{args.labels_dir}/frame_timestamps.npy")
    mask_labels = np.load(f"{args.labels_dir}/mask_labels.npy")
    keypoint_heatmaps = np.load(f"{args.labels_dir}/keypoint_heatmaps.npy")
    keypoint_coords = np.load(f"{args.labels_dir}/keypoint_coords.npy")
    keypoint_visible = np.load(f"{args.labels_dir}/keypoint_visible.npy")

    dataset = SiWiSDataset(
        feature_seq, frame_timestamps, mask_labels, keypoint_heatmaps, keypoint_coords, keypoint_visible
    )
    train_idx, test_idx = chronological_split(len(dataset))
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False)

    model = SiWiSNet().to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()

    all_pred_masks, all_gt_masks = [], []
    all_pred_poses, all_gt_coords, all_gt_visible = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            signal = batch["signal"].to(args.device)
            out = model(signal)
            all_pred_masks.append(torch.sigmoid(out["mask"])[:, 0].cpu().numpy())
            all_gt_masks.append(batch["mask_label"][:, 0].numpy())
            all_pred_poses.append(out["pose_heatmaps"].cpu().numpy())
            all_gt_coords.append(batch["keypoint_coords"][:, 0].numpy())
            all_gt_visible.append(batch["keypoint_visible"][:, 0].numpy())

    pred_masks = np.concatenate(all_pred_masks)
    gt_masks = np.concatenate(all_gt_masks)
    pred_poses = np.concatenate(all_pred_poses)
    gt_coords = np.concatenate(all_gt_coords)
    gt_visible = np.concatenate(all_gt_visible)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Evaluated on {len(test_idx)} held-out (chronologically last 20%) test frames.\n")

    m_ap, ap_by_thr, _ = mask_map(pred_masks, gt_masks)
    print("Mask segmentation (IoU-based, Eqn. 15):")
    print(f"  mAP    = {m_ap:.4f}")
    for a in [0.5, 0.6, 0.7, 0.8]:
        print(f"  AP@{a:.2f} = {ap_by_thr[a]:.4f}")

    pck, n_used = pose_pck(pred_poses, gt_coords, gt_visible)
    print(f"\nPose estimation (single-person PCK, Eqn. 17; {n_used}/{len(test_idx)} frames "
          f"had both right_shoulder+left_hip visible for normalization):")
    for a, v in pck.items():
        print(f"  PCK@{a} = {v:.4f}")


if __name__ == "__main__":
    main()
