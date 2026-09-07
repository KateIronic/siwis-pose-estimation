"""
Training loop for the SiWiS reproduction (Fig. 13).

Pipeline wiring:
  data/csi_pcap_loader.py -> data/dataset.py -> models/siwis_net.py -> losses/losses.py

Labels (mask_labels.npy, keypoint_heatmaps.npy, keypoint_coords.npy,
keypoint_visible.npy, frame_timestamps.npy) must first be produced by
data/build_labels.py from your video frames.

Run:
  python train.py --pcap capture.pcap --src_mac 8c:86:dd:b7:46:73 \
      --labels_dir ./labels_out --epochs 50
"""

from __future__ import annotations

import argparse
import os

import config
import numpy as np
import torch
from data.csi_pcap_loader import build_feature_sequence
from data.dataset import SiWiSDataset, chronological_split
from losses.losses import total_loss
from models.siwis_net import SiWiSNet
from torch.utils.data import DataLoader, Subset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pcap", required=True, help="CSI capture, see data/csi_pcap_loader.py")
    p.add_argument("--bw", type=int, default=80)
    p.add_argument("--chip", default="43455c0")
    p.add_argument("--variant", default="legacy", choices=["legacy", "pull46", "pull256"])
    p.add_argument("--src_mac", default=None,
                    help="Only use frames from this transmitter's MAC. Strongly recommended "
                         "-- see data/csi_pcap_loader.py point 2.")
    p.add_argument("--labels_dir", required=True, help="output directory of data/build_labels.py")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="siwis_checkpoint.pt")
    return p.parse_args()


def main():
    args = parse_args()

    feature_seq = build_feature_sequence(
        args.pcap, chip=args.chip, bw=args.bw, variant=args.variant, src_mac=args.src_mac
    )

    frame_timestamps = np.load(os.path.join(args.labels_dir, "frame_timestamps.npy"))
    mask_labels = np.load(os.path.join(args.labels_dir, "mask_labels.npy"))
    keypoint_heatmaps = np.load(os.path.join(args.labels_dir, "keypoint_heatmaps.npy"))
    keypoint_coords = np.load(os.path.join(args.labels_dir, "keypoint_coords.npy"))
    keypoint_visible = np.load(os.path.join(args.labels_dir, "keypoint_visible.npy"))

    dataset = SiWiSDataset(
        feature_seq, frame_timestamps, mask_labels, keypoint_heatmaps, keypoint_coords, keypoint_visible
    )
    train_idx, test_idx = chronological_split(len(dataset))  # §4.2: 80/20 chronological split
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False)

    model = SiWiSNet().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)  # §4.1.2

    for epoch in range(args.epochs):
        model.train()
        running = {"mask_loss": 0.0, "pose_loss": 0.0}
        for batch in train_loader:
            signal = batch["signal"].to(args.device)
            mask_target = batch["mask_label"].to(args.device)
            pose_target = batch["keypoint_label"].to(args.device)
            kp_coords = batch["keypoint_coords"].to(args.device)
            kp_visible = batch["keypoint_visible"].to(args.device)

            out = model(signal)
            mask_pred = torch.sigmoid(out["mask"])  # Eqn. 12 expects x_i in (0,1)

            loss, parts = total_loss(
                mask_pred,
                mask_target,
                out["pose_heatmaps"],
                pose_target,
                out["pose_tags"],
                kp_coords,
                kp_visible,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running["mask_loss"] += parts["mask_loss"]
            running["pose_loss"] += parts["pose_loss"]

        n_batches = max(len(train_loader), 1)
        print(
            f"epoch {epoch}: mask_loss={running['mask_loss'] / n_batches:.4f} "
            f"pose_loss={running['pose_loss'] / n_batches:.4f}"
        )

    torch.save(model.state_dict(), args.out)
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()