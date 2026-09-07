"""
Inference: load a trained SiWiSNet and run it on a fresh CSI capture
to produce mask segmentation and pose estimation outputs (Fig. 1's
"DNN output 1: human skeleton" / "DNN output 2: human body mask").
At inference time SiWiS "relies solely on WiFi-based sensing signals"
(§3.1) -- no video/vision branch is used here.

Run:
  python infer.py --pcap live_capture.pcap --src_mac 8c:86:dd:b7:46:73 \
      --checkpoint siwis_checkpoint.pt --out predictions.npz
"""

from __future__ import annotations

import argparse

import config
import numpy as np
import torch
from data.csi_pcap_loader import build_feature_sequence
from models.siwis_net import SiWiSNet


def windows_from_feature_seq(feature_seq: np.ndarray, window_sec: float, hop_sec: float):
    t_k = feature_seq[:, 0]
    y_k = feature_seq[:, 1:]
    if len(t_k) == 0:
        return
    t0 = t_k[0]
    while t0 + window_sec <= t_k[-1]:
        mask = (t_k >= t0) & (t_k < t0 + window_sec)
        window = y_k[mask]
        if len(window) > 0:
            window = window - window.mean(axis=0, keepdims=True)  # [.]_ac, Eqn. 10
            feat = window.T.astype(np.float32)
            T = feat.shape[1]
            target_len = config.SIGNAL_SEQ_LEN_SAMPLES
            if T >= target_len:
                feat = feat[:, -target_len:]
            else:
                pad = np.zeros((feat.shape[0], target_len - T), dtype=np.float32)
                feat = np.concatenate([pad, feat], axis=1)
            yield t0 + window_sec, feat
        t0 += hop_sec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pcap", required=True)
    p.add_argument("--bw", type=int, default=80)
    p.add_argument("--chip", default="43455c0")
    p.add_argument("--variant", default="legacy", choices=["legacy", "pull46", "pull256"])
    p.add_argument("--src_mac", default=None,
                    help="Only use frames from this transmitter's MAC. Strongly recommended "
                         "-- see data/csi_pcap_loader.py point 2.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="siwis_predictions.npz")
    args = p.parse_args()

    feature_seq = build_feature_sequence(
        args.pcap, chip=args.chip, bw=args.bw, variant=args.variant, src_mac=args.src_mac
    )

    model = SiWiSNet().to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()

    timestamps, masks, poses = [], [], []
    with torch.no_grad():
        for t_end, feat in windows_from_feature_seq(
            feature_seq, config.SIGNAL_DURATION_SEC, hop_sec=1.0 / config.VIDEO_FPS
        ):
            signal = torch.from_numpy(feat).unsqueeze(0).to(args.device)
            out = model(signal)
            timestamps.append(t_end)
            masks.append(torch.sigmoid(out["mask"]).cpu().numpy()[0])
            poses.append(out["pose_heatmaps"].cpu().numpy()[0])

    np.savez(
        args.out,
        timestamps=np.array(timestamps),
        masks=np.stack(masks) if masks else np.empty((0,)),
        pose_heatmaps=np.stack(poses) if poses else np.empty((0,)),
    )
    print(f"Saved {len(timestamps)} predictions to {args.out}")


if __name__ == "__main__":
    main()