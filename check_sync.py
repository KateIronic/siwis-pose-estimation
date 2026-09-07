"""
Preflight check for CSI/video timestamp alignment, before spending
GPU time running Mask R-CNN + HRNet over every frame in
data/build_labels.py.

Loads a CSI .pcap (data/csi_pcap_loader.load_csi_pcap) and a frames
directory produced by capture/record_frames.py (files named
"<epoch_seconds>.jpg"), then reports:

  - capture time range and average rate for each stream
  - how much of the video's time range actually falls inside the
    CSI capture's time range (frames outside it get an empty/zero-
    padded signal window in data/dataset.py -- SiWiSDataset._window_indices
    -- i.e. wasted training samples)
  - for each frame, how many CSI packets land in the
    [t_frame - SIGNAL_DURATION_SEC, t_frame] window dataset.py uses;
    flags frames with zero packets
  - largest single gap in each stream's timestamps (a stall, dropped
    frames, or a clock jump)

Run from the repo root:
  python check_sync.py --pcap capture.pcap --frames_dir data/raw_frames/walk01 \
      --src_mac 8c:86:dd:b7:46:73

Note: this checks raw capture-wide timestamp coverage, before transmit-
chain splitting (data/csi_pcap_loader.split_tx_chains) -- it's a coarse
preflight check on overall sync, not a per-stream analysis.
"""

from __future__ import annotations

import argparse
import bisect
import glob
import os

import numpy as np

import config
from data.csi_pcap_loader import load_csi_pcap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pcap", required=True)
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--bw", type=int, default=80)
    p.add_argument("--chip", default="43455c0")
    p.add_argument("--variant", default="legacy", choices=["legacy", "pull46", "pull256"])
    p.add_argument("--src_mac", default=None,
                    help="Only use frames from this transmitter's MAC. Strongly recommended "
                         "-- see data/csi_pcap_loader.py point 2.")
    p.add_argument("--signal_duration", type=float, default=config.SIGNAL_DURATION_SEC)
    args = p.parse_args()

    print(f"Loading CSI packets from {args.pcap} ...")
    _, t_csi = load_csi_pcap(args.pcap, chip=args.chip, bw=args.bw, variant=args.variant,
                              src_mac=args.src_mac)
    if len(t_csi) == 0:
        raise SystemExit("No CSI packets parsed from pcap -- check --src_mac/--bw/--chip "
                          "and that this is a Nexmon CSI UDP-in-pcap capture.")
    t_csi = np.sort(t_csi)

    frame_paths = glob.glob(os.path.join(args.frames_dir, "*"))
    if not frame_paths:
        raise SystemExit(f"No frames found in {args.frames_dir}")
    t_frames = np.array(sorted(
        float(os.path.splitext(os.path.basename(fp))[0]) for fp in frame_paths
    ))

    def describe(name: str, t: np.ndarray):
        span = t[-1] - t[0]
        rate = (len(t) - 1) / span if span > 0 else float("nan")
        gaps = np.diff(t)
        biggest_gap = gaps.max() if len(gaps) else 0.0
        print(f"{name}: n={len(t)}  range=[{t[0]:.6f}, {t[-1]:.6f}]  "
              f"span={span:.3f}s  avg_rate={rate:.2f}/s  biggest_gap={biggest_gap*1000:.1f}ms")
        return span, biggest_gap

    print()
    describe("CSI packets ", t_csi)
    describe("Video frames", t_frames)
    print()

    overlap_start = max(t_csi[0], t_frames[0])
    overlap_end = min(t_csi[-1], t_frames[-1])
    frames_in_range = np.sum((t_frames >= t_csi[0]) & (t_frames <= t_csi[-1]))
    print(f"Overlap window: [{overlap_start:.6f}, {overlap_end:.6f}] "
          f"({max(0.0, overlap_end - overlap_start):.3f}s)")
    print(f"Frames inside CSI time range: {frames_in_range}/{len(t_frames)}")
    if frames_in_range < len(t_frames):
        print(f"  Warning: {len(t_frames) - frames_in_range} frame(s) fall outside "
              f"CSI coverage and will get an empty/zero-padded signal window.")

    # Mirrors data/dataset.py SiWiSDataset._window_indices exactly:
    # for each frame timestamp t_end, count CSI packets in
    # [t_end - signal_duration, t_end].
    t_csi_list = t_csi.tolist()
    counts = []
    for t_end in t_frames:
        i0 = bisect.bisect_left(t_csi_list, t_end - args.signal_duration)
        i1 = bisect.bisect_right(t_csi_list, t_end)
        counts.append(i1 - i0)
    counts = np.array(counts)

    print()
    print(f"CSI packets per {args.signal_duration}s frame window: "
          f"min={counts.min()} mean={counts.mean():.1f} max={counts.max()}")
    n_empty = int(np.sum(counts == 0))
    if n_empty:
        print(f"  Warning: {n_empty}/{len(counts)} frame(s) have ZERO CSI packets "
              f"in their window and will train on an all-zero signal.")
    else:
        print("  Every frame has at least one CSI packet in its window.")

    print()
    if n_empty == 0 and frames_in_range == len(t_frames):
        print("OK: video and CSI streams look aligned and ready for data/build_labels.py.")
    else:
        print("ISSUES FOUND: see warnings above before running data/build_labels.py.")


if __name__ == "__main__":
    main()
