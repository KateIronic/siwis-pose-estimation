"""
On-Pi webcam capture for SiWiS-style CSI/video sync.

Run this on the same Raspberry Pi that is running nexmon_csi (e.g.
`tcpdump -i wlan0 dst port 5500 -w csi.pcap`), started at roughly the
same time as this script. Because both processes read the Pi's own
CLOCK_REALTIME, absolute clock correctness does not matter -- only
that CSI packet timestamps (read via `csiread`, see
data/csi_pcap_loader.py) and these frame timestamps come from the same
clock. No NTP/PTP sync across devices is required with this single-Pi
setup.

Each frame is timestamped the instant it is grabbed (cv2.grab(), before
the slower decode/write path) and saved as "<epoch_seconds>.jpg" --
the exact filename convention data/build_labels.py expects ("a
directory of frames named '<timestamp_seconds>.png' (or .jpg)").
Point build_labels.py's --frames_dir straight at --out_dir from this
script; no separate matching step is needed there, since
data/dataset.py already bisects CSI packets into the window ending at
each frame's timestamp.

An optional --preview_video writes a second, human-watchable .mp4 with
a burned-in wall-clock overlay, for eyeballing sync (e.g. confirming
task start/stop) while reviewing footage. That overlay is NOT written
onto the canonical .jpg frames -- burning text into the frames used
for Mask R-CNN / HRNet label extraction would risk the models picking
up the overlay as a spurious feature.

Usage:
    python capture/record_frames.py --out_dir data/raw_frames/walk01 \
        --fps 10 --preview_video data/raw_frames/walk01_preview.mp4
"""

from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from datetime import datetime

import cv2


def _writer_loop(q: "queue.Queue", out_dir: str, jpeg_quality: int,
                  preview_writer, stats: dict, stop_event: threading.Event):
    while not (stop_event.is_set() and q.empty()):
        try:
            item = q.get(timeout=0.5)
        except queue.Empty:
            continue
        frame, t = item
        fname = os.path.join(out_dir, f"{t:.6f}.jpg")
        cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        stats["written"] += 1

        if preview_writer is not None:
            overlay = frame.copy()
            label = datetime.fromtimestamp(t).strftime("%H:%M:%S.%f")
            cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(overlay, f"t={t:.6f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)
            preview_writer.write(overlay)

        q.task_done()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True, help="Directory for timestamped frames (feed to build_labels.py --frames_dir)")
    p.add_argument("--camera_index", type=int, default=0)
    p.add_argument("--fps", type=float, default=10.0, help="Target capture rate (config.VIDEO_FPS default is 10, matching the paper's setup)")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--jpeg_quality", type=int, default=95)
    p.add_argument("--preview_video", default=None, help="Optional path to a human-watchable .mp4 with a burned-in timestamp overlay")
    p.add_argument("--duration_sec", type=float, default=None, help="Stop automatically after this many seconds (omit to run until Ctrl+C)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_index)
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    preview_writer = None
    if args.preview_video:
        ok, probe = cap.read()
        if not ok:
            raise RuntimeError("Could not read a frame from the camera to size the preview video")
        h, w = probe.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        preview_writer = cv2.VideoWriter(args.preview_video, fourcc, args.fps, (w, h))

    frame_q: "queue.Queue" = queue.Queue()
    stats = {"written": 0}
    stop_event = threading.Event()
    writer_thread = threading.Thread(
        target=_writer_loop,
        args=(frame_q, args.out_dir, args.jpeg_quality, preview_writer, stats, stop_event),
        daemon=True,
    )
    writer_thread.start()

    min_interval = 1.0 / args.fps
    next_due = time.time()
    start_time = time.time()
    n_grabbed = 0

    print(f"Recording to {args.out_dir} at target {args.fps} fps. Press Ctrl+C to stop.")
    try:
        while True:
            now = time.time()
            if now < next_due:
                time.sleep(next_due - now)

            ok = cap.grab()
            t = time.time()  # timestamp taken as close to sensor capture as possible
            if not ok:
                print("Warning: frame grab failed, skipping")
                continue
            ok, frame = cap.retrieve()
            if not ok:
                print("Warning: frame retrieve failed, skipping")
                continue

            frame_q.put((frame, t))
            n_grabbed += 1
            next_due += min_interval

            if n_grabbed % max(1, int(args.fps) * 5) == 0:
                elapsed = now - start_time
                achieved_fps = n_grabbed / elapsed if elapsed > 0 else 0.0
                backlog = frame_q.qsize()
                print(f"[{elapsed:6.1f}s] grabbed={n_grabbed} written={stats['written']} "
                      f"achieved_fps={achieved_fps:.2f} disk_backlog={backlog}")
                if backlog > args.fps * 3:
                    print("  Warning: disk writes are falling behind capture rate; "
                          "consider lowering --fps or --jpeg_quality")

            if args.duration_sec is not None and (now - start_time) >= args.duration_sec:
                break

    except KeyboardInterrupt:
        print("\nStopping capture...")
    finally:
        stop_event.set()
        cap.release()
        writer_thread.join()
        if preview_writer is not None:
            preview_writer.release()
        elapsed = time.time() - start_time
        print(f"Done. {stats['written']} frames written to {args.out_dir} "
              f"over {elapsed:.1f}s (achieved {stats['written']/elapsed:.2f} fps).")


if __name__ == "__main__":
    main()
