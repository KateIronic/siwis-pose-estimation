"""
Extract individual timestamped frames from an already-recorded video
(e.g. poses.mkv), for data/build_labels.py --frames_dir.

This is the offline counterpart to record_frames.py in this directory:
that script saves frames one-by-one during a *live* capture; this one
is for video already recorded as a single container by this project's
existing capture workflow (collect_csi.sh: ffmpeg MJPEG stream-copy to
.mkv, run alongside tcpdump for the CSI capture -- see
Collision-detection/PROJECT_NOTES.md, and the *_meta.json file it
writes alongside each *.mkv).

Frame timestamps are computed as:
    t_i = epoch_first_frame + i / native_fps
where:
  - i is the index of the frame as actually decoded (0-based) -- NOT
    cv2's CAP_PROP_FRAME_COUNT, which is a container-level estimate.
    Measured on poses.mkv: CAP_PROP_FRAME_COUNT reports 2472, but only
    2468 frames actually decode, matching the original ffmpeg encode
    log's final "frame=" count exactly. Trusting the estimate would
    silently mislabel the tail of the recording.
  - native_fps is read directly from the container (CAP_PROP_FPS).
    Measured on poses.mkv: exactly 30.0, matching the encode log's
    "30 tbr" -- reliable here because it's an MJPEG stream-copy at a
    genuinely constant capture rate, not a re-encode.
  - epoch_first_frame is the already-corrected timestamp of frame 0,
    written to <name>_meta.json by this project's capture tooling per
    PROJECT_NOTES.md 3.5 ("epoch_first_frame must be computed as
    epoch_video_stop - video_duration, not from when ffmpeg launched --
    getting this wrong shifts every label by 1.4s").

This intentionally does NOT derive per-frame spacing from
video_duration / frame_count: video_duration includes teardown time
after the last frame (measured ~0.13s of slop on poses.mkv) and is a
noisier estimate than the container's own reported frame rate.

Usage:
    python capture/extract_frames_from_video.py --video poses.mkv \
        --meta poses_meta.json --out_dir data/raw_frames/poses01 \
        --target_fps 10
"""

from __future__ import annotations

import argparse
import json
import os

import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Path to the recorded .mkv/.mp4")
    p.add_argument("--meta", default=None,
                    help="<name>_meta.json written by this project's capture workflow; "
                         "reads epoch_first_frame from it.")
    p.add_argument("--epoch_first_frame", type=float, default=None,
                    help="Epoch timestamp of frame 0. Overrides --meta if both are given; "
                         "required if --meta is omitted.")
    p.add_argument("--out_dir", required=True, help="Feed this straight to build_labels.py --frames_dir")
    p.add_argument("--target_fps", type=float, default=None,
                    help="Downsample to roughly this rate (e.g. 10, matching config.VIDEO_FPS / "
                         "the paper's setup) by keeping every Nth decoded frame. Omit to keep "
                         "every frame at the video's native rate.")
    p.add_argument("--ext", default="jpg", choices=["jpg", "png"])
    p.add_argument("--jpeg_quality", type=int, default=95)
    args = p.parse_args()

    epoch_first_frame = args.epoch_first_frame
    if epoch_first_frame is None:
        if args.meta is None:
            raise SystemExit("Need --epoch_first_frame or --meta to anchor frame timestamps.")
        with open(args.meta) as f:
            meta = json.load(f)
        epoch_first_frame = meta["epoch_first_frame"]

    os.makedirs(args.out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.video}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    reported_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if native_fps <= 0:
        raise RuntimeError(
            f"Container reports an invalid fps ({native_fps}) for {args.video}; "
            "can't compute per-frame timestamps without it."
        )
    print(f"{args.video}: native_fps={native_fps:.3f}  "
          f"container-reported frame count={reported_count} (estimate; actual count "
          f"is determined by decoding below, not trusted directly)")

    stride = 1
    if args.target_fps is not None:
        stride = max(1, round(native_fps / args.target_fps))
        print(f"Keeping every {stride}th frame (~{native_fps / stride:.2f} fps effective)")

    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality] if args.ext == "jpg" else []

    i = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            t = epoch_first_frame + i / native_fps
            fname = os.path.join(args.out_dir, f"{t:.6f}.{args.ext}")
            cv2.imwrite(fname, frame, jpeg_params)
            written += 1
        i += 1
    cap.release()

    print(f"Done. Decoded {i} frames, wrote {written} to {args.out_dir}.")
    if abs(i - reported_count) > max(5, 0.01 * reported_count):
        print(f"  Warning: decoded frame count ({i}) differs substantially from the "
              f"container's reported count ({reported_count}) -- worth checking the "
              f"source file isn't truncated/corrupt before trusting these timestamps.")


if __name__ == "__main__":
    main()
