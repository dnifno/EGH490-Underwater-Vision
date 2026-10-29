#!/usr/bin/env python3
"""Extract the 11 frames that were annotated in CVAT from the onboard BlueROV2 video.

Run from the repository root:

    python utils/extract_cvat_frames.py

Frames are saved at the video's native resolution (1920x1080) as PNG, so no further lossy compression is introduced before annotation.
"""
import argparse
from pathlib import Path

import cv2

# Zero-based frame numbers of the annotated frames (CVAT frame numbers are zero-based).
FRAME_NUMBERS = [946, 2035, 2276, 2421, 3076, 3799, 4284, 4644, 5782, 8478, 9529]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/bluerov_videos/Myora-BlueROV.mp4")
    ap.add_argument("--out", default="data/cvat_original_frames")
    ap.add_argument("--frames", type=int, nargs="*", default=FRAME_NUMBERS)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    print(f"Video resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

    for frame_number in args.frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()
        if not success:
            print(f"FAILED: frame {frame_number}")
            continue
        output_path = out_dir / f"frame_{frame_number:06d}.png"
        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        print(f"Saved frame {frame_number}: {output_path}")

    cap.release()
    print("Finished.")


if __name__ == "__main__":
    main()
