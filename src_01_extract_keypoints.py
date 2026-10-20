"""
Step 1 of the pipeline: video -> per-frame keypoints of the near-court player.

Usage (from the badminton_analysis folder):
    python src/01_extract_keypoints.py --video raw_videos/match1.mp4
    python src/01_extract_keypoints.py --video clips/clip_001.mp4 --show

Output: keypoints/<video_name>.npy with shape (T, 17, 3) -> (x, y, confidence), COCO-17 order
        keypoints/<video_name>_fps.txt with the video frame rate
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# COCO-17 indices (useful later in feature extraction)
# 5 L-shoulder, 6 R-shoulder, 7 L-elbow, 8 R-elbow, 9 L-wrist, 10 R-wrist, 11 L-hip, 12 R-hip


def pick_near_player(result):
    """Pick the person with the largest bounding box.
    In a broadcast back-view the near-court player appears biggest. Crude, but a fine start."""
    if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        return None
    xyxy = result.boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    idx = int(np.argmax(areas))
    xy = result.keypoints.xy.cpu().numpy()[idx]          # (17, 2)
    conf = result.keypoints.conf.cpu().numpy()[idx]      # (17,)
    return np.concatenate([xy, conf[:, None]], axis=1)   # (17, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="yolo11n-pose.pt", help="n/s/m/l/x: bigger = more accurate, slower")
    ap.add_argument("--out", default="keypoints")
    ap.add_argument("--show", action="store_true", help="preview skeleton overlay (press q to quit)")
    args = ap.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    model = YOLO(args.model)

    all_kp = []
    last = np.full((17, 3), np.nan)

    for _ in tqdm(range(n_frames), desc=video_path.name):
        ok, frame = cap.read()
        if not ok:
            break
        res = model(frame, verbose=False)[0]
        kp = pick_near_player(res)
        if kp is None:
            kp = np.full((17, 3), np.nan)   # missing frame -> interpolate later
        all_kp.append(kp)

        if args.show:
            cv2.imshow("pose", res.plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    arr = np.stack(all_kp)
    np.save(out_dir / f"{video_path.stem}.npy", arr)
    (out_dir / f"{video_path.stem}_fps.txt").write_text(str(fps))
    print(f"Saved {arr.shape} keypoints at {fps:.1f} fps -> {out_dir / (video_path.stem + '.npy')}")


if __name__ == "__main__":
    main()
