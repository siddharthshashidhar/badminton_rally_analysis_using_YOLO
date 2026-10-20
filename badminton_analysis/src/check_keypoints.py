"""
Quick quality check of extracted keypoints.

Usage (from badminton_analysis folder):
    python src/check_keypoints.py --name test

Prints missing-frame %, mean confidence per joint, and saves
results/plots/<name>_check.png (wrist/elbow/shoulder tracks and a jump detector).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

JOINTS = {5: "L-shoulder", 6: "R-shoulder", 7: "L-elbow", 8: "R-elbow",
          9: "L-wrist", 10: "R-wrist", 11: "L-hip", 12: "R-hip"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="file name without extension, e.g. test")
    args = ap.parse_args()

    kp = np.load(Path("keypoints") / f"{args.name}.npy")
    fps = float(Path("keypoints", f"{args.name}_fps.txt").read_text())
    T = len(kp)
    t = np.arange(T) / fps

    missing = np.isnan(kp[:, 0, 0]).mean() * 100
    print(f"Frames: {T}   fps: {fps:.1f}   frames with no person detected: {missing:.1f}%")
    print("\nMean confidence per joint (want > 0.5):")
    for j, name in JOINTS.items():
        print(f"  {name:11s} {np.nanmean(kp[:, j, 2]):.2f}")

    # Frame-to-frame jump of the hip midpoint: large spikes = tracking switched to another person
    hip = (kp[:, 11, :2] + kp[:, 12, :2]) / 2
    jump = np.linalg.norm(np.diff(hip, axis=0), axis=1)
    thr = 80  # pixels per frame at 720p; tune if needed
    n_jumps = int(np.nansum(jump > thr))
    print(f"\nLarge hip jumps (> {thr}px/frame): {n_jumps}   <- should be ~0; "
          f"many means the wrong person is being picked")

    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for j, c in [(9, "tab:blue"), (10, "tab:red")]:
        ax[0].plot(t, kp[:, j, 0], color=c, label=JOINTS[j] + " x")
        ax[1].plot(t, kp[:, j, 1], color=c, label=JOINTS[j] + " y")
    ax[0].set_ylabel("x (px)"); ax[0].legend()
    ax[1].set_ylabel("y (px)"); ax[1].legend()
    ax[2].plot(t[1:], jump, color="k")
    ax[2].axhline(thr, color="r", ls="--")
    ax[2].set_ylabel("hip jump (px/frame)"); ax[2].set_xlabel("time (s)")
    fig.suptitle(f"Keypoint check: {args.name}")
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    out = Path("results/plots") / f"{args.name}_check.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved plot -> {out}")


if __name__ == "__main__":
    main()
