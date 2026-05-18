import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPONENT_NEW_DIR = REPO_ROOT / "src" / "models" / "component_new"
sys.path.insert(0, str(COMPONENT_NEW_DIR))

from viclip import _frame_from_video, frames2tensor, get_viclip, get_vid_feat  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ViCLIP features for THUMOS videos."
    )
    parser.add_argument(
        "--anno-path",
        default=str(REPO_ROOT / "data" / "thumos_annotations" / "thumos_anno_action.json"),
        help="Path to THUMOS annotation json.",
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        help="Directory containing THUMOS videos.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "thumos_viclip_online"),
        help="Directory to save extracted .npy features.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(COMPONENT_NEW_DIR / "viclip" / "ViCLIP-L_InternVid-FLT-10M.pth"),
        help="Path to ViCLIP checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device used for feature extraction, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index in the annotation video list.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index in the annotation video list.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files.",
    )
    return parser.parse_args()


def load_thumos_video_names(anno_path):
    with open(anno_path, "r") as f:
        annotations = json.load(f)
    return list(annotations.keys())


def open_video(video_dir, video_name):
    for ext in [".mp4", ".mkv", ".webm", ".avi"]:
        video_path = Path(video_dir) / f"{video_name}{ext}"
        if video_path.exists():
            return cv2.VideoCapture(str(video_path)), video_path
    return None, None


def split_frames_into_online_chunks(frames, chunk_size=8):
    chunks = []
    for i in range(len(frames)):
        start_idx = max(0, i - (chunk_size - 1))
        chunk = frames[start_idx : i + 1]
        if len(chunk) < chunk_size:
            chunk = [chunk[0]] * (chunk_size - len(chunk)) + chunk
        chunks.append(chunk)
    return chunks


def extract_video_features(frames, clip_model, device):
    chunks = split_frames_into_online_chunks(frames)
    features = []

    with torch.no_grad():
        for chunk in chunks:
            frames_tensor = frames2tensor(chunk, device=device)
            feature = get_vid_feat(frames_tensor, clip_model)
            features.append(feature)

    return torch.cat(features, dim=0).cpu().numpy()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_viclip("l", args.checkpoint)
    clip_model = model["viclip"].float().to(device)
    clip_model.eval()
    print("Loaded ViCLIP model")

    video_names = load_thumos_video_names(args.anno_path)[args.start : args.end]
    print(f"Found {len(video_names)} videos to process")

    for index, video_name in enumerate(video_names, start=args.start):
        save_path = output_dir / f"{video_name}.npy"
        if save_path.exists() and not args.overwrite:
            print(f"[{index}] {video_name}: exists, skip")
            continue

        start_time = time.time()
        video, video_path = open_video(args.video_dir, video_name)
        if video is None:
            print(f"[{index}] {video_name}: video not found, skip")
            continue

        frames = [frame for frame in _frame_from_video(video)]
        video.release()
        if not frames:
            print(f"[{index}] {video_name}: empty video, skip")
            continue

        features = extract_video_features(frames, clip_model, device)
        np.save(save_path, features)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        print(
            f"[{index}] {video_path.name}: saved {save_path.name}, "
            f"shape={features.shape}, time={elapsed:.2f}s"
        )


if __name__ == "__main__":
    main()
