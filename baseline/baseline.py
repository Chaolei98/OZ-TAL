import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the hard-threshold visual-text similarity baseline."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("baseline_config.yaml")),
        help="Path to baseline yaml config.",
    )
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_class_names(split_file):
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def make_prompt(class_name):
    class_name = re.sub(r"([a-z])([A-Z])", r"\1 \2", class_name)
    return f"a video of action {class_name}"


def load_viclip_model(cfg, device):
    viclip_code_dir = cfg.get("code_dir")
    if viclip_code_dir:
        sys.path.insert(0, str(resolve_path(viclip_code_dir)))

    from viclip import get_viclip

    model_bundle = get_viclip(cfg.get("size", "l"), cfg["checkpoint"])
    model = model_bundle["viclip"].float().to(device)
    model.eval()
    return model, model_bundle["tokenizer"]


def encode_text_features(model, tokenizer, class_names, device):
    prompts = [make_prompt(name) for name in class_names]
    text_feature_cache = {}
    text_features = []
    with torch.no_grad():
        for prompt in prompts:
            feature = model.get_text_features(prompt, tokenizer, text_feature_cache)
            text_features.append(feature.to(device))
    return normalize(torch.cat(text_features, dim=0))


def normalize(features):
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def load_video_features(feature_path, device):
    features = np.load(feature_path)
    features = torch.from_numpy(features).float().to(device)
    return features.squeeze()


def compute_similarity(image_features, text_features):
    image_features = normalize(image_features)
    return image_features @ text_features.T


def get_video_fps(fps_data, video_name):
    if "database" in fps_data and video_name in fps_data["database"]:
        return float(fps_data["database"][video_name]["fps"])
    return 30.0


def group_actions(scores, video_name, class_names, fps, threshold):
    scores_np = scores.detach().cpu().numpy()
    num_classes = scores_np.shape[-1]
    outputs = []

    for class_idx in range(num_classes):
        active = scores_np[:, class_idx] >= threshold
        start = None

        for frame_idx, is_active in enumerate(active):
            is_last = frame_idx == len(active) - 1

            if is_active and start is None:
                start = frame_idx

            if start is not None and ((not is_active) or is_last):
                end = frame_idx if is_active and is_last else frame_idx - 1
                segment_scores = scores_np[start : end + 1, class_idx]
                outputs.append(
                    {
                        "video-id": video_name,
                        "t-start": start / fps,
                        "t-end": end / fps,
                        "label": class_names[class_idx],
                        "label_id": class_idx,
                        "score": float(segment_scores.mean()),
                    }
                )
                start = None

    return outputs


def save_outputs(outputs, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "predictions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    csv_path = output_dir / "predictions.csv"
    fieldnames = ["video-id", "t-start", "t-end", "label", "label_id", "score"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(outputs)

    return json_path, csv_path


def main():
    args = parse_args()
    cfg = load_config(args.config)

    feature_dir = resolve_path(cfg["paths"]["feature_dir"])
    split_file = resolve_path(cfg["paths"]["split_file"])
    fps_json = resolve_path(cfg["paths"]["fps_json"])
    output_dir = resolve_path(cfg["paths"]["output_dir"])

    requested_device = cfg["inference"].get("device", "cuda:0")
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    threshold = float(cfg["inference"]["threshold"])
    start = cfg["inference"].get("start", 0)
    end = cfg["inference"].get("end")

    class_names = load_class_names(split_file)
    with open(fps_json, "r", encoding="utf-8") as f:
        fps_data = json.load(f)

    model, tokenizer = load_viclip_model(cfg["model"], device)
    text_features = encode_text_features(model, tokenizer, class_names, device)

    feature_paths = sorted(feature_dir.glob("*.npy"))[start:end]
    print(f"Using device: {device}")
    print(f"Classes: {class_names}")
    print(f"Feature files: {len(feature_paths)}")

    all_outputs = []
    for idx, feature_path in enumerate(feature_paths, start=start):
        video_name = feature_path.stem
        image_features = load_video_features(feature_path, device)
        scores = compute_similarity(image_features, text_features)
        fps = get_video_fps(fps_data, video_name)
        outputs = group_actions(scores, video_name, class_names, fps, threshold)
        all_outputs.extend(outputs)
        print(f"[{idx}] {video_name}: {len(outputs)} segments")

    json_path, csv_path = save_outputs(all_outputs, output_dir)
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
