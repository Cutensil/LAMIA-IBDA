"""Run Lamia on one image pair or on two paired image directories."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils

from mm.cv.transforms import Compose
from mm.det.registry import MODELS, TRANSFORMS
from mm.engine.config import Config, DictAction
from mm.engine.dataset import pseudo_collate
from mm.engine.registry import init_default_scope
from mm.engine.runner import load_checkpoint


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
TILE_SIZE = 512


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("pre_input", type=Path)
    parser.add_argument("post_input", type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="output directory, default work_dirs/<config>/<timestamp>",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="legacy single-pair JSON output path",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="save a mask and bounding-box overlay on the pre-event image",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="number of 512x512 windows processed at once",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.05,
        help="discard tile predictions below this score before fusion",
    )
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    return parser.parse_args()


def image_files(directory):
    files = sorted(
        path.relative_to(directory)
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise ValueError(f"No supported image files found in {directory}")
    return files


def make_pairs(pre_input, post_input):
    pre_is_dir = pre_input.is_dir()
    post_is_dir = post_input.is_dir()
    if pre_is_dir != post_is_dir:
        raise ValueError(
            "pre_input and post_input must both be files or both be directories")
    if not pre_is_dir:
        if not pre_input.is_file() or not post_input.is_file():
            raise FileNotFoundError("Both image inputs must exist")
        return [(pre_input, post_input, Path(pre_input.name))]

    pre_names = set(image_files(pre_input))
    post_names = set(image_files(post_input))
    if pre_names != post_names:
        raise ValueError(
            "Directory image names do not match. "
            f"Only in pre_input: {sorted(pre_names - post_names)[:5]}. "
            f"Only in post_input: {sorted(post_names - pre_names)[:5]}.")
    return [
        (pre_input / name, post_input / name, name)
        for name in sorted(pre_names)
    ]


def default_output_dir(config_path):
    timestamp = os.environ.get("LAMIA_INFER_TIMESTAMP")
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("work_dirs") / config_path.stem / timestamp


def inference_device():
    """Select the device visible to this process.

    ``CUDA_VISIBLE_DEVICES`` controls which physical GPUs are visible. During
    a ``torchrun`` launch, ``LOCAL_RANK`` selects one of those visible GPUs.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_count = torch.cuda.device_count()
    if local_rank < 0 or local_rank >= device_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is invalid for {device_count} visible GPUs")
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def output_paths(args, pairs, output_dir):
    if args.out is not None and args.out_dir is not None:
        raise ValueError("--out and --out-dir cannot be used together")
    if args.out is not None:
        if len(pairs) != 1:
            raise ValueError("--out can only be used with one image pair")
        if args.out.suffix.lower() != ".json":
            raise ValueError("--out must be a JSON file for a single image pair")
        return args.out.parent, [args.out]

    json_paths = []
    for _, _, relative_name in pairs:
        json_path = output_dir / relative_name.with_suffix(".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_paths.append(json_path)
    return output_dir, json_paths


def starts(length):
    if length <= TILE_SIZE:
        return [0]
    values = list(range(0, length - TILE_SIZE + 1, TILE_SIZE))
    values.append(length - TILE_SIZE)
    for boundary in range(TILE_SIZE, length, TILE_SIZE):
        values.append(max(0, min(length - TILE_SIZE, boundary - TILE_SIZE // 2)))
    return sorted(set(values))


def load_image_pipeline(cfg):
    pipeline = cfg.test_dataloader.dataset.datasets[0].pipeline
    base_steps = [
        step for step in pipeline
        if step.type not in {"LoadAnnotations", "PackDetInputs",
                             "FixedQuadCrop", "Resize"}
    ]
    load_pipeline = Compose(base_steps)
    pack_step = next(step for step in pipeline if step.type == "PackDetInputs")
    return load_pipeline, TRANSFORMS.build(pack_step)


def padding_values(cfg, channels):
    mean = list(cfg.model.data_preprocessor.mean)
    if len(mean) != channels:
        return [0] * channels
    if channels == 4 and cfg.model.data_preprocessor.get("bgr_to_rgb", False):
        return [mean[2], mean[1], mean[0], mean[3]]
    return mean


def prepare_full_image(cfg, load_pipeline, pre_path, post_path):
    loaded = load_pipeline(dict(
        img_path=str(post_path),
        pre_img_path=str(pre_path),
        post_img_path=str(post_path),
    ))
    image = loaded["img"]
    original_height, original_width = image.shape[:2]
    padded_height = max(TILE_SIZE, (
        (original_height + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE)
    padded_width = max(TILE_SIZE, (
        (original_width + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE)
    if (padded_height, padded_width) == (original_height, original_width):
        return image, (original_height, original_width)

    padding = np.empty(
        (padded_height, padded_width, image.shape[2]), dtype=image.dtype)
    for channel, value in enumerate(padding_values(cfg, image.shape[2])):
        padding[:, :, channel] = value
    padding[:original_height, :original_width] = image
    return padding, (original_height, original_width)


def make_samples(image, original_shape, pre_path, post_path, pack, positions):
    samples = []
    for index, (top, left) in enumerate(positions):
        tile = image[top:top + TILE_SIZE, left:left + TILE_SIZE]
        samples.append(pack(dict(
            img=tile,
            img_id=index,
            img_path=str(post_path),
            pre_img_path=str(pre_path),
            post_img_path=str(post_path),
            ori_shape=original_shape,
            img_shape=tile.shape[:2],
            scale_factor=(1.0, 1.0),
        )))
    return samples


def mask_overlap(first, second):
    first_left, first_top = first["origin"]
    second_left, second_top = second["origin"]
    first_height, first_width = first["mask"].shape
    second_height, second_width = second["mask"].shape
    left = max(first_left, second_left)
    top = max(first_top, second_top)
    right = min(first_left + first_width, second_left + second_width)
    bottom = min(first_top + first_height, second_top + second_height)
    if left >= right or top >= bottom:
        return 0.0, 0.0
    first_mask = first["mask"][
        top - first_top:bottom - first_top,
        left - first_left:right - first_left]
    second_mask = second["mask"][
        top - second_top:bottom - second_top,
        left - second_left:right - second_left]
    intersection = np.logical_and(first_mask, second_mask).sum()
    first_area = first["mask"].sum()
    second_area = second["mask"].sum()
    union = first_area + second_area - intersection
    ios = intersection / min(first_area, second_area) if min(
        first_area, second_area) else 0.0
    iou = intersection / union if union else 0.0
    return float(iou), float(ios)


def merge_masks(first, second):
    left = min(first["origin"][0], second["origin"][0])
    top = min(first["origin"][1], second["origin"][1])
    right = max(
        first["origin"][0] + first["mask"].shape[1],
        second["origin"][0] + second["mask"].shape[1])
    bottom = max(
        first["origin"][1] + first["mask"].shape[0],
        second["origin"][1] + second["mask"].shape[0])
    shape = (bottom - top, right - left)
    weighted = np.zeros(shape, dtype=np.float32)
    weights = np.zeros(shape, dtype=np.float32)
    for item in (first, second):
        y, x = item["origin"][1] - top, item["origin"][0] - left
        height, width = item["mask"].shape
        weighted[y:y + height, x:x + width] += (
            item["mask"].astype(np.float32) * item["score"])
        weights[y:y + height, x:x + width] += item["score"]
    return weighted / np.maximum(weights, 1e-6) >= 0.5, (left, top)


def merge_detections(detections):
    cell_size = 256
    merged = []
    grid = {}

    def cells(item):
        left, top = item["origin"]
        height, width = item["mask"].shape
        right = left + width - 1
        bottom = top + height - 1
        return {
            (x, y)
            for x in range(left // cell_size, right // cell_size + 1)
            for y in range(top // cell_size, bottom // cell_size + 1)
        }

    def add_to_grid(index, item):
        for cell in cells(item):
            grid.setdefault(cell, set()).add(index)

    def remove_from_grid(index, item):
        for cell in cells(item):
            values = grid.get(cell)
            if values is None:
                continue
            values.discard(index)
            if not values:
                grid.pop(cell)

    for candidate in sorted(detections, key=lambda item: item["score"],
                            reverse=True):
        match = None
        candidate_indices = set()
        for cell in cells(candidate):
            candidate_indices.update(grid.get(cell, ()))
        for index in candidate_indices:
            current = merged[index]
            iou, ios = mask_overlap(candidate, current)
            same_label = candidate["label"] == current["label"]
            if (same_label and (ios >= 0.5 or iou >= 0.3)) or ios >= 0.8:
                match = index
                break
        if match is None:
            add_to_grid(len(merged), candidate)
            merged.append(candidate)
            continue
        current = merged[match]
        remove_from_grid(match, current)
        current["mask"], current["origin"] = merge_masks(current, candidate)
        total = current["score"] + candidate["score"]
        current["bbox"] = [
            (current["bbox"][index] * current["score"]
             + candidate["bbox"][index] * candidate["score"]) / total
            for index in range(4)
        ]
        if candidate["score"] > current["score"]:
            current["score"] = candidate["score"]
            current["label"] = candidate["label"]
        add_to_grid(match, current)
    return merged


def collect_predictions(
        model, samples, positions, original_shape, batch_size, score_thr):
    detections = []
    for start in range(0, len(samples), batch_size):
        predictions = model.test_step(
            pseudo_collate(samples[start:start + batch_size]))
        for prediction, (top, left) in zip(
                predictions, positions[start:start + batch_size]):
            instances = prediction.pred_instances.cpu()
            for box, score, label, mask in zip(
                    instances.bboxes.tolist(),
                    instances.scores.tolist(),
                    instances.labels.tolist(),
                    instances.masks.numpy()):
                if score < score_thr:
                    continue
                detections.append(dict(
                    bbox=[
                        box[0] + left, box[1] + top,
                        box[2] + left, box[3] + top,
                    ],
                    score=float(score),
                    label=int(label),
                    mask=mask.astype(bool),
                    origin=(left, top),
                ))
    height, width = original_shape
    output = []
    for item in merge_detections(detections):
        left, top = item["origin"]
        mask = item["mask"]
        visible_left = max(0, left)
        visible_top = max(0, top)
        visible_right = min(width, left + mask.shape[1])
        visible_bottom = min(height, top + mask.shape[0])
        if visible_left >= visible_right or visible_top >= visible_bottom:
            continue
        cropped = mask[
            visible_top - top:visible_bottom - top,
            visible_left - left:visible_right - left]
        if not cropped.any():
            continue
        box = [
            max(0.0, min(width, item["bbox"][0])),
            max(0.0, min(height, item["bbox"][1])),
            max(0.0, min(width, item["bbox"][2])),
            max(0.0, min(height, item["bbox"][3])),
        ]
        full_mask = np.zeros((height, width), dtype=bool)
        full_mask[
            visible_top:visible_bottom,
            visible_left:visible_right,
        ] = cropped
        rle = mask_utils.encode(np.asfortranarray(full_mask, dtype=np.uint8))
        rle["counts"] = rle["counts"].decode("ascii")
        output.append(dict(
            bbox=box,
            score=item["score"],
            label=item["label"],
            mask=rle,
            visualization_mask=cropped,
            visualization_origin=(visible_left, visible_top),
        ))
    return output


def json_instances(instances):
    return [
        {
            key: value
            for key, value in item.items()
            if key not in {"visualization_mask", "visualization_origin"}
        }
        for item in instances
    ]


def draw_visualization(pre_path, instances, palette, image_shape, output_path):
    image = cv2.imread(str(pre_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read pre-event image: {pre_path}")
    height, width = image_shape
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    thickness = max(2, round(min(height, width) / 512))
    for item in instances:
        left, top = item["visualization_origin"]
        mask = item["visualization_mask"]
        color_rgb = tuple(int(value) for value in palette[item["label"]])
        color_bgr = color_rgb[::-1]
        region = image[top:top + mask.shape[0], left:left + mask.shape[1]]
        overlay = region.copy()
        overlay[mask] = color_bgr
        region[:] = cv2.addWeighted(region, 0.62, overlay, 0.38, 0)
        x1, y1, x2, y2 = item["bbox"]
        box = (round(x1), round(y1), round(x2), round(y2))
        cv2.rectangle(image, box[:2], box[2:], color_bgr, thickness)
        cv2.putText(
            image,
            f"{item['label']}:{item['score']:.2f}",
            (box[0], max(16, box[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color_bgr,
            max(1, thickness - 1),
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise IOError(f"Could not write visualization: {output_path}")


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not 0 <= args.score_thr <= 1:
        raise ValueError("--score-thr must be between 0 and 1")
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.default_scope)
    pairs = make_pairs(
        args.pre_input.expanduser().resolve(),
        args.post_input.expanduser().resolve(),
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        pairs = pairs[rank::world_size]
    if not pairs:
        print(f"Rank {rank} has no assigned image pairs")
        return
    output_dir = args.out_dir or default_output_dir(Path(args.config))
    output_dir, json_paths = output_paths(args, pairs, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu", strict=True)
    model.to(inference_device()).eval()
    load_pipeline, pack = load_image_pipeline(cfg)
    palette = cfg.metainfo.palette

    for (pre_path, post_path, relative_name), json_path in zip(
            pairs, json_paths):
        full_image, original_shape = prepare_full_image(
            cfg, load_pipeline, pre_path, post_path)
        positions = [
            (top, left)
            for top in starts(full_image.shape[0])
            for left in starts(full_image.shape[1])
        ]
        samples = make_samples(
            full_image, original_shape, pre_path, post_path, pack, positions)
        with torch.inference_mode():
            instances = collect_predictions(
                model,
                samples,
                positions,
                original_shape,
                args.batch_size,
                args.score_thr,
            )
        payload = dict(
            classes=list(cfg.metainfo.classes),
            image=dict(
                pre_path=str(pre_path),
                post_path=str(post_path),
                width=original_shape[1],
                height=original_shape[0],
                padded_width=full_image.shape[1],
                padded_height=full_image.shape[0],
            ),
            windows=[
                dict(top=top, left=left, size=TILE_SIZE)
                for top, left in positions
            ],
            instances=json_instances(instances),
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False)
        if args.visualize:
            draw_visualization(
                pre_path,
                instances,
                palette,
                original_shape,
                output_dir / relative_name.with_suffix(".png"),
            )
        print(f"Saved predictions to {json_path}")


if __name__ == "__main__":
    main()
