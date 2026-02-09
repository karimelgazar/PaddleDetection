"""
This script converts YOLO-format detection/segmentation labels to COCO JSON format,
and generates a PaddleDetection dataset YAML.

Supports both YOLO detection (5 values per line: class_id xc yc w h)
and YOLO segmentation (>=7 values per line: class_id x1 y1 x2 y2 ... xN yN)
annotation formats. Segmentation polygons are automatically converted to bounding boxes.

Usage:
    python3.13 -m pip install Pillow PyYAML tqdm numpy
    python3.13 yolo2coco.py --yolo_yaml /workspace/open-images-v7/data.yaml --num_workers 30
"""

import argparse
import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from queue import SimpleQueue

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert YOLO detection/segmentation labels to COCO JSON format")
    parser.add_argument(
        "--yolo_yaml",
        type=str,
        required=True,
        help="Path to YOLO data.yaml (dataset_dir is inferred from its parent folder)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for COCO JSON files. Defaults to dataset_dir/coco_annotations",
    )
    parser.add_argument(
        "--output_yaml_path",
        type=str,
        default=None,
        help="Output PaddleDetection yaml path (default: <output_dir>/<dataset_name>_coco.yml)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Thread count for conversion. 0 means all CPU threads",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise an error immediately when invalid annotations are found",
    )
    parser.add_argument(
        "--no_tqdm",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    return parser.parse_args()


def clamp(value, min_value, max_value):
    return round(max(min_value, min(max_value, value)), 2)


def load_data_yaml(data_yaml_path):
    if not data_yaml_path.exists():
        raise FileNotFoundError("data.yaml does not exist: {}".format(data_yaml_path))
    with data_yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("data.yaml must be a dictionary-like YAML")
    return data


def load_class_names(data_cfg, data_yaml_path):
    names = data_cfg.get("names", None)
    if names is None:
        raise ValueError("Class names are missing. Include `names` in {}".format(data_yaml_path))

    if isinstance(names, list):
        return [str(name) for name in names]

    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names.keys(), key=lambda x: int(x))]

    raise ValueError("Unsupported `names` format in {}".format(data_yaml_path))


def build_categories(class_names):
    categories = []
    for idx, name in enumerate(class_names, start=1):
        categories.append({"id": idx, "name": str(name), "supercategory": "none"})
    return categories


def resolve_data_root(dataset_dir, data_cfg, data_yaml_path):
    data_root_value = data_cfg.get("path")
    if data_root_value is None:
        return dataset_dir

    data_root = Path(str(data_root_value)).expanduser()
    if data_root.is_absolute():
        data_root = data_root.resolve()
    else:
        data_root = (data_yaml_path.parent / data_root).resolve()

    if data_root.exists():
        return data_root

    print("[WARN] data.yaml 'path' does not exist: {}. Falling back to dataset_dir: {}".format(data_root, dataset_dir))
    return dataset_dir


def resolve_split_path(path_value, data_root, data_yaml_path):
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path.resolve()

    from_data_root = (data_root / path).resolve()
    if from_data_root.exists():
        return from_data_root

    from_yaml_dir = (data_yaml_path.parent / path).resolve()
    if from_yaml_dir.exists():
        return from_yaml_dir

    return from_data_root


def resolve_split_image_dir(split_name, data_cfg, data_root, data_yaml_path):
    candidate_keys = [split_name]
    if split_name == "val":
        candidate_keys = ["val", "valid", "validation"]

    split_key = None
    for candidate_key in candidate_keys:
        if candidate_key in data_cfg:
            split_key = candidate_key
            break

    if split_key is None:
        print("[INFO] Split '{}' not found in data.yaml. Skipping.".format(split_name))
        return None

    split_value = data_cfg[split_key]
    if not isinstance(split_value, str):
        print("[WARN] Split '{}' path must be a string. Skipping.".format(split_key))
        return None

    image_dir = resolve_split_path(split_value, data_root, data_yaml_path)
    if not image_dir.exists():
        print("[INFO] Split '{}' path does not exist: {}. Skipping.".format(split_key, image_dir))
        return None
    if not image_dir.is_dir():
        print("[WARN] Split '{}' path is not a directory: {}. Skipping.".format(split_key, image_dir))
        return None

    return image_dir


def infer_label_dir(split_name, image_dir, dataset_dir, data_root):
    candidates = []

    image_parts = list(image_dir.parts)
    image_positions = [idx for idx, part in enumerate(image_parts) if part == "images"]
    if image_positions:
        idx = image_positions[-1]
        replaced = image_parts[:]
        replaced[idx] = "labels"
        candidates.append(Path(*replaced))

    if image_dir.parent.name == "images":
        candidates.append(image_dir.parent.parent / "labels" / image_dir.name)

    candidates.append(dataset_dir / "labels" / split_name)
    candidates.append(data_root / "labels" / split_name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0].resolve()


def count_images_in_dir(image_dir):
    """Count image files using the `find | wc -l` command for speed, without loading paths into memory."""
    name_args = []
    for suffix in sorted(IMAGE_SUFFIXES):
        if name_args:
            name_args.append("-o")
        # Match both lower and upper case extensions
        name_args.extend(["-iname", "*{}".format(suffix)])

    cmd = "find {} -type f \\( {} \\) | wc -l".format(str(image_dir), " ".join(name_args))
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        return int(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        # Fallback: count via rglob (still lazy, just counts)
        count = 0
        for suffix in IMAGE_SUFFIXES:
            for path in image_dir.rglob("*{}".format(suffix)):
                if path.is_file():
                    count += 1
        return count


def iter_images(image_dir):
    """Yield image paths lazily using pathlib.rglob (generator, no list in memory)."""
    for suffix in IMAGE_SUFFIXES:
        for path in image_dir.rglob("*{}".format(suffix)):
            if path.is_file():
                yield path


def mask_polygon_to_bbox(normalized_coords, img_width, img_height):
    """Convert YOLO segmentation polygon (normalized x,y pairs) to a clamped bbox.

    Args:
        normalized_coords: flat list of normalized floats [x1, y1, x2, y2, ..., xN, yN].
        img_width: image width in pixels.
        img_height: image height in pixels.

    Returns:
        (x1, y1, box_w, box_h) in pixel coordinates, or None if degenerate.
    """
    coords = np.array(normalized_coords, dtype=np.float64)
    xs = coords[0::2] * img_width
    ys = coords[1::2] * img_height

    x1 = clamp(float(np.min(xs)), 0.0, float(img_width))
    y1 = clamp(float(np.min(ys)), 0.0, float(img_height))
    x2 = clamp(float(np.max(xs)), 0.0, float(img_width))
    y2 = clamp(float(np.max(ys)), 0.0, float(img_height))

    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0.0 or box_h <= 0.0:
        return None

    return x1, y1, box_w, box_h


def process_single_image(job):
    image_id, image_path_str, rel_file_name, label_path_str, num_classes, strict = job
    image_path = Path(image_path_str)
    label_path = Path(label_path_str)

    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except Exception as ex:
        raise ValueError("Failed to read image '{}': {}".format(image_path, ex))

    image_record = {
        "id": image_id,
        "file_name": rel_file_name,
        "width": width,
        "height": height,
    }

    result = {
        "image": image_record,
        "annotations": [],
        "missing_labels": 0,
        "empty_labels": 0,
        "invalid_annotations": 0,
        "warning": None,
    }

    # skip if label file doesn't exist - it will be counted in stats and can be used to identify missing annotations
    if not label_path.exists():
        result["missing_labels"] = 1
        result["image"] = None  # skip it
        return result

    # In the COCO dataset format, "negative" images (images without any objects of interest) are included by listing
    # them in the images section of the COCO JSON file but providing no corresponding entries in the annotations section.
    raw = label_path.read_text(encoding="utf-8").strip()
    if not raw:
        result["empty_labels"] = 1
        return result

    for line_id, line in enumerate(raw.splitlines(), start=1):
        parts = line.strip().split()
        num_parts = len(parts)

        # YOLO segmentation: class_id followed by polygon coordinates (>=7 values)
        is_segmentation = num_parts >= 7 and num_parts % 2 == 1
        # YOLO detection: class_id xc yc w h (exactly 5 values)
        is_detection = num_parts == 5

        if not is_detection and not is_segmentation:
            message = "{}:{} expected 5 columns (detection) or >=7 odd columns (segmentation), got {}".format(
                label_path, line_id, num_parts
            )
            if strict:
                raise ValueError(message)
            result["invalid_annotations"] += 1
            if result["warning"] is None:
                result["warning"] = message
            continue

        try:
            class_id = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
        except ValueError:
            message = "{}:{} contains non-numeric values".format(label_path, line_id)
            if strict:
                raise ValueError(message)
            result["invalid_annotations"] += 1
            if result["warning"] is None:
                result["warning"] = message
            continue

        if not all(math.isfinite(x) for x in values):
            message = "{}:{} contains non-finite values".format(label_path, line_id)
            if strict:
                raise ValueError(message)
            result["invalid_annotations"] += 1
            if result["warning"] is None:
                result["warning"] = message
            continue

        if class_id < 0 or class_id >= num_classes:
            message = "{}:{} class_id {} out of range [0, {})".format(label_path, line_id, class_id, num_classes)
            if strict:
                raise ValueError(message)
            result["invalid_annotations"] += 1
            if result["warning"] is None:
                result["warning"] = message
            continue

        segmentation = []

        if is_segmentation:
            bbox_result = mask_polygon_to_bbox(values, width, height)
            if bbox_result is None:
                message = "{}:{} has degenerate polygon bbox".format(label_path, line_id)
                if strict:
                    raise ValueError(message)
                result["invalid_annotations"] += 1
                if result["warning"] is None:
                    result["warning"] = message
                continue
            x1, y1, box_w, box_h = [round(x, 2) for x in bbox_result]
            # Store the pixel-coordinate polygon as COCO segmentation
            pixel_coords = []
            for i in range(0, len(values), 2):
                pixel_coords.append(clamp(values[i] * width, 0.0, float(width)))
                pixel_coords.append(clamp(values[i + 1] * height, 0.0, float(height)))
            segmentation = [pixel_coords]
        else:
            xc, yc, bw, bh = values

            x1 = (xc - bw / 2.0) * width
            y1 = (yc - bh / 2.0) * height
            x2 = (xc + bw / 2.0) * width
            y2 = (yc + bh / 2.0) * height

            x1 = clamp(x1, 0.0, float(width))
            y1 = clamp(y1, 0.0, float(height))
            x2 = clamp(x2, 0.0, float(width))
            y2 = clamp(y2, 0.0, float(height))

            box_w = round(x2 - x1, 2)
            box_h = round(y2 - y1, 2)
            if box_w <= 0.0 or box_h <= 0.0:
                message = "{}:{} has non-positive box after clipping".format(label_path, line_id)
                if strict:
                    raise ValueError(message)
                result["invalid_annotations"] += 1
                if result["warning"] is None:
                    result["warning"] = message
                continue

        result["annotations"].append(
            {
                "image_id": image_id,
                "category_id": class_id + 1,
                "bbox": [x1, y1, box_w, box_h],
                "area": round(box_w * box_h, 2),
                "iscrowd": 0,
                "segmentation": segmentation,
            }
        )

    return result


def job_generator(image_dir, label_dir, num_classes, strict):
    """Yield (image_id, image_path, rel_name, label_path, num_classes, strict) lazily."""
    for image_id, image_path in enumerate(iter_images(image_dir), start=1):
        rel_file = image_path.relative_to(image_dir)
        rel_file_name = rel_file.as_posix()
        label_path = label_dir / rel_file.with_suffix(".txt")
        yield (
            image_id,
            str(image_path),
            rel_file_name,
            str(label_path),
            num_classes,
            strict,
        )


def convert_split(split_name, image_dir, label_dir, class_names, num_workers, use_tqdm, strict):
    if not label_dir.exists():
        print("[WARN] Label directory not found: {}. All images will have empty annotations.".format(label_dir))

    # Count images using `find` command (fast, no memory allocation for paths)
    print("[INFO] Counting images in '{}' ...".format(image_dir))
    total_images = count_images_in_dir(image_dir)
    if total_images == 0:
        raise ValueError("No images found in {}".format(image_dir))
    print(f"[INFO]   - Found {total_images:,} images")

    # Build a lazy generator of jobs -- paths are never held in a list
    jobs = job_generator(image_dir, label_dir, len(class_names), strict)

    images = []
    annotations = []
    warning_messages = []
    max_warning_messages = 20
    stats = {
        "num_images": 0,
        "num_annotations": 0,
        "num_missing_labels": 0,
        "num_empty_labels": 0,
        "num_invalid_annotations": 0,
    }

    batch_size = max(num_workers * 4, 16)
    pbar = None
    if use_tqdm:
        pbar = tqdm(total=total_images, desc="Converting '{}'".format(split_name), unit="img")

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Sliding-window: keep `batch_size` futures in flight at all times.
            # A SimpleQueue receives results as futures complete via callbacks,
            # so we never block waiting for the entire batch to finish.
            result_queue = SimpleQueue()
            in_flight = 0

            def _done_callback(fut):
                result_queue.put(fut.result())

            # Submit initial window from the generator
            for job in islice(jobs, batch_size):
                fut = executor.submit(process_single_image, job)
                fut.add_done_callback(_done_callback)
                in_flight += 1

            while in_flight > 0:
                result = result_queue.get()
                in_flight -= 1

                # Refill: submit next job to replace the one that just finished
                next_job = next(jobs, None)
                if next_job is not None:
                    fut = executor.submit(process_single_image, next_job)
                    fut.add_done_callback(_done_callback)
                    in_flight += 1
                if result["image"] is not None:
                    images.append(result["image"])
                annotations.extend(result["annotations"])
                stats["num_missing_labels"] += result["missing_labels"]
                stats["num_empty_labels"] += result["empty_labels"]
                stats["num_invalid_annotations"] += result["invalid_annotations"]
                warning = result["warning"]
                if warning and len(warning_messages) < max_warning_messages:
                    warning_messages.append(warning)

                if pbar is not None:
                    pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()

    for ann_id, ann in enumerate(annotations, start=1):
        ann["id"] = ann_id

    stats["num_images"] = len(images)
    stats["num_annotations"] = len(annotations)

    for message in warning_messages:
        print("[WARN] {}".format(message))
    if stats["num_invalid_annotations"] > len(warning_messages):
        print(
            "[WARN] {:,} additional invalid annotations were skipped.".format(
                stats["num_invalid_annotations"] - len(warning_messages)
            )
        )

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": build_categories(class_names),
    }
    return coco_dict, stats


def quote_yaml_value(value):
    return "'{}'".format(str(value).replace("'", "''"))


def write_dataset_yaml(dataset_yaml_path, dataset_dir, num_classes, split_outputs):
    train_output = split_outputs.get("train")
    if train_output is None:
        first_split = sorted(split_outputs.keys())[0]
        train_output = split_outputs[first_split]
        print("[WARN] Train split not available. Using '{}' for TrainDataset.".format(first_split))

    eval_output = split_outputs.get("val")
    if eval_output is None:
        eval_output = split_outputs.get("test")
        if eval_output is not None:
            print("[WARN] Val split not available. Using test split for EvalDataset.")
        else:
            eval_output = train_output
            print("[WARN] Val/Test split not available. Using TrainDataset for EvalDataset.")

    test_output = split_outputs.get("test")
    if test_output is None:
        test_output = split_outputs.get("val")
        if test_output is not None:
            print("[WARN] Test split not available. Using val split annotation for TestDataset.")
        else:
            test_output = train_output
            print("[WARN] Test/Val split not available. Using train split annotation for TestDataset.")

    yaml_text = "\n".join(
        [
            "metric: COCO",
            "num_classes: {}".format(num_classes),
            "",
            "TrainDataset:",
            "  !COCODataSet",
            "    image_dir: {}".format(quote_yaml_value(train_output["image_dir"])),
            "    anno_path: {}".format(quote_yaml_value(train_output["anno_path"])),
            "    dataset_dir: {}".format(quote_yaml_value(dataset_dir)),
            "    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']",
            "",
            "EvalDataset:",
            "  !COCODataSet",
            "    image_dir: {}".format(quote_yaml_value(eval_output["image_dir"])),
            "    anno_path: {}".format(quote_yaml_value(eval_output["anno_path"])),
            "    dataset_dir: {}".format(quote_yaml_value(dataset_dir)),
            "",
            "TestDataset:",
            "  !ImageFolder",
            "    image_dir: {}".format(quote_yaml_value(test_output["image_dir"])),
            "    anno_path: {}".format(quote_yaml_value(test_output["anno_path"])),
            "    dataset_dir: {}".format(quote_yaml_value(dataset_dir)),
            "",
        ]
    )

    dataset_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml_path.write_text(yaml_text, encoding="utf-8")


def main():
    args = parse_args()

    yolo_yaml_path = Path(args.yolo_yaml).expanduser().resolve()
    data_cfg = load_data_yaml(yolo_yaml_path)

    dataset_dir = yolo_yaml_path.parent.resolve()
    class_names = load_class_names(data_cfg, yolo_yaml_path)
    data_root = resolve_data_root(dataset_dir, data_cfg, yolo_yaml_path)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else dataset_dir / "coco_annotations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_yaml_path = (
        Path(args.output_yaml_path).expanduser().resolve()
        if args.output_yaml_path
        else output_dir / "{}_coco.yml".format(dataset_dir.name)
    )

    num_workers = args.num_workers if (0 < args.num_workers <= os.cpu_count()) else (os.cpu_count() or 1)

    print("Dataset dir:  '{}'".format(dataset_dir))
    print("YOLO yaml:    '{}'".format(yolo_yaml_path))
    print("Data root:    '{}'".format(data_root))
    print("Classes ({:,}):  {}".format(len(class_names), class_names))
    print("Output dir:   '{}'".format(output_dir))
    print("Output YAML:  '{}'".format(output_yaml_path))
    print("Workers:      {:,}".format(num_workers))
    print()

    split_outputs = {}

    for split_name in ["train", "val", "test"]:
        image_dir = resolve_split_image_dir(split_name, data_cfg, data_root, yolo_yaml_path)
        if image_dir is None:
            continue

        label_dir = infer_label_dir(split_name, image_dir, dataset_dir, data_root)
        if not label_dir.exists():
            print("[WARN] Label directory not found for split '{}': {}".format(split_name, label_dir))

        print("=" * 50)
        print("[INFO] Processing split '{}'".format(split_name))
        print("[INFO]   - Image dir: '{}'".format(image_dir))
        print("[INFO]   - Label dir: '{}'".format(label_dir))

        coco_dict, stats = convert_split(
            split_name=split_name,
            image_dir=image_dir,
            label_dir=label_dir,
            class_names=class_names,
            num_workers=num_workers,
            use_tqdm=not args.no_tqdm,
            strict=args.strict,
        )

        output_json = output_dir / "instances_{}.json".format(split_name)
        print("[INFO] Writing COCO JSON to '{}' ...".format(output_json))
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(coco_dict, f)

        split_outputs[split_name] = {
            "image_dir": str(image_dir.resolve()),
            "anno_path": str(output_json.resolve()),
        }

        print(
            f"[INFO] ✅ Done '{split_name}':",
            f"- total_images = {stats['num_images']:,}",
            f"- annotations = {stats['num_annotations']:,}",
            f"- missing_labels(removed from the final dataset) = {stats['num_missing_labels']:,}",
            f"- empty_labels = {stats['num_empty_labels']:,}",
            f"- invalid = {stats['num_invalid_annotations']:,}",
            sep="\n\t\t",
        )
        print("=" * 50)
        print()

    if not split_outputs:
        raise RuntimeError("No split was converted. Make sure train/val/test paths exist in data.yaml.")

    num_classes = len(class_names)
    write_dataset_yaml(
        dataset_yaml_path=output_yaml_path,
        dataset_dir=str(dataset_dir),
        num_classes=num_classes,
        split_outputs=split_outputs,
    )
    print("[INFO] Generated PaddleDetection dataset YAML: {}".format(output_yaml_path))


if __name__ == "__main__":
    main()
