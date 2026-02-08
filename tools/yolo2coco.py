"""
This script converts YOLO-format detection labels to COCO JSON format, and generates a PaddleDetection dataset YAML.

Usage:
    pip install Pillow PyYAML tqdm
    python tools/yolo2coco.py \
    --dataset_dir /workspace/open-images-v7 \
    --data_yaml /workspace/open-images-v7/data.yaml \
    --splits train val \
    --num_workers 0
"""

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "train": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert YOLO detection labels to COCO JSON format")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Dataset root path",
    )
    parser.add_argument(
        "--data_yaml",
        type=str,
        default=None,
        help="Path to data.yaml. Defaults to dataset_dir/data.yaml",
    )
    parser.add_argument(
        "--class_names",
        type=str,
        nargs="*",
        default=None,
        help="Class names. Overrides names from data.yaml",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="Split names to convert (e.g. train val test). Auto-inferred if omitted",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for COCO JSON files. Defaults to dataset_dir/coco_annotations",
    )
    parser.add_argument(
        "--dataset_yaml_path",
        type=str,
        default=None,
        help=(
            "Path to auto-generated PaddleDetection COCO dataset yaml. "
            "Defaults to <repo_root>/configs/datasets/<dataset_name>_coco.yml"
        ),
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


def canonical_split_name(split_name):
    return SPLIT_ALIASES.get(split_name.lower(), split_name.lower())


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def load_data_yaml(data_yaml_path):
    if not data_yaml_path.exists():
        return {}
    with data_yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("data.yaml must be a dictionary-like YAML")
    return data


def load_class_names(data_cfg, data_yaml_path, class_names):
    if class_names:
        return [str(name) for name in class_names]

    names = data_cfg.get("names", None)
    if names is None:
        raise ValueError(
            "Class names are missing. Provide --class_names or include `names` in {}".format(data_yaml_path)
        )

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


def infer_requested_splits(args_splits, data_cfg, dataset_dir):
    if args_splits:
        return args_splits

    split_candidates = []
    for key in ["train", "val", "valid", "test"]:
        if key in data_cfg:
            split_candidates.append(key)
    if split_candidates:
        return split_candidates

    for key in ["train", "val", "valid", "test"]:
        if (dataset_dir / key).exists() or (dataset_dir / "images" / key).exists():
            split_candidates.append(key)
    if split_candidates:
        return split_candidates

    return ["train", "val", "test"]


def resolve_data_root(dataset_dir, data_cfg, data_yaml_path):
    data_root = data_cfg.get("path", None)
    if data_root is None:
        return dataset_dir

    data_root_path = Path(str(data_root)).expanduser()
    if not data_root_path.is_absolute():
        data_root_path = (data_yaml_path.parent / data_root_path).resolve()
    else:
        data_root_path = data_root_path.resolve()

    if data_root_path.exists():
        return data_root_path

    print(
        "[WARN] data.yaml path '{}' does not exist. Falling back to dataset_dir '{}'".format(
            data_root_path, dataset_dir
        )
    )
    return dataset_dir


def resolve_split_value_path(split_value, data_root, data_yaml_path):
    if isinstance(split_value, list):
        raise ValueError("List-based split paths are not supported yet. Got: {}".format(split_value))
    if not isinstance(split_value, str):
        raise ValueError("Split path must be a string. Got: {}".format(type(split_value)))

    path = Path(split_value).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidate_from_data_root = (data_root / path).resolve()
    if candidate_from_data_root.exists():
        return candidate_from_data_root

    candidate_from_yaml = (data_yaml_path.parent / path).resolve()
    if candidate_from_yaml.exists():
        return candidate_from_yaml

    stripped_parts = [part for part in path.parts if part not in (".", "..")]
    if stripped_parts:
        candidate_stripped = (data_root / Path(*stripped_parts)).resolve()
        if candidate_stripped.exists():
            return candidate_stripped

    return candidate_from_data_root


def infer_label_dir(image_dir, dataset_dir, data_root, split_name, canonical_split):
    candidates = []

    image_parts = list(image_dir.parts)
    if "images" in image_parts:
        idx = image_parts.index("images")
        replaced = image_parts[:]
        replaced[idx] = "labels"
        candidates.append(Path(*replaced))

    if image_dir.parent.name == "images":
        candidates.append(image_dir.parent.parent / "labels" / image_dir.name)

    candidates.append(dataset_dir / "labels" / split_name)
    candidates.append(dataset_dir / "labels" / canonical_split)
    candidates.append(data_root / "labels" / image_dir.name)

    uniq_candidates = []
    seen = set()
    for c in candidates:
        c_key = str(c)
        if c_key not in seen:
            uniq_candidates.append(c)
            seen.add(c_key)

    for c in uniq_candidates:
        if c.exists():
            return c.resolve()

    return uniq_candidates[0].resolve()


def resolve_split_dirs(split_name, dataset_dir, data_root, data_cfg, data_yaml_path):
    canonical = canonical_split_name(split_name)

    candidate_keys = []
    if split_name in data_cfg:
        candidate_keys.append(split_name)
    if canonical == "val":
        for key in ["val", "valid", "validation"]:
            if key in data_cfg and key not in candidate_keys:
                candidate_keys.append(key)
    elif canonical in data_cfg and canonical not in candidate_keys:
        candidate_keys.append(canonical)

    image_dir = None
    label_dir = None
    chosen_split_key = split_name

    for key in candidate_keys:
        image_dir = resolve_split_value_path(data_cfg[key], data_root, data_yaml_path)
        chosen_split_key = key
        label_dir = infer_label_dir(
            image_dir=image_dir,
            dataset_dir=dataset_dir,
            data_root=data_root,
            split_name=key,
            canonical_split=canonical,
        )
        break

    if image_dir is None:
        fs_names = [split_name]
        if canonical == "val":
            fs_names.extend(["val", "valid", "validation"])
        elif canonical != split_name:
            fs_names.append(canonical)

        dedup_fs_names = []
        seen = set()
        for name in fs_names:
            if name not in seen:
                dedup_fs_names.append(name)
                seen.add(name)

        for fs_name in dedup_fs_names:
            split_style_dir = dataset_dir / fs_name
            direct_image_dir = split_style_dir / "images"
            grouped_image_dir = dataset_dir / "images" / fs_name
            if direct_image_dir.exists():
                image_dir = direct_image_dir
                label_dir = split_style_dir / "labels"
                chosen_split_key = fs_name
                break
            if grouped_image_dir.exists():
                image_dir = grouped_image_dir
                label_dir = dataset_dir / "labels" / fs_name
                chosen_split_key = fs_name
                break

    if image_dir is None or not image_dir.exists():
        raise FileNotFoundError(
            "Cannot resolve image directory for split '{}'. Check --dataset_dir, --data_yaml and --splits.".format(
                split_name
            )
        )

    if label_dir is None:
        label_dir = infer_label_dir(
            image_dir=image_dir,
            dataset_dir=dataset_dir,
            data_root=data_root,
            split_name=chosen_split_key,
            canonical_split=canonical,
        )

    return {
        "requested": split_name,
        "canonical": canonical,
        "split_key": chosen_split_key,
        "image_dir": image_dir.resolve(),
        "label_dir": label_dir.resolve(),
    }


def collect_images(image_dir):
    image_paths = []
    for root, _, files in os.walk(image_dir):
        root_path = Path(root)
        for file_name in files:
            suffix = Path(file_name).suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                image_paths.append(root_path / file_name)

    image_paths.sort(key=lambda p: p.relative_to(image_dir).as_posix())
    return image_paths


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

    if not label_path.exists():
        result["missing_labels"] = 1
        return result

    raw = label_path.read_text(encoding="utf-8").strip()
    if not raw:
        result["empty_labels"] = 1
        return result

    for line_id, line in enumerate(raw.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            message = "{}:{} expected 5 columns, got {}".format(label_path, line_id, len(parts))
            if strict:
                raise ValueError(message)
            result["invalid_annotations"] += 1
            if result["warning"] is None:
                result["warning"] = message
            continue

        try:
            class_id = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
        except ValueError:
            message = "{}:{} contains non-numeric values".format(label_path, line_id)
            if strict:
                raise ValueError(message)
            result["invalid_annotations"] += 1
            if result["warning"] is None:
                result["warning"] = message
            continue

        if not all(math.isfinite(x) for x in [xc, yc, bw, bh]):
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

        x1 = (xc - bw / 2.0) * width
        y1 = (yc - bh / 2.0) * height
        x2 = (xc + bw / 2.0) * width
        y2 = (yc + bh / 2.0) * height

        x1 = clamp(x1, 0.0, float(width))
        y1 = clamp(y1, 0.0, float(height))
        x2 = clamp(x2, 0.0, float(width))
        y2 = clamp(y2, 0.0, float(height))

        box_w = x2 - x1
        box_h = y2 - y1
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
                "area": box_w * box_h,
                "iscrowd": 0,
                "segmentation": [],
            }
        )

    return result


def convert_split(split_info, class_names, num_workers, use_tqdm, strict):
    image_dir = split_info["image_dir"]
    label_dir = split_info["label_dir"]

    image_paths = collect_images(image_dir)
    if not image_paths:
        raise ValueError("No images found in {}".format(image_dir))

    if not label_dir.exists():
        print("[WARN] Label directory not found: {}. All images in this split become empty samples.".format(label_dir))

    jobs = []
    for image_id, image_path in enumerate(image_paths, start=1):
        rel_file = image_path.relative_to(image_dir)
        rel_file_name = rel_file.as_posix()
        label_path = label_dir / rel_file.with_suffix(".txt")
        jobs.append(
            (
                image_id,
                str(image_path),
                rel_file_name,
                str(label_path),
                len(class_names),
                strict,
            )
        )

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

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        iterator = executor.map(process_single_image, jobs)
        if use_tqdm:
            iterator = tqdm(
                iterator,
                total=len(jobs),
                desc="Converting '{}'".format(split_info["canonical"]),
                unit="img",
            )
        for result in iterator:
            images.append(result["image"])
            annotations.extend(result["annotations"])
            stats["num_missing_labels"] += result["missing_labels"]
            stats["num_empty_labels"] += result["empty_labels"]
            stats["num_invalid_annotations"] += result["invalid_annotations"]
            warning = result["warning"]
            if warning and len(warning_messages) < max_warning_messages:
                warning_messages.append(warning)

    for ann_id, ann in enumerate(annotations, start=1):
        ann["id"] = ann_id

    stats["num_images"] = len(images)
    stats["num_annotations"] = len(annotations)

    for message in warning_messages:
        print("[WARN] {}".format(message))
    if stats["num_invalid_annotations"] > len(warning_messages):
        print(
            "[WARN] {} additional invalid annotations were skipped.".format(
                stats["num_invalid_annotations"] - len(warning_messages)
            )
        )

    return {
        "images": images,
        "annotations": annotations,
        "categories": build_categories(class_names),
    }, stats


def quote_yaml_value(value):
    return "'{}'".format(str(value).replace("'", "''"))


def write_dataset_yaml(dataset_yaml_path, dataset_dir, num_classes, split_artifacts):
    if "train" in split_artifacts:
        train_artifact = split_artifacts["train"]
    else:
        first_key = sorted(split_artifacts.keys())[0]
        train_artifact = split_artifacts[first_key]
        print("[WARN] Train split was not found. Using '{}' for TrainDataset.".format(first_key))

    if "val" in split_artifacts:
        eval_artifact = split_artifacts["val"]
    elif "test" in split_artifacts:
        eval_artifact = split_artifacts["test"]
        print("[WARN] Val split was not found. Using 'test' for EvalDataset.")
    else:
        eval_artifact = train_artifact
        print("[WARN] Val/Test split was not found. Using TrainDataset for EvalDataset.")

    yaml_text = "\n".join(
        [
            "metric: COCO",
            "num_classes: {}".format(num_classes),
            "",
            "TrainDataset:",
            "  !COCODataSet",
            "    image_dir: {}".format(quote_yaml_value(train_artifact["image_dir"])),
            "    anno_path: {}".format(quote_yaml_value(train_artifact["anno_path"])),
            "    dataset_dir: {}".format(quote_yaml_value(dataset_dir)),
            "    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']",
            "",
            "EvalDataset:",
            "  !COCODataSet",
            "    image_dir: {}".format(quote_yaml_value(eval_artifact["image_dir"])),
            "    anno_path: {}".format(quote_yaml_value(eval_artifact["anno_path"])),
            "    dataset_dir: {}".format(quote_yaml_value(dataset_dir)),
            "",
            "TestDataset:",
            "  !ImageFolder",
            "    anno_path: {}".format(quote_yaml_value(eval_artifact["anno_path"])),
            "    dataset_dir: {}".format(quote_yaml_value(dataset_dir)),
            "",
        ]
    )

    dataset_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml_path.write_text(yaml_text, encoding="utf-8")


def main():
    args = parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError("dataset_dir does not exist: {}".format(dataset_dir))

    data_yaml_path = Path(args.data_yaml).expanduser().resolve() if args.data_yaml else dataset_dir / "data.yaml"
    data_cfg = load_data_yaml(data_yaml_path)
    class_names = load_class_names(data_cfg, data_yaml_path, args.class_names)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else dataset_dir / "coco_annotations"
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    dataset_yaml_path = (
        Path(args.dataset_yaml_path).expanduser().resolve()
        if args.dataset_yaml_path
        else repo_root / "configs" / "datasets" / "{}_coco.yml".format(dataset_dir.name)
    )

    num_workers = args.num_workers if args.num_workers > 0 else (os.cpu_count() or 1)
    data_root = resolve_data_root(dataset_dir, data_cfg, data_yaml_path)
    requested_splits = infer_requested_splits(args.splits, data_cfg, dataset_dir)

    print("Dataset dir: {}".format(dataset_dir))
    print("Data yaml: {}".format(data_yaml_path))
    print("Data root: {}".format(data_root))
    print("Class names ({}): {}".format(len(class_names), class_names))
    print("Output dir: {}".format(output_dir))
    print("Dataset yaml output: {}".format(dataset_yaml_path))
    print("Using worker threads: {}".format(num_workers))
    print("Requested splits: {}".format(requested_splits))

    split_artifacts = {}
    for split in requested_splits:
        split_info = resolve_split_dirs(
            split_name=split,
            dataset_dir=dataset_dir,
            data_root=data_root,
            data_cfg=data_cfg,
            data_yaml_path=data_yaml_path,
        )

        canonical = split_info["canonical"]
        if canonical in split_artifacts:
            print(
                "[WARN] Split '{}' maps to canonical '{}', which is already converted. Skipping duplicate.".format(
                    split, canonical
                )
            )
            continue

        print(
            "Resolving split '{}' -> image_dir='{}', label_dir='{}'".format(
                split, split_info["image_dir"], split_info["label_dir"]
            )
        )
        coco_dict, stats = convert_split(
            split_info=split_info,
            class_names=class_names,
            num_workers=num_workers,
            use_tqdm=not args.no_tqdm,
            strict=args.strict,
        )

        output_file = output_dir / "instances_{}.json".format(canonical)
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(coco_dict, f)

        split_artifacts[canonical] = {
            "image_dir": split_info["image_dir"],
            "anno_path": output_file,
        }
        print(
            "Converted '{}' -> '{}' (images={}, annotations={}, missing_labels={}, empty_labels={}, invalid_annotations={})".format(
                split,
                output_file,
                stats["num_images"],
                stats["num_annotations"],
                stats["num_missing_labels"],
                stats["num_empty_labels"],
                stats["num_invalid_annotations"],
            )
        )

    if not split_artifacts:
        raise RuntimeError("No split was converted. Check --splits and dataset layout.")

    write_dataset_yaml(
        dataset_yaml_path=dataset_yaml_path,
        dataset_dir=dataset_dir,
        num_classes=len(class_names),
        split_artifacts=split_artifacts,
    )
    print("Generated dataset yaml: {}".format(dataset_yaml_path))


if __name__ == "__main__":
    main()
