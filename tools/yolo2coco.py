"""
Convert a YOLO dataset to COCO JSON files using supervision,
then auto-generate a PaddleDetection COCO dataset YAML.

Expected flow:
1) Read train/val/test image paths from data.yaml.
2) Build one supervision DetectionDataset per existing split.
3) Save instances_train.json / instances_val.json / instances_test.json.
4) Auto-create a PaddleDetection COCO yaml file.

Usage:
    pip install supervision==0.27.0 PyYAML
    python yolo2coco.py --dataset_dir /workspace/open-images-v7 --data_yaml /workspace/open-images-v7/data.yaml
"""

import argparse
import json
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Convert YOLO splits to COCO JSON with supervision")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="YOLO dataset root directory",
    )
    parser.add_argument(
        "--data_yaml",
        type=str,
        default="data.yaml",
        help="Path to YOLO data.yaml (absolute, or relative to --dataset_dir)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for COCO json outputs (default: <dataset_dir>/coco_annotations)",
    )
    parser.add_argument(
        "--dataset_yaml_path",
        type=str,
        default=None,
        help="Output PaddleDetection yaml path (default: <output_dir>/<dataset_name>_coco.yml)",
    )
    return parser.parse_args()


def resolve_data_yaml_path(dataset_dir, data_yaml_arg):
    data_yaml_path = Path(data_yaml_arg).expanduser()
    if data_yaml_path.is_absolute():
        return data_yaml_path.resolve()
    return (dataset_dir / data_yaml_path).resolve()


def load_data_yaml(data_yaml_path):
    if not data_yaml_path.exists():
        raise FileNotFoundError("data.yaml does not exist: {}".format(data_yaml_path))
    with data_yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("data.yaml must be a dictionary-like YAML")
    return data


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


def convert_categories_to_one_based(coco_json_path):
    with coco_json_path.open("r", encoding="utf-8") as f:
        coco_data = json.load(f)

    categories = coco_data.get("categories", [])
    if not categories:
        return

    category_ids = [int(category.get("id", 0)) for category in categories]
    if min(category_ids) != 0:
        return

    for category in categories:
        category["id"] = int(category["id"]) + 1

    for annotation in coco_data.get("annotations", []):
        annotation["category_id"] = int(annotation["category_id"]) + 1

    with coco_json_path.open("w", encoding="utf-8") as f:
        json.dump(coco_data, f)


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

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError("dataset_dir does not exist: {}".format(dataset_dir))

    data_yaml_path = resolve_data_yaml_path(dataset_dir, args.data_yaml)
    data_cfg = load_data_yaml(data_yaml_path)
    data_root = resolve_data_root(dataset_dir, data_cfg, data_yaml_path)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else dataset_dir / "coco_annotations"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_yaml_path = (
        Path(args.dataset_yaml_path).expanduser().resolve()
        if args.dataset_yaml_path
        else output_dir / "{}_coco.yml".format(dataset_dir.name)
    )

    try:
        import supervision as sv
    except ImportError as ex:
        raise ImportError("Missing dependency `supervision`. Please run: pip install supervision") from ex

    print("Dataset dir: {}".format(dataset_dir))
    print("data.yaml: {}".format(data_yaml_path))
    print("data_root: {}".format(data_root))
    print("Output dir: {}".format(output_dir))
    print("Dataset yaml output: {}".format(dataset_yaml_path))

    split_outputs = {}
    class_names = None

    for split_name in ["train", "val", "test"]:
        image_dir = resolve_split_image_dir(split_name, data_cfg, data_root, data_yaml_path)
        if image_dir is None:
            continue

        label_dir = infer_label_dir(split_name, image_dir, dataset_dir, data_root)
        if not label_dir.exists():
            print("[WARN] Label directory not found for split '{}': {}".format(split_name, label_dir))

        print("-" * 40)
        print(
            "Converting split '{}' with images='{}' labels='{}'".format(
                split_name,
                image_dir,
                label_dir,
            )
        )

        dataset = sv.DetectionDataset.from_yolo(
            images_directory_path=str(image_dir),
            annotations_directory_path=str(label_dir),
            data_yaml_path=str(data_yaml_path),
        )

        output_json = output_dir / "instances_{}.json".format(split_name)
        dataset.as_coco(annotations_path=str(output_json))
        convert_categories_to_one_based(output_json)

        split_outputs[split_name] = {
            "image_dir": str(image_dir.resolve()),
            "anno_path": str(output_json.resolve()),
        }

        if class_names is None:
            class_names = list(dataset.classes)

        print(
            "✅ Saved '{}' (images={}, classes={})".format(
                output_json,
                len(dataset),
                len(dataset.classes),
            )
        )
        print("-" * 40 + "\n")

    if not split_outputs:
        raise RuntimeError("No split was converted. Make sure train/val/test paths exist in data.yaml.")

    num_classes = len(class_names) if class_names is not None else 0
    write_dataset_yaml(
        dataset_yaml_path=dataset_yaml_path,
        dataset_dir=str(dataset_dir),
        num_classes=num_classes,
        split_outputs=split_outputs,
    )
    print("Generated dataset yaml: {}".format(dataset_yaml_path))


if __name__ == "__main__":
    main()
