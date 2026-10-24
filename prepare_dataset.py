
#!/usr/bin/env python3
"""
Inspect and combine:
  1. Traffic Cones dataset
  2. Self-driving Cars dataset

Unified classes:
  0: cone
  1: barrier (reserved for future dataset)
  2: stop_sign

Run:
  python prepare_dataset.py inspect
  python prepare_dataset.py organize
"""

from pathlib import Path
from collections import Counter
import argparse
import hashlib
import random
import shutil
import sys

try:
    import yaml
except ImportError:
    sys.exit("Install PyYAML first: pip install pyyaml")


# ---------------------------------------------------------
# CONFIGURATION: update OUTPUT if your project is elsewhere
# ---------------------------------------------------------

SOURCES = {
    "cones": Path(
        "/home/ammaar/Downloads/"
        "Traffic Cones.v2-augmentated-version.yolov11"
    ),
    "cars": Path("/home/ammaar/Downloads/archive/car"),
}

OUTPUT = Path("/home/ammaar/Documents/zzz HACKS&PROJS/eric_assisment/robot-nav-distance/data")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

# Desired class IDs in the combined dataset
TARGET_CLASSES = {
    "cone": 0,
    "barrier": 1,
    "stop_sign": 2,
}

# Class-name aliases used in source dataset YAML files.
ALIASES = {
    "cone": {"cone", "traffic cone", "traffic-cone", "traffic_cone"},
    "barrier": {
        "barrier", "road barrier", "road-barrier",
        "construction barrier", "construction-barrier",
        "traffic barrier", "traffic barrel", "traffic-barrel",
    },
    "stop_sign": {"stop", "stop sign", "stop-sign", "stop_sign"},
}

SPLITS = ("train", "val", "test")
RATIOS = (0.70, 0.20, 0.10)
SEED = 42


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def normalize_name(name):
    return str(name).strip().lower().replace("_", " ")


def read_yaml(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_class_names(data):
    names = data.get("names", {})

    if isinstance(names, list):
        return {i: str(name) for i, name in enumerate(names)}

    if isinstance(names, dict):
        return {int(i): str(name) for i, name in names.items()}

    return {}


def resolve_target_class(name):
    normalized = normalize_name(name)

    for target, aliases in ALIASES.items():
        if normalized in {normalize_name(a) for a in aliases}:
            return TARGET_CLASSES[target]

    return None


def find_yaml_files(root):
    # Prefer the standard Roboflow data.yaml filename.
    files = list(root.rglob("data.yaml"))
    if not files:
        files = list(root.rglob("*.yaml"))
    return files


def find_label_for_image(image_path):
    """
    Common YOLO layout:
      train/images/example.jpg
      train/labels/example.txt
    """
    parts = list(image_path.parts)

    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            label_parts = parts.copy()
            label_parts[i] = "labels"
            label_path = Path(*label_parts).with_suffix(".txt")
            if label_path.is_file():
                return label_path

    # Also support labels stored next to images.
    sibling = image_path.with_suffix(".txt")
    if sibling.is_file():
        return sibling

    return None


def get_existing_split(image_path):
    for part in image_path.parts:
        part = part.lower()
        if part in ("train", "training"):
            return "train"
        if part in ("val", "valid", "validation"):
            return "val"
        if part in ("test", "testing"):
            return "test"
    return None


def read_yolo_labels(label_path, class_map):
    """
    Return target-class annotation lines with remapped IDs.
    Keep only annotations with valid target class IDs.
    """
    remapped = []
    counts = Counter()

    if not label_path or not label_path.is_file():
        return remapped, counts

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split()

            if len(fields) != 5:
                continue

            try:
                old_id = int(fields[0])
                coords = [float(v) for v in fields[1:]]
            except ValueError:
                continue

            if not all(0 <= v <= 1 for v in coords):
                continue

            if old_id not in class_map:
                continue

            new_id = class_map[old_id]
            remapped.append(
                f"{new_id} " + " ".join(f"{v:.6f}" for v in coords)
            )
            counts[new_id] += 1

    return remapped, counts


def discover_dataset(source_name, root):
    """
    Discover images and annotations and determine source class names
    from the nearest applicable dataset YAML.
    """
    if not root.is_dir():
        print(f"\n[ERROR] Missing source directory: {root}")
        return []

    yaml_files = find_yaml_files(root)
    print(f"\n{'=' * 65}")
    print(f"DATASET: {source_name}")
    print(f"ROOT:    {root}")
    print(f"YAML files found: {len(yaml_files)}")

    for path in yaml_files[:10]:
        print(f"  - {path.relative_to(root)}")

    # Find images, excluding hidden directories.
    images = [
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not any(part.startswith(".") for part in p.parts)
    ]

    print(f"Images found: {len(images)}")

    if not yaml_files:
        print("[ERROR] No YAML class mapping found.")
        return []

    # Read class mappings from each YAML.
    yaml_info = []
    for yaml_path in yaml_files:
        try:
            data = read_yaml(yaml_path)
            names = get_class_names(data)
            yaml_info.append((yaml_path, names))
            if names:
                print(f"\nClasses in {yaml_path.relative_to(root)}:")
                for class_id, name in names.items():
                    print(f"  {class_id}: {name}")
        except Exception as exc:
            print(f"[WARNING] Could not read {yaml_path}: {exc}")

    records = []
    annotation_counts = Counter()
    matched = 0
    missing_labels = 0
    skipped = 0

    for image_path in images:
        label_path = find_label_for_image(image_path)

        if label_path is None:
            missing_labels += 1
            continue

        # Select the closest YAML whose directory contains this image.
        candidates = [
            (yp, names) for yp, names in yaml_info
            if yp.parent in image_path.parents
        ]

        if not candidates:
            candidates = yaml_info

        candidates.sort(key=lambda item: len(item[0].parent.parts),
                        reverse=True)
        _, names = candidates[0]

        # Only keep source classes that match our target classes.
        class_map = {}
        for old_id, class_name in names.items():
            new_id = resolve_target_class(class_name)
            if new_id is not None:
                class_map[old_id] = new_id

        if not class_map:
            skipped += 1
            continue

        labels, counts = read_yolo_labels(label_path, class_map)

        if not labels:
            skipped += 1
            continue

        matched += 1
        annotation_counts.update(counts)

        records.append({
            "source": source_name,
            "image": image_path,
            "label": label_path,
            "labels": labels,
            "counts": counts,
            "existing_split": get_existing_split(image_path),
        })

    print(f"\nImages with target annotations: {matched}")
    print(f"Images missing labels:          {missing_labels}")
    print(f"Images skipped/no target class: {skipped}")
    print("Target object annotations:")
    for target, class_id in TARGET_CLASSES.items():
        print(f"  {target}: {annotation_counts[class_id]}")

    return records


def inspect_all():
    all_records = []

    for source_name, root in SOURCES.items():
        all_records.extend(discover_dataset(source_name, root))

    print(f"\n{'=' * 65}")
    print("INSPECTION SUMMARY")
    print(f"Total usable images: {len(all_records)}")

    source_counts = Counter(r["source"] for r in all_records)
    object_counts = Counter()
    for record in all_records:
        object_counts.update(record["counts"])

    for source_name in SOURCES:
        print(f"  {source_name}: {source_counts[source_name]} images")

    for target, class_id in TARGET_CLASSES.items():
        print(f"  {target}: {object_counts[class_id]} annotations")

    print(f"\nOutput directory configured as:\n  {OUTPUT}")
    print("No files were changed.")
    return all_records


# ---------------------------------------------------------
# SPLITTING AND ORGANIZATION
# ---------------------------------------------------------

def split_records(records, rng):
    """
    Split records separately by source so that the cone dataset
    and stop-sign dataset each receive the requested ratio.
    """
    assignments = {}

    for source_name in SOURCES:
        subset = [r for r in records if r["source"] == source_name]
        rng.shuffle(subset)

        n = len(subset)
        n_train = int(n * RATIOS[0])
        n_val = int(n * RATIOS[1])

        for record in subset[:n_train]:
            assignments[id(record)] = "train"

        for record in subset[n_train:n_train + n_val]:
            assignments[id(record)] = "val"

        for record in subset[n_train + n_val:]:
            assignments[id(record)] = "test"

    return assignments


def safe_filename(record):
    """
    Prefix filenames with source and a stable hash to prevent collisions.
    """
    relative = str(record["image"])
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:10]
    stem = record["image"].stem
    return f'{record["source"]}_{stem}_{digest}'


def organize(records, force=False):
    if not records:
        print("No usable annotated images found. Nothing to organize.")
        return

    if OUTPUT.exists() and any(OUTPUT.iterdir()) and not force:
        print(f"\nOutput directory is not empty: {OUTPUT}")
        print("Choose a new OUTPUT path, or use --force to replace it.")
        print("--force deletes the existing OUTPUT directory first.")
        return

    if force and OUTPUT.exists():
        print(f"Removing existing output directory: {OUTPUT}")
        shutil.rmtree(OUTPUT)

    rng = random.Random(SEED)
    assignments = split_records(records, rng)

    for split in SPLITS:
        (OUTPUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_counts = Counter()
    class_counts = Counter()

    for record in records:
        split = assignments[id(record)]
        name = safe_filename(record)

        image_dest = OUTPUT / "images" / split / (
            name + record["image"].suffix.lower()
        )
        label_dest = OUTPUT / "labels" / split / (name + ".txt")

        shutil.copy2(record["image"], image_dest)
        label_dest.write_text(
            "\n".join(record["labels"]) + "\n",
            encoding="utf-8",
        )

        split_counts[split] += 1
        class_counts.update(record["counts"])

    yaml_content = f"""# Unified robotics navigation dataset
path: {OUTPUT}
train: images/train
val: images/val
test: images/test

nc: 3
names:
  0: cone
  1: barrier
  2: stop_sign
"""

    (OUTPUT / "dataset.yaml").write_text(
        yaml_content, encoding="utf-8"
    )

    print(f"\n{'=' * 65}")
    print("DATASET ORGANIZATION COMPLETE")
    print(f"Output: {OUTPUT}")

    total = sum(split_counts.values())
    for split in SPLITS:
        count = split_counts[split]
        pct = 100 * count / total if total else 0
        print(f"  {split:5}: {count:6} images ({pct:.1f}%)")

    print("\nObject annotations:")
    for target, class_id in TARGET_CLASSES.items():
        print(f"  {target}: {class_counts[class_id]}")

    print(f"\nDataset YAML: {OUTPUT / 'dataset.yaml'}")
    print("\nOriginal source datasets have not been modified.")


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inspect and organize robotics YOLO datasets."
    )
    parser.add_argument(
        "mode",
        choices=["inspect", "organize"],
        help="inspect dataset contents or organize into a unified dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="delete and rebuild the output directory (organize mode only)",
    )
    args = parser.parse_args()

    records = inspect_all()

    if args.mode == "organize":
        organize(records, force=args.force)


if __name__ == "__main__":
    main()
