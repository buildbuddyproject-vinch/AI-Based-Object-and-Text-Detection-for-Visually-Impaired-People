"""
Dataset Preparation Tool
===========================
Converts each raw dataset under dataset/ into a ready-to-train YOLO
layout under dataset_prepared/<domain>/, WITHOUT ever modifying the
original files in dataset/. Every conversion is informed by the actual
findings from tools/analyze_datasets.py - see each domain's docstring
below for exactly what it does and why.

Run:
    python tools/prepare_datasets.py --dataset indoor
    python tools/prepare_datasets.py --dataset household --min-class-count 20
    python tools/prepare_datasets.py --dataset outdoor
    python tools/prepare_datasets.py --dataset footpath
    python tools/prepare_datasets.py --dataset all
"""
import argparse
import json
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_common import is_watermark_class, voc_box_to_yolo  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
PREPARED_DIR = os.path.join(BASE_DIR, "dataset_prepared")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _write_yaml(path, train_dir, val_dir, class_names):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"train: {train_dir}\n")
        f.write(f"val: {val_dir}\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names:\n")
        for name in class_names:
            f.write(f"- {name}\n")


# ---------------------------------------------------------------------------
# INDOOR - already a clean Roboflow YOLO export. The only real problem
# (per analyze_datasets.py) is data.yaml pointing at a Google Colab
# path. Copies the already-correct train/valid/test folders as-is and
# writes a corrected data.yaml - no format conversion needed.
# ---------------------------------------------------------------------------
def prepare_indoor():
    src = os.path.join(DATASET_DIR, "indoor")
    dst = os.path.join(PREPARED_DIR, "indoor")
    if not os.path.isdir(src):
        print("indoor: source dataset not found, skipping")
        return

    yaml_src = os.path.join(src, "data.yaml")
    with open(yaml_src, "r", encoding="utf-8") as f:
        content = f.read()
    class_names = []
    in_names = False
    for line in content.splitlines():
        if line.strip().startswith("names"):
            in_names = True
            continue
        if in_names:
            stripped = line.strip()
            if stripped.startswith("-"):
                class_names.append(stripped.lstrip("- ").strip())
            elif stripped:
                break

    os.makedirs(dst, exist_ok=True)
    for split in ("train", "valid", "test"):
        split_src = os.path.join(src, split)
        split_dst = os.path.join(dst, split)
        if not os.path.isdir(split_src):
            continue
        if os.path.isdir(split_dst):
            shutil.rmtree(split_dst)
        shutil.copytree(split_src, split_dst)

    yaml_dst = os.path.join(dst, "data.yaml")
    with open(yaml_dst, "w", encoding="utf-8") as f:
        f.write(f"train: {os.path.join(dst, 'train', 'images')}\n")
        f.write(f"val: {os.path.join(dst, 'valid', 'images')}\n")
        f.write(f"test: {os.path.join(dst, 'test', 'images')}\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names:\n")
        for name in class_names:
            f.write(f"- {name}\n")

    print(f"indoor: prepared at {dst} ({len(class_names)} classes) - original dataset/indoor/ untouched")


# ---------------------------------------------------------------------------
# HOUSEHOLD - Supervisely JSON -> YOLO. Drops images with zero
# annotated objects (9825 of them, per analyze_datasets.py) and any
# class with fewer than --min-class-count examples (too few to train
# reliably). Images are COPIED (not moved) into the prepared layout;
# originals in dataset/household/ are untouched.
# ---------------------------------------------------------------------------
def prepare_household(min_class_count=20, val_ratio=0.1, seed=42, max_images=None):
    src = os.path.join(DATASET_DIR, "household")
    dst = os.path.join(PREPARED_DIR, "household")
    if not os.path.isdir(src):
        print("household: source dataset not found, skipping")
        return

    # Pass 1: count class frequency across the whole dataset so we know
    # which classes clear min_class_count BEFORE deciding what to keep.
    class_counter = Counter()
    theme_folders = sorted(d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d)))
    for folder in theme_folders:
        ann_dir = os.path.join(src, folder, "ann")
        if not os.path.isdir(ann_dir):
            continue
        for fname in os.listdir(ann_dir):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(ann_dir, fname), "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    continue
            for obj in data.get("objects", []):
                class_counter[obj.get("classTitle", "?")] += 1

    kept_classes = sorted(c for c, n in class_counter.items() if n >= min_class_count)
    class_index = {name: i for i, name in enumerate(kept_classes)}
    dropped_classes = sorted(c for c, n in class_counter.items() if n < min_class_count)
    print(f"household: keeping {len(kept_classes)} classes (>= {min_class_count} examples each), "
          f"dropping {len(dropped_classes)} rare classes: {dropped_classes}")

    # Pass 2: build (image_path, yolo_lines) pairs for every image that
    # ends up with at least one kept-class annotation.
    samples = []
    for folder in theme_folders:
        img_dir = os.path.join(src, folder, "img")
        ann_dir = os.path.join(src, folder, "ann")
        if not os.path.isdir(img_dir) or not os.path.isdir(ann_dir):
            continue
        for fname in os.listdir(ann_dir):
            if not fname.endswith(".json"):
                continue
            img_name = fname[:-5]  # strip ".json", leaves "<file>.jpg"
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                continue
            with open(os.path.join(ann_dir, fname), "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    continue
            img_w = data.get("size", {}).get("width")
            img_h = data.get("size", {}).get("height")
            if not img_w or not img_h:
                continue
            lines = []
            for obj in data.get("objects", []):
                title = obj.get("classTitle", "?")
                if title not in class_index or obj.get("geometryType") != "rectangle":
                    continue
                (x1, y1), (x2, y2) = obj["points"]["exterior"]
                cx, cy, w, h = voc_box_to_yolo(x1, y1, x2, y2, img_w, img_h)
                lines.append(f"{class_index[title]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            if lines:
                samples.append((img_path, img_name, lines))

    if max_images:
        samples = samples[:max_images]

    random.Random(seed).shuffle(samples)
    val_count = max(1, int(len(samples) * val_ratio))
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]

    for split_name, split_samples in (("train", train_samples), ("val", val_samples)):
        img_out = os.path.join(dst, split_name, "images")
        lbl_out = os.path.join(dst, split_name, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for img_path, img_name, lines in split_samples:
            shutil.copy2(img_path, os.path.join(img_out, img_name))
            stem = os.path.splitext(img_name)[0]
            with open(os.path.join(lbl_out, f"{stem}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

    _write_yaml(
        os.path.join(dst, "data.yaml"),
        os.path.join(dst, "train", "images"),
        os.path.join(dst, "val", "images"),
        kept_classes,
    )
    print(f"household: prepared {len(train_samples)} train / {len(val_samples)} val images "
          f"at {dst} - original dataset/household/ untouched")


# ---------------------------------------------------------------------------
# OUTDOOR - Pascal VOC XML -> YOLO, using ONLY 'XML Files/' (the OBB
# 'labels/' set is deliberately skipped - see analyze_datasets.py for
# why its class-id mapping can't be trusted). Roboflow watermark
# "objects" are filtered out entirely. Images referenced by these XML
# files are NOT present in this dataset folder (they were exported
# without their source images) - see the printed warning.
# ---------------------------------------------------------------------------
def prepare_outdoor(val_ratio=0.2, seed=42):
    src = os.path.join(DATASET_DIR, "outdoor", "XML Files")
    dst = os.path.join(PREPARED_DIR, "outdoor")
    if not os.path.isdir(src):
        print("outdoor: XML Files/ not found, skipping")
        return

    class_counter = Counter()
    xml_paths = []
    for dirpath, _, filenames in os.walk(src):
        for fname in filenames:
            if fname.lower().endswith(".xml"):
                xml_paths.append(os.path.join(dirpath, fname))

    parsed = []
    for xml_path in xml_paths:
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        size = root.find("size")
        if size is None:
            continue
        try:
            img_w = int(size.findtext("width"))
            img_h = int(size.findtext("height"))
        except (TypeError, ValueError):
            continue
        objs = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "?").strip()
            if is_watermark_class(name):
                continue
            box = obj.find("bndbox")
            if box is None:
                continue
            try:
                xmin, ymin = float(box.findtext("xmin")), float(box.findtext("ymin"))
                xmax, ymax = float(box.findtext("xmax")), float(box.findtext("ymax"))
            except (TypeError, ValueError):
                continue
            objs.append((name, xmin, ymin, xmax, ymax))
            class_counter[name] += 1
        if objs:
            parsed.append((xml_path, root.findtext("filename", ""), img_w, img_h, objs))

    class_names = sorted(class_counter)
    class_index = {name: i for i, name in enumerate(class_names)}
    print(f"outdoor: {len(class_names)} real classes after filtering watermark labels: {class_names}")

    # Check whether the source images this XML references actually
    # exist anywhere alongside it - if not, we can still produce YOLO
    # label files (useful for documentation/inspection) but training
    # needs real images, which must be sourced separately.
    images_found = 0
    for xml_path, filename, *_ in parsed:
        candidate = os.path.join(os.path.dirname(xml_path), filename)
        if filename and os.path.exists(candidate):
            images_found += 1
    if images_found == 0:
        print("outdoor: WARNING - no corresponding source images found next to the XML "
              "files (checked <filename> paths). Writing YOLO label files anyway for "
              "inspection, but training needs the original images added to "
              "dataset/outdoor/ before training/train_outdoor.py can run.")

    random.Random(seed).shuffle(parsed)
    val_count = max(1, int(len(parsed) * val_ratio))
    val_set, train_set = parsed[:val_count], parsed[val_count:]

    for split_name, split_items in (("train", train_set), ("val", val_set)):
        img_out = os.path.join(dst, split_name, "images")
        lbl_out = os.path.join(dst, split_name, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for xml_path, filename, img_w, img_h, objs in split_items:
            stem = os.path.splitext(os.path.basename(xml_path))[0]
            lines = []
            for name, xmin, ymin, xmax, ymax in objs:
                cx, cy, w, h = voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
                lines.append(f"{class_index[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            with open(os.path.join(lbl_out, f"{stem}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            src_img = os.path.join(os.path.dirname(xml_path), filename)
            if filename and os.path.exists(src_img):
                shutil.copy2(src_img, os.path.join(img_out, filename))

    _write_yaml(
        os.path.join(dst, "data.yaml"),
        os.path.join(dst, "train", "images"),
        os.path.join(dst, "val", "images"),
        class_names,
    )
    print(f"outdoor: prepared {len(train_set)} train / {len(val_set)} val label files at {dst} "
          f"({images_found} had a matching source image copied) - original dataset/outdoor/ untouched")


# ---------------------------------------------------------------------------
# FOOTPATH - Pascal VOC XML -> YOLO, single class "footpath". Only 50
# images total (per analyze_datasets.py) - expect this model to
# overfit without more data; heavy augmentation is left to
# training/train_footpath.py's Ultralytics config rather than
# duplicated here.
# ---------------------------------------------------------------------------
def prepare_footpath(val_ratio=0.2, seed=42):
    img_dir = os.path.join(DATASET_DIR, "Footpath", "footpath_images", "footpath_images")
    ann_dir = os.path.join(DATASET_DIR, "Footpath", "footpath_annotation", "footpath_annotation")
    dst = os.path.join(PREPARED_DIR, "footpath")
    if not os.path.isdir(img_dir) or not os.path.isdir(ann_dir):
        print("footpath: source dataset not found, skipping")
        return

    samples = []
    for fname in os.listdir(ann_dir):
        if not fname.lower().endswith(".xml"):
            continue
        stem = os.path.splitext(fname)[0]
        img_candidates = [f"{stem}{ext}" for ext in (".jpg", ".jpeg", ".png")]
        img_name = next((c for c in img_candidates if os.path.exists(os.path.join(img_dir, c))), None)
        if img_name is None:
            continue
        try:
            root = ET.parse(os.path.join(ann_dir, fname)).getroot()
        except ET.ParseError:
            continue
        size = root.find("size")
        try:
            img_w, img_h = int(size.findtext("width")), int(size.findtext("height"))
        except (TypeError, ValueError, AttributeError):
            continue
        lines = []
        for obj in root.findall("object"):
            box = obj.find("bndbox")
            if box is None:
                continue
            try:
                xmin, ymin = float(box.findtext("xmin")), float(box.findtext("ymin"))
                xmax, ymax = float(box.findtext("xmax")), float(box.findtext("ymax"))
            except (TypeError, ValueError):
                continue
            cx, cy, w, h = voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        if lines:
            samples.append((os.path.join(img_dir, img_name), img_name, lines))

    random.Random(seed).shuffle(samples)
    val_count = max(1, int(len(samples) * val_ratio))
    val_set, train_set = samples[:val_count], samples[val_count:]

    for split_name, split_items in (("train", train_set), ("val", val_set)):
        img_out = os.path.join(dst, split_name, "images")
        lbl_out = os.path.join(dst, split_name, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for img_path, img_name, lines in split_items:
            shutil.copy2(img_path, os.path.join(img_out, img_name))
            stem = os.path.splitext(img_name)[0]
            with open(os.path.join(lbl_out, f"{stem}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

    _write_yaml(
        os.path.join(dst, "data.yaml"),
        os.path.join(dst, "train", "images"),
        os.path.join(dst, "val", "images"),
        ["footpath"],
    )
    print(f"footpath: prepared {len(train_set)} train / {len(val_set)} val images at {dst} "
          f"(only {len(samples)} total - expect overfitting) - original dataset/Footpath/ untouched")


def main():
    parser = argparse.ArgumentParser(description="Convert raw datasets into ready-to-train YOLO layouts")
    parser.add_argument("--dataset", choices=["indoor", "household", "outdoor", "footpath", "all"],
                         default="all")
    parser.add_argument("--min-class-count", type=int, default=20,
                         help="household: drop classes with fewer than this many examples")
    parser.add_argument("--max-images", type=int, default=None,
                         help="household: cap total images used (for a quick trial run)")
    args = parser.parse_args()

    os.makedirs(PREPARED_DIR, exist_ok=True)

    if args.dataset in ("indoor", "all"):
        prepare_indoor()
    if args.dataset in ("household", "all"):
        prepare_household(min_class_count=args.min_class_count, max_images=args.max_images)
    if args.dataset in ("outdoor", "all"):
        prepare_outdoor()
    if args.dataset in ("footpath", "all"):
        prepare_footpath()


if __name__ == "__main__":
    main()
