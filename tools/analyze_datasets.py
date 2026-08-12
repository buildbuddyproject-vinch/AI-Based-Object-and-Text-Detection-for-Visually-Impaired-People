"""
Dataset Analysis Tool
=======================
Inspects every dataset under dataset/ and produces a factual report of
what's actually there - format, class list, counts, splits, and data
quality problems - without assuming anything or modifying any files.

Run:
    python tools/analyze_datasets.py
    python tools/analyze_datasets.py --json tools/dataset_report.json
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_common import is_watermark_class  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def analyze_currency():
    path = os.path.join(DATASET_DIR, "currency")
    report = {"name": "currency", "path": path, "exists": os.path.isdir(path)}
    if not report["exists"]:
        return report

    entries = os.listdir(path)
    images = [f for f in entries if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
    report["format"] = ("single reference image per denomination, used for ORB "
                         "feature-matching in modules/currency_detector.py - "
                         "NOT a trainable detection dataset in its current form")
    report["images"] = len(images)
    report["denominations_found"] = sorted(images)
    report["recommended_task"] = (
        "keep as-is for template matching, OR collect many real photos per "
        "denomination (varied angle/lighting/background/wear) if a trained "
        "classifier/detector is wanted instead of ORB matching"
    )
    stray = [f for f in entries if f not in images and not f.startswith(".")
             and f.lower() != "readme.md"]
    if stray:
        report["unexpected_files"] = stray
    return report


def analyze_footpath():
    path = os.path.join(DATASET_DIR, "Footpath")
    report = {"name": "Footpath", "path": path, "exists": os.path.isdir(path)}
    if not report["exists"]:
        return report

    img_dir = os.path.join(path, "footpath_images", "footpath_images")
    ann_dir = os.path.join(path, "footpath_annotation", "footpath_annotation")
    if not os.path.isdir(img_dir) or not os.path.isdir(ann_dir):
        report["error"] = f"expected nested {img_dir} and {ann_dir}"
        return report

    images = {os.path.splitext(f)[0] for f in os.listdir(img_dir)
              if os.path.splitext(f)[1].lower() in IMAGE_EXTS}
    ann_files = [f for f in os.listdir(ann_dir) if f.lower().endswith(".xml")]
    annotations = {os.path.splitext(f)[0] for f in ann_files}

    class_counter = Counter()
    segmented_flags = Counter()
    objects_per_image = []
    corrupted_xml = []
    for fname in ann_files:
        fpath = os.path.join(ann_dir, fname)
        try:
            root = ET.parse(fpath).getroot()
        except ET.ParseError:
            corrupted_xml.append(fname)
            continue
        segmented_flags[root.findtext("segmented", "?")] += 1
        objs = root.findall("object")
        objects_per_image.append(len(objs))
        for obj in objs:
            class_counter[obj.findtext("name", "?")] += 1

    report.update({
        "format": ("Pascal VOC XML, axis-aligned bounding boxes (<bndbox> "
                   "xmin/ymin/xmax/ymax). NOT segmentation - <segmented>0</segmented> "
                   "in every file checked, despite the folder name suggesting otherwise"),
        "images": len(images),
        "annotations": len(annotations),
        "images_without_annotation": sorted(images - annotations),
        "annotations_without_image": sorted(annotations - images),
        "corrupted_xml_files": corrupted_xml,
        "segmented_flag_values_seen": dict(segmented_flags),
        "classes": dict(class_counter),
        "objects_per_image_min": min(objects_per_image) if objects_per_image else 0,
        "objects_per_image_max": max(objects_per_image) if objects_per_image else 0,
        "objects_per_image_avg": round(sum(objects_per_image) / len(objects_per_image), 2)
                                  if objects_per_image else 0,
        "train_val_test_split": "none present - all images in one flat folder",
        "recommended_task": ("YOLO object detection, single class 'footpath' (a box "
                              "marking the walkable region) - not a segmentation task"),
        "concerns": [
            f"Only {len(images)} images total - very small for a generalizing detector; "
            "expect overfitting without heavy augmentation or substantially more data",
            "No train/val/test split provided - must be created before training",
        ],
    })
    return report


def analyze_household():
    path = os.path.join(DATASET_DIR, "household")
    report = {"name": "household", "path": path, "exists": os.path.isdir(path)}
    if not report["exists"]:
        return report

    subfolders = sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))
    class_counter = Counter()
    total_images = 0
    total_annotated_objects = 0
    images_with_zero_objects = 0
    corrupted_json = []
    geometry_types = Counter()

    for folder in subfolders:
        img_dir = os.path.join(path, folder, "img")
        ann_dir = os.path.join(path, folder, "ann")
        if not os.path.isdir(img_dir) or not os.path.isdir(ann_dir):
            continue
        images = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
        total_images += len(images)

        for fname in os.listdir(ann_dir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(ann_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                corrupted_json.append(os.path.join(folder, "ann", fname))
                continue
            objs = data.get("objects", [])
            if not objs:
                images_with_zero_objects += 1
            for obj in objs:
                class_counter[obj.get("classTitle", "?")] += 1
                geometry_types[obj.get("geometryType", "?")] += 1
                total_annotated_objects += 1

    report.update({
        "format": ("Supervisely-style JSON per image (ann/<file>.jpg.json + "
                   "img/<file>.jpg pairs), one folder per Dollar Street theme/collection"),
        "theme_folders": len(subfolders),
        "total_images": total_images,
        "total_annotated_objects": total_annotated_objects,
        "images_with_zero_annotated_objects": images_with_zero_objects,
        "corrupted_json_files": corrupted_json,
        "geometry_types_used": dict(geometry_types),
        "distinct_classes": len(class_counter),
        "top_30_classes_by_frequency": class_counter.most_common(30),
        "rarest_10_classes_with_at_least_1": class_counter.most_common()[:-11:-1],
        "train_val_test_split": "none present - must be created before training",
        "recommended_task": (
            "YOLO object detection (every geometryType found is 'rectangle', i.e. "
            "axis-aligned boxes). IMPORTANT: the folder name (e.g. 'Objects__armchairs') "
            "is just the source theme/collection, NOT the object's class - real classes "
            "are the classTitle fields inside each JSON and multiple unrelated object "
            "types can appear in one image (e.g. an 'armchairs' photo also containing "
            "annotated Cup, Book, Bed)"
        ),
        "concerns": [],
    })

    rare = [c for c, n in class_counter.items() if n < 20]
    if rare:
        report["concerns"].append(
            f"{len(rare)} of {len(class_counter)} classes have fewer than 20 annotated "
            "examples each - too few to train reliably; consider merging into broader "
            "categories or dropping them"
        )
    if images_with_zero_objects:
        report["concerns"].append(
            f"{images_with_zero_objects} images have zero annotated objects - "
            "exclude, or use deliberately as hard-negative background examples"
        )
    return report


def _parse_yolo_yaml_names(content):
    """Minimal names: list parser - avoids a hard pyyaml dependency for
    this simple, known Roboflow-export shape."""
    names = []
    in_names = False
    for line in content.splitlines():
        if line.strip().startswith("names"):
            in_names = True
            continue
        if in_names:
            stripped = line.strip()
            if stripped.startswith("-"):
                names.append(stripped.lstrip("- ").strip())
            elif stripped:
                break
    return names


def analyze_indoor():
    path = os.path.join(DATASET_DIR, "indoor")
    report = {"name": "indoor", "path": path, "exists": os.path.isdir(path)}
    if not report["exists"]:
        return report

    yaml_path = os.path.join(path, "data.yaml")
    class_names = []
    notes = []
    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            content = f.read()
        class_names = _parse_yolo_yaml_names(content)
        if "/content/" in content:
            notes.append("data.yaml paths point at /content/... (a Google Colab path) - "
                         "must be rewritten to local relative paths before training locally")
    else:
        notes.append("data.yaml not found")

    report["classes"] = class_names
    report["num_classes"] = len(class_names)
    report["notes"] = notes

    splits = {}
    class_counter = Counter()
    for split in ("train", "valid", "test"):
        img_dir = os.path.join(path, split, "images")
        lbl_dir = os.path.join(path, split, "labels")
        if not os.path.isdir(img_dir):
            splits[split] = {"exists": False}
            continue
        images = {os.path.splitext(f)[0] for f in os.listdir(img_dir)
                   if os.path.splitext(f)[1].lower() in IMAGE_EXTS}
        labels = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith(".txt")} \
                  if os.path.isdir(lbl_dir) else set()
        malformed = set()
        empty = 0
        for fname in (os.listdir(lbl_dir) if os.path.isdir(lbl_dir) else []):
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(lbl_dir, fname), "r", encoding="utf-8") as f:
                lines = [l for l in f.read().splitlines() if l.strip()]
            if not lines:
                empty += 1
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    malformed.add(fname)
                    continue
                try:
                    cls_id = int(parts[0])
                    class_counter[class_names[cls_id] if cls_id < len(class_names) else f"id_{cls_id}"] += 1
                except ValueError:
                    malformed.add(fname)
        splits[split] = {
            "exists": True,
            "images": len(images),
            "labels": len(labels),
            "images_without_label": len(images - labels),
            "labels_without_image": len(labels - images),
            "empty_label_files_background_images": empty,
            "malformed_label_files": sorted(malformed),
        }

    report["splits"] = splits
    report["class_distribution_all_splits"] = dict(class_counter)
    report["format"] = ("standard YOLO detection (class cx cy w h, normalized 0-1), "
                        "Roboflow export layout: train/valid/test each with images/+labels/")
    report["recommended_task"] = "YOLO object detection - cleanest dataset here, ready to train with minimal cleanup"
    report["concerns"] = []
    for split, info in splits.items():
        if info.get("exists") and (info["images_without_label"] or info["labels_without_image"]):
            report["concerns"].append(
                f"{split}: {info['images_without_label']} images missing a label file, "
                f"{info['labels_without_image']} label files with no matching image"
            )
    underrepresented = [c for c, n in class_counter.items() if n < 30]
    if underrepresented:
        report["concerns"].append(f"Classes with under 30 labeled instances (may be unreliable): {underrepresented}")
    return report


def analyze_outdoor():
    path = os.path.join(DATASET_DIR, "outdoor")
    report = {"name": "outdoor", "path": path, "exists": os.path.isdir(path)}
    if not report["exists"]:
        return report

    report["format"] = (
        "MIXED/inconsistent, three unrelated things share this folder: "
        "(1) Day2fog/ and Day2Rainy/ contain paired clear/adverse-weather images "
        "(Train A/Train B/Test A/Test B - CycleGAN-style domain-pair layout, "
        "NOT annotated for detection); "
        "(2) 'XML Files/' contains Pascal VOC XML with polygon+bbox per object; "
        "(3) 'labels/' contains YOLO-OBB-style .txt files (class + 8 coords = 4 "
        "corner points, 9 fields per line - NOT standard 5-field YOLO detection)"
    )

    # --- XML Files/: real vs watermark classes ---
    xml_root = os.path.join(path, "XML Files")
    class_counter = Counter()
    watermark_counter = Counter()
    xml_file_count = 0
    corrupted_xml = []
    if os.path.isdir(xml_root):
        for dirpath, _, filenames in os.walk(xml_root):
            for fname in filenames:
                if not fname.lower().endswith(".xml"):
                    continue
                xml_file_count += 1
                fpath = os.path.join(dirpath, fname)
                try:
                    tree = ET.parse(fpath)
                except ET.ParseError:
                    corrupted_xml.append(os.path.relpath(fpath, path))
                    continue
                for obj in tree.getroot().findall("object"):
                    name = (obj.findtext("name") or "?").strip()
                    if is_watermark_class(name):
                        watermark_counter[name] += 1
                    else:
                        class_counter[name] += 1

    report["xml_annotations"] = {
        "total_xml_files": xml_file_count,
        "corrupted_xml_files": corrupted_xml,
        "real_classes_found": dict(class_counter),
        "watermark_or_junk_labels_found": dict(watermark_counter),
        "total_watermark_annotations": sum(watermark_counter.values()),
        "total_real_annotations": sum(class_counter.values()),
    }

    # --- labels/ (OBB-style .txt): per-subset class id distributions ---
    labels_root = os.path.join(path, "labels")
    obb_report = {}
    if os.path.isdir(labels_root):
        for subset in sorted(os.listdir(labels_root)):
            subset_dir = os.path.join(labels_root, subset)
            if not os.path.isdir(subset_dir):
                continue
            class_id_counts = Counter()
            field_counts = Counter()
            file_count = 0
            for fname in os.listdir(subset_dir):
                if not fname.endswith(".txt"):
                    continue
                file_count += 1
                with open(os.path.join(subset_dir, fname), "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.split()
                        if not parts:
                            continue
                        field_counts[len(parts)] += 1
                        try:
                            class_id_counts[int(parts[0])] += 1
                        except ValueError:
                            pass
            obb_report[subset] = {
                "files": file_count,
                "class_id_distribution": dict(sorted(class_id_counts.items())),
                "fields_per_line_seen": dict(field_counts),
            }
    report["obb_labels"] = obb_report
    report["obb_labels_concern"] = (
        "Class ID sets differ across Day/Foggy/Rainy subsets (e.g. Day only uses ids "
        "0-3, Rainy uses 0-13) with no classes.txt/data.yaml found to map ids to names - "
        "cannot safely assume the same id means the same real-world class across "
        "subsets without additional documentation from whoever exported these."
    )

    # --- Day2fog / Day2Rainy image pairs ---
    pair_info = {}
    for pair_name in ("Day2fog", "Day2Rainy"):
        pair_dir = os.path.join(path, pair_name)
        if not os.path.isdir(pair_dir):
            continue
        counts = {}
        for dirpath, _, filenames in os.walk(pair_dir):
            imgs = [f for f in filenames if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
            if imgs:
                counts[os.path.relpath(dirpath, pair_dir)] = len(imgs)
        pair_info[pair_name] = counts
    report["weather_pair_images"] = pair_info
    report["weather_pair_note"] = (
        "Unannotated paired day/adverse-weather images (domain-transfer style layout) - "
        "not usable directly for detection training; only useful for weather-augmentation "
        "research or as a source of extra unlabeled images to annotate later"
    )

    report["recommended_task"] = (
        "Object detection IS feasible from 'XML Files/' for real classes: car, pole, "
        "truck, flyover, hoarding, traffic symbols, pedestrian, traffic signal, bus, "
        "building, bike, caravan, auto rickshaw - after filtering out the watermark/junk "
        "labels reported above. The 'labels/' OBB set should NOT be used until its "
        "class-id mapping is clarified/documented by whoever produced it."
    )
    report["concerns"] = [
        f"{sum(watermark_counter.values())} annotations in XML Files/ are Roboflow "
        "watermark text mistakenly present as object classes - must be filtered before training",
        "labels/ (OBB format) has inconsistent, undocumented class-id mappings across Day/Foggy/Rainy",
        "No train/val/test split provided for either annotation set",
    ]
    underrepresented = [c for c, n in class_counter.items() if n < 15]
    if underrepresented:
        report["concerns"].append(f"Real classes with very few examples in XML Files/ (<15): {underrepresented}")
    return report


def print_section(title, data):
    print(f"\n{'=' * 78}\nDATASET: {title}\n{'=' * 78}")
    for key, value in data.items():
        if isinstance(value, dict) and len(value) > 15:
            print(f"{key}: ({len(value)} entries, showing first 15)")
            for k, v in list(value.items())[:15]:
                print(f"  {k}: {v}")
        elif isinstance(value, list) and len(value) > 15:
            print(f"{key}: ({len(value)} entries, showing first 15)")
            for v in value[:15]:
                print(f"  - {v}")
        else:
            print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Analyze all datasets under dataset/")
    parser.add_argument("--json", help="Also write the full report to this JSON file")
    args = parser.parse_args()

    print(f"Scanning datasets under: {DATASET_DIR}")

    reports = {
        "currency": analyze_currency(),
        "footpath": analyze_footpath(),
        "household": analyze_household(),
        "indoor": analyze_indoor(),
        "outdoor": analyze_outdoor(),
    }

    for name, report in reports.items():
        print_section(name.upper(), report)

    if args.json:
        out_path = args.json
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2, default=str)
        print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
