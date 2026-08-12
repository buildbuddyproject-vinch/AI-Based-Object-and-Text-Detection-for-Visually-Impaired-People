"""
Footpath Detection Model Training
=====================================
Trains a YOLOv8n detector on the prepared footpath dataset - a single
class, "footpath", marking the walkable region as a bounding box (per
tools/analyze_datasets.py, the source data is Pascal VOC bounding
boxes, NOT segmentation, despite what the folder name suggests).

Only ~50 source images total - this WILL overfit and should be treated
as a proof-of-concept / starting point, not a production-ready model.
Collecting substantially more footpath images (different locations,
lighting, obstructions) is the single highest-value next step for this
domain; see the final report for this call-out.

Prerequisite:
    python tools/prepare_datasets.py --dataset footpath

Run:
    python training/train_footpath.py [--epochs 50]
"""
import argparse
import os
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(BASE_DIR, "dataset_prepared", "footpath", "data.yaml")
RUNS_DIR = os.path.join(BASE_DIR, "runs", "footpath")
MODEL_OUT = os.path.join(BASE_DIR, "models", "footpath", "best.pt")


def main():
    parser = argparse.ArgumentParser(description="Train the footpath detection model")
    parser.add_argument("--epochs", type=int, default=50,
                         help="Higher than other domains since the dataset is tiny (~50 "
                              "images) - more epochs + augmentation partially compensates, "
                              "but cannot substitute for more real data.")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not os.path.exists(DATA_YAML):
        print(f"ERROR: {DATA_YAML} not found.")
        print("Run `python tools/prepare_datasets.py --dataset footpath` first.")
        return

    from ultralytics import YOLO

    print(f"Training footpath model: epochs={args.epochs} imgsz={args.imgsz} "
          f"batch={args.batch} device={args.device}")
    model = YOLO(args.base_model)
    model.train(
        data=DATA_YAML,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=RUNS_DIR,
        name="train",
        exist_ok=True,
        patience=15,
        # Heavier augmentation than the default, specifically because
        # the dataset is so small - this is the one place augmentation
        # is intentionally cranked up to partially compensate.
        degrees=10.0,
        translate=0.15,
        scale=0.3,
        fliplr=0.5,
        verbose=True,
    )

    metrics = model.val(data=DATA_YAML, device=args.device)
    print("\n--- Validation metrics ---")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision (mean): {metrics.box.mp:.4f}")
    print(f"Recall (mean):    {metrics.box.mr:.4f}")
    print("\nNOTE: with ~50 source images, these numbers reflect a small, likely "
          "overfit validation set - treat as a proof-of-concept baseline, not a "
          "reliable accuracy claim. See README.md's Remaining Work section.")

    best_path = os.path.join(RUNS_DIR, "train", "weights", "best.pt")
    if os.path.exists(best_path):
        os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
        shutil.copy2(best_path, MODEL_OUT)
        print(f"\nCopied best model to {MODEL_OUT}")
    else:
        print(f"\nWARNING: expected best.pt at {best_path} but it wasn't found.")


if __name__ == "__main__":
    main()
