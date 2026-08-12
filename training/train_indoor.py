"""
Indoor Detection Model Training
===================================
Trains a YOLOv8n detector on the prepared indoor dataset (door,
cabinetDoor, refrigeratorDoor, window, chair, table, cabinet, couch,
openedDoor, pole - the actual classes present, per
tools/analyze_datasets.py; nothing invented).

Prerequisite:
    python tools/prepare_datasets.py --dataset indoor

Run:
    python training/train_indoor.py [--epochs 30] [--imgsz 640] [--batch 16]

Outputs (Ultralytics' standard run layout):
    runs/indoor/train/weights/best.pt   <- what models/indoor/best.pt gets copied from
    runs/indoor/train/weights/last.pt
    runs/indoor/train/results.png, confusion_matrix.png, PR curves, etc.
    runs/indoor/train/results.csv       <- per-epoch metrics, no invented numbers

After training, copies best.pt to models/indoor/best.pt so
modules/model_router.py picks it up automatically on the next app
restart.
"""
import argparse
import os
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(BASE_DIR, "dataset_prepared", "indoor", "data.yaml")
RUNS_DIR = os.path.join(BASE_DIR, "runs", "indoor")
MODEL_OUT = os.path.join(BASE_DIR, "models", "indoor", "best.pt")


def main():
    parser = argparse.ArgumentParser(description="Train the indoor detection model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--base-model", default="yolov8n.pt",
                         help="Starting weights - yolov8n.pt (COCO-pretrained backbone, "
                              "standard transfer-learning starting point, not used for "
                              "inference) fine-tuned entirely on the indoor classes above.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not os.path.exists(DATA_YAML):
        print(f"ERROR: {DATA_YAML} not found.")
        print("Run `python tools/prepare_datasets.py --dataset indoor` first.")
        return

    from ultralytics import YOLO

    print(f"Training indoor model: epochs={args.epochs} imgsz={args.imgsz} "
          f"batch={args.batch} device={args.device}")
    model = YOLO(args.base_model)
    results = model.train(
        data=DATA_YAML,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=RUNS_DIR,
        name="train",
        exist_ok=True,
        patience=10,
        verbose=True,
    )

    # Validate on the held-out val split for real precision/recall/mAP
    # numbers (never hand-waved - see the run's results.csv/confusion
    # matrix for what's actually printed here).
    metrics = model.val(data=DATA_YAML, device=args.device)
    print("\n--- Validation metrics ---")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision (mean): {metrics.box.mp:.4f}")
    print(f"Recall (mean):    {metrics.box.mr:.4f}")

    best_path = os.path.join(RUNS_DIR, "train", "weights", "best.pt")
    if os.path.exists(best_path):
        os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
        shutil.copy2(best_path, MODEL_OUT)
        print(f"\nCopied best model to {MODEL_OUT}")
        print("model_router.py will pick this up automatically on the app's next restart.")
    else:
        print(f"\nWARNING: expected best.pt at {best_path} but it wasn't found - training may have failed.")


if __name__ == "__main__":
    main()
