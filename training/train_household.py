"""
Household Detection Model Training
======================================
Trains a YOLOv8n detector on the prepared household dataset - 93 real
household object classes (Shoe, Cup, Cooking pot, Hand, Toothbrush,
Plate, Toy, Cutlery, Book, Power outlet, ... - the actual classTitle
values found in dataset/household/*/ann/*.json, per
tools/analyze_datasets.py; nothing invented, and the folder names like
"Objects__armchairs" were NOT used as classes since they're just the
source theme/collection, not the object's label).

"fan" was NOT found anywhere in this dataset - see the final report's
MISSING CLASS section. This model will never announce "fan" because
it was never trained to recognize one.

Much larger and slower to train than indoor (5,400 train images across
93 classes vs. ~1,000 images across 10 classes for indoor) - expect
noticeably longer per-epoch time on CPU.

Prerequisite:
    python tools/prepare_datasets.py --dataset household --min-class-count 20 --max-images 6000

Run:
    python training/train_household.py [--epochs 25] [--imgsz 416] [--batch 16]
"""
import argparse
import os
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(BASE_DIR, "dataset_prepared", "household", "data.yaml")
RUNS_DIR = os.path.join(BASE_DIR, "runs", "household")
MODEL_OUT = os.path.join(BASE_DIR, "models", "household", "best.pt")


def main():
    parser = argparse.ArgumentParser(description="Train the household detection model")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=416,
                         help="Smaller than indoor's 640 default - 93 classes over "
                              "6,000 images is already CPU-heavy; 416 keeps epoch time "
                              "reasonable at some cost to small-object accuracy.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not os.path.exists(DATA_YAML):
        print(f"ERROR: {DATA_YAML} not found.")
        print("Run `python tools/prepare_datasets.py --dataset household` first.")
        return

    from ultralytics import YOLO

    print(f"Training household model: epochs={args.epochs} imgsz={args.imgsz} "
          f"batch={args.batch} device={args.device}")
    print("NOTE: 93 classes, several with as few as ~20 examples - expect weak "
          "per-class accuracy on the rarest classes; see results.csv/confusion_matrix.png "
          "for the real per-class numbers once training finishes.")
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
        patience=10,
        verbose=True,
    )

    metrics = model.val(data=DATA_YAML, device=args.device)
    print("\n--- Validation metrics ---")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision (mean): {metrics.box.mp:.4f}")
    print(f"Recall (mean):    {metrics.box.mr:.4f}")
    print("\nNOTE: with several classes at the 20-example minimum, per-class metrics "
          "will vary widely - check results.csv/confusion_matrix.png for which classes "
          "are actually reliable before trusting them in the field.")

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
