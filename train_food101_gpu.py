"""
Train YOLOv8 Classification on Food-101 with explicit GPU selection.

Usage (recommended):
    python train_food101_gpu.py --data dataset/food101 --device 0 --epochs 30 --batch 32

If you leave --device auto, the script will pick GPU 0 when available, else CPU.
"""

import os
import argparse
import torch
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 Classification on Food-101 (GPU-aware).")
    p.add_argument("--data", type=str, default="dataset/food101",
                   help="Path to dataset root containing train/ and val/ dirs.")
    p.add_argument("--model", type=str, default="yolov8n-cls.pt",
                   help="Base YOLOv8 classification model to fine-tune.")
    p.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    p.add_argument("--imgsz", type=int, default=224, help="Training image size.")
    p.add_argument("--batch", type=int, default=32,
                   help="Batch size (reduce if you hit out-of-memory on GPU). Use -1 or 0 for auto.")
    p.add_argument("--device", type=str, default="auto",
                   help="GPU index (e.g., 0), 'cpu', or 'auto' to pick automatically.")
    p.add_argument("--workers", type=int, default=4,
                   help="Dataloader workers. On Windows, set 0 if you see issues.")
    p.add_argument("--cache", type=str, default="ram",
                   help="Cache images in 'ram', 'disk', or 'None' (string). Speeds training if RAM allows.")
    p.add_argument("--project", type=str, default="runs/classify",
                   help="Training output project directory.")
    p.add_argument("--name", type=str, default="food101-cls-gpu",
                   help="Run name (subfolder under project).")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the last checkpoint in project/name.")
    return p.parse_args()


def resolve_device(arg_device: str):
    """Return device argument appropriate for Ultralytics."""
    if arg_device == "cpu":
        return "cpu"
    if arg_device == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    # user passed index like "0" or "1"
    try:
        return int(arg_device)
    except ValueError:
        return "cpu"


def main():
    args = parse_args()

    # Pick device
    device = resolve_device(args.device)

    print("-" * 60)
    print("GPU / CUDA Check")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Detected GPU 0: {torch.cuda.get_device_name(0)}")
    else:
        print("No CUDA GPU detected; training will run on CPU (slow).")
    print(f"Using device={device}")
    print("-" * 60)

    # Load model
    print(f"Loading base model: {args.model}")
    model = YOLO(args.model)

    # Build training kwargs
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=device,
        batch=args.batch,
        workers=args.workers,
        cache=None if args.cache.lower() == "none" else args.cache,
        project=args.project,
        name=args.name,
        verbose=True,
        resume=args.resume,
    )

    print("Starting training with args:")
    for k, v in train_kwargs.items():
        print(f"  {k}: {v}")

    model.train(**train_kwargs)

    # Locate best weights
    weights_dir = os.path.join(args.project, args.name, "weights")
    best_path = os.path.join(weights_dir, "best.pt")
    last_path = os.path.join(weights_dir, "last.pt")

    print("\nTraining complete.")
    if os.path.exists(best_path):
        print(f"Best weights: {best_path}")
    else:
        print("best.pt not found; check run folder. (Maybe training stopped early?)")

    if os.path.exists(last_path):
        print(f"Last weights: {last_path}")

    print("\nTo test your model on an image:")
    print(f"  yolo classify predict model={best_path} source=path/to/image.jpg device={device}")
    print("\nOr from Python:")
    print("  from ultralytics import YOLO")
    print(f"  model = YOLO(r'{best_path}')")
    print("  results = model('image.jpg', device=0)")
    print("  print(results[0].names[results[0].probs.top1])")


if __name__ == "__main__":
    main()
