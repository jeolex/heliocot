#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch
from ultralytics import YOLO


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    p = argparse.ArgumentParser(description="Train YOLOv8 segmentation model (Ultralytics).")
    p.add_argument("--data", default="model/data.yaml")
    p.add_argument("--pretrained", default="yolov8m-seg.pt")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--imgsz", type=int, default=768)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default=None)  # auto if empty
    p.add_argument("--project", default="runs_leaf_seg")
    p.add_argument("--name", default="yolov8m_leafseg_final")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--lr0", type=float, default=0.001)
    p.add_argument("--mosaic", type=float, default=0.0)
    p.add_argument("--erasing", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--cache", default="disk")
    args = p.parse_args()

    device = args.device or detect_device()
    print(f"Using device: {device}")

    model = YOLO(args.pretrained)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        mosaic=args.mosaic,
        erasing=args.erasing,
        seed=args.seed,
        deterministic=args.deterministic,
        cache=args.cache,
    )


if __name__ == "__main__":
    main()
