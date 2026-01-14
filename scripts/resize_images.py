#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def resize_long_edge(img, target_long_edge: int):
    h, w = img.shape[:2]
    long_edge = max(h, w)

    if long_edge <= target_long_edge:
        return img, 1.0

    scale = target_long_edge / long_edge
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def main():
    p = argparse.ArgumentParser(
        description="Resize images by fixing the long edge (aspect ratio preserved)."
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("workflow/images/raw"),
        help="Input folder containing raw images",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("workflow/images/resized"),
        help="Output folder for resized images",
    )
    p.add_argument("--long-edge", type=int, default=3000, help="Target long edge in pixels")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--recursive", action="store_true", help="Scan subfolders recursively")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.recursive:
        image_paths = sorted([p for p in args.raw_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
    else:
        image_paths = sorted([p for p in args.raw_dir.glob("*") if p.suffix.lower() in IMG_EXTS])

    print(f"Found {len(image_paths)} images in: {args.raw_dir}")
    print(f"Target long edge: {args.long_edge}px")
    print(f"Output dir: {args.out_dir}\n")

    n_done = 0
    for img_path in image_paths:
        out_path = args.out_dir / img_path.name
        if out_path.exists() and not args.overwrite:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skipped (unreadable): {img_path.name}")
            continue

        resized_img, scale = resize_long_edge(img, args.long_edge)
        ok = cv2.imwrite(str(out_path), resized_img)
        if not ok:
            raise RuntimeError(f"Failed to write: {out_path}")

        h0, w0 = img.shape[:2]
        h1, w1 = resized_img.shape[:2]
        print(f"{img_path.name} | {w0}x{h0} -> {w1}x{h1} | scale={scale:.3f}")
        n_done += 1

    print(f"\nDone. Resized/wrote {n_done} images to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
