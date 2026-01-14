#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def make_group(df: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in metadata: {missing}")
    return df[group_cols].astype(str).agg("_".join, axis=1)


def split_groups(df, group_col, train_size, val_size, test_size, seed):
    s = train_size + val_size + test_size
    if abs(s - 1.0) > 1e-9:
        raise ValueError(f"train/val/test must sum to 1.0 (got {s})")

    gss1 = GroupShuffleSplit(n_splits=1, test_size=(1.0 - train_size), random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, groups=df[group_col]))
    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    val_frac = val_size / (val_size + test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=(1.0 - val_frac), random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df[group_col]))
    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    return train_df, val_df, test_df


def transfer_file(src: Path, dst: Path, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return
    if dst.exists() and overwrite:
        dst.unlink()
    shutil.copy2(src, dst)


def write_data_yaml(base_dir: Path, class_name: str = "leaf") -> Path:
    yaml_path = base_dir / "data.yaml"
    content = (
        f"path: {base_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: 1\n"
        f"names: ['{class_name}']\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def write_filelists(base_dir: Path) -> None:
    for split in ["train", "val", "test"]:
        img_dir = base_dir / "images" / split
        out = base_dir / f"{split}.txt"
        imgs = sorted([p for p in img_dir.glob("*") if p.is_file()])
        out.write_text("\n".join([p.as_posix() for p in imgs]) + ("\n" if imgs else ""), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Split CVAT YOLO export (raw pool -> train/val/test).")

    p.add_argument("--base-dir", type=Path, default=Path("model"))
    p.add_argument("--metadata", type=Path, default=Path("model/metadata.xlsx"))

    # CVAT export pool (after you moved train -> raw)
    p.add_argument("--src-images", type=Path, default=Path("model/images/raw"))
    p.add_argument("--src-labels", type=Path, default=Path("model/labels/raw"))

    # destination
    p.add_argument("--dst-images", type=Path, default=Path("model/images"))
    p.add_argument("--dst-labels", type=Path, default=Path("model/labels"))

    p.add_argument("--image-col", type=str, default="image_name")
    p.add_argument("--group-cols", type=str, default="variety,stage")

    p.add_argument("--train-size", type=float, default=0.70)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--class-name", type=str, default="leaf")
    p.add_argument("--write-filelists", action="store_true")

    args = p.parse_args()

    if not args.src_images.exists():
        raise FileNotFoundError(f"Missing folder: {args.src_images}")
    if not args.src_labels.exists():
        raise FileNotFoundError(f"Missing folder: {args.src_labels}")

    df = pd.read_excel(args.metadata)
    if args.image_col not in df.columns:
        raise ValueError(f"Column not found in metadata: {args.image_col}")

    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    df["group"] = make_group(df, group_cols)

    train_df, val_df, test_df = split_groups(
        df=df,
        group_col="group",
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    for split in ["train", "val", "test"]:
        (args.dst_images / split).mkdir(parents=True, exist_ok=True)
        (args.dst_labels / split).mkdir(parents=True, exist_ok=True)

    def do_split(split_df: pd.DataFrame, split_name: str) -> None:
        for img_name in split_df[args.image_col].astype(str):
            src_img = args.src_images / img_name
            if not src_img.exists():
                raise FileNotFoundError(f"Missing image: {src_img}")

            lbl_name = Path(img_name).with_suffix(".txt").name
            src_lbl = args.src_labels / lbl_name
            if not src_lbl.exists():
                raise FileNotFoundError(f"Missing label: {src_lbl}")

            dst_img = args.dst_images / split_name / img_name
            dst_lbl = args.dst_labels / split_name / lbl_name

            transfer_file(src_img, dst_img, overwrite=args.overwrite)
            transfer_file(src_lbl, dst_lbl, overwrite=args.overwrite)

        print(f"{split_name}: done")

    do_split(train_df, "train")
    do_split(val_df, "val")
    do_split(test_df, "test")

    out_csv = args.base_dir / "splits_manifest.csv"
    pd.concat(
        [train_df.assign(split="train"), val_df.assign(split="val"), test_df.assign(split="test")],
        ignore_index=True,
    ).to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")

    yaml_path = write_data_yaml(args.base_dir, class_name=args.class_name)
    print(f"Wrote: {yaml_path}")

    if args.write_filelists:
        write_filelists(args.base_dir)
        print("Wrote: model/train.txt, model/val.txt, model/test.txt")


if __name__ == "__main__":
    main()
