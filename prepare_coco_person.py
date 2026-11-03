#!/usr/bin/env python3
"""
Pregătește date din COCO (doar clasa „person”) pentru train_tiny_cnn.py.

Structură COCO așteptată după ce descarci arhivele oficiale:
  coco/
    annotations/instances_train2017.json   (sau instances_val2017.json)
    train2017/*.jpg                        (sau val2017/*.jpg)

Ieșire (implicit):
  data/images/train/*.jpg   (copii sau symlink-uri)
  data/labels/train/*.txt   (YOLO: 0 cx cy w h)

Descărcări oficiale: https://cocodataset.org/#download
  - Annotations: „2017 Train/Val annotations” (~241 MB)
  - Imagini: „2017 Train images” (~18 GB) sau „2017 Val images” (~1 GB) pentru test rapid

Exemplu (ai deja coco/ cu train2017 + annotations):
  python3 prepare_coco_person.py --coco-root coco --split train

Exemplu val mai mic:
  python3 prepare_coco_person.py --coco-root coco --split val --out-image-dir data/images/train --out-label-dir data/labels/train
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path


# COCO: person are category_id == 1
PERSON_CAT_ID = 1
# În YOLO pentru antrenarea noastră folosim clasa 0 = person
YOLO_CLASS_PERSON = 0


def load_coco_instances(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = {im["id"]: im for im in data["images"]}
    anns_by_image: dict = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("category_id") != PERSON_CAT_ID:
            continue
        anns_by_image[ann["image_id"]].append(ann)
    return images, anns_by_image


def coco_bbox_to_yolo_line(bbox, img_w: int, img_h: int) -> str:
    """bbox COCO: [x, y, width, height] pixeli, colț stânga-sus."""
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 1e-6), 1.0)
    nh = min(max(nh, 1e-6), 1.0)
    return f"{YOLO_CLASS_PERSON} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst, target_is_directory=False)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description="COCO person → YOLO pentru TinyCNN")
    ap.add_argument(
        "--coco-root",
        type=Path,
        default=Path("coco"),
        help="Folder rădăcină unde ai annotations/ și train2017/ sau val2017/",
    )
    ap.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Folosește train2017 sau val2017 (val e mai mic, bun la început)",
    )
    ap.add_argument(
        "--out-image-dir",
        type=Path,
        default=Path("data/images/train"),
        help="Unde se pun imaginile",
    )
    ap.add_argument(
        "--out-label-dir",
        type=Path,
        default=Path("data/labels/train"),
        help="Unde se scriu .txt YOLO",
    )
    ap.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Opțional: limitează la primele N imagini cu persoane (0 = toate)",
    )
    ap.add_argument(
        "--copy",
        action="store_true",
        help="Copiază imaginile în loc de symlink (mai lent, dar merge peste tot)",
    )
    args = ap.parse_args()

    split_dir = "train2017" if args.split == "train" else "val2017"
    ann_name = (
        "instances_train2017.json"
        if args.split == "train"
        else "instances_val2017.json"
    )
    ann_path = args.coco_root / "annotations" / ann_name
    img_root = args.coco_root / split_dir

    if not ann_path.is_file():
        print(
            f"Lipsește fișierul de anotații:\n  {ann_path}\n\n"
            "Descarcă de pe https://cocodataset.org/#download pachetul\n"
            "  „2017 Train/Val annotations” (zip)\n"
            "Dezarhivează astfel încât să existe:\n"
            f"  {args.coco_root}/annotations/{ann_name}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not img_root.is_dir():
        print(
            f"Lipsește folderul cu imagini:\n  {img_root}\n\n"
            "Descarcă de pe același site:\n"
            f"  „2017 {'Train' if args.split == 'train' else 'Val'} images”\n"
            f"Dezarhivează în: {args.coco_root}/",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Citesc {ann_path} ...")
    images, anns_by_image = load_coco_instances(ann_path)

    args.out_image_dir.mkdir(parents=True, exist_ok=True)
    args.out_label_dir.mkdir(parents=True, exist_ok=True)

    count_img = 0
    count_skip = 0

    for image_id, anns in anns_by_image.items():
        if not anns:
            continue
        im = images.get(image_id)
        if not im:
            continue
        file_name = im["file_name"]
        iw, ih = im["width"], im["height"]
        src = img_root / file_name
        if not src.is_file():
            count_skip += 1
            continue

        stem = Path(file_name).stem
        lines = []
        for ann in anns:
            line = coco_bbox_to_yolo_line(ann["bbox"], iw, ih)
            lines.append(line)

        lbl_path = args.out_label_dir / f"{stem}.txt"
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        dst_img = args.out_image_dir / file_name
        if args.copy:
            shutil.copy2(src, dst_img)
        else:
            link_or_copy(src, dst_img)

        count_img += 1
        if args.max_images and count_img >= args.max_images:
            break

    print(f"Gata: {count_img} imagini cu persoane → {args.out_image_dir} + {args.out_label_dir}")
    if count_skip:
        print(f"Avertisment: {count_skip} anotații sărite (fișier imagine lipsă).")


if __name__ == "__main__":
    main()
