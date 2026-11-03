"""
Antrenare TinyCNNDetector pe imagini cu etichete YOLO (.txt).

Structură:
  --images folder/*.jpg
  --labels folder/*.txt  (același stem; linii: class cx cy w h normalizat)

Exemplu:
  python3 train_tiny_cnn.py --images data/images/train --labels data/labels/train --epochs 30
"""
from __future__ import annotations

import argparse
import math
import os
from typing import List, Tuple

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tiny_cnn_detector import TinyCNNDetector

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def list_pairs(img_dir: str, lbl_dir: str) -> List[Tuple[str, str]]:
    pairs = []
    for p in sorted(os.listdir(img_dir)):
        stem, ext = os.path.splitext(p)
        if ext.lower() not in IMG_EXT:
            continue
        lp = os.path.join(lbl_dir, stem + ".txt")
        if os.path.isfile(lp):
            pairs.append((os.path.join(img_dir, p), lp))
    return pairs


def parse_yolo_label(path: str) -> List[Tuple[float, float, float, float, int]]:
    out = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            out.append((cx, cy, w, h, cid))
    return out


class YoloGridDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], input_size: int = 320, class_id: int = 0):
        self.pairs = pairs
        self.input_size = input_size
        self.class_id = class_id

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        ip, lp = self.pairs[idx]
        img = cv2.imread(ip)
        if img is None:
            raise RuntimeError(f"Nu pot citi {ip}")
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        rgb = img[:, :, ::-1].astype("float32") / 255.0
        chw = torch.from_numpy(rgb).permute(2, 0, 1)

        labels = parse_yolo_label(lp)
        objs = [(cx, cy, w, h) for cx, cy, w, h, cid in labels if cid == self.class_id]
        return chw, objs


def collate_fn(batch):
    return batch


def build_targets(
    objs_batch: List[List[Tuple[float, float, float, float]]],
    S: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returnează:
      box_tgt [B,4,S,S] — valori țintă pentru tx,ty și log-scale tw,th (ca în model);
      obj_tgt [B,S,S] — 1 dacă există obiect în celulă, altfel 0;
      cls_tgt [B,S,S] — 1 pentru person în celulă activă.
    """
    B = len(objs_batch)
    box_tgt = torch.zeros(B, 4, S, S, device=device)
    obj_tgt = torch.zeros(B, S, S, device=device)
    cls_tgt = torch.zeros(B, S, S, device=device)

    Sf = float(S)
    eps = 1e-6

    for b, objs in enumerate(objs_batch):
        for cx, cy, w, h in objs:
            j = min(S - 1, max(0, int(cx * S)))
            i = min(S - 1, max(0, int(cy * S)))
            tx = min(max(cx * Sf - j, eps), 1.0 - eps)
            ty = min(max(cy * Sf - i, eps), 1.0 - eps)
            tw_lin = max(w * Sf / 1.8, eps)
            th_lin = max(h * Sf / 2.4, eps)
            tw_log = min(max(math.log(tw_lin + eps), -4.0), 4.0)
            th_log = min(max(math.log(th_lin + eps), -4.0), 4.0)

            box_tgt[b, 0, i, j] = math.log(tx / (1.0 - tx + eps))
            box_tgt[b, 1, i, j] = math.log(ty / (1.0 - ty + eps))
            box_tgt[b, 2, i, j] = tw_log
            box_tgt[b, 3, i, j] = th_log
            obj_tgt[b, i, j] = 1.0
            cls_tgt[b, i, j] = 1.0

    return box_tgt, obj_tgt, cls_tgt


def train_epoch(model, loader, opt, device, S: int):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        imgs = []
        objs_batch = []
        for chw, objs in batch:
            imgs.append(chw)
            objs_batch.append(objs)
        x = torch.stack(imgs, dim=0).to(device)
        pred = model(x)
        box_tgt, obj_tgt, cls_tgt = build_targets(objs_batch, S, device)

        mask = obj_tgt.unsqueeze(1).expand_as(box_tgt) > 0.5  # bool (nu float) pentru indexare
        if mask.any():
            loss_box = F.smooth_l1_loss(pred[:, :4][mask], box_tgt[mask])
        else:
            loss_box = pred.new_tensor(0.0)
        loss_obj = F.binary_cross_entropy_with_logits(pred[:, 4], obj_tgt)
        loss_cls = F.binary_cross_entropy_with_logits(pred[:, 5], cls_tgt)

        loss = loss_box + loss_obj + loss_cls
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Folder cu imagini train")
    ap.add_argument("--labels", required=True, help="Folder cu .txt YOLO")
    ap.add_argument("--out", default="tiny_cnn_person.pt", help="Fișier weights ieșire")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--class_id", type=int, default=0, help="ID clasă (person COCO = 0)")
    args = ap.parse_args()

    pairs = list_pairs(args.images, args.labels)
    if not pairs:
        raise SystemExit("Nu am găsit perechi imagine/label.")

    ds = YoloGridDataset(pairs, input_size=320, class_id=args.class_id)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCNNDetector(num_classes=1, grid_size=10, in_size=320).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    S = model.grid_size

    print(f"Antrenare pe {len(ds)} imagini, device={device}")
    for ep in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, opt, device, S)
        print(f"epoch {ep}/{args.epochs}  loss={loss:.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"Salvat: {args.out}")


if __name__ == "__main__":
    main()
