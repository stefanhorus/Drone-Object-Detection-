"""
Detector CNN ușor (person / o clasă): backbone convoluțional + head pe grilă SxS.
Compatibil cu antrenare în PyTorch și inferență pe frame-uri redimensionate.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, p: Optional[int] = None):
        if p is None:
            p = k // 2
        super().__init__(
            nn.Conv2d(c_in, c_out, k, s, p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


class TinyCNNDetector(nn.Module):
    """
    Intrare: [B, 3, H, H] (recomandat H=320).
    Ieșire: [B, C, S, S] unde C = 5 + num_classes (tx, ty, tw, th, obj, cls...).

    Decodare: stil YOLO pe celule — fiecare celulă propune o cutie față de grilă.
    """

    def __init__(self, num_classes: int = 1, grid_size: int = 10, in_size: int = 320):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.in_size = in_size
        self.out_ch = 5 + num_classes

        # Backbone: 320 -> 160 -> 80 -> 40 -> 20 -> 10
        self.stem = ConvBNReLU(3, 32, s=2)
        self.b1 = ConvBNReLU(32, 64, s=2)
        self.b2 = ConvBNReLU(64, 128, s=2)
        self.b3 = ConvBNReLU(128, 256, s=2)
        self.b4 = ConvBNReLU(256, 256, s=2)
        self.head = nn.Conv2d(256, self.out_ch, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        return self.head(x)


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def decode_predictions(
    pred: torch.Tensor,
    orig_w: int,
    orig_h: int,
    conf_thres: float = 0.35,
    grid_size: Optional[int] = None,
    min_box_area: Optional[float] = None,
    max_aspect_ratio: Optional[float] = None,
) -> List[Tuple[int, int, int, int, float, int]]:
    """
    pred: [1, C, S, S] pe CPU sau CUDA
    Returnează liste de (x1,y1,x2,y2, score, class_id) în coordonate orig_w x orig_h.
    """
    if pred.dim() != 4:
        raise ValueError("pred trebuie [1, C, S, S]")
    _, C, S, _ = pred.shape
    if grid_size is not None:
        S = grid_size
    num_classes = C - 5
    pred = pred[0]  # [C, S, S]

    tx = pred[0]
    ty = pred[1]
    tw = pred[2]
    th = pred[3]
    tobj = pred[4]
    tcls = pred[5:] if num_classes > 0 else pred[4:5]

    boxes: List[Tuple[int, int, int, int, float, int]] = []

    sig_tx = _sigmoid(tx)
    sig_ty = _sigmoid(ty)
    sig_obj = _sigmoid(tobj)

    for i in range(S):
        for j in range(S):
            if num_classes == 1:
                cls_score = _sigmoid(tcls[0, i, j])
                conf = sig_obj[i, j] * cls_score
                cls_id = 0
            else:
                probs = _sigmoid(tcls[:, i, j])
                cls_id = int(torch.argmax(probs).item())
                conf = sig_obj[i, j] * probs[cls_id]

            if conf.item() < conf_thres:
                continue

            cx_n = (j + sig_tx[i, j]) / S
            cy_n = (i + sig_ty[i, j]) / S
            bw_n = (1.8 / S) * torch.exp(torch.clamp(tw[i, j], -4.0, 4.0))
            bh_n = (2.4 / S) * torch.exp(torch.clamp(th[i, j], -4.0, 4.0))

            cx_pix = cx_n * orig_w
            cy_pix = cy_n * orig_h
            w_pix = bw_n * orig_w
            h_pix = bh_n * orig_h

            x1 = int(max(0, cx_pix - w_pix / 2))
            y1 = int(max(0, cy_pix - h_pix / 2))
            x2 = int(min(orig_w - 1, cx_pix + w_pix / 2))
            y2 = int(min(orig_h - 1, cy_pix + h_pix / 2))

            if x2 <= x1 or y2 <= y1:
                continue
            bw = float(x2 - x1)
            bh = float(y2 - y1)
            if min_box_area is not None and bw * bh < min_box_area:
                continue
            if max_aspect_ratio is not None and bw > 0 and bh > 0:
                ar = max(bw / bh, bh / bw)
                if ar > max_aspect_ratio:
                    continue
            boxes.append((x1, y1, x2, y2, float(conf.item()), cls_id))

    return nms_boxes(boxes, iou_thres=0.45)


def nms_boxes(
    boxes: List[Tuple[int, int, int, int, float, int]], iou_thres: float = 0.45
) -> List[Tuple[int, int, int, int, float, int]]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep: List[Tuple[int, int, int, int, float, int]] = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if iou_xyxy(best[:4], b[:4]) < iou_thres]
    return keep


def iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_detector(weights_path: Optional[str], device: torch.device) -> TinyCNNDetector:
    model = TinyCNNDetector(num_classes=1, grid_size=10, in_size=320)
    if weights_path and os.path.isfile(weights_path):
        try:
            state = torch.load(weights_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(weights_path, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
        print(f"Încărcat: {weights_path}")
    else:
        print(
            "Avertisment: nu există weights antrenate — predicțiile vor fi slabe. "
            f"Antrenează cu train_tiny_cnn.py sau pune tiny_cnn_person.pt lângă script."
        )
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def infer_frame(
    model: TinyCNNDetector,
    frame_bgr,
    device: torch.device,
    conf_thres: float = 0.35,
    input_size: int = 320,
    min_box_area: Optional[float] = None,
    max_aspect_ratio: Optional[float] = None,
) -> List[Tuple[int, int, int, int, float, int]]:
    """frame BGR OpenCV; return cutii în coordonate frame original."""
    h0, w0 = frame_bgr.shape[:2]
    resized = cv2.resize(frame_bgr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb = resized[:, :, ::-1].astype("float32") / 255.0
    chw = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    pred = model(chw)
    return decode_predictions(
        pred,
        w0,
        h0,
        conf_thres=conf_thres,
        grid_size=model.grid_size,
        min_box_area=min_box_area,
        max_aspect_ratio=max_aspect_ratio,
    )
