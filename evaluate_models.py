"""
Evaluare Precision, Recall, F1 pentru detecție persoană pe un set cu etichete YOLO.

Ground truth: fișiere .txt cu linii `class cx cy w h` (normalizat), aceeași structură ca la antrenare.

Exemplu:
  python3 evaluate_models.py --images data/images/val --labels data/labels/val

Opțional: doar anumite modele
  python3 evaluate_models.py --images val/img --labels val/lbl --models ssd yolo tiny
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from train_tiny_cnn import list_pairs, parse_yolo_label

# --- IoU (aceeași geometrie ca tiny_cnn_detector) ---


def iou_xyxy(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def yolo_norm_to_xyxy(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    x_c, y_c = cx * img_w, cy * img_h
    bw, bh = w * img_w, h * img_h
    x1 = x_c - bw / 2.0
    y1 = y_c - bh / 2.0
    x2 = x_c + bw / 2.0
    y2 = y_c + bh / 2.0
    x1 = max(0.0, min(float(img_w), x1))
    y1 = max(0.0, min(float(img_h), y1))
    x2 = max(0.0, min(float(img_w), x2))
    y2 = max(0.0, min(float(img_h), y2))
    return x1, y1, x2, y2


def load_gt_boxes(
    label_path: str, img_w: int, img_h: int, person_class_id: int
) -> List[Tuple[float, float, float, float]]:
    raw = parse_yolo_label(label_path)
    boxes = []
    for cx, cy, w, h, cid in raw:
        if cid != person_class_id:
            continue
        boxes.append(yolo_norm_to_xyxy(cx, cy, w, h, img_w, img_h))
    return boxes


def match_image(
    gt: List[Tuple[float, float, float, float]],
    preds: List[Tuple[float, float, float, float, float]],
    iou_thresh: float,
) -> Tuple[int, int, int]:
    """
    preds: (x1,y1,x2,y2, conf) sortate desc după conf.
    Returnează (tp, fp, fn) pentru această imagine.
    """
    preds_sorted = sorted(preds, key=lambda p: p[4], reverse=True)
    gt_matched = [False] * len(gt)
    tp = 0
    fp = 0
    for pred in preds_sorted:
        best_j = -1
        best_iou = 0.0
        pb = pred[:4]
        for j, g in enumerate(gt):
            if gt_matched[j]:
                continue
            ov = iou_xyxy(pb, g)
            if ov > best_iou:
                best_iou = ov
                best_j = j
        if best_iou >= iou_thresh and best_j >= 0:
            gt_matched[best_j] = True
            tp += 1
        else:
            fp += 1
    fn = sum(1 for m in gt_matched if not m)
    return tp, fp, fn


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


# --- Detectori ---

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def build_ssd_net():
    config_path = os.path.join(_SCRIPT_DIR, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    weights_path = os.path.join(_SCRIPT_DIR, "frozen_inference_graph.pb")
    if not os.path.isfile(config_path) or not os.path.isfile(weights_path):
        return None
    net = cv2.dnn_DetectionModel(weights_path, config_path)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    return net


def predict_ssd(
    net, img: np.ndarray, conf_thres: float, nms_thres: float, person_class_id: int
) -> List[Tuple[float, float, float, float, float]]:
    """person_class_id: 1 pentru ieșirea OpenCV COCO (person = primul index → id 1)."""
    class_ids, confs, bbox = net.detect(img, confThreshold=conf_thres, nmsThreshold=nms_thres)
    out: List[Tuple[float, float, float, float, float]] = []
    if class_ids is None or len(class_ids) == 0:
        return out
    class_ids = np.array(class_ids).flatten()
    confs = np.array(confs).flatten()
    for cid, cf, box in zip(class_ids, confs, bbox):
        if int(cid) != person_class_id:
            continue
        x, y, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        out.append((x, y, x + w, y + h, float(cf)))
    return out


def predict_yolo(model, img: np.ndarray, conf_thres: float, person_class_id: int):
    out: List[Tuple[float, float, float, float, float]] = []
    results = model.predict(img, conf=conf_thres, verbose=False)
    if not results:
        return out
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return out
    for box in boxes:
        cls_id = int(box.cls[0].item())
        if cls_id != person_class_id:
            continue
        cf = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        out.append((float(x1), float(y1), float(x2), float(y2), cf))
    return out


def predict_tiny(detector, device, img: np.ndarray, conf_thres: float, min_area: Optional[float], max_ar: Optional[float]):
    from tiny_cnn_detector import infer_frame

    dets = infer_frame(
        detector,
        img,
        device,
        conf_thres=conf_thres,
        min_box_area=min_area,
        max_aspect_ratio=max_ar,
    )
    out: List[Tuple[float, float, float, float, float]] = []
    for x1, y1, x2, y2, cf, _cid in dets:
        out.append((float(x1), float(y1), float(x2), float(y2), float(cf)))
    return out


def run_eval(
    pairs: Sequence[Tuple[str, str]],
    predict_fn: Callable[[np.ndarray], List[Tuple[float, float, float, float, float]]],
    iou_thresh: float,
) -> Tuple[int, int, int]:
    tp_t = fp_t = fn_t = 0
    for ip, lp in pairs:
        img = cv2.imread(ip)
        if img is None:
            continue
        h0, w0 = img.shape[:2]
        gt = load_gt_boxes(lp, w0, h0, person_class_id=0)
        preds = predict_fn(img)
        tp, fp, fn_ = match_image(gt, preds, iou_thresh)
        tp_t += tp
        fp_t += fp
        fn_t += fn_
    return tp_t, fp_t, fn_t


def main():
    ap = argparse.ArgumentParser(description="Precision / Recall / F1 pentru modelele din proiect")
    ap.add_argument("--images", required=True, help="Folder imagini (aceeași structură ca la train_tiny_cnn)")
    ap.add_argument("--labels", required=True, help="Folder etichete YOLO .txt")
    ap.add_argument(
        "--models",
        nargs="*",
        default=["ssd", "yolo", "tiny"],
        choices=["ssd", "yolo", "tiny"],
        help="Ce modele să ruleze (implicit: toate)",
    )
    ap.add_argument("--iou", type=float, default=0.5, help="Prag IoU pentru TP (standard 0.5)")
    ap.add_argument("--ssd-conf", type=float, default=0.55)
    ap.add_argument("--ssd-nms", type=float, default=0.2)
    ap.add_argument("--yolo-conf", type=float, default=0.4)
    ap.add_argument("--tiny-conf", type=float, default=0.55)
    ap.add_argument("--tiny-min-area", type=float, default=3500.0)
    ap.add_argument("--tiny-max-ar", type=float, default=2.8)
    ap.add_argument(
        "--tiny-weights",
        default=os.path.join(_SCRIPT_DIR, "tiny_cnn_person.pt"),
        help="Checkpoint TinyCNN",
    )
    ap.add_argument("--yolo-weights", default=os.path.join(_SCRIPT_DIR, "yolov8n.pt"))
    args = ap.parse_args()

    img_dir = os.path.abspath(os.path.expanduser(args.images))
    lbl_dir = os.path.abspath(os.path.expanduser(args.labels))
    if not os.path.isdir(img_dir):
        ex_img = os.path.join(_SCRIPT_DIR, "data", "images", "train")
        ex_lbl = os.path.join(_SCRIPT_DIR, "data", "labels", "train")
        print(
            "Eroare: folderul --images nu există:\n"
            f"  {img_dir}\n\n"
            "Textul «CALE/CATRE/...» din exemplu era doar un placeholder — pune calea ta reală.\n"
            "În acest proiect poți folosi de exemplu (train):\n"
            f'  --images "{ex_img}" --labels "{ex_lbl}"',
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.path.isdir(lbl_dir):
        print(
            "Eroare: folderul --labels nu există:\n" f"  {lbl_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    pairs = list_pairs(img_dir, lbl_dir)
    if not pairs:
        print("Nu am găsit perechi imagine/label.", file=sys.stderr)
        sys.exit(1)

    print(f"Imagini evaluate: {len(pairs)}")
    print(f"Prag IoU (TP): {args.iou}\n")

    results: Dict[str, Tuple[int, int, int]] = {}

    if "ssd" in args.models:
        net = build_ssd_net()
        if net is None:
            print("SSD: lipsesc frozen_inference_graph.pb sau pbtxt — sarit.\n")
        else:
            # COCO OpenCV: class id pentru person = 1
            tp, fp, fn_ = run_eval(
                pairs,
                lambda im: predict_ssd(net, im, args.ssd_conf, args.ssd_nms, person_class_id=1),
                args.iou,
            )
            results["SSD MobileNet v3 (COCO)"] = (tp, fp, fn_)

    if "yolo" in args.models:
        try:
            from ultralytics import YOLO
        except ImportError:
            print("YOLO: ultralytics nu e instalat — sarit.\n")
        else:
            wpath = args.yolo_weights
            if not os.path.isfile(wpath):
                print(f"YOLO: nu există {wpath} — sarit.\n")
            else:
                yolo = YOLO(wpath)
                tp, fp, fn_ = run_eval(
                    pairs,
                    lambda im: predict_yolo(yolo, im, args.yolo_conf, person_class_id=0),
                    args.iou,
                )
                results["YOLOv8n (COCO)"] = (tp, fp, fn_)

    if "tiny" in args.models:
        import torch
        from tiny_cnn_detector import load_detector

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wpath = args.tiny_weights if os.path.isfile(args.tiny_weights) else None
        tiny = load_detector(wpath, device)
        min_a = args.tiny_min_area if args.tiny_min_area > 0 else None
        max_ar = args.tiny_max_ar if args.tiny_max_ar > 0 else None
        tp, fp, fn_ = run_eval(
            pairs,
            lambda im: predict_tiny(
                tiny, device, im, args.tiny_conf, min_area=min_a, max_ar=max_ar
            ),
            args.iou,
        )
        results["TinyCNN (custom)"] = (tp, fp, fn_)

    if not results:
        print("Niciun model evaluat. Verifică fișierele și dependențele.")
        sys.exit(1)

    # Tabel
    col_w = 28
    print(f"{'Model':<{col_w}} {'TP':>8} {'FP':>8} {'FN':>8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * (col_w + 8 * 3 + 10 * 3 + 6))
    for name, (tp, fp, fn_) in results.items():
        p, r, f1 = precision_recall_f1(tp, fp, fn_)
        print(
            f"{name:<{col_w}} {tp:>8} {fp:>8} {fn_:>8} {p:>10.4f} {r:>10.4f} {f1:>10.4f}"
        )


if __name__ == "__main__":
    main()
