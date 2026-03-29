"""
prepare.py — Fixed setup & evaluation harness.
===============================================
AutoResearch-DET | DO NOT EDIT.

Supported dataset structures
-----------------------------
A) Roboflow / split-level COCO:
   train/_annotations.coco.json  +  train/images/
   valid/_annotations.coco.json  +  valid/images/

B) Root-level COCO (e.g. VisDrone):
   annotations_*_train.json  +  train/   (images directly inside)
   annotations_*_val.json    +  val/

C) YOLO flat (labels already converted):
   images/train/  +  labels/train/
   data.yaml or classes.txt

Usage
-----
    python prepare.py --dataset-dir /path/to/dataset
    python prepare.py --eval --weights output/train/exp/weights/best.pt
    python prepare.py --verify
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml


# ── Constants ─────────────────────────────────────────────────────────────────
EVAL_IMGSZ   = 640
EVAL_BATCH   = 16
EVAL_WORKERS = 4
EVAL_CONF    = 0.001
EVAL_IOU     = 0.6


# ═══════════════════════════════════════════════════════════════════════════════
# Class-name helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _names_from_yaml(root: Path) -> dict | None:
    for p in [root / "data.yaml", root / "dataset.yaml", *root.glob("*.yaml")]:
        try:
            cfg   = yaml.safe_load(p.read_text())
            names = cfg.get("names")
            if names is None:
                continue
            if isinstance(names, list):
                return {i: n for i, n in enumerate(names)}
            if isinstance(names, dict):
                return {int(k): v for k, v in names.items()}
        except Exception:
            continue
    txt = root / "classes.txt"
    if txt.exists():
        lines = [l.strip() for l in txt.read_text().splitlines() if l.strip()]
        return {i: n for i, n in enumerate(lines)}
    return None


def _names_from_json(json_path: Path) -> dict | None:
    try:
        data = json.loads(json_path.read_text())
        cats = sorted(data.get("categories", []), key=lambda c: int(c["id"]))
        if cats:
            return {i: c["name"] for i, c in enumerate(cats)}
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset structure detection
# ═══════════════════════════════════════════════════════════════════════════════

def _has_images(p: Path) -> bool:
    if not p.is_dir():
        return False
    return any(p.glob("*.jpg")) or any(p.glob("*.png")) or any(p.glob("*.jpeg"))


def _find_images_dir(root: Path, split: str) -> Path | None:
    """Resolve image directory for a split, trying multiple layout conventions."""
    aliases = {"val": ["val", "valid"], "valid": ["val", "valid"]}.get(split, [split])
    for alias in aliases:
        for p in [
            root / alias / "images",   # Roboflow: train/images/
            root / "images" / alias,   # YOLO flat: images/train/
            root / alias,              # VisDrone: train/ (images directly)
        ]:
            if _has_images(p):
                return p
    return None


def _find_annotation_json(root: Path, split: str) -> Path | None:
    """
    Locate COCO JSON for a split.
    Supports root-level naming (VisDrone) and split-folder naming (Roboflow).
    """
    aliases = {"val": ["val", "valid"], "valid": ["val", "valid"]}.get(split, [split])
    for alias in aliases:
        candidates = [
            # Root-level (VisDrone): annotations_VisDrone_train.json
            *root.glob(f"annotations_*_{alias}.json"),
            *root.glob(f"*_{alias}.json"),
            root / f"annotations_{alias}.json",
            # Standard COCO annotations folder
            root / "annotations" / f"instances_{alias}.json",
            root / "annotations" / f"{alias}.json",
            # Roboflow split-folder
            root / alias / "_annotations.coco.json",
            root / alias / "annotations.json",
        ]
        for p in candidates:
            if p.is_file():
                return p
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# COCO → YOLO label conversion
# ═══════════════════════════════════════════════════════════════════════════════

def _convert_coco_to_yolo(json_path: Path, labels_dir: Path) -> int:
    """
    Convert COCO JSON to YOLO .txt label files.

    Always uses manual conversion (NOT ultralytics built-in) because
    ultralytics convert_coco computes class = category_id - 1, which
    produces class -1 when any category has id=0 (e.g. VisDrone).

    Safe mapping: sort categories by id, enumerate from 0 regardless
    of original id values — guaranteed no negative class indices.
    Returns number of label files written.
    """
    data = json.loads(json_path.read_text())

    # Safe 0-based index: sort by original id, enumerate from 0
    cats   = sorted(data.get("categories", []), key=lambda c: int(c["id"]))
    id2idx = {c["id"]: i for i, c in enumerate(cats)}

    print(f"          categories ({len(cats)}): "
          + ", ".join(f"{c['id']}->{id2idx[c['id']]}:{c['name']}" for c in cats[:6])
          + ("..." if len(cats) > 6 else ""))

    id2meta: dict = {img["id"]: img for img in data.get("images", [])}

    anns_by_image: dict = defaultdict(list)
    skipped_crowd = skipped_cat = 0
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            skipped_crowd += 1
            continue
        if ann["category_id"] not in id2idx:
            skipped_cat += 1
            continue
        anns_by_image[ann["image_id"]].append(ann)

    if skipped_crowd: print(f"          skipped {skipped_crowd} crowd annotations")
    if skipped_cat:   print(f"          skipped {skipped_cat} unknown category_id")

    labels_dir.mkdir(parents=True, exist_ok=True)
    written = skipped_bbox = 0
    for img_id, anns in anns_by_image.items():
        meta = id2meta.get(img_id)
        if not meta:
            continue
        W, H  = meta["width"], meta["height"]
        stem  = Path(meta["file_name"]).stem
        lines = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                skipped_bbox += 1
                continue
            cx  = max(0.0, min(1.0, (x + w / 2) / W))
            cy  = max(0.0, min(1.0, (y + h / 2) / H))
            nw  = max(0.0, min(1.0, w / W))
            nh  = max(0.0, min(1.0, h / H))
            cls = id2idx[ann["category_id"]]  # guaranteed >= 0
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        if lines:
            (labels_dir / f"{stem}.txt").write_text("\n".join(lines))
            written += 1

    if skipped_bbox: print(f"          skipped {skipped_bbox} zero-size bboxes")
    return written

def _labels_dir_for(root: Path, split: str, images_dir: Path) -> Path:
    """
    Derive the labels directory path from the images directory.

    If root is read-only (e.g. /kaggle/input), labels are redirected to
    a writable sibling: /kaggle/working/labels/<dataset_name>/<split>
    Otherwise:
      - train/images/ → train/labels/
      - images/train/ → labels/train/
      - train/        → labels/train/  (VisDrone style)
    """
    # ── Read-only root guard ──────────────────────────────────────────────────
    try:
        root.stat()
        test_file = root / ".write_test"
        test_file.touch()
        test_file.unlink()
    except (OSError, PermissionError):
        # Root is read-only → write labels next to the working script
        working_dir = Path(os.environ.get("LABELS_DIR", "labels"))
        return working_dir / root.name / split

    # ── Writable root: keep labels near images ────────────────────────────────
    img_str = str(images_dir)
    if "/images/" in img_str or img_str.endswith("/images"):
        return Path(img_str.replace("/images", "/labels", 1))
    return root / "labels" / split


def _ensure_yolo_labels(root: Path, split: str, images_dir: Path) -> Path | None:
    """Convert COCO annotations to YOLO labels if not already done."""
    labels_dir = _labels_dir_for(root, split, images_dir)

    if labels_dir.exists() and len(list(labels_dir.glob("*.txt"))) > 0:
        return labels_dir  # already converted

    json_path = _find_annotation_json(root, split)
    if json_path is None:
        print(f"  [prepare] WARNING: no annotation JSON found for split '{split}'")
        return None

    print(f"[prepare] Converting '{split}': {json_path.name} → {labels_dir}")
    n = _convert_coco_to_yolo(json_path, labels_dir)
    print(f"          {n} label files written → {labels_dir}")
    return labels_dir


# ═══════════════════════════════════════════════════════════════════════════════
# Main: build data.yaml
# ═══════════════════════════════════════════════════════════════════════════════

def build_data_yaml(dataset_dir: str, output: str = "data.yaml") -> dict:
    root = Path(dataset_dir).resolve()
    assert root.exists(), f"Dataset not found: {root}"

    # ── Detect class names ────────────────────────────────────────────────────
    class_map = _names_from_yaml(root)
    if class_map is None:
        for split in ("train", "val", "valid"):
            jp = _find_annotation_json(root, split)
            if jp:
                class_map = _names_from_json(jp)
                if class_map:
                    break
    if class_map is None:
        raise FileNotFoundError(
            f"Cannot find class names in {root}.\n"
            "Expected: data.yaml, classes.txt, annotations_*_train.json, "
            "or train/_annotations.coco.json"
        )

    nc    = len(class_map)
    names = [class_map[i] for i in range(nc)]

    # ── Resolve image directories ─────────────────────────────────────────────
    train_imgs = _find_images_dir(root, "train")
    val_imgs   = _find_images_dir(root, "val")
    test_imgs  = _find_images_dir(root, "test") or _find_images_dir(root, "test-dev")

    assert train_imgs, f"Cannot find train images under {root}"
    assert val_imgs,   f"Cannot find val images under {root}"

    # ── Convert COCO → YOLO labels for each split ─────────────────────────────
    _ensure_yolo_labels(root, "train", train_imgs)
    _ensure_yolo_labels(root, "val",   val_imgs)
    if test_imgs:
        _ensure_yolo_labels(root, "test", test_imgs)

    # ── Write data.yaml ───────────────────────────────────────────────────────
    cfg: dict = {
        "path" : str(root),
        "train": str(train_imgs),
        "val"  : str(val_imgs),
        "nc"   : nc,
        "names": names,
    }
    if test_imgs:
        cfg["test"] = str(test_imgs)

    Path(output).write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    print(f"\n[prepare] data.yaml → {output}")
    print(f"          nc={nc}  classes={names[:6]}{'...' if nc > 6 else ''}")
    print(f"          train : {train_imgs}")
    print(f"          val   : {val_imgs}")
    if test_imgs:
        print(f"          test  : {test_imgs}")
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset verification
# ═══════════════════════════════════════════════════════════════════════════════

def verify_dataset(data_yaml: str = "data.yaml") -> bool:
    cfg  = yaml.safe_load(Path(data_yaml).read_text())
    root = Path(cfg.get("path", "."))
    ok   = True

    print("\n[verify] Checking dataset integrity…")
    for split in ("train", "val"):
        img_dir = Path(cfg.get(split, ""))
        if not img_dir.is_absolute():
            img_dir = root / img_dir

        imgs = (list(img_dir.glob("*.jpg")) +
                list(img_dir.glob("*.jpeg")) +
                list(img_dir.glob("*.png")))

        lbl_dir = _labels_dir_for(root, split, img_dir)
        lbls    = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        status = "✓" if (imgs and lbls) else "✗"
        rel    = img_dir.relative_to(root) if img_dir.is_relative_to(root) else img_dir
        print(f"  {status}  {split:5s}: {len(imgs):5d} images | {len(lbls):5d} labels  ({rel})")

        if not imgs:
            print(f"     WARNING: no images found in {img_dir}")
            ok = False
        if not lbls:
            print(f"     WARNING: no labels found in {lbl_dir}")
            print(f"     Hint: run  python prepare.py --dataset-dir {root}")
            ok = False
        elif abs(len(imgs) - len(lbls)) > 5:
            print(f"     WARNING: image/label count mismatch ({len(imgs)} vs {len(lbls)})")

    print(f"\n  {'[OK] Dataset looks good.' if ok else '[FAIL] Fix warnings above before training.'}")
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(weights: str, data_yaml: str = "data.yaml") -> dict:
    try:
        import torch
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ERROR: pip install ultralytics torch")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    results = YOLO(weights).val(
        data    = data_yaml,
        imgsz   = EVAL_IMGSZ,
        batch   = EVAL_BATCH,
        workers = EVAL_WORKERS,
        conf    = EVAL_CONF,
        iou     = EVAL_IOU,
        verbose = False,
    )

    elapsed   = time.time() - t0
    peak_vram = (
        torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available() else 0.0
    )
    box     = results.box
    map5095 = float(box.map)
    map50   = float(box.map50)

    print("---")
    print(f"val_mAP5095:  {map5095:.6f}")
    print(f"val_mAP50:    {map50:.6f}")
    print(f"eval_seconds: {elapsed:.1f}")
    print(f"peak_vram_mb: {peak_vram:.1f}")
    return {"val_mAP5095": map5095, "val_mAP50": map50, "peak_vram_mb": peak_vram}


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AutoResearch-DET setup & eval")
    parser.add_argument("--dataset-dir", default=os.environ.get("DATASET_DIR", ""))
    parser.add_argument("--output",      default="data.yaml")
    parser.add_argument("--eval",        action="store_true")
    parser.add_argument("--weights",     default="")
    parser.add_argument("--verify",      action="store_true",
                        help="Check dataset integrity only")
    args = parser.parse_args()

    if args.verify:
        ok = verify_dataset(args.output)
        sys.exit(0 if ok else 1)
    elif args.eval:
        assert args.weights, "Pass --weights /path/to/best.pt"
        evaluate(args.weights, args.output)
    else:
        assert args.dataset_dir, "Pass --dataset-dir or set DATASET_DIR"
        build_data_yaml(args.dataset_dir, args.output)


if __name__ == "__main__":
    main()