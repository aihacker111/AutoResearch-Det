"""
prepare.py — Fixed setup & evaluation harness.
===============================================
AutoResearch-DET | DO NOT EDIT.

Responsibilities
----------------
1. Resolve dataset directory → generate data.yaml
2. Auto-detect classes; support YOLO flat, COCO JSON, Roboflow formats
3. Evaluate a trained checkpoint and print metrics

Usage
-----
    python prepare.py --dataset-dir /path/to/dataset
    python prepare.py --eval --weights output/train/exp/weights/best.pt
    python prepare.py --verify          # check dataset integrity only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml


# ── Constants (used by train.py as well) ──────────────────────────────────────
EVAL_IMGSZ   = 640
EVAL_BATCH   = 16
EVAL_WORKERS = 4
EVAL_CONF    = 0.001
EVAL_IOU     = 0.6


# ── Class-name detection ──────────────────────────────────────────────────────

def _names_from_yaml(root: Path) -> dict | None:
    for p in [root / "data.yaml", root / "dataset.yaml", *root.glob("*.yaml")]:
        try:
            cfg = yaml.safe_load(p.read_text())
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


def _names_from_coco(root: Path) -> dict | None:
    candidates = [
        root / "train" / "_annotations.coco.json",
        root / "valid" / "_annotations.coco.json",
        root / "val"   / "_annotations.coco.json",
        *(sorted((root / "annotations").glob("*.json"))
          if (root / "annotations").is_dir() else []),
        *sorted(root.glob("*.json")),
    ]
    for p in candidates:
        if not p.is_file():
            continue
        try:
            data  = json.loads(p.read_text())
            cats  = sorted(data.get("categories", []), key=lambda c: int(c["id"]))
            if cats:
                return {i: c["name"] for i, c in enumerate(cats)}
        except Exception:
            continue
    return None


# ── COCO JSON → YOLO txt conversion ──────────────────────────────────────────

def _convert_coco_split(json_path: Path, images_dir: Path, labels_dir: Path) -> None:
    """
    Convert a single COCO-format JSON split to YOLO .txt labels.
    Uses ultralytics built-in if available, falls back to manual conversion.
    """
    try:
        from ultralytics.data.converter import convert_coco
        # ultralytics expects a directory containing the JSON file
        convert_coco(
            labels_dir  = str(json_path.parent),
            save_dir    = str(labels_dir.parent),  # writes into <save_dir>/labels/
            use_segments= False,
            cls91to80   = False,
        )
        return
    except Exception:
        pass  # fallback below

    # ── Manual fallback ───────────────────────────────────────────────────────
    data     = json.loads(json_path.read_text())
    cats     = sorted(data["categories"], key=lambda c: int(c["id"]))
    id2idx   = {c["id"]: i for i, c in enumerate(cats)}
    id2meta  = {img["id"]: img for img in data["images"]}

    from collections import defaultdict
    anns_by_image: dict = defaultdict(list)
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        anns_by_image[ann["image_id"]].append(ann)

    labels_dir.mkdir(parents=True, exist_ok=True)
    for img_id, anns in anns_by_image.items():
        meta = id2meta.get(img_id)
        if not meta:
            continue
        W, H   = meta["width"], meta["height"]
        stem   = Path(meta["file_name"]).stem
        lines  = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            cls = id2idx.get(ann["category_id"], 0)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines))


# ── Dataset resolution ────────────────────────────────────────────────────────

def _find_images(root: Path, split: str) -> Path | None:
    for p in [root / split / "images", root / "images" / split, root / split]:
        if p.is_dir() and (list(p.glob("*.jpg")) or list(p.glob("*.png"))):
            return p
    return None


def _ensure_yolo_labels(root: Path, split: str) -> None:
    """Auto-convert COCO JSON labels to YOLO .txt if labels/ is missing."""
    images_dir = _find_images(root, split)
    if images_dir is None:
        return

    # Labels expected at sibling path images/ → labels/
    labels_dir = Path(str(images_dir).replace("images", "labels"))
    if labels_dir.exists() and any(labels_dir.glob("*.txt")):
        return  # already converted

    # Find COCO JSON for this split
    split_aliases = {"val": ["valid", "val"], "valid": ["valid", "val"]}.get(split, [split])
    json_candidates = []
    for alias in split_aliases:
        json_candidates += [
            root / alias / "_annotations.coco.json",
            root / "annotations" / f"instances_{alias}.json",
            root / "annotations" / f"{alias}.json",
        ]
    for jpath in json_candidates:
        if jpath.is_file():
            print(f"[prepare] Converting COCO JSON → YOLO labels: {jpath}")
            _convert_coco_split(jpath, images_dir, labels_dir)
            return


def build_data_yaml(dataset_dir: str, output: str = "data.yaml") -> dict:
    root = Path(dataset_dir).resolve()
    assert root.exists(), f"Dataset not found: {root}"

    class_map = _names_from_yaml(root) or _names_from_coco(root)
    if class_map is None:
        raise FileNotFoundError(
            f"Cannot find class names in {root}.\n"
            "Expected: data.yaml, classes.txt, or *_annotations.coco.json"
        )

    # Auto-convert COCO labels for each split
    for split in ("train", "valid", "val", "test"):
        _ensure_yolo_labels(root, split)

    nc    = len(class_map)
    names = [class_map[i] for i in range(nc)]

    train_path = _find_images(root, "train")
    val_path   = _find_images(root, "valid") or _find_images(root, "val")
    test_path  = _find_images(root, "test")

    assert train_path, f"Cannot find train images under {root}"
    assert val_path,   f"Cannot find val/valid images under {root}"

    cfg: dict = {
        "path" : str(root),
        "train": str(train_path),
        "val"  : str(val_path),
        "nc"   : nc,
        "names": names,
    }
    if test_path:
        cfg["test"] = str(test_path)

    Path(output).write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    print(f"[prepare] data.yaml → {output}")
    print(f"          nc={nc}  classes={names[:5]}{'...' if nc > 5 else ''}")
    print(f"          train={train_path}  val={val_path}")
    return cfg


# ── Dataset verification ──────────────────────────────────────────────────────

def verify_dataset(data_yaml: str = "data.yaml") -> bool:
    """Sanity-check that images/labels are paired and non-empty."""
    cfg   = yaml.safe_load(Path(data_yaml).read_text())
    ok    = True
    root  = Path(cfg.get("path", "."))
    for split in ("train", "val"):
        img_dir = Path(cfg.get(split, ""))
        if not img_dir.is_absolute():
            img_dir = root / img_dir
        lbl_dir = Path(str(img_dir).replace("images", "labels"))
        imgs    = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        lbls    = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
        print(f"[verify] {split}: {len(imgs)} images, {len(lbls)} labels  ({img_dir})")
        if len(imgs) == 0:
            print(f"  WARNING: no images found in {img_dir}")
            ok = False
        if len(lbls) == 0:
            print(f"  WARNING: no labels found in {lbl_dir}")
            ok = False
    return ok


# ── Evaluation ────────────────────────────────────────────────────────────────

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


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AutoResearch-DET setup & eval")
    parser.add_argument("--dataset-dir", default=os.environ.get("DATASET_DIR", ""))
    parser.add_argument("--output",      default="data.yaml")
    parser.add_argument("--eval",        action="store_true")
    parser.add_argument("--weights",     default="")
    parser.add_argument("--verify",      action="store_true",
                        help="Verify dataset integrity only")
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