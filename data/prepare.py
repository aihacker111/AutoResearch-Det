import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
import yaml

_ROOT = Path(__file__).resolve().parent.parent

def _resolve_output_path(output: str) -> str:
    p = Path(output)
    return str(p.resolve()) if p.is_absolute() else str((_ROOT / p).resolve())

EVAL_IMGSZ = 640
EVAL_BATCH = 16
EVAL_WORKERS = 4
EVAL_CONF = 0.001
EVAL_IOU = 0.6

def _names_from_yaml(root: Path) -> dict | None:
    for p in [root / "data.yaml", root / "dataset.yaml", *root.glob("*.yaml")]:
        try:
            cfg = yaml.safe_load(p.read_text())
            names = cfg.get("names")
            if isinstance(names, list): return {i: n for i, n in enumerate(names)}
            if isinstance(names, dict): return {int(k): v for k, v in names.items()}
        except: pass
    txt = root / "classes.txt"
    if txt.exists():
        lines = [l.strip() for l in txt.read_text().splitlines() if l.strip()]
        return {i: n for i, n in enumerate(lines)}
    return None

def _names_from_json(json_path: Path) -> dict | None:
    try:
        data = json.loads(json_path.read_text())
        cats = sorted(data.get("categories", []), key=lambda c: int(c["id"]))
        if cats: return {i: c["name"] for i, c in enumerate(cats)}
    except: pass
    return None

def _has_images(p: Path) -> bool:
    return p.is_dir() and (any(p.glob("*.jpg")) or any(p.glob("*.png")) or any(p.glob("*.jpeg")))

def _find_images_dir(root: Path, split: str) -> Path | None:
    aliases = {"val": ["val", "valid"], "valid": ["val", "valid"]}.get(split, [split])
    for alias in aliases:
        for p in [root / alias / "images", root / "images" / alias, root / alias]:
            if _has_images(p): return p
    return None

def _find_annotation_json(root: Path, split: str) -> Path | None:
    aliases = {"val": ["val", "valid"], "valid": ["val", "valid"]}.get(split, [split])
    for alias in aliases:
        for p in [*root.glob(f"annotations_*_{alias}.json"), *root.glob(f"*_{alias}.json"), root / f"annotations_{alias}.json", root / "annotations" / f"instances_{alias}.json", root / "annotations" / f"{alias}.json", root / alias / "_annotations.coco.json", root / alias / "annotations.json"]:
            if p.is_file(): return p
    return None

def _convert_coco_to_yolo(json_path: Path, labels_dir: Path) -> int:
    data = json.loads(json_path.read_text())
    cats = sorted(data.get("categories", []), key=lambda c: int(c["id"]))
    id2idx = {c["id"]: i for i, c in enumerate(cats)}
    id2meta = {img["id"]: img for img in data.get("images", [])}
    anns_by_image = defaultdict(list)
    
    for ann in data.get("annotations", []):
        if not ann.get("iscrowd", 0) and ann["category_id"] in id2idx:
            anns_by_image[ann["image_id"]].append(ann)

    labels_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for img_id, anns in anns_by_image.items():
        meta = id2meta.get(img_id)
        if not meta: continue
        W, H = meta["width"], meta["height"]
        stem = Path(meta["file_name"]).stem
        lines = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0: continue
            cx, cy = max(0.0, min(1.0, (x + w / 2) / W)), max(0.0, min(1.0, (y + h / 2) / H))
            nw, nh = max(0.0, min(1.0, w / W)), max(0.0, min(1.0, h / H))
            cls = id2idx[ann["category_id"]]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        if lines:
            (labels_dir / f"{stem}.txt").write_text("\n".join(lines))
            written += 1
    return written

def _rel_under_root(base: Path, p: Path) -> str:
    base, p = base.resolve(), p.resolve()
    try: return str(p.relative_to(base))
    except ValueError: return str(p)

def _stage_images_into(src_images: Path, staging_root: Path, split: str) -> Path:
    dst = staging_root / "images" / split
    dst.mkdir(parents=True, exist_ok=True)
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for src in sorted(src_images.glob(ext)):
            out = dst / src.name
            if out.exists(): continue
            try: out.symlink_to(src.resolve())
            except OSError: shutil.copy2(src, out)
    return dst

def _labels_dir_for(root: Path, split: str, images_dir: Path) -> Path:
    img_str = str(images_dir.resolve())
    if "/images/" in img_str or img_str.endswith("/images"): return Path(img_str.replace("/images", "/labels", 1))
    
    roboflow_split = root / split / "images"
    try:
        if images_dir.resolve() == roboflow_split.resolve(): return root / split / "labels"
    except OSError: pass

    default = root / "labels" / split
    try:
        test_file = root / ".write_test_prepare"
        test_file.touch(); test_file.unlink()
        return default
    except (OSError, PermissionError): return Path(os.environ.get("LABELS_DIR", "labels")) / root.name / split

def _ensure_yolo_labels(root: Path, split: str, images_dir: Path) -> Path | None:
    labels_dir = _labels_dir_for(root, split, images_dir)
    if labels_dir.exists() and len(list(labels_dir.glob("*.txt"))) > 0: return labels_dir
    json_path = _find_annotation_json(root, split)
    if not json_path: return None
    _convert_coco_to_yolo(json_path, labels_dir)
    return labels_dir

def build_data_yaml(dataset_dir: str, output: str = "data.yaml", staging_dir: str | None = None) -> dict:
    root = Path(dataset_dir).resolve()
    class_map = _names_from_yaml(root)
    if not class_map:
        for split in ("train", "val", "valid"):
            jp = _find_annotation_json(root, split)
            if jp:
                class_map = _names_from_json(jp)
                if class_map: break
    
    nc = len(class_map)
    names = [class_map[i] for i in range(nc)]

    train_imgs = _find_images_dir(root, "train")
    val_imgs = _find_images_dir(root, "val")
    test_imgs = _find_images_dir(root, "test") or _find_images_dir(root, "test-dev")

    yaml_root = root
    if staging_dir:
        yaml_root = Path(staging_dir).resolve()
        yaml_root.mkdir(parents=True, exist_ok=True)
        train_imgs = _stage_images_into(train_imgs, yaml_root, "train")
        val_imgs = _stage_images_into(val_imgs, yaml_root, "val")
        if test_imgs: test_imgs = _stage_images_into(test_imgs, yaml_root, "test")

    _ensure_yolo_labels(root, "train", train_imgs)
    _ensure_yolo_labels(root, "val", val_imgs)
    if test_imgs: _ensure_yolo_labels(root, "test", test_imgs)

    cfg = {"path": str(yaml_root), "train": _rel_under_root(yaml_root, train_imgs), "val": _rel_under_root(yaml_root, val_imgs), "nc": nc, "names": names}
    if test_imgs: cfg["test"] = _rel_under_root(yaml_root, test_imgs)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    return cfg

def verify_dataset(data_yaml: str = "data.yaml") -> bool:
    p = Path(data_yaml)
    if not p.is_file(): return False
    cfg = yaml.safe_load(p.read_text())
    root = Path(cfg.get("path", "."))
    ok = True

    for split in ("train", "val"):
        img_dir = Path(cfg.get(split, ""))
        if not img_dir.is_absolute(): img_dir = root / img_dir
        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))
        lbl_dir = _labels_dir_for(root, split, img_dir)
        lbls = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        if not imgs or not lbls: ok = False
    return ok

def evaluate(weights: str, data_yaml: str = "data.yaml") -> dict:
    import torch
    from ultralytics import YOLO
    torch.cuda.reset_peak_memory_stats()
    results = YOLO(weights).val(data=data_yaml, imgsz=EVAL_IMGSZ, batch=EVAL_BATCH, workers=EVAL_WORKERS, conf=EVAL_CONF, iou=EVAL_IOU, verbose=False)
    box = results.box
    return {"val_mAP5095": float(box.map), "val_mAP50": float(box.map50)}

def main():
    parser = argparse.ArgumentParser(description="AutoResearch-DET setup & eval")
    parser.add_argument("--dataset-dir", default=os.environ.get("DATASET_DIR", ""))
    parser.add_argument("--output", default="data.yaml")
    parser.add_argument("--staging-dir", default=os.environ.get("PREPARE_STAGING_DIR", ""))
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--weights", default="")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    staging = (args.staging_dir or "").strip()
    if staging:
        out = Path(args.output)
        if not out.is_absolute(): args.output = str(Path(staging) / out.name)
    output_resolved = _resolve_output_path(args.output)

    if args.verify: sys.exit(0 if verify_dataset(output_resolved) else 1)
    elif args.eval: evaluate(args.weights, output_resolved)
    else: build_data_yaml(args.dataset_dir, output_resolved, staging_dir=staging or None)

if __name__ == "__main__":
    main()