import argparse
import json
import os
import shutil
import sys
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

def _stage_split(src_images: Path, root: Path, split: str, prepared_dir: Path) -> bool:
    """Safely stages images into images/{split} and converts/moves labels into labels/{split}."""
    if not src_images: return False

    img_dst = prepared_dir / "images" / split
    lbl_dst = prepared_dir / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    print(f"  [{split}] Staging images -> {img_dst}")
    img_count = 0
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for img in src_images.glob(ext):
            dest = img_dst / img.name
            if not dest.exists():
                try: dest.symlink_to(img.resolve())
                except OSError: shutil.copy2(img, dest)
            img_count += 1

    # Look for COCO json to convert
    json_path = _find_annotation_json(root, split)
    if json_path:
        print(f"  [{split}] Converting COCO JSON -> {lbl_dst}")
        lbl_count = _convert_coco_to_yolo(json_path, lbl_dst)
    else:
        # Fallback: Maybe it's already YOLO format? Try to find existing txt labels
        existing_labels_dir = None
        img_str = str(src_images.resolve())
        if "/images/" in img_str or img_str.endswith("/images"):
            existing_labels_dir = Path(img_str.replace("/images", "/labels", 1))
        elif (root / split / "labels").exists():
            existing_labels_dir = root / split / "labels"

        lbl_count = 0
        if existing_labels_dir and existing_labels_dir.exists():
            print(f"  [{split}] Copying YOLO labels -> {lbl_dst}")
            for txt in existing_labels_dir.glob("*.txt"):
                dest = lbl_dst / txt.name
                if not dest.exists():
                    try: dest.symlink_to(txt.resolve())
                    except OSError: shutil.copy2(txt, dest)
                lbl_count += 1

    print(f"  [{split}] Done: {img_count} images, {lbl_count} labels.")
    return True

def build_data_yaml(dataset_dir: str, output: str = "data.yaml") -> dict:
    root = Path(dataset_dir).resolve()
    out_yaml = Path(output).resolve()
    
    # Force unified YOLO output directory right next to the generated data.yaml
    prepared_dir = out_yaml.parent / "yolo_dataset"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[prepare] Building unified YOLO dataset at: {prepared_dir}")

    class_map = _names_from_yaml(root)
    if not class_map:
        for split in ("train", "val", "valid"):
            jp = _find_annotation_json(root, split)
            if jp:
                class_map = _names_from_json(jp)
                if class_map: break
    
    if not class_map:
        sys.exit(f"ERROR: Cannot find class names in {root}.")

    nc = len(class_map)
    names = [class_map[i] for i in range(nc)]

    train_imgs = _find_images_dir(root, "train")
    val_imgs = _find_images_dir(root, "val")
    test_imgs = _find_images_dir(root, "test") or _find_images_dir(root, "test-dev")

    if not train_imgs or not val_imgs:
        sys.exit(f"ERROR: Could not find train or val image directories in {root}")

    # Stage data into strict YOLO layout
    _stage_split(train_imgs, root, "train", prepared_dir)
    _stage_split(val_imgs, root, "val", prepared_dir)
    if test_imgs:
        _stage_split(test_imgs, root, "test", prepared_dir)

    # YOLO requires paths to be relative to the data.yaml directory OR absolute
    # By making them absolute, we guarantee YOLO never gets confused.
    cfg = {
        "path": str(prepared_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": nc,
        "names": names
    }
    if test_imgs: 
        cfg["test"] = "images/test"

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    
    print(f"\n[prepare] Successfully wrote {out_yaml}")
    return cfg

def verify_dataset(data_yaml: str = "data.yaml") -> bool:
    p = Path(data_yaml)
    if not p.is_file(): return False
    cfg = yaml.safe_load(p.read_text())
    root = Path(cfg.get("path", "."))
    ok = True

    print(f"\n[verify] Checking dataset integrity at {root}...")
    for split in ("train", "val"):
        img_dir = root / cfg.get(split, f"images/{split}")
        lbl_dir = root / f"labels/{split}"
        
        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))
        lbls = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        status = "✓" if (imgs and lbls) else "✗"
        print(f"  {status}  {split:5s}: {len(imgs):5d} images | {len(lbls):5d} labels")

        if not imgs or not lbls: ok = False
    return ok

def main():
    parser = argparse.ArgumentParser(description="AutoResearch-DET setup")
    parser.add_argument("--dataset-dir", default=os.environ.get("DATASET_DIR", ""))
    parser.add_argument("--output", default="data.yaml")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    output_resolved = _resolve_output_path(args.output)

    if args.verify: 
        sys.exit(0 if verify_dataset(output_resolved) else 1)
    else: 
        if not args.dataset_dir:
            sys.exit("ERROR: Pass --dataset-dir or set DATASET_DIR environment variable.")
        build_data_yaml(args.dataset_dir, output_resolved)

if __name__ == "__main__":
    main()