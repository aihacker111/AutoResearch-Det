"""
train.py — The ONLY file the agent is allowed to edit.
=======================================================
AutoResearch-DET | YOLO11 fine-tuning on small datasets.

Metric  : val/mAP50-95  (higher = better)
Backbone: YOLO11 (ultralytics)
Task    : object detection, fine-tuning from COCO pretrained weights
"""

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_SIZE = "n"        # "n" | "s" | "m" | "l" | "x"
PRETRAINED = True       # start from COCO pretrained weights

# ── Dataset ───────────────────────────────────────────────────────────────────
DATA_YAML  = "data.yaml"

# ── Training schedule ─────────────────────────────────────────────────────────
EPOCHS        = 1
PATIENCE      = 15          # ↑ slightly more tolerance
IMGSZ         = 640
CLOSE_MOSAIC  = 15          # ↑ disable mosaic earlier for small-data stability

# ── Optimiser ─────────────────────────────────────────────────────────────────
OPTIMIZER     = "AdamW"     # AdamW generally best for fine-tuning
LR0           = 1e-3
LRF           = 0.01        # cosine decay: final LR = LR0 * LRF
MOMENTUM      = 0.937
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS   = 3.0
WARMUP_BIAS_LR  = 0.1

# ── Augmentation ──────────────────────────────────────────────────────────────
HSV_H       = 0.015
HSV_S       = 0.7
HSV_V       = 0.4
DEGREES     = 0.0
TRANSLATE   = 0.1
SCALE       = 0.5
SHEAR       = 0.0
PERSPECTIVE = 0.0
FLIPUD      = 0.0
FLIPLR      = 0.5
MOSAIC      = 1.0
MIXUP       = 0.0
COPY_PASTE  = 0.0

# ── Regularisation ────────────────────────────────────────────────────────────
DROPOUT         = 0.0
LABEL_SMOOTHING = 0.0

# ── Hardware ──────────────────────────────────────────────────────────────────
BATCH   = 16
WORKERS = 4
AMP     = True

# ─────────────────────────────────────────────────────────────────────────────
# Do NOT edit below this line
# ─────────────────────────────────────────────────────────────────────────────
import json, os, sys, time
from pathlib import Path


def _resolve_weights_arg() -> str:
    """Absolute path to weights under this script's directory (stable for download + load)."""
    name = f"yolo11{MODEL_SIZE}.pt" if PRETRAINED else f"yolo11{MODEL_SIZE}.yaml"
    p = Path(name)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    return str(p.resolve())


def _load_yolo(weights_arg: str):
    """
    Construct YOLO(weights). Uses an exclusive lock so torch.distributed workers
    do not download the same .pt concurrently (corrupts the zip / PytorchStreamReader).
    """
    from ultralytics import YOLO

    root = Path(__file__).resolve().parent
    lock_path = root / ".ultralytics_yolo_init.lock"
    lock_f = open(lock_path, "a+", encoding="utf-8")
    try:
        if sys.platform != "win32":
            import fcntl

            fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            try:
                return YOLO(weights_arg)
            except RuntimeError as exc:
                err = str(exc).lower()
                if "pytorchstreamreader" in err or "failed reading" in err:
                    wp = Path(weights_arg)
                    if wp.suffix == ".pt" and wp.is_file():
                        wp.unlink()
                        return YOLO(weights_arg)
                raise
        finally:
            if sys.platform != "win32":
                fcntl.flock(lock_f, fcntl.LOCK_UN)
    finally:
        lock_f.close()


def _resolve_data_yaml() -> str:
    """Resolve DATA_YAML relative to this file; optional DATA_YAML env override."""
    raw = os.environ.get("DATA_YAML", DATA_YAML)
    p = Path(raw)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    p = p.resolve()
    if not p.is_file():
        root = Path(__file__).resolve().parent
        sys.exit(
            f"ERROR: Dataset config not found: {p}\n"
            "  Generate it from your dataset root, e.g.:\n"
            f"    python prepare.py --dataset-dir $DATASET_DIR\n"
            f"  (run inside {root} so data.yaml is created next to train.py)\n"
            "  Or: export DATA_YAML=/path/to/your/data.yaml"
        )
    return str(p)


def main():
    try:
        import torch
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ERROR: pip install ultralytics torch")

    weights = _resolve_weights_arg()
    model   = _load_yolo(weights)

    output_dir = os.environ.get("OUTPUT_DIR", "output/train")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    results = model.train(
        data            = _resolve_data_yaml(),
        epochs          = EPOCHS,
        patience        = PATIENCE,
        imgsz           = IMGSZ,
        batch           = BATCH,
        workers         = WORKERS,
        optimizer       = OPTIMIZER,
        lr0             = LR0,
        lrf             = LRF,
        momentum        = MOMENTUM,
        weight_decay    = WEIGHT_DECAY,
        warmup_epochs   = WARMUP_EPOCHS,
        warmup_bias_lr  = WARMUP_BIAS_LR,
        hsv_h           = HSV_H,
        hsv_s           = HSV_S,
        hsv_v           = HSV_V,
        degrees         = DEGREES,
        translate       = TRANSLATE,
        scale           = SCALE,
        shear           = SHEAR,
        perspective     = PERSPECTIVE,
        flipud          = FLIPUD,
        fliplr          = FLIPLR,
        mosaic          = MOSAIC,
        mixup           = MIXUP,
        copy_paste      = COPY_PASTE,
        dropout         = DROPOUT,
        label_smoothing = LABEL_SMOOTHING,
        amp             = AMP,
        close_mosaic    = CLOSE_MOSAIC,
        project         = output_dir,
        name            = "exp",
        exist_ok        = True,
        verbose         = True,
        plots           = False,
    )

    elapsed   = time.time() - t0
    peak_vram = (
        torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available() else 0.0
    )

    metrics = results.results_dict if hasattr(results, "results_dict") else {}
    map5095 = float(metrics.get("metrics/mAP50-95(B)", 0.0))
    map50   = float(metrics.get("metrics/mAP50(B)",    0.0))

    summary = {
        "model"        : Path(weights).name,
        "val_mAP5095"  : map5095,
        "val_mAP50"    : map50,
        "epochs"       : EPOCHS,
        "training_sec" : round(elapsed, 1),
        "peak_vram_mb" : round(peak_vram, 1),
    }
    exp_dir = Path(output_dir) / "exp"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("---")
    print(f"val_mAP5095:      {map5095:.6f}")
    print(f"val_mAP50:        {map50:.6f}")
    print(f"training_epochs:  {EPOCHS}")
    print(f"training_seconds: {elapsed:.1f}")
    print(f"peak_vram_mb:     {peak_vram:.1f}")
    print(f"checkpoint:       {exp_dir / 'weights/best.pt'}")


if __name__ == "__main__":
    main()