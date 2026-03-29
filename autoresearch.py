"""
run_autoresearch.py — Autonomous hyperparameter research loop.
==============================================================
AutoResearch-DET | Object Detection with YOLO11

The agent (LLM) proposes ONE hyperparameter change per iteration.
This runner executes the experiment, records the result, and keeps
or discards the change — completely autonomously.

Usage
-----
    python run_autoresearch.py --experiments 10
    python run_autoresearch.py --experiments 10 --cuda-devices 0,1
    python run_autoresearch.py --resume          # continue from last run
    python run_autoresearch.py --dry-run         # LLM proposals only, no training

Environment variables
---------------------
    OPENROUTER_API_KEY   API key for OpenRouter
    DATASET_DIR          Path to dataset root
    OUTPUT_DIR           Base output directory (default: output/train)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────
_DEFAULT_MODEL  = "openrouter/meta-llama/llama-3.3-70b-instruct:free"
_ROOT           = Path(__file__).resolve().parent
_TRAIN_FILE     = _ROOT / "train.py"
_RESULTS_FILE   = _ROOT / "results.tsv"
_LOG_FILE       = _ROOT / "run.log"
# Git paths are relative to repo root (= _ROOT when layout is standard)
_GIT_TRAIN_PATH = "train.py"
_TIMEOUT_FACTOR = 2.5
_LLM_RETRIES    = 3
_LLM_RETRY_WAIT = 5   # seconds between retries

# Runtime globals
OPENROUTER_API_KEY: str = "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"
LLM_MODEL:          str = _DEFAULT_MODEL
NUM_GPUS:           int = 2
_shutdown_requested:bool = False


# ── Graceful Ctrl-C ───────────────────────────────────────────────────────────

def _handle_sigint(sig, frame):
    global _shutdown_requested
    print("\n[!] Interrupt received — finishing current experiment then stopping.")
    _shutdown_requested = True

signal.signal(signal.SIGINT, _handle_sigint)


# ── GPU detection ─────────────────────────────────────────────────────────────

def detect_gpus() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return max(len([l for l in out.strip().splitlines() if l.strip()]), 1)
    except Exception:
        return 1


def build_cmd(num_gpus: int) -> list[str]:
    train = str(_TRAIN_FILE)
    if num_gpus <= 1:
        return [sys.executable, train]
    return [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        "--master_port=29500",
        train,
    ]


# ── LLM interaction ───────────────────────────────────────────────────────────

def _call_api(messages: list[dict], max_tokens: int = 3000) -> str:
    payload = json.dumps({
        "model"     : LLM_MODEL,
        "max_tokens": max_tokens,
        "messages"  : messages,
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data    = payload,
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type" : "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def _call_api_with_retry(messages: list[dict], max_tokens: int = 3000) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, _LLM_RETRIES + 1):
        try:
            return _call_api(messages, max_tokens)
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
            last_exc = e
            if attempt < _LLM_RETRIES:
                print(f"  [LLM] attempt {attempt} failed ({e}), retrying in {_LLM_RETRY_WAIT}s…")
                time.sleep(_LLM_RETRY_WAIT * attempt)
    raise RuntimeError(f"LLM API failed after {_LLM_RETRIES} attempts: {last_exc}")


def _parse_json(text: str) -> dict:
    """Parse JSON from LLM response, stripping markdown fences."""
    text = re.sub(r"```(?:json|python)?\s*", "", text)
    text = re.sub(r"```\s*",                  "", text)
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        desc = re.search(r'"description"\s*:\s*"([^"]+)"',    text)
        code = re.search(r'"new_code"\s*:\s*"(.*?)"\s*[,}]', text, re.DOTALL)
        if not (desc and code):
            raise ValueError(f"Cannot parse LLM JSON:\n{text[:300]}")
        return {
            "description": desc.group(1),
            "new_code"   : code.group(1).replace("\\n", "\n").replace('\\"', '"'),
        }


def propose_experiment(history: list[dict], current_code: str) -> dict:
    history_block = "\n".join(
        f"  exp{i+1:02d}: {h['description']}  →  "
        f"mAP50-95={h['val_mAP5095']:.4f}  ({h['status']})"
        for i, h in enumerate(history)
    ) or "  (none yet — this will be the baseline)"

    messages = [
        {
            "role"   : "system",
            "content": (
                "You are an expert computer vision researcher optimising YOLO11 "
                "fine-tuning on a small custom dataset. "
                "Your goal is to maximise val/mAP50-95. "
                f"Training runs on {NUM_GPUS} GPU(s). "
                "Reply ONLY with a single JSON object — "
                '{"description": "one-line summary", "new_code": "full train.py content"}. '
                "No markdown, no explanation outside the JSON."
            ),
        },
        {
            "role"   : "user",
            "content": (
                f"Experiment history:\n{history_block}\n\n"
                f"Current train.py:\n{current_code}\n\n"
                "Pick ONE change most likely to improve val/mAP50-95 for small-data "
                "fine-tuning. Good candidates: LR schedule, augmentation strength, "
                "mosaic/mixup, close_mosaic, optimizer, warmup, weight decay, "
                "dropout, label smoothing, model size, imgsz.\n"
                "Return the complete modified train.py as new_code."
            ),
        },
    ]
    return _parse_json(_call_api_with_retry(messages))


# ── Git helpers ───────────────────────────────────────────────────────────────

def git_is_clean() -> bool:
    r = subprocess.run(
        ["git", "-C", str(_ROOT), "status", "--porcelain", _GIT_TRAIN_PATH],
        capture_output=True, text=True,
    )
    return r.stdout.strip() == ""


def git_commit(msg: str) -> str:
    subprocess.run(
        ["git", "-C", str(_ROOT), "add", _GIT_TRAIN_PATH],
        capture_output=True,
    )
    r = subprocess.run(
        ["git", "-C", str(_ROOT), "commit", "-m", msg],
        capture_output=True,
    )
    if r.returncode != 0 and b"nothing to commit" in r.stdout + r.stderr:
        # Nothing changed (e.g. baseline); return current HEAD
        pass
    return subprocess.check_output(
        ["git", "-C", str(_ROOT), "rev-parse", "--short", "HEAD"], text=True
    ).strip()


def git_reset_last() -> None:
    """Discard the last commit (experiment that didn't improve)."""
    subprocess.run(
        ["git", "-C", str(_ROOT), "reset", "--hard", "HEAD~1"],
        capture_output=True,
    )


# ── Training & metrics ────────────────────────────────────────────────────────

def run_training(num_gpus: int, timeout: float | None) -> tuple[bool, float]:
    cmd = build_cmd(num_gpus)
    t0  = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd     = str(_ROOT),
            stdout  = open(_LOG_FILE, "w"),
            stderr  = subprocess.STDOUT,
            timeout = timeout,
        )
        return proc.returncode == 0, time.time() - t0
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  [TIMEOUT] killed after {elapsed:.0f}s")
        return False, elapsed


def parse_metrics() -> dict[str, float]:
    metrics: dict[str, float] = {
        "val_mAP5095" : 0.0,
        "val_mAP50"   : 0.0,
        "peak_vram_mb": 0.0,
    }
    try:
        for line in Path(_LOG_FILE).read_text(errors="ignore").splitlines():
            for key in metrics:
                m = re.match(rf"^{key}:\s+([\d.]+)", line)
                if m:
                    metrics[key] = float(m.group(1))
    except FileNotFoundError:
        pass
    return metrics


# ── Results file ──────────────────────────────────────────────────────────────

_TSV_HEADER = "commit\tval_mAP5095\tval_mAP50\tmemory_gb\tstatus\tdescription\n"

def _ensure_results_file() -> None:
    if not Path(_RESULTS_FILE).exists():
        Path(_RESULTS_FILE).write_text(_TSV_HEADER)


def load_history() -> list[dict]:
    """Load completed experiments from results.tsv for --resume."""
    history = []
    p = Path(_RESULTS_FILE)
    if not p.exists():
        return history
    lines = p.read_text().splitlines()[1:]  # skip header
    for line in lines:
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        history.append({
            "description" : parts[5],
            "val_mAP5095" : float(parts[1]),
            "status"      : parts[4],
        })
    return history


def append_result(commit: str, metrics: dict, status: str, description: str) -> None:
    mem_gb = round(metrics["peak_vram_mb"] / 1024, 1)
    row    = (
        f"{commit}\t{metrics['val_mAP5095']:.6f}\t{metrics['val_mAP50']:.6f}"
        f"\t{mem_gb}\t{status}\t{description}\n"
    )
    with open(_RESULTS_FILE, "a") as f:
        f.write(row)


# ── Progress plot ─────────────────────────────────────────────────────────────

def update_plot() -> None:
    plot = _ROOT / "plot_progress.py"
    if plot.exists():
        subprocess.run(
            [sys.executable, str(plot)],
            cwd=str(_ROOT),
            capture_output=True,
        )


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    global LLM_MODEL, NUM_GPUS, OPENROUTER_API_KEY

    parser = argparse.ArgumentParser(
        description="AutoResearch-DET: autonomous YOLO11 hyperparameter search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiments",  type=int, default=10)
    parser.add_argument("--model",        default=_DEFAULT_MODEL)
    parser.add_argument("--gpus",         type=int, default=0)
    parser.add_argument("--cuda-devices", default=None)
    parser.add_argument("--resume",       action="store_true",
                        help="Continue from existing results.tsv")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Propose experiments without training")
    args = parser.parse_args()

    # ── Environment ----------------------------------------------------------
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "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")
    if not OPENROUTER_API_KEY:
        sys.exit("ERROR: set OPENROUTER_API_KEY environment variable")

    LLM_MODEL = args.model

    cuda = args.cuda_devices or os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda
        NUM_GPUS = args.gpus or len(cuda.split(","))
    else:
        NUM_GPUS = args.gpus or detect_gpus()

    # ── Git branch ----------------------------------------------------------
    tag    = datetime.now().strftime("%b%d").lower()
    branch = f"autoresearch/{tag}"
    r = subprocess.run(
        ["git", "-C", str(_ROOT), "checkout", "-b", branch],
        capture_output=True,
    )
    if r.returncode != 0:
        subprocess.run(
            ["git", "-C", str(_ROOT), "checkout", branch],
            capture_output=True,
        )

    if not _TRAIN_FILE.is_file():
        sys.exit(
            f"ERROR: {_TRAIN_FILE} not found. Add train.py next to autoresearch.py "
            "(or clone the full AutoResearch-Det repo)."
        )

    _ensure_results_file()

    # ── Resume: reload history -----------------------------------------------
    history       = load_history() if args.resume else []
    completed     = len(history)
    remaining     = args.experiments - completed
    best_mAP      = max((h["val_mAP5095"] for h in history), default=0.0)
    baseline_time: float | None = None

    if args.resume and completed > 0:
        print(f"[resume] Loaded {completed} completed experiments, best mAP={best_mAP:.4f}")

    cmd_str = " ".join(build_cmd(NUM_GPUS))
    print(f"\n{'='*62}")
    print(f"  AutoResearch-DET  |  YOLO11 fine-tuning")
    print(f"  Branch      : {branch}")
    print(f"  LLM model   : {LLM_MODEL}")
    print(f"  GPUs        : {NUM_GPUS}  →  {cmd_str}")
    print(f"  Experiments : {remaining} remaining / {args.experiments} total")
    if args.dry_run:
        print(f"  Mode        : DRY-RUN (no training)")
    print(f"{'='*62}\n")

    for exp_num in range(completed + 1, args.experiments + 1):
        if _shutdown_requested:
            print("\n[!] Graceful shutdown — results saved.")
            break

        print(f"\n── Experiment {exp_num:02d}/{args.experiments:02d} " + "─" * 38)

        current_code = Path(_TRAIN_FILE).read_text(errors="ignore")

        # ── Propose ---------------------------------------------------------
        if exp_num == 1:
            description = "baseline — default hyperparameters"
            print(f"  Idea  : {description}")
        else:
            print("  LLM   : proposing next experiment…")
            try:
                proposal    = propose_experiment(history, current_code)
                description = proposal["description"]
                if not args.dry_run:
                    Path(_TRAIN_FILE).write_text(proposal["new_code"])
                print(f"  Idea  : {description}")
            except Exception as exc:
                print(f"  LLM error: {exc} — skipping")
                continue

        if args.dry_run:
            print("  [dry-run] skipping training")
            history.append({"description": description, "val_mAP5095": 0.0, "status": "dry-run"})
            continue

        # ── Commit ----------------------------------------------------------
        commit  = git_commit(f"experiment: {description}")
        timeout = baseline_time * _TIMEOUT_FACTOR if baseline_time else None
        print(f"  Commit: {commit}  |  timeout: "
              f"{f'{timeout:.0f}s' if timeout else 'none'}")

        # ── Train -----------------------------------------------------------
        print("  Training…")
        success, elapsed = run_training(NUM_GPUS, timeout)
        if exp_num == 1:
            baseline_time = elapsed

        metrics     = parse_metrics()
        val_mAP5095 = metrics["val_mAP5095"]

        if not success or val_mAP5095 == 0.0:
            print(f"  ✗  CRASH  ({elapsed:.0f}s)  — see {_LOG_FILE}")
            append_result(commit, metrics, "crash", description)
            history.append({"description": description, "val_mAP5095": 0.0, "status": "crash"})
            git_reset_last()
            continue

        prev = history[-1]["val_mAP5095"] if history else 0.0
        print(
            f"  mAP50-95={val_mAP5095:.4f}  "
            f"mAP50={metrics['val_mAP50']:.4f}  "
            f"VRAM={metrics['peak_vram_mb']/1024:.1f}GB  "
            f"time={elapsed:.0f}s"
        )

        # ── Keep / discard --------------------------------------------------
        if val_mAP5095 > best_mAP:
            best_mAP = val_mAP5095
            status   = "keep"
            print(f"  ✓  KEEP   (Δ={val_mAP5095 - prev:+.4f})")
        else:
            status = "discard"
            print(f"  ✗  DISCARD (Δ={val_mAP5095 - prev:+.4f})")
            if exp_num > 1:  # never reset baseline
                git_reset_last()

        append_result(commit, metrics, status, description)
        history.append({"description": description, "val_mAP5095": val_mAP5095, "status": status})
        update_plot()

    # ── Final summary --------------------------------------------------------
    if not history:
        print("No experiments completed.")
        return

    done          = [h for h in history if h["status"] not in ("crash", "dry-run")]
    best          = max(history, key=lambda h: h["val_mAP5095"])
    baseline_map  = history[0]["val_mAP5095"]
    delta         = best["val_mAP5095"] - baseline_map
    pct           = delta / baseline_map * 100 if baseline_map > 0 else 0.0

    print(f"\n{'='*62}")
    print(f"  AUTORESEARCH COMPLETE  ({len(history)} experiments, {len(done)} successful)")
    print(f"  GPUs used    : {NUM_GPUS}")
    print(f"  Baseline mAP : {baseline_map:.4f}")
    print(f"  Best mAP     : {best['val_mAP5095']:.4f}")
    print(f"  Improvement  : {delta:+.4f}  ({pct:+.1f}%)")
    print(f"  Best config  : {best['description']}")
    print(f"  Results      : {_RESULTS_FILE}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()