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
    python run_autoresearch.py --quiet           # training logs only in run.log (no live print)

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
import shlex
import signal
import subprocess
import sys
import threading
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
OPENROUTER_API_KEY: str = ""
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
    """-u = unbuffered stdout/stderr (needed for live logs under tqdm / notebooks)."""
    train = str(_TRAIN_FILE)
    py = sys.executable
    if num_gpus <= 1:
        return [py, "-u", train]
    return [
        py, "-u", "-m", "torch.distributed.run",
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


def _normalize_llm_text(text: str) -> str:
    t = text.strip()
    if t.startswith("\ufeff"):
        t = t[1:]
    for a, b in (
        ("\u201c", '"'),
        ("\u201d", '"'),
        ("\u2018", "'"),
        ("\u2019", "'"),
    ):
        t = t.replace(a, b)
    return t.encode("utf-8", errors="replace").decode("utf-8")


def _strip_outer_markdown_fence(text: str) -> str:
    """If the whole reply is one ```json / ```python block, return inner text."""
    t = text.strip()
    m = re.match(r"^```(?:json|python|py)?\s*\r?\n(.*)\r?\n```\s*$", t, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else t


def _looks_like_train_py(s: str) -> bool:
    if not s or len(s.strip()) < 80:
        return False
    head = s[:4000]
    return (
        "train.py" in head
        or "MODEL_SIZE" in head
        or ("def main" in s and "YOLO" in s)
        or "ultralytics" in head
        or ("model.train(" in s and "EPOCHS" in head)
    )


def _try_json_raw_decode(text: str) -> dict | None:
    """Parse first JSON object in text (ignores prose before/after)."""
    i = text.find("{")
    if i < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, i)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or "new_code" not in obj:
        return None
    nc = obj["new_code"]
    if nc is None or (isinstance(nc, str) and not nc.strip()):
        return None
    desc = obj.get("description") or "LLM proposal"
    return {"description": str(desc).strip(), "new_code": str(nc)}


def _try_json_loads_whole(text: str) -> dict | None:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict) or "new_code" not in obj:
        return None
    nc = obj["new_code"]
    if nc is None or (isinstance(nc, str) and not str(nc).strip()):
        return None
    desc = obj.get("description") or "LLM proposal"
    return {"description": str(desc).strip(), "new_code": str(nc)}


def _try_triple_quote_new_code(text: str) -> dict | None:
    tri = re.search(r'"new_code"\s*:\s*"""(.*?)"""', text, re.DOTALL)
    if not tri:
        return None
    new_code = tri.group(1).strip()
    desc_m = re.search(r'"description"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if not desc_m:
        desc_m = re.search(r'"description"\s*:\s*"([^"]*)"', text)
    if not desc_m:
        return {"description": "LLM proposal (triple-quoted new_code)", "new_code": new_code}
    desc = desc_m.group(1).replace("\\n", "\n").replace('\\"', '"')
    return {"description": desc, "new_code": new_code}


def _unescape_json_string(s: str) -> str:
    """Decode a JSON string body (without outer quotes). Fallback on failure."""
    try:
        return json.loads('"' + s + '"')
    except (json.JSONDecodeError, ValueError):
        return (
            s.replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace('\\"', '"')
            .replace("\\\\", "\\")
        )


def _try_regex_quoted_fields(text: str) -> dict | None:
    desc = re.search(r'"description"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    code = re.search(r'"new_code"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if not (desc and code):
        return None
    body = _unescape_json_string(code.group(1))
    d = _unescape_json_string(desc.group(1))
    if not body.strip():
        return None
    return {"description": d, "new_code": body}


def _try_fenced_json_inner(text: str) -> dict | None:
    """Parse first ```json ... ``` block as JSON (may contain extra keys)."""
    m = re.search(r"```(?:json)?\s*\r?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    inner = m.group(1).strip()
    return _try_json_raw_decode(inner) or _try_json_loads_whole(inner)


def _try_markdown_code_blocks(text: str) -> dict | None:
    blocks = re.findall(
        r"```(?:python|py)?\s*\r?\n(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    # Also ```python ... ``` without newline after fence
    if not blocks:
        blocks = re.findall(
            r"```(?:python|py)\s+(.*?)```",
            text,
            re.DOTALL | re.IGNORECASE,
        )
    best: str | None = None
    for b in blocks:
        b = b.strip()
        if _looks_like_train_py(b) and (best is None or len(b) > len(best)):
            best = b
    if best:
        m = re.search(
            r"(?:description|change|summary)\s*[:#]\s*(.+?)(?:\n|$)",
            text,
            re.IGNORECASE,
        )
        desc = m.group(1).strip() if m else "LLM proposal (markdown code block)"
        return {"description": desc[:200], "new_code": best}
    return None


def _try_plain_train_py(text: str) -> dict | None:
    t = text.strip()
    if t.startswith("{") or t.startswith("["):
        return None
    if _looks_like_train_py(t):
        line = t.splitlines()[0].strip("# ")[:120]
        desc = line if line else "LLM returned raw train.py"
        return {"description": desc, "new_code": t}
    return None


def _parse_llm_proposal(text: str) -> dict:
    """
    Extract {"description", "new_code"} from arbitrary LLM output.
    Tries strict JSON first, then common malformed patterns, then markdown / raw code.
    """
    raw = text
    text = _normalize_llm_text(text)
    text = _strip_outer_markdown_fence(text)

    for fn in (
        _try_json_raw_decode,
        _try_json_loads_whole,
        _try_fenced_json_inner,
        _try_triple_quote_new_code,
        _try_regex_quoted_fields,
    ):
        out = fn(text)
        if out:
            return out

    text2 = _normalize_llm_text(raw)
    out = _try_markdown_code_blocks(text2)
    if out:
        return out
    out = _try_markdown_code_blocks(text)
    if out:
        return out

    out = _try_plain_train_py(text2)
    if out:
        return out

    out = _try_plain_train_py(text)
    if out:
        return out

    raise ValueError(
        "Cannot parse LLM response (no JSON with new_code, no train.py-like code block).\n"
        f"First 500 chars:\n{text2[:500]}"
    )


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
                "Reply ONLY with a single JSON object: "
                '{"description": "one-line summary", "new_code": "<entire train.py as one JSON string>"}. '
                "new_code MUST be valid JSON: escape every newline inside the string as \\n "
                'and quotes as \\". Do NOT use Python triple quotes \"\"\" around new_code. '
                "No markdown fences, no text outside the JSON object."
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
    return _parse_llm_proposal(_call_api_with_retry(messages))


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

def _run_training_unix_tee(
    cmd: list[str],
    timeout: float | None,
    env: dict[str, str],
) -> tuple[bool, float]:
    """
    Linux/macOS/Kaggle: run through `tee` so output inherits the real TTY.
    `torch.distributed` worker logs often do not show up when the parent uses
    subprocess.PIPE (Jupyter then stays blank until the run ends).
    """
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    logf = shlex.quote(str(_LOG_FILE))
    print("  ── training log (live; also saved to run.log) ──")
    sys.stdout.flush()
    bash_cmd = f"set -o pipefail && {cmd_str} 2>&1 | tee {logf}"
    t0 = time.time()
    try:
        r = subprocess.run(
            bash_cmd,
            shell=True,
            cwd=str(_ROOT),
            env=env,
            executable="/bin/bash",
            timeout=timeout,
        )
        return r.returncode == 0, time.time() - t0
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"\n  [TIMEOUT] killed after {elapsed:.0f}s")
        return False, elapsed


def _run_training_pipe_tee(
    cmd: list[str],
    timeout: float | None,
    env: dict[str, str],
) -> tuple[bool, float]:
    """Windows fallback: copy pipe output to stdout and run.log."""
    t0 = time.time()
    timed_out = False
    proc: subprocess.Popen | None = None

    def _kill_if_stale() -> None:
        nonlocal timed_out, proc
        if timeout is None or timeout <= 0:
            return
        time.sleep(timeout)
        if proc is not None and proc.poll() is None:
            timed_out = True
            proc.kill()

    with open(_LOG_FILE, "w") as log_f:
        print("  ── training log (live; also saved to run.log) ──")
        sys.stdout.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=0,
            env=env,
        )
        threading.Thread(target=_kill_if_stale, daemon=True).start()
        assert proc.stdout is not None
        try:
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                sys.stdout.write(chunk)
                sys.stdout.flush()
                log_f.write(chunk)
                log_f.flush()
        except BrokenPipeError:
            pass
        rc = proc.wait()
        elapsed = time.time() - t0
        if timed_out and timeout is not None:
            print(f"\n  [TIMEOUT] killed after {timeout:.0f}s")
            return False, elapsed
        return rc == 0, elapsed


def run_training(
    num_gpus: int,
    timeout: float | None,
    *,
    quiet: bool = False,
) -> tuple[bool, float]:
    """
    Run train.py. By default streams stdout/stderr to the terminal and run.log.
    On Linux/macOS, uses `tee` so notebooks (e.g. Kaggle) show distributed logs live.
    Set quiet=True to only write run.log (no live output).
    """
    cmd = build_cmd(num_gpus)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    if quiet:
        t0 = time.time()
        try:
            with open(_LOG_FILE, "w") as log_out:
                proc = subprocess.run(
                    cmd,
                    cwd=str(_ROOT),
                    stdout=log_out,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    env=env,
                )
            return proc.returncode == 0, time.time() - t0
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"  [TIMEOUT] killed after {elapsed:.0f}s")
            return False, elapsed

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except (OSError, ValueError):
            pass

    if sys.platform != "win32" and os.path.isfile("/bin/bash"):
        return _run_training_unix_tee(cmd, timeout, env)
    return _run_training_pipe_tee(cmd, timeout, env)


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
    parser.add_argument("--quiet",        action="store_true",
                        help="Do not print training logs to the terminal (still write run.log)")
    args = parser.parse_args()

    # ── Environment ----------------------------------------------------------
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
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
    if args.quiet:
        print(f"  Training log: quiet (see {_LOG_FILE})")
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
        success, elapsed = run_training(NUM_GPUS, timeout, quiet=args.quiet)
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