"""
run_autoresearch.py -- Autonomous hyperparameter research loop.
==============================================================
AutoResearch-DET | Object Detection with YOLO11

The agent (LLM) proposes ONE hyperparameter change per iteration.
This runner executes the experiment, records the result, and keeps
or discards the change -- completely autonomously.

Usage
-----
    python run_autoresearch.py --experiments 10
    python run_autoresearch.py --experiments 10 --cuda-devices 0,1
    python run_autoresearch.py --resume          # continue from last run
    python run_autoresearch.py --dry-run         # LLM proposals only, no training
    python run_autoresearch.py --quiet           # training logs only in run.log (no live print)

Environment variables
---------------------
    OPENROUTER_API_KEY      API key for OpenRouter
    DATASET_DIR             Path to dataset root
    OUTPUT_DIR              Base output directory (default: output/train)
    AUTORESEARCH_PLOT_DIR   Where plot_progress writes progress.png (default: project root)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
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
_GIT_TRAIN_PATH = "train.py"
_TIMEOUT_FACTOR = 2.5
_LLM_RETRIES    = 3
_LLM_RETRY_WAIT = 5    # seconds between retries
_LLM_MAX_TOKENS = 6000

# Parameters that must never be changed by the LLM -- changing these would
# make experiment comparisons unfair (different training budget / hardware
# settings / numerical precision).
_FIXED_PARAMS = ("MODEL_SIZE", "PRETRAINED", "EPOCHS", "IMGSZ", "BATCH", "WORKERS", "AMP")

# Runtime globals
OPENROUTER_API_KEY: str = ""
LLM_MODEL:          str = _DEFAULT_MODEL
NUM_GPUS:           int = 2
_shutdown_requested: bool = False


# ── Graceful Ctrl-C ───────────────────────────────────────────────────────────

def _handle_sigint(sig, frame):
    global _shutdown_requested
    print("\n[!] Interrupt received -- finishing current experiment then stopping.")
    _shutdown_requested = True

signal.signal(signal.SIGINT, _handle_sigint)


# ── Stdout helpers ────────────────────────────────────────────────────────────

def _force_unbuffered_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(line_buffering=True)
            except (OSError, ValueError, io.UnsupportedOperation):
                pass


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


# ── LLM API ───────────────────────────────────────────────────────────────────

def _call_api(messages: list[dict], max_tokens: int = _LLM_MAX_TOKENS) -> str:
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
        body    = json.loads(r.read())
        content = body["choices"][0]["message"].get("content")
        if not content:
            # Surface finish_reason (e.g. "length", "content_filter") if present
            reason = body["choices"][0].get("finish_reason", "unknown")
            raise ValueError(f"API returned empty content (finish_reason={reason!r})")
        return content.strip()


def _call_api_with_retry(messages: list[dict], max_tokens: int = _LLM_MAX_TOKENS) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, _LLM_RETRIES + 1):
        try:
            return _call_api(messages, max_tokens)
        except (urllib.error.URLError, KeyError, json.JSONDecodeError,
                AttributeError, ValueError) as e:
            last_exc = e
            if attempt < _LLM_RETRIES:
                print(f"  [LLM] attempt {attempt} failed ({e}), retrying in {_LLM_RETRY_WAIT}s...")
                time.sleep(_LLM_RETRY_WAIT * attempt)
    raise RuntimeError(f"LLM API failed after {_LLM_RETRIES} attempts: {last_exc}")


# ── Unicode / encoding helpers ────────────────────────────────────────────────

_UNICODE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    # Quotation marks (all variants LLMs emit)
    ("\u2018", "'"),   # left single quotation mark
    ("\u2019", "'"),   # right single quotation mark
    ("\u201a", "'"),   # single low-9 quotation mark
    ("\u201b", "'"),   # single high-reversed-9 quotation mark
    ("\u201c", '"'),   # left double quotation mark
    ("\u201d", '"'),   # right double quotation mark
    ("\u201e", '"'),   # double low-9 quotation mark
    ("\u201f", '"'),   # double high-reversed-9 quotation mark
    ("\u2032", "'"),   # prime
    ("\u2033", '"'),   # double prime
    # Dashes
    ("\u2014", "-"),   # em dash
    ("\u2013", "-"),   # en dash
    ("\u2012", "-"),   # figure dash
    ("\u2015", "-"),   # horizontal bar
    # Ellipsis
    ("\u2026", "..."), # horizontal ellipsis
    # Spaces / invisible characters
    ("\u00a0", " "),   # non-breaking space
    ("\u200b", ""),    # zero-width space
    ("\u200c", ""),    # zero-width non-joiner
    ("\u200d", ""),    # zero-width joiner
    ("\u2060", ""),    # word joiner
    ("\ufeff", ""),    # BOM / zero-width no-break space
    # Misc
    ("\u00b4", "'"),   # acute accent used as apostrophe
)


def _normalize_llm_text(text: str) -> str:
    """Strip BOM, replace typographic Unicode, re-encode as clean UTF-8."""
    t = text.strip()
    for bad, good in _UNICODE_REPLACEMENTS:
        t = t.replace(bad, good)
    return t.encode("utf-8", errors="replace").decode("utf-8")


def _sanitize_train_py(src: str) -> str:
    """Full sanitization of LLM-generated train.py before writing to disk."""
    for bad, good in _UNICODE_REPLACEMENTS:
        src = src.replace(bad, good)
    # Windows -> Unix line endings
    src = src.replace("\r\n", "\n").replace("\r", "\n")
    # Trailing whitespace on each line (cosmetic, avoids diff noise)
    src = "\n".join(line.rstrip() for line in src.splitlines())
    # Ensure exactly one trailing newline
    return src.rstrip("\n") + "\n"


# ── LLM output parsing ────────────────────────────────────────────────────────

def _strip_outer_markdown_fence(text: str) -> str:
    """If the entire reply is wrapped in one code fence, unwrap it."""
    t = text.strip()
    m = re.match(r"^```\S*\s*\r?\n(.*)\r?\n```\s*$", t, re.DOTALL)
    return m.group(1).strip() if m else t


def _looks_like_train_py(s: str) -> bool:
    """Returns True if s looks like a YOLO11 train.py source file."""
    if not s or len(s.strip()) < 200:
        return False
    has_python = (
        "\ndef " in s or s.startswith("def ")
        or '"""' in s or "'''" in s
        or "import " in s
    )
    if not has_python:
        return False
    return (
        "MODEL_SIZE"      in s
        or "EPOCHS"       in s
        or "LR0"          in s
        or "BATCH"        in s
        or "ultralytics"  in s
        or "YOLO"         in s
        or "model.train(" in s
        or "train.py"     in s
        or ("def main"    in s and ("train" in s.lower() or "yolo" in s.lower()))
    )


def _unescape_json_string(s: str) -> str:
    try:
        return json.loads('"' + s + '"')
    except (json.JSONDecodeError, ValueError):
        return (
            s.replace("\\n", "\n")
             .replace("\\t", "\t")
             .replace('\\"', '"')
             .replace("\\\\", "\\")
        )


def _smart_decode_code(raw: str) -> str:
    if "\n" in raw:
        return (
            raw.replace("\\t", "\t")
               .replace('\\"', '"')
               .replace("\\\\", "\\")
        )
    return _unescape_json_string(raw)


# -- Stage 1 ------------------------------------------------------------------

def _try_json_raw_decode(text: str) -> dict | None:
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


# -- Stage 2 ------------------------------------------------------------------

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


# -- Stage 3 pre-processor ----------------------------------------------------

def _fix_json_literal_newlines(text: str) -> str:
    start_m = re.search(r'"new_code"\s*:\s*"', text)
    if not start_m:
        return text

    prefix = text[: start_m.end()]
    rest   = text[start_m.end():]

    fixed: list[str] = []
    i = 0
    closed = False

    while i < len(rest):
        c = rest[i]
        if c == "\\" and i + 1 < len(rest):
            fixed.append(rest[i : i + 2])
            i += 2
        elif c == '"':
            fixed.append('"')
            i += 1
            closed = True
            break
        elif c == "\r" and i + 1 < len(rest) and rest[i + 1] == "\n":
            fixed.append("\\n")
            i += 2
        elif c in ("\n", "\r"):
            fixed.append("\\n")
            i += 1
        else:
            fixed.append(c)
            i += 1

    if not closed:
        return text

    return prefix + "".join(fixed) + rest[i:]


# -- Stage 4 ------------------------------------------------------------------

def _try_fenced_json_inner(text: str) -> dict | None:
    m = re.search(r"```(?:json)?\s*\r?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    inner = m.group(1).strip()

    for fn in (_try_json_raw_decode, _try_json_loads_whole):
        out = fn(inner)
        if out:
            return out

    fixed = _fix_json_literal_newlines(inner)
    if fixed != inner:
        for fn in (_try_json_raw_decode, _try_json_loads_whole):
            out = fn(fixed)
            if out:
                return out

    return (
        _try_triple_quote_new_code(inner)
        or _try_regex_quoted_fields(inner)
        or _try_greedy_new_code_extract(inner)
    )


# -- Stage 5 ------------------------------------------------------------------

def _try_triple_quote_new_code(text: str) -> dict | None:
    start_m = re.search(r'"new_code"\s*:\s*"""', text, re.DOTALL)
    if not start_m:
        return None

    after_open = text[start_m.end():]
    triple_positions = [m.start() for m in re.finditer(r'"""', after_open)]
    if not triple_positions:
        return None

    new_code: str | None = None

    for pos in reversed(triple_positions):
        candidate = after_open[:pos]
        tail = after_open[pos + 3:].lstrip()
        if not tail or tail[0] in ("}", ",", "\n"):
            new_code = candidate
            break

    if new_code is None:
        new_code = after_open[: triple_positions[-1]]

    new_code = new_code.strip()
    if not new_code:
        return None

    desc_m = (
        re.search(r'"description"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        or re.search(r'"description"\s*:\s*"([^"]*)"', text)
    )
    desc = (
        _unescape_json_string(desc_m.group(1)) if desc_m
        else "LLM proposal (triple-quoted)"
    )
    return {"description": desc, "new_code": new_code}


# -- Stage 6 ------------------------------------------------------------------

def _try_regex_quoted_fields(text: str) -> dict | None:
    desc_m = re.search(r'"description"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    code_m = re.search(r'"new_code"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if not (desc_m and code_m):
        return None
    body = _unescape_json_string(code_m.group(1))
    d    = _unescape_json_string(desc_m.group(1))
    if not body.strip():
        return None
    return {"description": d, "new_code": body}


# -- Stage 7 ------------------------------------------------------------------

def _try_greedy_new_code_extract(text: str) -> dict | None:
    start_m = re.search(r'"new_code"\s*:\s*"', text)
    if not start_m:
        return None

    raw_from_start = text[start_m.end():]

    desc_m = re.search(r'"description"\s*:\s*"((?:[^"\\]|\\.){0,400})"', text)
    desc = (
        _unescape_json_string(desc_m.group(1)) if desc_m
        else "LLM proposal (greedy)"
    )

    end_patterns = [
        r'"\s*\n\s*\}\s*$',
        r'"\s*\}\s*$',
        r'"\s*\n\s*\}',
        r'"\s*\}',
        r'"\s*$',
    ]

    raw_code: str | None = None
    for pat in end_patterns:
        end_m = re.search(pat, raw_from_start)
        if end_m:
            raw_code = raw_from_start[: end_m.start()]
            break

    if raw_code is None:
        raw_code = raw_from_start.rstrip('"}\n\r ')

    if not raw_code:
        return None

    code = _smart_decode_code(raw_code)
    code = code.replace("\r\n", "\n").replace("\r", "\n")

    if not _looks_like_train_py(code):
        return None

    return {"description": desc, "new_code": code}


# -- Stage 8 ------------------------------------------------------------------

def _try_markdown_code_blocks(text: str) -> dict | None:
    blocks = re.findall(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```[^\n]*(.*?)```", text, re.DOTALL)

    best: str | None = None
    for b in blocks:
        b = b.strip()
        if _looks_like_train_py(b) and (best is None or len(b) > len(best)):
            best = b

    if not best:
        return None

    m = re.search(
        r"(?:description|change|summary)\s*[:#]\s*(.+?)(?:\n|$)",
        text, re.IGNORECASE,
    )
    desc = m.group(1).strip()[:200] if m else "LLM proposal (code block)"
    return {"description": desc, "new_code": best}


# -- Stage 9 ------------------------------------------------------------------

def _try_plain_train_py(text: str) -> dict | None:
    t = text.strip()
    if t.startswith("{") or t.startswith("["):
        return None
    if not _looks_like_train_py(t):
        return None
    for line in t.splitlines():
        line = line.strip().lstrip("#").strip().strip('"').strip("'").strip()
        if line and len(line) > 5:
            return {"description": line[:120], "new_code": t}
    return {"description": "LLM returned raw train.py", "new_code": t}


# ── Master parser dispatcher ──────────────────────────────────────────────────

def _parse_llm_proposal(raw_text: str) -> dict:
    """Extract {"description": str, "new_code": str} from ANY LLM response."""
    text = _normalize_llm_text(raw_text)
    text = _strip_outer_markdown_fence(text)

    for fn in (_try_json_raw_decode, _try_json_loads_whole):
        out = fn(text)
        if out:
            return out

    fixed_nl = _fix_json_literal_newlines(text)
    if fixed_nl != text:
        for fn in (_try_json_raw_decode, _try_json_loads_whole):
            out = fn(fixed_nl)
            if out:
                return out

    for fn in (
        _try_fenced_json_inner,
        _try_triple_quote_new_code,
        _try_regex_quoted_fields,
        _try_greedy_new_code_extract,
    ):
        out = fn(text)
        if out:
            return out

    raw_normalized = _normalize_llm_text(raw_text)
    for fn in (_try_markdown_code_blocks, _try_plain_train_py):
        out = fn(raw_normalized) or fn(text)
        if out:
            return out

    raise ValueError(
        "Cannot parse LLM response -- all 10 parser stages failed.\n"
        f"First 600 chars:\n{raw_normalized[:600]}"
    )


# ── Experiment proposal ───────────────────────────────────────────────────────

def propose_experiment(history: list[dict], current_code: str) -> dict:
    
    # ── 1. Phân tích history để tạo context thông minh hơn ──
    kept    = [h for h in history if h["status"] == "keep"]
    crashed = [h for h in history if h["status"] == "crash"]
    discarded = [h for h in history if h["status"] == "discard"]
    best_map = max((h["val_mAP5095"] for h in history), default=0.0)
    baseline = history[0]["val_mAP5095"] if history else 0.0
    
    # Tính average delta của 3 lần gần nhất → detect plateau
    recent = history[-3:]
    if len(recent) >= 2:
        deltas = [recent[i]["val_mAP5095"] - recent[i-1]["val_mAP5095"]
                  for i in range(1, len(recent))]
        avg_delta = sum(deltas) / len(deltas)
    else:
        avg_delta = 0.0

    # ── 2. History block có cấu trúc, bao gồm trend ──
    history_block = "\n".join(
        f"  exp{i+1:02d}: [{h['status'].upper():7s}] "
        f"mAP={h['val_mAP5095']:.4f}  "
        f"delta={h['val_mAP5095'] - (history[i-1]['val_mAP5095'] if i > 0 else baseline):+.4f}  "
        f"| {h['description']}"
        for i, h in enumerate(history)
    ) or "  (none yet — this is the baseline run)"

    trend_note = (
        f"Recent avg Δ={avg_delta:+.4f}. "
        + ("PLATEAU detected — consider a bigger/bolder change." if avg_delta < 0.001 else
           "Positive trend — fine-tune around what's working.")
    )

    params_tried = [h["description"] for h in history[1:]]  # skip baseline
    params_tried_str = "; ".join(params_tried) if params_tried else "none yet"

    # ── 3. Extract dataset info từ current_code nếu có ──
    import re
    def _extract(pattern, src, default="unknown"):
        m = re.search(pattern, src, re.MULTILINE)
        return m.group(1).strip() if m else default

    num_classes  = _extract(r"^NC\s*=\s*(\d+)",       current_code)
    dataset_size = _extract(r"^DATASET_SIZE\s*=\s*(.+)", current_code)

    code_chars = len(current_code)
    fixed_str  = ", ".join(_FIXED_PARAMS)

    # ── 4. System prompt với domain knowledge mạnh hơn ──
    system_prompt = (
        "You are a expert computer vision researcher specialising in YOLO fine-tuning "
        "Your goal is to maximise val/mAP50-95 "
        f"on a dataset with {num_classes} classes (~{dataset_size} images) "
        f"using {NUM_GPUS} GPU(s).\n\n"

        # --- Chain-of-thought yêu cầu ---
        "REASONING PROTOCOL:\n"
        "Before choosing a hyperparameter, think step by step:\n"
        "  1. What does the current best mAP tell you about underfitting vs overfitting?\n"
        "  2. What have the kept/discarded experiments revealed so far?\n"
        "  3. Is the search plateauing? (use the trend note provided)\n"
        "  4. Which single change has the highest expected impact given the above?\n"
        "Encode your reasoning in the 'description' field (max 200 chars), "
        "formatted as: 'REASON: <why> | CHANGE: <what> | EXPECTED: <effect>'\n\n"

        # --- Format output ---
        "OUTPUT FORMAT — reply with ONLY this JSON object, nothing else:\n"
        '  {"description": "REASON: ... | CHANGE: ... | EXPECTED: ...", '
        '"new_code": "<complete train.py>"}\n\n'

        "CRITICAL JSON ENCODING RULES for new_code:\n"
        "  1. Every newline -> \\n\n"
        "  2. Every double-quote -> \\\"\n"
        "  3. Every backslash -> \\\\\n"
        "  4. Must be valid json.loads()\n"
        "  5. Do NOT use triple-quotes or markdown fences\n\n"

        # --- Domain knowledge về từng param ---
        "HYPERPARAMETER PRIORITY TABLE (for small datasets):\n"
        "┌─────────────────────┬──────────┬──────┬─────────────────────────────────┐\n"
        "│ Parameter           │ Impact   │ Risk │ When to use                     │\n"
        "├─────────────────────┼──────────┼──────┼─────────────────────────────────┤\n"
        "│ LR0                 │ HIGH     │ MED  │ First to tune; try 1e-3..1e-4   │\n"
        "│ LR0+LRF (group)     │ HIGH     │ MED  │ When LR0 alone doesn't converge │\n"
        "│ MOSAIC              │ HIGH     │ LOW  │ Overfitting? Disable (0.0)       │\n"
        "│ CLOSE_MOSAIC        │ MED      │ LOW  │ Tune when mosaic hurts final ep  │\n"
        "│ WEIGHT_DECAY        │ MED      │ LOW  │ High when overfitting            │\n"
        "│ OPTIMIZER           │ MED      │ LOW  │ SGD vs AdamW on small data       │\n"
        "│ WARMUP_EPOCHS       │ MED      │ LOW  │ Increase if loss spikes early    │\n"
        "│ DROPOUT             │ LOW-MED  │ LOW  │ Only if clear overfitting        │\n"
        "│ LABEL_SMOOTHING     │ LOW      │ LOW  │ Class imbalance present          │\n"
        "│ HSV_H/S/V (group)   │ LOW-MED  │ LOW  │ When images vary in lighting     │\n"
        "│ MIXUP/COPY_PASTE    │ MED      │ LOW  │ After mosaic is tuned            │\n"
        "│ PATIENCE            │ NONE     │ NONE │ Don't tune — affects budget      │\n"
        "└─────────────────────┴──────────┴──────┴─────────────────────────────────┘\n\n"

        "CONTENT RULES (STRICT):\n"
        f"  - Do NOT change: {fixed_str}\n"
        "  - Change EXACTLY ONE param OR one allowed group (HSV_H/S/V; "
        "DEGREES/TRANSLATE/SCALE/SHEAR; MOSAIC/MIXUP/COPY_PASTE; "
        "LR0/LRF; WARMUP_EPOCHS/WARMUP_BIAS_LR; DROPOUT/LABEL_SMOOTHING)\n"
        f"  - new_code must be {code_chars}±200 characters\n"
        "  - ASCII only (no Unicode)\n"
        "  - File must start with a triple-quoted docstring\n"
    )

    # ── 5. User prompt với context đầy đủ ──
    user_prompt = (
        f"=== EXPERIMENT HISTORY ({len(history)} runs) ===\n"
        f"{history_block}\n\n"
        f"=== TREND ANALYSIS ===\n"
        f"  Baseline mAP  : {baseline:.4f}\n"
        f"  Best mAP so far: {best_map:.4f}  (+{best_map-baseline:+.4f})\n"
        f"  Kept / Discarded / Crashed: {len(kept)} / {len(discarded)} / {len(crashed)}\n"
        f"  {trend_note}\n"
        f"  Already tried  : {params_tried_str}\n\n"
        f"=== CURRENT train.py ({code_chars} chars) ===\n"
        f"{current_code}\n\n"
        "=== YOUR TASK ===\n"
        "Step 1 — Diagnose: Is the model underfitting or overfitting? "
        "What does the kept/discarded pattern suggest?\n"
        "Step 2 — Select: Pick ONE change (or one group) with highest expected mAP50-95 gain. "
        "Do NOT repeat a change already attempted.\n"
        "Step 3 — Output: Return the complete modified train.py as new_code.\n"
        "Your description MUST follow: 'REASON: <why> | CHANGE: <what> | EXPECTED: <effect>'"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    return _parse_llm_proposal(_call_api_with_retry(messages))


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_train_py_source(src: str) -> None:
    """Reject LLM output that is not syntactically valid Python."""
    s = src.lstrip("\ufeff \t\r\n")
    if not (s.startswith('"""') or s.startswith("'''")):
        raise ValueError(
            "train.py must start with a triple-quoted docstring on line 1.\n"
            f"Got first 80 chars: {s[:80]!r}"
        )
    try:
        compile(src, str(_TRAIN_FILE), "exec")
    except SyntaxError as e:
        lines = src.splitlines()
        lo  = max(0, (e.lineno or 1) - 3)
        hi  = min(len(lines), (e.lineno or 1) + 2)
        ctx = "\n".join(
            f"  {'>>>' if idx + 1 == e.lineno else '   '} {idx+1:4d}  {lines[idx]}"
            for idx in range(lo, hi)
        )
        raise ValueError(
            f"SyntaxError on line {e.lineno}: {e.msg}\n{ctx}"
        ) from e


def _validate_fixed_params(new_src: str, original_src: str) -> None:
    """
    Ensure the LLM did not touch any of the fixed parameters defined in
    _FIXED_PARAMS.  These must stay identical across all experiments so that
    results are directly comparable (same training budget, resolution, hardware
    settings, and numerical precision).
    """
    for param in _FIXED_PARAMS:
        orig_m = re.search(rf"^{param}\s*=\s*(.+)$", original_src, re.MULTILINE)
        new_m  = re.search(rf"^{param}\s*=\s*(.+)$", new_src,      re.MULTILINE)
        if orig_m and new_m and orig_m.group(1).strip() != new_m.group(1).strip():
            raise ValueError(
                f"LLM changed fixed param {param}: "
                f"{orig_m.group(1).strip()!r} -> {new_m.group(1).strip()!r}"
            )



def _validate_single_change(new_src: str, original_src: str) -> None:
    """Buộc agent chỉ được thay ĐÚNG 1 hyperparameter 
    HOẶC 1 nhóm liên quan chặt chẽ (định nghĩa rõ ràng).
    Mục tiêu: attribution rõ ràng, experiment dễ trace."""
    
    param_re = re.compile(r'^([A-Z0-9_]+)\s*=\s*(.+?)(?:\s*#.*)?$', re.MULTILINE)

    orig_params = dict(param_re.findall(original_src))
    new_params  = dict(param_re.findall(new_src))

    changed: list[str] = []
    for k in set(orig_params) & set(new_params):
        if orig_params[k].strip() != new_params[k].strip():
            changed.append(k)

    if len(changed) <= 1:
        return  # chỉ thay 1 tham số → OK

    # ================== NHÓM LIÊN QUAN CHẶT CHẼ ĐƯỢC PHÉP ==================
    allowed_groups = [
        {"HSV_H", "HSV_S", "HSV_V"},                    # Color augmentation
        {"DEGREES", "TRANSLATE", "SCALE", "SHEAR"},     # Geometry augmentation
        {"MOSAIC", "MIXUP", "COPY_PASTE"},              # Advanced mosaic & mix
        {"LR0", "LRF"},                                 # Learning rate schedule
        {"WARMUP_EPOCHS", "WARMUP_BIAS_LR"},            # Warmup strategy
        {"DROPOUT", "LABEL_SMOOTHING"},                 # Regularization
    ]

    changed_set = set(changed)
    for group in allowed_groups:
        if changed_set <= group:        # toàn bộ changed nằm trong 1 group
            return  # hợp lệ

    # Nếu không thuộc bất kỳ group nào → reject
    raise ValueError(
        f"LLM changed {len(changed)} parameters: {changed}.\n"
        "Quy tắc: Chỉ được thay ĐÚNG 1 hyperparameter "
        "hoặc 1 nhóm liên quan chặt chẽ (ví dụ: cả 3 HSV_*, hoặc LR0+LRF).\n"
        "Không được thay nhiều tham số không liên quan trong cùng 1 experiment."
    )

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
        pass
    return subprocess.check_output(
        ["git", "-C", str(_ROOT), "rev-parse", "--short", "HEAD"], text=True
    ).strip()


def git_reset_last() -> None:
    """Discard the last commit (experiment that did not improve)."""
    subprocess.run(
        ["git", "-C", str(_ROOT), "reset", "--hard", "HEAD~1"],
        capture_output=True,
    )


# ── Progress-line detection ───────────────────────────────────────────────────

# Matches a YOLO/rich per-batch progress line, e.g.:
#   "  1/1  3.8G  1.948 ...  78% ━━━━━─── 316/405 4.7it/s 1:24<19.3s"
_PROGRESS_RE = re.compile(
    r"(?:[\u2501\u2578\u2579\u257a\u257b\u254b\u2503\u2500\u2580\u2584\u2588]"
    r"|\b\d+/\d+\s+\d+\.\d+it/s"
    r"|\d+%\s*[|\u2500-\u259f])",
)

# Width used to pad/clear the overwritten progress line
_TERM_WIDTH = 120


def _is_progress_line(line: str) -> bool:
    """Return True for YOLO per-batch lines that should overwrite in place."""
    stripped = line.strip()
    return bool(stripped and _PROGRESS_RE.search(stripped))


# ── Training & metrics ────────────────────────────────────────────────────────

def _stream_process(
    proc: subprocess.Popen,
    log_f,
    timeout: float | None,
) -> tuple[bool, float]:
    """
    Read process stdout LINE BY LINE.

    Batch-level progress lines (the ones YOLO emits hundreds of times per
    epoch) are collapsed into a single terminal line overwritten in place
    with \\r.  All other lines (epoch summaries, val metrics, etc.) are
    printed normally so nothing important is lost.

    Every line -- including progress noise -- is still written to run.log.
    """
    t0 = time.time()
    timed_out    = False
    on_prog_line = False   # True when the last terminal write used \\r

    def _watchdog():
        nonlocal timed_out
        if not timeout:
            return
        time.sleep(timeout)
        if proc.poll() is None:
            timed_out = True
            proc.kill()

    if timeout:
        threading.Thread(target=_watchdog, daemon=True).start()

    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            # Always write full line to log file
            log_f.write(line)
            log_f.flush()

            if _is_progress_line(line):
                # Overwrite the same terminal line in place
                display = line.rstrip("\n\r")[:_TERM_WIDTH].ljust(_TERM_WIDTH)
                sys.stdout.write("\r" + display)
                sys.stdout.flush()
                on_prog_line = True
            else:
                # Normal line -- move to a fresh line first if needed
                if on_prog_line:
                    sys.stdout.write("\n")
                    on_prog_line = False
                sys.stdout.write(line)
                sys.stdout.flush()
    except BrokenPipeError:
        pass

    # Ensure the cursor lands on a fresh line after the loop ends
    if on_prog_line:
        sys.stdout.write("\n")
        sys.stdout.flush()

    rc      = proc.wait()
    elapsed = time.time() - t0

    if timed_out:
        print(f"\n  [TIMEOUT] killed after {timeout:.0f}s")
        return False, elapsed

    return rc == 0, elapsed


def _launch_training(
    cmd: list[str],
    timeout: float | None,
    env: dict[str, str],
) -> tuple[bool, float]:
    print("  -- training log (live; also saved to run.log) --")
    sys.stdout.flush()

    with open(_LOG_FILE, "w", buffering=1) as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,    # line-buffered on OS side
            env=env,
        )
        return _stream_process(proc, log_f, timeout)


def run_training(
    num_gpus: int,
    timeout: float | None,
    *,
    quiet: bool = False,
) -> tuple[bool, float]:
    cmd = build_cmd(num_gpus)
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
    }

    if quiet:
        t0 = time.time()
        try:
            with open(_LOG_FILE, "w") as log_out:
                proc = subprocess.run(
                    cmd, cwd=str(_ROOT),
                    stdout=log_out, stderr=subprocess.STDOUT,
                    timeout=timeout, env=env,
                )
            return proc.returncode == 0, time.time() - t0
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"  [TIMEOUT] killed after {elapsed:.0f}s")
            return False, elapsed

    _force_unbuffered_stdout()
    return _launch_training(cmd, timeout, env)


def _resolve_train_output_dir() -> Path:
    """Same root as training subprocess (cwd=_ROOT); fixes Kaggle when autoresearch cwd != repo."""
    raw = os.environ.get("OUTPUT_DIR", "output/train")
    p = Path(raw)
    if not p.is_absolute():
        p = _ROOT / p
    return p.resolve()


def _parse_metrics_from_run_log() -> tuple[dict[str, float], bool]:
    """Fallback: lines printed by train.py (--- / val_mAP5095: / ...)."""
    metrics: dict[str, float] = {
        "val_mAP5095" : 0.0,
        "val_mAP50"   : 0.0,
        "peak_vram_mb": 0.0,
    }
    try:
        text = Path(_LOG_FILE).read_text(errors="ignore")
    except FileNotFoundError:
        return metrics, False
    for line in text.splitlines():
        for key in metrics:
            m = re.match(rf"^{re.escape(key)}:\s+([\d.]+)", line.strip())
            if m:
                metrics[key] = float(m.group(1))
    # Training finished and printed the summary block (mAP may legitimately be 0)
    ok = re.search(r"^val_mAP5095:\s+[\d.]+", text, re.MULTILINE) is not None
    return metrics, ok


def parse_metrics() -> tuple[dict[str, float], bool]:
    """
    Read train.py metrics: summary.json under OUTPUT_DIR (anchored to _ROOT), then run.log.
    Second value is True if metrics were read successfully (including mAP legitimately 0).
    """
    metrics: dict[str, float] = {
        "val_mAP5095" : 0.0,
        "val_mAP50"   : 0.0,
        "peak_vram_mb": 0.0,
    }

    base = _resolve_train_output_dir()
    # Ultralytics may create exp, exp2, ... — prefer newest summary.json
    candidates = sorted(
        base.glob("exp*/summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for summary_path in candidates:
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics["val_mAP5095"] = float(data.get("val_mAP5095", 0.0))
            metrics["val_mAP50"] = float(data.get("val_mAP50", 0.0))
            metrics["peak_vram_mb"] = float(data.get("peak_vram_mb", 0.0))
            return metrics, True
        except Exception as e:
            print(f"  [parse_metrics] warning: cannot read {summary_path} ({e})")

    log_metrics, log_ok = _parse_metrics_from_run_log()
    if log_ok:
        print(
            "  [parse_metrics] note: using metrics from run.log "
            f"(no usable summary.json under {base})"
        )
        return log_metrics, True

    print(
        f"  [parse_metrics] warning: no summary.json under {base} "
        f"(exp*/summary.json) and no metrics in {_LOG_FILE}"
    )
    return metrics, False


# ── Results file ──────────────────────────────────────────────────────────────

_TSV_HEADER = "commit\tval_mAP5095\tval_mAP50\tmemory_gb\tstatus\tdescription\n"


def _ensure_results_file() -> None:
    if not Path(_RESULTS_FILE).exists():
        Path(_RESULTS_FILE).write_text(_TSV_HEADER)


def load_history() -> list[dict]:
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

def _plot_output_dir() -> Path:
    """Directory for progress.png / progress_detail.png (see plot_progress.py -o)."""
    p = Path(os.environ.get("AUTORESEARCH_PLOT_DIR", str(_ROOT))).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def update_plot() -> None:
    """Regenerate charts via plot_progress.py after each experiment (same CLI as manual run)."""
    script = _ROOT / "plot_progress.py"
    if not script.is_file():
        return
    out_dir = _plot_output_dir()
    r = subprocess.run(
        [
            sys.executable,
            str(script),
            "-i",
            str(_RESULTS_FILE),
            "-o",
            str(out_dir),
        ],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(
            f"  [plot] warning: plot_progress.py exited {r.returncode} "
            f"(see stderr below)"
        )
        err = (r.stderr or "").strip()
        if err:
            print(err[:800])


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
                        help="Do not print training logs to terminal (still writes run.log)")
    args = parser.parse_args()

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
        sys.exit(f"ERROR: {_TRAIN_FILE} not found.")

    _ensure_results_file()

    history      = load_history() if args.resume else []
    completed    = len(history)
    best_mAP     = max((h["val_mAP5095"] for h in history), default=0.0)
    baseline_time: float | None = None

    if args.resume and completed > 0:
        print(f"[resume] Loaded {completed} experiments, best mAP={best_mAP:.4f}")

    cmd_str = " ".join(build_cmd(NUM_GPUS))
    print(f"\n{'='*62}")
    print(f"  AutoResearch-DET  |  YOLO11 fine-tuning")
    print(f"  Branch      : {branch}")
    print(f"  LLM model   : {LLM_MODEL}")
    print(f"  GPUs        : {NUM_GPUS}  ->  {cmd_str}")
    print(f"  Experiments : {args.experiments - completed} remaining / {args.experiments} total")
    print(f"  Fixed params: {', '.join(_FIXED_PARAMS)}")
    if args.dry_run:
        print(f"  Mode        : DRY-RUN (no training)")
    if args.quiet:
        print(f"  Training log: quiet (see {_LOG_FILE})")
    print(f"{'='*62}\n")

    for exp_num in range(completed + 1, args.experiments + 1):
        if _shutdown_requested:
            print("\n[!] Graceful shutdown -- results saved.")
            break

        print(f"\n{'='*62}")
        print(f"  Experiment {exp_num:02d} / {args.experiments:02d}")
        print(f"{'='*62}")

        current_code = Path(_TRAIN_FILE).read_text(errors="ignore")

        if exp_num == 1:
            description = "baseline -- default hyperparameters"
            print(f"  Idea  : {description}")
        else:
            print("  LLM   : proposing next experiment...")
            try:
                proposal    = propose_experiment(history, current_code)
                description = proposal["description"]
                if not args.dry_run:
                    clean_code = _sanitize_train_py(proposal["new_code"])
                    _validate_train_py_source(clean_code)
                    _validate_fixed_params(clean_code, current_code)
                    _validate_fixed_params(clean_code, current_code)
                    Path(_TRAIN_FILE).write_text(clean_code, encoding="utf-8")
                print(f"  Idea  : {description}")
            except Exception as exc:
                print(f"  LLM error: {exc} -- skipping")
                continue

        if args.dry_run:
            print("  [dry-run] skipping training")
            history.append({"description": description, "val_mAP5095": 0.0, "status": "dry-run"})
            continue

        commit  = git_commit(f"experiment: {description}")
        timeout = baseline_time * _TIMEOUT_FACTOR if baseline_time else None
        print(f"  Commit: {commit}  |  timeout: "
              f"{f'{timeout:.0f}s' if timeout else 'none'}")

        print("  Training...\n")
        sys.stdout.flush()
        success, elapsed = run_training(NUM_GPUS, timeout, quiet=args.quiet)
        if exp_num == 1:
            baseline_time = elapsed

        metrics, metrics_ok = parse_metrics()
        val_mAP5095 = metrics["val_mAP5095"]

        print()
        if not metrics_ok:
            print(f"  X  CRASH  ({elapsed:.0f}s)  -- see {_LOG_FILE}")
            append_result(commit, metrics, "crash", description)
            history.append({"description": description, "val_mAP5095": 0.0, "status": "crash"})
            git_reset_last()
            continue

        if not success:
            print(
                "  [!] Training subprocess exited non-zero but metrics were recovered "
                "(often NCCL / DDP teardown). Continuing with parsed metrics."
            )

        prev = history[-1]["val_mAP5095"] if history else 0.0
        print(
            f"  mAP50-95={val_mAP5095:.4f}  "
            f"mAP50={metrics['val_mAP50']:.4f}  "
            f"VRAM={metrics['peak_vram_mb']/1024:.1f}GB  "
            f"time={elapsed:.0f}s"
        )

        if val_mAP5095 > best_mAP:
            best_mAP = val_mAP5095
            status   = "keep"
            print(f"  v  KEEP   (delta={val_mAP5095 - prev:+.4f})")
        else:
            status = "discard"
            print(f"  X  DISCARD (delta={val_mAP5095 - prev:+.4f})")
            if exp_num > 1:
                git_reset_last()

        append_result(commit, metrics, status, description)
        history.append({"description": description, "val_mAP5095": val_mAP5095, "status": status})
        update_plot()

    if not history:
        print("No experiments completed.")
        return

    done         = [h for h in history if h["status"] not in ("crash", "dry-run")]
    best         = max(history, key=lambda h: h["val_mAP5095"])
    baseline_map = history[0]["val_mAP5095"]
    delta        = best["val_mAP5095"] - baseline_map
    pct          = delta / baseline_map * 100 if baseline_map > 0 else 0.0

    print(f"\n{'='*62}")
    print(f"  AUTORESEARCH COMPLETE  ({len(history)} experiments, {len(done)} successful)")
    print(f"  GPUs used    : {NUM_GPUS}")
    print(f"  Baseline mAP : {baseline_map:.4f}")
    print(f"  Best mAP     : {best['val_mAP5095']:.4f}")
    print(f"  Improvement  : {delta:+.4f}  ({pct:+.1f}%)")
    print(f"  Best config  : {best['description']}")
    print(f"  Results      : {_RESULTS_FILE}")
    plot_png = _plot_output_dir() / "progress.png"
    if plot_png.is_file():
        print(f"  Progress plot: {plot_png}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()