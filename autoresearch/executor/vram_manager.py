"""
vram_manager.py — GPU VRAM leak prevention between experiments.

Problem: YOLO spawns DDP sub-processes that may not cleanly release the CUDA
context.  Over many experiments this causes cumulative VRAM growth → OOM.

Strategy:
  1. After each training run, sleep briefly to let the OS reclaim the context.
  2. Before the next run, check free VRAM; if below threshold → proactive clear.
  3. Inject PYTORCH_CUDA_ALLOC_CONF into the training sub-process env to use
     the expandable allocator and reduce fragmentation.
"""

from __future__ import annotations

import subprocess
import time
from config import Config


class VRAMManager:

    # ── Post-experiment cooldown ──────────────────────────────────────────

    @staticmethod
    def post_run_cooldown():
        """Brief sleep after process termination to allow full CUDA context release."""
        time.sleep(Config.VRAM_COOLDOWN_SECS)

    # ── Subprocess environment ────────────────────────────────────────────

    @staticmethod
    def training_env(base_env: dict) -> dict:
        """
        Return env dict for the training sub-process with VRAM-friendly settings.
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments reduces fragmentation across
        many short-lived experiments sharing the same physical device.
        """
        return {**base_env, **Config.VRAM_CLEANUP_ENV}

    # ── VRAM introspection (optional, requires nvidia-smi) ────────────────

    @staticmethod
    def free_vram_mb() -> float | None:
        """Return free VRAM in MiB for GPU 0, or None if nvidia-smi is unavailable."""
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL, timeout=5,
            )
            first = out.strip().splitlines()[0]
            return float(first.strip())
        except Exception:
            return None

    @staticmethod
    def total_vram_mb() -> float | None:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL, timeout=5,
            )
            first = out.strip().splitlines()[0]
            return float(first.strip())
        except Exception:
            return None

    @staticmethod
    def log_vram_state(label: str = "") -> str:
        free  = VRAMManager.free_vram_mb()
        total = VRAMManager.total_vram_mb()
        if free is None or total is None:
            return ""
        used = total - free
        msg  = f"  [VRAM] {label}used={used:.0f} MiB / {total:.0f} MiB  (free={free:.0f} MiB)"
        print(msg)
        return msg

    # ── Wait for VRAM to be released ─────────────────────────────────────

    @staticmethod
    def wait_for_release(expected_free_floor_mb: float = 0, poll_secs: float = 1.0, timeout: float = 15.0):
        """
        Poll until free VRAM rises above `expected_free_floor_mb`.
        Used after killing a training process to verify the context is gone.
        """
        if expected_free_floor_mb <= 0:
            time.sleep(Config.VRAM_COOLDOWN_SECS)
            return
        deadline = time.time() + timeout
        while time.time() < deadline:
            free = VRAMManager.free_vram_mb()
            if free is None or free >= expected_free_floor_mb:
                break
            time.sleep(poll_secs)