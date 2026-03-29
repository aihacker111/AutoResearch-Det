"""
plot_progress.py — Visualise experiment progress.
==================================================
AutoResearch-DET | Run anytime: python plot_progress.py

Outputs
-------
    progress.png        — bar chart per experiment, coloured by status
    progress_detail.png — step chart showing only improvements
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

RESULTS_FILE = "results.tsv"
STATUS_COLOR = {
    "keep"   : "#2ecc71",
    "discard": "#e74c3c",
    "crash"  : "#95a5a6",
}


def load_results(path: str = RESULTS_FILE) -> list[dict]:
    rows = []
    with open(path) as f:
        for i, row in enumerate(csv.DictReader(f, delimiter="\t")):
            rows.append({
                "idx"        : i + 1,
                "commit"     : row["commit"],
                "val_mAP5095": float(row["val_mAP5095"]),
                "val_mAP50"  : float(row["val_mAP50"]),
                "memory_gb"  : float(row["memory_gb"]),
                "status"     : row["status"].strip(),
                "description": row["description"].strip(),
            })
    return rows


def plot_overview(rows: list[dict], out: str = "progress.png") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), facecolor="#1a1a2e")
    fig.suptitle(
        "AutoResearch-DET  ·  YOLO11 Fine-tuning Progress",
        color="white", fontsize=14, fontweight="bold", y=0.98,
    )

    idxs    = [r["idx"]         for r in rows]
    maps    = [r["val_mAP5095"] for r in rows]
    maps50  = [r["val_mAP50"]   for r in rows]
    mems    = [r["memory_gb"]   for r in rows]
    colors  = [STATUS_COLOR.get(r["status"], "#aaa") for r in rows]

    # ── Top: mAP per experiment ──────────────────────────────────────────────
    ax1.set_facecolor("#16213e")
    ax1.set_title("val/mAP50-95 per experiment", color="white", fontsize=11, pad=6)
    ax1.bar(idxs, maps, color=colors, alpha=0.85, zorder=3)
    ax1.plot(idxs, maps,   color="#f0e6ff", lw=1.5, marker="o", ms=4,
             zorder=4, label="mAP50-95")
    ax1.plot(idxs, maps50, color="#74b9ff", lw=1.2, ls="--", marker="s",
             ms=3, zorder=4, alpha=0.7, label="mAP50")
    ax1.axhline(maps[0], color="#fdcb6e", lw=1.2, ls=":",
                label=f"baseline {maps[0]:.4f}")

    best_i = int(np.argmax(maps))
    ax1.annotate(
        f"best\n{maps[best_i]:.4f}",
        xy=(idxs[best_i], maps[best_i]),
        xytext=(idxs[best_i] + 0.3, maps[best_i] + 0.005),
        color="#2ecc71", fontsize=8, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=1.2),
    )

    ax1.set_ylabel("mAP50-95", color="white")
    ax1.set_xticks(idxs)
    ax1.set_xticklabels([r["commit"][:7] for r in rows],
                        rotation=30, ha="right", fontsize=7, color="#aaccee")
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#444466")
    ax1.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", loc="upper left")

    # ── Bottom: VRAM ─────────────────────────────────────────────────────────
    ax2.set_facecolor("#16213e")
    ax2.set_title("Peak VRAM (GB) per experiment", color="white", fontsize=11, pad=6)
    ax2.bar(idxs, mems, color=colors, alpha=0.75, zorder=3)
    ax2.set_ylabel("GB", color="white")
    ax2.set_xticks(idxs)
    ax2.set_xticklabels([r["commit"][:7] for r in rows],
                        rotation=30, ha="right", fontsize=7, color="#aaccee")
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#444466")

    legend = [mpatches.Patch(color=v, label=k) for k, v in STATUS_COLOR.items()]
    ax2.legend(handles=legend, fontsize=8,
               facecolor="#1a1a2e", labelcolor="white", loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] {out}")


def plot_detail(rows: list[dict], out: str = "progress_detail.png") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keeps = [r for r in rows if r["status"] == "keep"]
    if len(keeps) < 2:
        print("[plot] Not enough 'keep' rows for detail chart yet.")
        return

    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.set_title(
        "Cumulative best mAP50-95 — what each improvement changed",
        color="white", fontsize=12, fontweight="bold", pad=10,
    )

    ki   = [r["idx"]         for r in keeps]
    km   = [r["val_mAP5095"] for r in keeps]

    ax.step(ki, km, where="post", color="#a29bfe", lw=2.5, zorder=3)
    ax.fill_between(ki, km, step="post", alpha=0.12, color="#a29bfe")
    ax.scatter(ki, km, color="#fdcb6e", s=60, zorder=5)

    for i, r in enumerate(keeps):
        delta = r["val_mAP5095"] - keeps[i - 1]["val_mAP5095"] if i > 0 else 0
        label = r["description"][:42] + "…" if len(r["description"]) > 42 else r["description"]
        ax.annotate(
            f"{delta:+.4f}\n{label}",
            xy=(r["idx"], r["val_mAP5095"]),
            xytext=(r["idx"] + 0.15,
                    r["val_mAP5095"] + 0.002 * (1 + i % 3)),
            color="#2ecc71" if delta >= 0 else "#e74c3c",
            fontsize=7.5,
            arrowprops=dict(arrowstyle="-", color="#666688", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.2",
                      fc="#16213e", ec="#444466", alpha=0.8),
        )

    ax.set_xlabel("Experiment #", color="white")
    ax.set_ylabel("mAP50-95",     color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444466")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] {out}")


def main() -> None:
    if not Path(RESULTS_FILE).exists():
        sys.exit(f"ERROR: {RESULTS_FILE} not found — run at least 1 experiment first.")

    rows = load_results()
    if not rows:
        sys.exit("No results yet.")

    print(f"[plot] Loaded {len(rows)} experiments from {RESULTS_FILE}")
    plot_overview(rows)
    plot_detail(rows)

    best     = max(rows, key=lambda r: r["val_mAP5095"])
    baseline = rows[0]["val_mAP5095"]
    delta    = best["val_mAP5095"] - baseline
    pct      = delta / baseline * 100 if baseline > 0 else 0

    print(f"\n{'─'*50}")
    print(f"  Experiments : {len(rows)}")
    print(f"  Baseline    : {baseline:.4f}")
    print(f"  Best mAP    : {best['val_mAP5095']:.4f}  "
          f"(exp #{best['idx']}, {best['commit']})")
    print(f"  Improvement : {delta:+.4f}  ({pct:+.1f}%)")
    print(f"  Best config : {best['description']}")
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()