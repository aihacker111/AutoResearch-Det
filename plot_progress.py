"""
plot_progress.py — Visualise experiment progress.
==================================================
AutoResearch-DET | Run anytime: python plot_progress.py

Example (demo TSV bundled in repo)
----------------------------------
    python plot_progress.py -i examples/sample_results.tsv -o examples/output

Outputs
-------
    progress.png        — mAP50-95 per experiment + LLM solution per bar
    progress_detail.png — step chart of improvements (optional)
"""

from __future__ import annotations

import argparse
import csv
import sys
import textwrap
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


def _desc_for_box(s: str, width: int = 44, max_lines: int = 4) -> str:
    s = " ".join(s.replace("\n", " ").split())
    lines = textwrap.wrap(s, width=width) if s else [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: width - 1] + "…"
    return "\n".join(lines)


def plot_overview(rows: list[dict], out: str = "progress.png") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    import numpy as np

    idxs    = [r["idx"]         for r in rows]
    maps    = [r["val_mAP5095"] for r in rows]
    maps50  = [r["val_mAP50"]   for r in rows]
    colors  = [STATUS_COLOR.get(r["status"], "#aaa") for r in rows]

    n = len(rows)
    fig_w = max(14.0, 1.4 * n + 6.0)
    fig_h = max(7.5, 4.0 + min(n * 0.45, 6.0))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="#1a1a2e")
    fig.suptitle(
        "AutoResearch-DET  ·  YOLO11  ·  val/mAP50-95 & LLM solution per experiment",
        color="white", fontsize=14, fontweight="bold", y=0.98,
    )

    ax.set_facecolor("#16213e")
    ax.bar(idxs, maps, color=colors, alpha=0.85, zorder=3, width=0.65)
    ax.plot(
        idxs, maps, color="#f0e6ff", lw=1.5, marker="o", ms=5,
        zorder=4,
    )
    ax.plot(
        idxs, maps50, color="#74b9ff", lw=1.2, ls="--", marker="s",
        ms=4, zorder=4, alpha=0.75,
    )
    ax.axhline(maps[0], color="#fdcb6e", lw=1.2, ls=":")

    best_exp = idxs[int(np.argmax(maps))]

    y_top = max(maps) if maps else 1.0
    ax.set_ylim(0, y_top * 1.42 + 1e-6)

    for r in rows:
        i = r["idx"]
        m = r["val_mAP5095"]
        desc = _desc_for_box(r["description"])
        is_best = i == best_exp
        txt = f"mAP50-95: {m:.4f}\n{desc}"
        ax.annotate(
            txt,
            xy=(i, m),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color="#eef6ff",
            fontweight="bold" if is_best else "normal",
            bbox=dict(
                boxstyle="round,pad=0.35",
                fc="#0f1729",
                ec="#2ecc71" if is_best else "#556688",
                lw=1.4 if is_best else 0.9,
                alpha=0.95,
            ),
            zorder=6,
        )

    ax.set_ylabel("mAP50-95", color="white", fontsize=11)
    ax.set_xlabel("Experiment #", color="white", fontsize=10)
    ax.set_xticks(idxs)
    ax.set_xticklabels(
        [f"#{r['idx']}\n{r['commit'][:7]}" for r in rows],
        rotation=0,
        ha="center",
        fontsize=8,
        color="#aaccee",
    )
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444466")

    line_handles = [
        Line2D([0], [0], color="#f0e6ff", lw=1.5, marker="o", label="mAP50-95"),
        Line2D(
            [0], [0], color="#74b9ff", lw=1.2, ls="--", marker="s",
            label="mAP50",
        ),
        Line2D(
            [0], [0], color="#fdcb6e", lw=1.2, ls=":",
            label=f"baseline {maps[0]:.4f}",
        ),
    ]
    status_handles = [
        mpatches.Patch(color=v, label=k) for k, v in STATUS_COLOR.items()
    ]
    ax.legend(
        handles=line_handles + status_handles,
        fontsize=7,
        facecolor="#1a1a2e",
        labelcolor="white",
        loc="upper left",
        ncol=2,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
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

    ki = [r["idx"] for r in keeps]
    km = [r["val_mAP5095"] for r in keeps]

    ax.step(ki, km, where="post", color="#a29bfe", lw=2.5, zorder=3)
    ax.fill_between(ki, km, step="post", alpha=0.12, color="#a29bfe")
    ax.scatter(ki, km, color="#fdcb6e", s=60, zorder=5)

    for i, r in enumerate(keeps):
        delta = r["val_mAP5095"] - keeps[i - 1]["val_mAP5095"] if i > 0 else 0
        label = r["description"][:42] + "…" if len(r["description"]) > 42 else r["description"]
        ax.annotate(
            f"{delta:+.4f}\n{label}",
            xy=(r["idx"], r["val_mAP5095"]),
            xytext=(r["idx"] + 0.15, r["val_mAP5095"] + 0.002 * (1 + i % 3)),
            color="#2ecc71" if delta >= 0 else "#e74c3c",
            fontsize=7.5,
            arrowprops=dict(arrowstyle="-", color="#666688", lw=0.8),
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="#16213e", ec="#444466", alpha=0.8,
            ),
        )

    ax.set_xlabel("Experiment #", color="white")
    ax.set_ylabel("mAP50-95", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444466")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot AutoResearch-DET results from results.tsv",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path(RESULTS_FILE),
        help=f"Tab-separated results (default: {RESULTS_FILE})",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Directory for progress.png and progress_detail.png",
    )
    args = parser.parse_args()
    in_path = args.input
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.is_file():
        sys.exit(f"ERROR: input file not found: {in_path}")

    rows = load_results(str(in_path))
    if not rows:
        sys.exit("No results yet.")

    print(f"[plot] Loaded {len(rows)} experiments from {in_path}")
    plot_overview(rows, str(out_dir / "progress.png"))
    plot_detail(rows, str(out_dir / "progress_detail.png"))

    best = max(rows, key=lambda r: r["val_mAP5095"])
    baseline = rows[0]["val_mAP5095"]
    delta = best["val_mAP5095"] - baseline
    pct = delta / baseline * 100 if baseline > 0 else 0

    print(f"\n{'-' * 50}")
    print(f"  Experiments : {len(rows)}")
    print(f"  Baseline    : {baseline:.4f}")
    print(
        f"  Best mAP    : {best['val_mAP5095']:.4f}  "
        f"(exp #{best['idx']}, {best['commit']})"
    )
    print(f"  Improvement : {delta:+.4f}  ({pct:+.1f}%)")
    print(f"  Best config : {best['description']}")
    print(f"  PNG written : {out_dir / 'progress.png'}")
    print(f"{'-' * 50}")


if __name__ == "__main__":
    main()
