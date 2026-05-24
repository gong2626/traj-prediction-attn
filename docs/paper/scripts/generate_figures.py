from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
    }
)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "tables"


METHOD_COLORS = {
    "input": "#dbeafe",
    "encoder": "#e0f2fe",
    "query": "#fef3c7",
    "decoder": "#dcfce7",
    "output": "#fce7f3",
    "line": "#334155",
}

MODEL_COLORS = {
    "Markov baseline": "#94a3b8",
    "Hierarchical GRU": "#38bdf8",
    "TrAISformer-H3": "#f59e0b",
    "FlightBERT++-H3": "#ef4444",
    "Transformer horizon decoder": "#a78bfa",
    "Proposed multiscale model": "#22c55e",
    "Final model": "#22c55e",
    "Refine a=0.05": "#60a5fa",
    "Refine a=0.10": "#f59e0b",
    "Refine a=0.20": "#ef4444",
}


def _save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _box(ax, xy, wh, text, facecolor, fontsize=9):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.2,
        edgecolor="#334155",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#0f172a",
        wrap=True,
    )
    return patch


def _arrow(ax, start, end, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color=METHOD_COLORS["line"],
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def method_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11.6, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(
        ax,
        (0.04, 0.60),
        (0.18, 0.20),
        "30-min flight history\nH3 cells + motion features\nobserved/gap masks",
        METHOD_COLORS["input"],
    )
    _box(
        ax,
        (0.28, 0.62),
        (0.18, 0.16),
        "Feature embeddings\nregion / parent H3\nnumeric + mask + context",
        METHOD_COLORS["input"],
    )
    _box(
        ax,
        (0.53, 0.70),
        (0.20, 0.14),
        "Fine-scale\nTransformer encoder",
        METHOD_COLORS["encoder"],
        fontsize=10,
    )
    _box(
        ax,
        (0.53, 0.43),
        (0.20, 0.14),
        "Stride pooling +\ncoarse-scale Transformer",
        METHOD_COLORS["encoder"],
        fontsize=10,
    )
    _box(
        ax,
        (0.28, 0.28),
        (0.18, 0.14),
        "Shared summary\n+ context feature",
        METHOD_COLORS["encoder"],
    )
    _box(
        ax,
        (0.53, 0.18),
        (0.20, 0.14),
        "Horizon queries\n10 / 30 / 60 / 120 min",
        METHOD_COLORS["query"],
        fontsize=10,
    )
    _box(
        ax,
        (0.78, 0.54),
        (0.18, 0.16),
        "Query attention over\nfine and coarse states",
        METHOD_COLORS["query"],
    )
    _box(
        ax,
        (0.78, 0.30),
        (0.18, 0.15),
        "Hierarchical H3 decoder\ncoarse parent -> child cell\n+ status head",
        METHOD_COLORS["decoder"],
    )
    _box(
        ax,
        (0.78, 0.08),
        (0.18, 0.13),
        "Per-horizon H3\nregion probabilities\nand candidate sets",
        METHOD_COLORS["output"],
    )

    _arrow(ax, (0.22, 0.70), (0.28, 0.70))
    _arrow(ax, (0.46, 0.70), (0.53, 0.77))
    _arrow(ax, (0.63, 0.70), (0.63, 0.57))
    _arrow(ax, (0.46, 0.66), (0.53, 0.50), rad=-0.08)
    _arrow(ax, (0.22, 0.63), (0.28, 0.35), rad=-0.16)
    _arrow(ax, (0.46, 0.35), (0.53, 0.25), rad=-0.08)
    _arrow(ax, (0.73, 0.77), (0.78, 0.62))
    _arrow(ax, (0.73, 0.50), (0.78, 0.59))
    _arrow(ax, (0.73, 0.25), (0.78, 0.57), rad=0.10)
    _arrow(ax, (0.87, 0.54), (0.87, 0.45))
    _arrow(ax, (0.87, 0.30), (0.87, 0.21))

    ax.text(
        0.04,
        0.93,
        "Multiscale horizon-query regional forecaster",
        fontsize=15,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        0.04,
        0.89,
        "The same hierarchical H3 output heads are kept across GRU, Transformer ablation, and final multiscale model.",
        fontsize=9.5,
        color="#475569",
    )
    _save(fig, "fig1_method_architecture")


def main_results_overview() -> None:
    df = pd.read_csv(TABLE_DIR / "main_results.csv")
    label_map = {
        "Markov baseline": "Markov",
        "Hierarchical GRU": "GRU",
        "TrAISformer-H3": "TrAIS",
        "FlightBERT++-H3": "FB++",
        "Transformer horizon decoder": "TF-dec",
        "Proposed multiscale model": "Ours",
    }
    labels = [label_map.get(name, name) for name in df["Model"]]
    colors = [MODEL_COLORS[v] for v in df["Model"]]
    metrics = [
        ("Top5", "Top-5 accuracy", True),
        ("NLL", "Negative log-likelihood", False),
        ("ECE", "Expected calibration error", False),
        ("UnionArea", "Candidate area (km$^2$)", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2))
    axes = axes.ravel()
    for ax, (col, title, higher) in zip(axes, metrics):
        values = df[col].to_numpy()
        ax.bar(labels, values, color=colors, edgecolor="#334155", linewidth=0.6)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        ax.set_axisbelow(True)
        if col == "UnionArea":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        for tick in ax.get_xticklabels():
            tick.set_rotation(18)
            tick.set_ha("right")
        best_idx = int(np.argmax(values) if higher else np.argmin(values))
        ax.patches[best_idx].set_hatch("//")
        ax.patches[best_idx].set_linewidth(1.2)
        ymin, ymax = ax.get_ylim()
        pad = (ymax - ymin) * 0.02
        for idx, val in enumerate(values):
            text = f"{val:.4f}" if col != "UnionArea" else f"{val/1000:.0f}k"
            ax.text(idx, val + pad, text, ha="center", va="bottom", fontsize=8)
    fig.suptitle("Full-scale macro performance across four forecast horizons", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "fig2_main_results_overview")


def h3_refinement_tradeoff() -> None:
    df = pd.read_csv(TABLE_DIR / "h3_refinement_sweep.csv")
    labels = ["Final", "a=0.05", "a=0.10", "a=0.20"]
    colors = [MODEL_COLORS[v] for v in df["Model"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))

    ax = axes[0]
    ax.plot(df["UnionCover"], df["ECE"], color="#334155", linewidth=1.0, zorder=1)
    for i, row in df.iterrows():
        ax.scatter(row["UnionCover"], row["ECE"], s=90, color=colors[i], edgecolor="#0f172a", zorder=2)
        ax.annotate(labels[i], (row["UnionCover"], row["ECE"]), textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.set_xlabel("Candidate union cover")
    ax.set_ylabel("ECE")
    ax.set_title("Recall-calibration trade-off", fontsize=11, fontweight="bold")
    ax.grid(True, color="#e2e8f0")

    ax = axes[1]
    ax.plot(df["UnionCover"], df["UnionArea"], color="#334155", linewidth=1.0, zorder=1)
    for i, row in df.iterrows():
        ax.scatter(row["UnionCover"], row["UnionArea"], s=90, color=colors[i], edgecolor="#0f172a", zorder=2)
        ax.annotate(labels[i], (row["UnionCover"], row["UnionArea"]), textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.set_xlabel("Candidate union cover")
    ax.set_ylabel("Candidate area (km$^2$)")
    ax.set_title("Recall-spatial compactness trade-off", fontsize=11, fontweight="bold")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(True, color="#e2e8f0")

    fig.suptitle("H3-neighbor refinement improves coverage by sacrificing calibration and compactness", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "fig3_h3_refinement_tradeoff")


def main() -> None:
    method_architecture()
    main_results_overview()
    h3_refinement_tradeoff()
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
