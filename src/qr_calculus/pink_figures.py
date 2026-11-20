"""Five reproducible experimental figures with explicit provenance."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .config import LEVELS
from .pink_analysis import curve
from .pink_experiment import sample_filename
from .pink_masks import actual_pct, covered_count
from .pink_payloads import MASK_TYPES, PCT

FIGURES = (
    "figure_01_occlusion_examples.png", "figure_02_success_by_level.png",
    "figure_03_shape_comparison.png", "figure_04_steepest_slopes.png",
    "figure_05_integrated_means.png",
)
SHAPES = {"scattered": "Scattered", "clustered": "Clustered"}
COLORS = ("#2f5690", "#12756d", "#9c4f31", "#8a3c69")
MARKERS = ("o", "s", "^", "D")
COVERAGE = "Actual coverage (% of 441 symbol cells)"
RECOVERY = "Exact decoding success (%)"
DATA = "Source: QRculus 2.0 · raw/trials.csv"
DESCRIPTIONS = {
    FIGURES[0]: "Original and pink masks; axis in image pixels; actual coverage (%); one QR and repeat.",
    FIGURES[1]: "Actual coverage (%), exact decoding (%), 100-payload measurements and 80-payload fitted lines, legend.",
    FIGURES[2]: "Actual coverage (%), exact decoding (%), 100-payload measurements by shape, legend.",
    FIGURES[3]: "Actual coverage (%), derivative (%p/%p), 80-payload linear models, legend and minimum.",
    FIGURES[4]: "Integrated mean success (%), 0 to maximum actual coverage, 80-payload model, legend.",
}


def setup(ax, xlabel=None, ylabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.24)
    ax.set_axisbelow(True)


def save(fig, out, identifier, sample):
    filename = FIGURES[identifier - 1]
    fig.text(0.015, 0.012, f"{DATA}  |  {sample}", fontsize=8, color="#354956")
    fig.tight_layout(rect=(0, 0.048, 1, 0.95))
    fig.savefig(out / filename, dpi=160, metadata={"Description": DESCRIPTIONS[filename]})
    plt.close(fig)


def examples(directory, out):
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.25))
    slots = [("none", 0), ("scattered", 20), ("clustered", 20)]
    for ax, (shape, target) in zip(axes, slots):
        path = directory / "samples" / sample_filename("qr001", "H", shape, target)
        with Image.open(path) as image:
            ax.imshow(image, interpolation="nearest")
        title = "Original" if shape == "none" else SHAPES[shape]
        ax.set(title=f"{title} · {actual_pct(covered_count(target)):.2f}%",
               xlabel="x (px)", xticks=(0, 145, 289), yticks=(0, 145, 289))
    axes[0].set_ylabel("y (px)")
    fig.suptitle("Same QR and repeat · 20% target occlusion")
    save(fig, out, 1, "qr001 · H · repeat 0 · two masks; baseline is one decode")


def success_by_level(out, summaries):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), sharey=True)
    for ax, shape in zip(axes, MASK_TYPES):
        for level, color, marker in zip(LEVELS, COLORS, MARKERS):
            values = curve(summaries["all"], level, shape)
            fitted = curve(summaries["train"], level, shape)
            x = [row["actual_pct"] for row in values]
            ax.plot(x, [row["success_rate_pct"] for row in fitted], color=color, linewidth=1.7)
            ax.scatter(x, [row["success_rate_pct"] for row in values], marker=marker,
                       color=color, s=17, label=level, clip_on=False, zorder=3)
        ax.set(title=SHAPES[shape], xlim=(0, actual_pct(covered_count(PCT[-1]))), ylim=(0, 100))
        setup(ax, COVERAGE)
        ax.legend(title="ECC level", loc="upper right", ncols=2, fontsize=8)
    axes[0].set_ylabel(RECOVERY)
    fig.suptitle("Decoding success by correction level · points are measurements, lines are models")
    save(fig, out, 2, "100 measured QR / 1,000 trials per nonzero condition · 80 train QR / 800 fit trials")


def shape_comparison(out, summaries):
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 5.9), sharex=True, sharey=True)
    for ax, level in zip(axes.flat, LEVELS):
        for shape, color, marker in zip(MASK_TYPES, ("#8a3c69", "#12756d"), ("o", "s")):
            rows = curve(summaries["all"], level, shape)
            ax.plot([r["actual_pct"] for r in rows], [r["success_rate_pct"] for r in rows],
                    color=color, marker=marker, markersize=4,
                    linestyle="-" if shape == "scattered" else "--", label=SHAPES[shape], clip_on=False)
        ax.set(title=f"Level {level}", ylim=(0, 100))
        setup(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel(RECOVERY)
    for ax in axes[1]:
        ax.set_xlabel(COVERAGE)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Scattered vs clustered · connections guide the eye between measured points")
    save(fig, out, 3, "100 QR · 10 masks per nonzero condition · 100 trials per level at 0%")


def slopes(out, models):
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 5.9), sharex=True, sharey=True)
    for ax, level in zip(axes.flat, LEVELS):
        for shape, color in zip(MASK_TYPES, ("#8a3c69", "#12756d")):
            model = models[f"{level}/{shape}"]["piecewise"]
            segments = model["segments"]
            ax.stairs([s["slope"] for s in segments], [segments[0]["x0"], *[s["x1"] for s in segments]],
                      color=color, linestyle="-" if shape == "scattered" else "--", baseline=None,
                      label=SHAPES[shape])
            segment = segments[model["steepest_index"]]
            ax.plot([segment["x0"], segment["x1"]], [segment["slope"]] * 2, color=color, linewidth=4)
        ax.set_title(f"Level {level}")
        setup(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Model derivative (%p / %p)")
    for ax in axes[1]:
        ax.set_xlabel(COVERAGE)
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Derivatives on open intervals · thick segments mark the steepest decline")
    save(fig, out, 4, "80 train QR · 800 trials per nonzero condition · 80 per level at 0% · model only")


def integrated(out, model_rows):
    fig, ax = plt.subplots(figsize=(8.7, 3.65))
    x = np.arange(len(LEVELS))
    ceiling = actual_pct(covered_count(PCT[-1]))
    for index, (shape, color, hatch) in enumerate(zip(MASK_TYPES, ("#8a3c69", "#12756d"), ("//", ""))):
        values = [r["mean_success_pct"] for r in model_rows if r["mask_type"] == shape]
        bars = ax.bar(x + (index - 0.5) * 0.35, values, width=0.33, label=SHAPES[shape], color=color, hatch=hatch)
        ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8)
    ax.set(title=f"Integrated means on the shared actual range 0–{ceiling:.4f}%",
           xticks=x, xticklabels=LEVELS, ylim=(0, max(r["mean_success_pct"] for r in model_rows) * 1.22))
    setup(ax, "Error correction level", "Integrated mean success (%)")
    ax.legend(loc="upper right")
    save(fig, out, 5, "80 train QR · 800 trials per nonzero condition · model comparison")


def build_figures(directory, summaries, models, model_rows):
    out = Path(directory) / "figures"
    out.mkdir(exist_ok=True)
    examples(directory, out)
    success_by_level(out, summaries)
    shape_comparison(out, summaries)
    slopes(out, models)
    integrated(out, model_rows)
