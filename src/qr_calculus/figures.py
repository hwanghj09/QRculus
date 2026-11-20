"""PNG outputs for observed recovery and the polynomial models."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

from .config import LEVELS, SAMPLES, actual_pct, changed_count
from .metrics import derivative, evaluate

STYLES = (
    ("#a63d32", "o", "-"), ("#17766d", "s", "--"),
    ("#40569f", "^", "-."), ("#6e3d81", "D", ":"),
)
X_LABEL = "Actual damage (% of 208 eligible modules)"
FIGURE_NAMES = (
    "figure_01_origin_and_damage.png", "figure_02_success_by_level.png",
    "figure_03_piecewise_analysis.png", "figure_04_local_cubic.png",
)
DESCRIPTIONS = (
    "Target and actual damage (%) are printed above each image.",
    "Actual damage (% of 208 eligible modules) vs. correct decoding (%).",
    "Actual damage (% of 208 eligible modules), with steepest segments emphasized.",
    "Actual damage (%) within selected intervals only. 선택 구간에서만 유효.",
)


def save_figure(fig, out, number):
    fig.savefig(out / FIGURE_NAMES[number - 1], dpi=165, metadata={"Description": DESCRIPTIONS[number - 1]})
    plt.close(fig)


def axis_style(ax):
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)


def plot_recovery(out, rows, models, count, show_slopes=False):
    fig, ax = plt.subplots(figsize=(7.4, 3.45), layout="constrained")
    for level, (color, marker, style) in zip(LEVELS, STYLES):
        items = sorted((row for row in rows if row["level"] == level), key=lambda row: float(row["actual_pct"]))
        ax.plot([float(row["actual_pct"]) for row in items],
                [float(row["success_pct"]) for row in items],
                color=color, marker=marker, linestyle=style, ms=4, label=level, clip_on=False)
        if show_slopes:
            model = models[level]["piecewise"]
            segment = model["segments"][model["steepest_index"]]
            ax.plot([segment["start_pct"], segment["end_pct"]],
                    [segment["start_success_pct"], segment["end_success_pct"]],
                    color=color, linewidth=4, alpha=0.65)
    title = "Piecewise linear model · steepest segments emphasized" if show_slopes else f"Measured QR recovery · {count} attempts per condition"
    ax.set(title=title, xlabel=X_LABEL, ylabel="Correct decoding (%)", ylim=(0, 100),
           xlim=(0, actual_pct(21)), xticks=range(0, 11), yticks=range(0, 101, 20))
    axis_style(ax)
    ax.legend(title="Correction level", loc="upper right")
    save_figure(fig, out, 3 if show_slopes else 2)


def plot_local(out, models):
    font = next((item.name for item in font_manager.fontManager.ttflist
                 if item.name in {"Noto Sans CJK KR", "Malgun Gothic", "AppleGothic", "NanumGothic"}), None)
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.7), layout="constrained")
    title = "Local polynomial and derivative · selected intervals only"
    if font:
        title += "\n선택 구간에서만 유효"
    fig.suptitle(title, fontfamily=font)
    offsets = np.linspace(0, 3, 181)
    for idx, (level, (color, marker, _)) in enumerate(zip(LEVELS, STYLES)):
        top, bottom = axes[:, idx]
        model = models[level]["cubic"]
        if model is None:
            top.set(title=level)
            top.text(0.5, 0.5, "No consecutive\nfour-point window", ha="center", va="center", transform=top.transAxes)
            bottom.set_axis_off()
            continue
        xs = actual_pct(model["k0"] + offsets)
        coefficients = model["coefficients"]
        linestyle = "-" if model["valid"] else "--"
        top.plot(xs, evaluate(coefficients, offsets), color=color, linestyle=linestyle)
        top.scatter([actual_pct(point["k"]) for point in model["points"]],
                    [point["success_pct"] for point in model["points"]],
                    marker=marker, color=color, zorder=3)
        bottom.plot(xs, derivative(coefficients, offsets) * 208 / 100, color=color, linestyle=linestyle)
        bottom.axhline(0, color="#777777", linewidth=0.8)
        if model["valid"]:
            location = model["min_derivative_pct"]
            for ax in (top, bottom):
                ax.axvline(location, linestyle=":", linewidth=1.3, color="#333333")
            bottom.scatter(location, model["min_derivative_per_damage_pct"], color=color,
                           marker="x", label=f"Min. at {location:.2f}%")
            bottom.legend(loc="best", fontsize=8)
        top.set(title=f"{level} · {'accepted' if model['valid'] else 'rejected'}")
        bottom.set(xlabel="Actual damage (%)")
        for ax in (top, bottom):
            ax.set(xlim=model["range_pct"])
            axis_style(ax)
    axes[0, 0].set(ylabel="Model success (%)")
    axes[1, 0].set(ylabel="Slope (success pp / damage pp)")
    save_figure(fig, out, 4)


def build_figures(input_dir, out, rows, models, count):
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(10, 2.6), layout="constrained")
    samples = [(None, "original_H.png"), *((rate, f"sample_H_{rate:g}pct.png") for rate in SAMPLES)]
    for ax, (rate, filename) in zip(axes, samples):
        with Image.open(input_dir / "images" / filename) as image:
            ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        if rate is None:
            ax.set_title("H original")
        else:
            k = changed_count(rate)
            ax.set_title(f"Target {rate:g}%\nActual {actual_pct(k):.2f}% · {k} modules", fontsize=9)
        ax.axis("off")
    fig.suptitle("H correction level · trial 0 · same damage order")
    save_figure(fig, out, 1)
    plot_recovery(out, rows, models, count)
    plot_recovery(out, rows, models, count, show_slopes=True)
    plot_local(out, models)
