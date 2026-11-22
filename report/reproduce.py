#!/usr/bin/env python3
"""Reproduce the 2026 calculus report from stored QRculus measurements.

Only the L error-correction, scattered pink occlusion condition is used.
The 0%-occlusion baseline has mask_type=none in the original data.
No network access or new QR decoding experiments are required.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "pink_v2" / "processed"
OUTPUT = Path(__file__).resolve().parent / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_points(filename: str) -> pd.DataFrame:
    data = pd.read_csv(SOURCE / filename, encoding="utf-8-sig")
    selected = data.loc[
        (data["level"] == "L")
        & (data["mask_type"].isin(["scattered", "none"]))
        & (data["actual_pct"] <= 10.0)
    ].copy()
    selected = selected.sort_values("actual_pct")
    if len(selected) != 10 or selected["actual_pct"].duplicated().any():
        raise ValueError(f"Expected 10 distinct L/scattered coverage values in {filename}")
    if selected.iloc[0]["mask_type"] != "none" or selected.iloc[0]["actual_pct"] != 0:
        raise ValueError("Missing undamaged baseline")
    return selected[["actual_pct", "success_rate_pct", "trials"]].reset_index(drop=True)


def logistic(x, slope, midpoint):
    return 100 * expit(-slope * (np.asarray(x) - midpoint))


def derivative(x, slope, midpoint):
    p = logistic(x, slope, midpoint) / 100
    return -100 * slope * p * (1 - p)


train = load_points("training_summary.csv")
holdout = load_points("validation_summary.csv")
x = train["actual_pct"].to_numpy(dtype=float)
xt = holdout["actual_pct"].to_numpy(dtype=float)
if not np.allclose(x, xt):
    raise ValueError("Train and holdout coverage grids do not match")
y = train["success_rate_pct"].to_numpy(dtype=float)
yt = holdout["success_rate_pct"].to_numpy(dtype=float)

# Unweighted least-squares fit to the ten TRAIN aggregated points.
# Each nonzero training point has 800 trials; the 0% baseline has 80.
(slope, midpoint), _ = curve_fit(
    logistic, x, y, p0=(1, 2.5),
    bounds=([0.00001, 0.0], [20.0, 10.0]),
    maxfev=100000,
)
pred = logistic(xt, slope, midpoint)
holdout_mae_pp = float(np.mean(np.abs(pred - yt)))
minimum_derivative = float(-25.0 * slope)
integral, _ = quad(lambda z: float(logistic(z, slope, midpoint)), 0.0, 10.0)
mean_success = float(integral / 10.0)
baseline_prediction = float(logistic(0, slope, midpoint))

summary = {
    "model": "R(x) = 100 / (1 + exp(a*(x-b)))",
    "scope": "error correction L, scattered pink occlusion, 0 to ~10% actual coverage",
    "training_payloads": 80,
    "validation_payloads": 20,
    "train_aggregated_points": int(len(x)),
    "a": float(slope),
    "b_pct": float(midpoint),
    "steepest_decline_at_pct": float(midpoint),
    "minimum_derivative_pp_per_pp": minimum_derivative,
    "mean_model_success_0_to_10_pct": mean_success,
    "validation_mae_pp_on_observed_coverages": holdout_mae_pp,
    "predicted_success_at_zero_pct": baseline_prediction,
    "measured_success_at_zero_pct": 100.0,
}
(OUTPUT / "fit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
pd.DataFrame({
    "actual_pct": xt,
    "train_rate_pct": y,
    "validation_rate_pct": yt,
    "model_prediction_pct": pred,
    "validation_absolute_error_pp": np.abs(pred - yt),
}).to_csv(OUTPUT / "observations_and_predictions.csv", index=False)

xs = np.linspace(0, 10, 401)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(xs, logistic(xs, slope, midpoint), label="Logistic fit (training)", linewidth=2.3)
ax.scatter(x, y, label="Training (80 QR)", s=28, zorder=3)
ax.scatter(xt, yt, label="Validation (20 QR)", marker="x", s=40, zorder=4)
ax.axvline(midpoint, linestyle="--", color="gray", alpha=0.7)
ax.set(xlim=(0, 10), ylim=(-3, 103), xlabel="Actual occlusion (%)", ylabel="Exact decode success (%)")
ax.grid(alpha=0.2)
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT / "success_curve.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.7))
ax.plot(xs, derivative(xs, slope, midpoint), linewidth=2.4)
ax.axvline(midpoint, linestyle="--", color="gray", alpha=0.7)
ax.scatter([midpoint], [minimum_derivative], s=45, zorder=4)
ax.set(xlim=(0, 10), xlabel="Actual occlusion (%)", ylabel="Derivative (pp/pp)")
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUTPUT / "derivative.png", dpi=180)
plt.close(fig)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nGenerated figures and measured-vs-predicted CSV: {OUTPUT}")
