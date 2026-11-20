"""Analyze recorded runs and measure same-k prediction errors on fresh seeds."""

from collections import Counter

import cv2

from .config import LEVELS, TARGETS, actual_pct, changed_count, read_json, validate_config, write_json
from .damage import damage_qr
from .decoder import decode_qr
from .experiment import environment, read_csv, write_csv
from .figures import build_figures
from .metrics import analyze_piecewise, evaluate, fit_local_cubic, select_local_points
from .module_mask import get_eligible_coords
from .qr_generator import make_qr, pixel_hash, render_qr

VALIDATION_TRIALS = 30


def measure_errors(models, config, input_dir, out):
    coords = get_eligible_coords()
    detector = cv2.QRCodeDetector()
    first_seed = config["seed_base"] + config["trials"]
    data = []
    recorded = read_json(input_dir / "original_matrices.json")
    for level, model in models.items():
        cubic = model["cubic"]
        if cubic is None:
            continue
        original = make_qr(level)
        pixels = render_qr(original)
        if pixel_hash(pixels) != recorded[level]["pixel_sha256"] or decode_qr(pixels, detector)[1] != "correct":
            raise ValueError(f"Supplementary experiment cannot reproduce the original {level} QR")
        observed = Counter()
        for seed in range(first_seed, first_seed + VALIDATION_TRIALS):
            for point in cubic["points"]:
                k = point["k"]
                damaged = render_qr(damage_qr(original, coords, k, seed))
                text, status = decode_qr(damaged, detector)
                correct = int(status == "correct")
                observed[k] += correct
                data.append({
                    "level": level, "seed": seed, "changed_modules": k, "actual_pct": actual_pct(k),
                    "decoded_text": text, "decode_status": status, "correct": correct,
                    "damaged_sha256": pixel_hash(damaged),
                })
        errors = []
        for point in cubic["points"]:
            k = point["k"]
            prediction = evaluate(cubic["coefficients"], k - cubic["k0"])
            measurement = 100 * observed[k] / VALIDATION_TRIALS
            errors.append({
                "k": k, "prediction_pct": prediction, "measured_pct": measurement,
                "absolute_error_pp": abs(prediction - measurement),
            })
        cubic["validation"] = {
            "trials_per_k": VALIDATION_TRIALS,
            "first_seed": first_seed,
            "last_seed": first_seed + VALIDATION_TRIALS - 1,
            "points": errors,
            "mae_pp": sum(row["absolute_error_pp"] for row in errors) / len(errors),
        }
    write_csv(out / "local_validation_trials.csv", (
        "level", "seed", "changed_modules", "actual_pct", "decoded_text", "decode_status",
        "correct", "damaged_sha256",
    ), data)
    write_json(out / "local_validation_environment.json", environment())


def analyze_run(input_dir, out):
    config = read_json(input_dir / "config.json")
    validate_config(config)
    summary = read_csv(input_dir / "success_rate.csv")
    models = {}
    for level in LEVELS:
        rows = sorted((row for row in summary if row["level"] == level), key=lambda row: float(row["target_pct"]))
        if [float(row["target_pct"]) for row in rows] != list(TARGETS):
            raise ValueError(f"Incomplete or duplicate summary grid for {level}")
        if any(int(row["trials"]) != config["trials"]
               or int(row["changed_modules"]) != changed_count(float(row["target_pct"])) for row in rows):
            raise ValueError(f"Summary metadata mismatch for {level}")
        piecewise = analyze_piecewise(rows)
        points = select_local_points(rows, piecewise["steepest_index"])
        models[level] = {"piecewise": piecewise, "cubic": fit_local_cubic(points) if points else None}

    out.mkdir(parents=True, exist_ok=True)
    if config["trials"] == 100:
        measure_errors(models, config, input_dir, out)
    write_json(out / "analysis_details.json", {
        "run_id": config["run_id"], "trials": config["trials"],
        "source_seed_base": config["seed_base"], "levels": models,
    })
    rows = []
    for level, model in models.items():
        piecewise, cubic = model["piecewise"], model["cubic"]
        steepest = piecewise["segments"][piecewise["steepest_index"]]
        rows.append({
            "level": level,
            "area_0_to_10_actual": piecewise["area"],
            "mean_success_pct": piecewise["mean_success_pct"],
            "steepest_interval_start_pct": steepest["start_pct"],
            "steepest_interval_end_pct": steepest["end_pct"],
            "steepest_slope": steepest["slope"],
            "local_cubic_equation": cubic["equation"] if cubic else "",
            "local_cubic_range": (" – ".join(f"{x:.8f}%" for x in cubic["range_pct"]) if cubic else ""),
            "local_max_drop_point_pct": (
                cubic["min_derivative_pct"] if cubic and cubic["valid"] and cubic["interior_minimum"] else ""
            ),
            "local_model_valid": cubic["valid"] if cubic else "",
            "local_validation_mae_pp": (
                cubic["validation"]["mae_pp"] if cubic and "validation" in cubic else ""
            ),
        })
    write_csv(out / "analysis_summary.csv", tuple(rows[0]), rows)
    table_rows = [
        {**row, "trials_per_condition": config["trials"],
         "actual_range_pct": f"0–{actual_pct(21):.8f}%", "run_id": config["run_id"]}
        for row in rows
    ]
    write_csv(out / "table_01_summary.csv", tuple(table_rows[0]), table_rows)
    build_figures(input_dir, out / "figures", summary, models, config["trials"])
    return models
