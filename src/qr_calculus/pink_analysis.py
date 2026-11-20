"""Group measured outcomes and derive train-only polynomial models."""

from collections import Counter, defaultdict
from fractions import Fraction
from itertools import product
from math import isclose

import numpy as np

from .config import LEVELS, read_json, write_json
from .experiment import read_csv, write_csv
from .metrics import derivative, evaluate, stationary_points
from .pink_experiment import ensure_result_destination
from .pink_masks import MODULES, actual_pct, covered_count
from .pink_payloads import MASK_TYPES, PCT, SELECT_RULE

SUMMARY_FIELDS = (
    "level", "mask_type", "target_pct", "actual_pct", "trials", "successes",
    "success_rate_pct", "detected_count", "detected_rate_pct", "wrong_decode_count",
    "mean_covered_reserved", "mean_covered_black", "mean_covered_white",
    "undetected_failure_count", "detected_failure_count",
)


def condition_keys():
    for level in LEVELS:
        yield level, "none", 0
        yield from ((level, shape, target) for shape, target in product(MASK_TYPES, PCT[1:]))


def update_counter(acc, row):
    success, detected = int(row["success"]), int(row["detected"])
    acc["trials"] += 1
    acc["successes"] += success
    acc["detected_count"] += detected
    acc["wrong_decode_count"] += int(row["wrong_decode"])
    acc["covered_reserved"] += int(row["covered_reserved"])
    acc["covered_black"] += int(row["covered_black"])
    acc["covered_white"] += int(row["covered_white"])
    acc["undetected_failure_count"] += (not detected and not success)
    acc["detected_failure_count"] += (detected and not success)


def summary_rows(acc):
    rows = []
    for level, shape, target in condition_keys():
        values = acc[level, shape, target]
        n = values["trials"]
        if not n:
            raise ValueError(f"Missing condition: {level}/{shape}/{target}")
        rows.append({
            "level": level, "mask_type": shape, "target_pct": target,
            "actual_pct": actual_pct(covered_count(target)), "trials": n,
            "successes": values["successes"], "success_rate_pct": 100 * values["successes"] / n,
            "detected_count": values["detected_count"], "detected_rate_pct": 100 * values["detected_count"] / n,
            "wrong_decode_count": values["wrong_decode_count"],
            "mean_covered_reserved": values["covered_reserved"] / n,
            "mean_covered_black": values["covered_black"] / n,
            "mean_covered_white": values["covered_white"] / n,
            "undetected_failure_count": values["undetected_failure_count"],
            "detected_failure_count": values["detected_failure_count"],
        })
    return rows


def finder_prefixes(path):
    """Count occluded cells in the three 7×7 finder squares for each stored mask prefix."""
    finder = np.zeros((21, 21), dtype=bool)
    finder[:7, :7] = True
    finder[:7, -7:] = True
    finder[-7:, :7] = True
    rows = {}
    for row in read_csv(path):
        order = np.fromstring(row["cells_row_major_index"], sep=" ", dtype=np.intp)
        cumulative = np.cumsum(finder.ravel()[order])
        rows[row["payload_id"], int(row["repeat_id"]), row["mask_type"]] = {
            target: int(cumulative[covered_count(target) - 1]) for target in PCT[1:]
        }
    return rows


def collect_data(directory):
    split = {row["payload_id"]: row["split"] for row in read_csv(directory / "payload_split.csv")}
    lookups = finder_prefixes(directory / "raw/mask_orders.csv")
    groups = {split_name: defaultdict(Counter) for split_name in ("all", "train", "validation")}
    coverage = defaultdict(Counter)
    rows = read_csv(directory / "raw/trials.csv")
    if len(rows) != 112_400:
        raise ValueError(f"Expected complete main run of 112,400 rows, found {len(rows):,}")
    for row in rows:
        pid, shape, target = row["payload_id"], row["mask_type"], int(row["target_pct"])
        key = row["level"], shape, target
        update_counter(groups["all"][key], row)
        update_counter(groups[split[pid]][key], row)
        if target:
            count = lookups[pid, int(row["repeat_id"]), shape][target]
            label = "at_least_one" if count else "zero"
            acc = coverage[(*key, label)]
            update_counter(acc, row)
            acc["covered_finder"] += count
    summaries = {name: summary_rows(acc) for name, acc in groups.items()}
    coverage_rows = []
    for (level, shape, target, label), acc in sorted(coverage.items()):
        n = acc["trials"]
        coverage_rows.append({
            "level": level, "mask_type": shape, "target_pct": target,
            "actual_pct": actual_pct(covered_count(target)), "finder_coverage": label, "trials": n,
            "mean_covered_reserved": acc["covered_reserved"] / n, "mean_covered_finder": acc["covered_finder"] / n,
            "undetected_failure_count": acc["undetected_failure_count"],
            "undetected_failure_pct": 100 * acc["undetected_failure_count"] / n,
            "detected_failure_count": acc["detected_failure_count"],
            "detected_failure_pct": 100 * acc["detected_failure_count"] / n,
        })
    return summaries, coverage_rows


def curve(summaries, level, shape):
    rows = [row for row in summaries if row["level"] == level and row["mask_type"] in ("none", shape)]
    return sorted(rows, key=lambda row: int(row["target_pct"]))


def piecewise_model(points):
    segments = []
    slopes = []
    area = Fraction(0)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        width = x1 - x0
        if width <= 0:
            raise ValueError("Piecewise model needs strictly increasing actual coverage")
        slope = (y1 - y0) / width
        antiderivative = y0 * width + slope * width**2 / 2
        trapezoid = (y0 + y1) * width / 2
        if antiderivative != trapezoid:
            raise ArithmeticError("Integral checks disagree")
        segments.append({
            "x0": float(x0), "x1": float(x1), "y0": float(y0), "y1": float(y1),
            "slope": float(slope), "area": float(antiderivative), "trapezoid": float(trapezoid),
        })
        slopes.append(slope)
        area += antiderivative
    steepest_index = min(range(len(slopes)), key=slopes.__getitem__)
    return {
        "segments": segments, "steepest_index": steepest_index, "area": float(area),
        "range_pct": [float(points[0][0]), float(points[-1][0])],
        "mean_success_pct": float(area / (points[-1][0] - points[0][0])),
    }


def select_window(points, index):
    left = max(0, index - 2)
    right = min(index, len(points) - 4)
    if left > right:
        raise ValueError("Four measured points are needed around the steepest interval")
    start = min(range(left, right + 1), key=lambda i: (abs((index - i) - (i + 3 - index - 1)), i))
    return start, points[start:start + 4]


def fit_cubic(points):
    if len(points) != 4 or any(a[0] >= b[0] for a, b in zip(points, points[1:])):
        raise ValueError("Cubic model needs four distinct increasing points")
    x0, d = points[0]
    x3 = points[-1][0]
    width = x3 - x0
    u1, u2, u3 = [x - x0 for x, _ in points[1:]]
    q1, q2, q3 = [(y - d) / (x - x0) for x, y in points[1:]]
    e12, e23 = (q2 - q1) / (u2 - u1), (q3 - q2) / (u3 - u2)
    a = (e23 - e12) / (u3 - u1)
    b = e12 - a * (u1 + u2)
    c = q1 - a * u1**2 - b * u1
    coefficients = a, b, c, d
    for x, y in points:
        if evaluate(coefficients, x - x0) != y:
            raise ArithmeticError("Cubic does not interpolate the measured points")

    bounds = [Fraction(0), width]
    vertex = -b / (3 * a) if a else None
    derivative_locations = [*bounds, *([vertex] if vertex is not None and 0 < vertex < width else [])]
    minimum_u = min(derivative_locations, key=lambda u: derivative(coefficients, u))
    max_slope = max(derivative(coefficients, u) for u in derivative_locations)
    minimum_slope = derivative(coefficients, minimum_u)
    extrema_u = [*bounds, *[u for u in stationary_points(a, b, c) if 0 < u < width]]
    values = [float(evaluate(coefficients, u)) for u in extrema_u]
    issues = []
    if min(values) < -1e-9 or max(values) > 100 + 1e-9:
        issues.append("success_outside_0_to_100")
    if max_slope > 0:
        issues.append("increasing_within_decline_window")
    if minimum_slope >= 0:
        issues.append("no_decrease")
    return {
        "x0": float(x0), "x1": float(x3), "degree": 3 if a else 2 if b else 1 if c else 0,
        "points": [[float(x), float(y)] for x, y in points],
        "coefficients": [float(val) for val in coefficients],
        "exact_coefficients": [str(val) for val in coefficients],
        "equations": [[float(x - x0), float(y)] for x, y in points],
        "derivative_coefficients": [float(3 * a), float(2 * b), float(c)],
        "endpoint_slopes": [float(derivative(coefficients, u)) for u in bounds],
        "vertex_u": float(vertex) if vertex is not None else None,
        "min_slope_pct": float(x0 + minimum_u), "min_slope": float(minimum_slope),
        "interior_min_slope": 0 < minimum_u < width,
        "model_min_pct": min(values), "model_max_pct": max(values), "max_derivative": float(max_slope),
        "valid": not issues, "issues": issues,
    }


def model_points(rows):
    return [
        (Fraction(100 * covered_count(row["target_pct"]), MODULES),
         Fraction(100 * row["successes"], row["trials"]))
        for row in rows
    ]


def build_models(summaries):
    result = {}
    holdout = []
    validation_predictions = []
    model_rows = []
    for level, shape in product(LEVELS, MASK_TYPES):
        train = curve(summaries["train"], level, shape)
        validation = curve(summaries["validation"], level, shape)
        all_data = curve(summaries["all"], level, shape)
        points = model_points(train)
        line = piecewise_model(points)
        empirical = piecewise_model(model_points(all_data))
        start, selected = select_window(points, line["steepest_index"])
        cubic = fit_cubic(selected)
        errors = []
        local_errors = []
        for i, ((x, y), val) in enumerate(zip(points, validation)):
            truth = Fraction(100 * val["successes"], val["trials"])
            error = float(abs(y - truth))
            is_local = start <= i < start + 4
            errors.append(error)
            if is_local:
                local_errors.append(error)
                estimate = evaluate(cubic["coefficients"], float(x - selected[0][0]))
                if not isclose(estimate, float(y), abs_tol=1e-8):
                    raise ArithmeticError("Stored cubic coefficients fail endpoint check")
            validation_predictions.append({
                "level": level, "mask_type": shape, "target_pct": val["target_pct"],
                "actual_pct": float(x), "validation_trials": val["trials"],
                "measured_pct": float(truth), "linear_prediction_pct": float(y),
                "linear_abs_error_pp": error, "in_cubic_window": int(is_local),
                "cubic_candidate_prediction_pct": float(y) if is_local else "",
                "cubic_candidate_abs_error_pp": error if is_local else "",
            })
        cubic["same_k_holdout_mae_pp"] = sum(local_errors) / len(local_errors)
        model_key = f"{level}/{shape}"
        result[model_key] = {"piecewise": line, "cubic": cubic, "all_measured_piecewise": empirical}
        holdout.append({
            "level": level, "mask_type": shape, "train_payloads": 80, "validation_payloads": 20,
            "n_coverage_knots": len(errors), "linear_mae_pp": sum(errors) / len(errors),
            "linear_max_error_pp": max(errors),
            "cubic_min_actual_pct": cubic["x0"], "cubic_max_actual_pct": cubic["x1"],
            "cubic_candidate_mae_pp": cubic["same_k_holdout_mae_pp"], "cubic_accepted": int(cubic["valid"]),
        })
        steep = line["segments"][line["steepest_index"]]
        all_steep = empirical["segments"][empirical["steepest_index"]]
        model_rows.append({
            "level": level, "mask_type": shape,
            "steepest_start_pct": steep["x0"], "steepest_end_pct": steep["x1"],
            "steepest_slope": steep["slope"], "steepest_start_success_pct": steep["y0"],
            "steepest_end_success_pct": steep["y1"], "area": line["area"],
            "mean_success_pct": line["mean_success_pct"],
            "all_100_steepest_start_pct": all_steep["x0"], "all_100_steepest_end_pct": all_steep["x1"],
            "all_100_steepest_slope": all_steep["slope"], "all_100_mean_success_pct": empirical["mean_success_pct"],
            "cubic_accepted": int(cubic["valid"]), "cubic_min_slope_pct": cubic["min_slope_pct"] if cubic["valid"] else "",
        })
    return result, model_rows, holdout, validation_predictions


def analyze_pink(out):
    from .pink_figures import build_figures

    ensure_result_destination(out)
    metadata = read_json(out / "run_metadata.json")
    if metadata.get("status") != "complete" or metadata.get("kind") != "main":
        raise ValueError("Analysis requires the completed main experiment")
    summaries, coverage = collect_data(out)
    models, model_rows, holdout, predictions = build_models(summaries)
    processed = out / "processed"
    processed.mkdir(exist_ok=True)
    for name, rows in (
        ("summary.csv", summaries["all"]), ("training_summary.csv", summaries["train"]),
        ("validation_summary.csv", summaries["validation"]), ("finder_coverage.csv", coverage),
        ("model_summary.csv", model_rows), ("holdout_metrics.csv", holdout),
        ("holdout_predictions.csv", predictions),
    ):
        write_csv(processed / name, tuple(rows[0]), rows)
    write_json(processed / "models.json", {
        "run_id": metadata["run_id"], "cubic_selection_rule": SELECT_RULE, "curves": models,
        "holdout_unit": "payload",
        "holdout_limits": "Same observed k only; does not test unseen intermediate points",
    })
    build_figures(out, summaries, models, model_rows)
    print(f"Analyzed {len(summaries['all'])} conditions and {len(model_rows)} curves", flush=True)
