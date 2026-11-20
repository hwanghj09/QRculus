"""QA for the stored experiment, mathematical results, and report."""

from collections import Counter
from math import hypot, isclose
from pathlib import Path

import numpy as np
from PIL import Image

from .checks import preflight, valid_pixels
from .config import (
    ELIGIBLE, HASH_RULE, LEVELS, MESSAGE, RNG_RULE, SAMPLES, TARGETS, actual_pct,
    changed_count, read_json, validate_config,
)
from .damage import damage_with_order, permutation
from .decoder import decode_status
from .experiment import read_csv
from .figures import DESCRIPTIONS, FIGURE_NAMES
from .metrics import analyze_piecewise, evaluate, fit_local_cubic, select_local_points
from .module_mask import get_eligible_coords
from .qr_generator import pixel_hash, render_qr


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def load_counts(directory, n):
    config = read_json(directory / "config.json")
    validate_config(config)
    require(config["trials"] == n and config["seed_base"] == 2026, "Unexpected experiment size or seed base")
    raw = read_csv(directory / "trials.csv")
    summary = read_csv(directory / "success_rate.csv")
    expected_keys = {(level, trial, rate) for level in LEVELS for trial in range(n) for rate in TARGETS}
    actual_keys = [(r["level"], int(r["trial"]), float(r["target_pct"])) for r in raw]
    require(len(actual_keys) == len(expected_keys) and set(actual_keys) == expected_keys, "Incomplete or duplicate trial grid")
    groups = [(r["level"], float(r["target_pct"])) for r in summary]
    require(len(groups) == 72 and set(groups) == {(level, rate) for level in LEVELS for rate in TARGETS},
            "Incomplete or duplicate summary grid")
    return config, raw, summary


def check_group_totals(directory, config, raw, summary):
    require(config["rng_rule"] == RNG_RULE and config["hash_rule"] == HASH_RULE, "Reproduction rule mismatch")
    metadata = read_json(directory / "original_matrices.json")
    versions = read_json(directory / "environment.json")
    require(all(versions[key] for key in ("python", "os", "architecture", "cpu")), "Environment is incomplete")
    require(all(versions["library_versions"][library] for library in
                ("opencv-python", "numpy", "qrcode", "matplotlib", "pillow")), "Library version missing")
    originals = {}
    hashes = {}
    for level in LEVELS:
        saved = metadata[level]
        base = np.array([list(map(int, row)) for row in saved["matrix"]], dtype=np.uint8)
        require(base.shape == (21, 21) and np.isin(base, [0, 1]).all(), f"Invalid matrix for {level}")
        require(pixel_hash(base) == saved["matrix_sha256"], f"Matrix hash mismatch for {level}")
        with Image.open(directory / "images" / f"original_{level}.png") as image:
            pixels = np.array(image)
            require(image.mode == "L" and valid_pixels(pixels), f"Invalid original PNG for {level}")
            require(np.array_equal(pixels, render_qr(base)), f"PNG/matrix mismatch for {level}")
        originals[level] = base
        hashes[level] = pixel_hash(pixels)
        require(hashes[level] == saved["pixel_sha256"], f"Original hash mismatch for {level}")

    counts = Counter()
    orders = {trial: permutation(ELIGIBLE, config["seed_base"] + trial) for trial in range(config["trials"])}
    coords = get_eligible_coords()
    for row in raw:
        level, target, trial = row["level"], float(row["target_pct"]), int(row["trial"])
        context = f"{level}, trial {trial}, target {target}"
        k = changed_count(target)
        correct = int(row["decoded_text"] == MESSAGE)
        require(
            row["run_id"] == config["run_id"] and int(row["seed"]) == config["seed_base"] + trial
            and int(row["eligible_modules"]) == ELIGIBLE and int(row["changed_modules"]) == k
            and isclose(float(row["actual_pct"]), actual_pct(k), abs_tol=1e-10)
            and int(row["correct"]) == correct and row["decode_status"] == decode_status(row["decoded_text"])
            and row["original_sha256"] == hashes[level],
            f"Trial metadata or status mismatch: {context}",
        )
        expected = render_qr(damage_with_order(originals[level], coords, k, orders[trial]))
        require(pixel_hash(expected) == row["damaged_sha256"], f"Damaged pixel SHA-256 mismatch: {context}")
        if level == "H" and trial == 0 and target in SAMPLES:
            with Image.open(directory / "images" / f"sample_H_{target:g}pct.png") as image:
                require(image.mode == "L" and np.array_equal(expected, np.array(image)),
                        f"Sample image mismatch: {context}")
        counts[level, target] += correct

    n = config["trials"]
    for row in summary:
        level, target = row["level"], float(row["target_pct"])
        correct_count = counts[level, target]
        k = changed_count(target)
        require(
            int(row["correct_count"]) == correct_count and int(row["trials"]) == n
            and int(row["changed_modules"]) == k and int(row["first_fail_count"]) == n - correct_count
            and isclose(float(row["actual_pct"]), actual_pct(k), abs_tol=1e-10)
            and isclose(float(row["success_pct"]), 100 * correct_count / n, abs_tol=1e-10),
            f"Group aggregate mismatch for {level} / {target}",
        )
        require(target != 0 or correct_count == n, f"0% damage not fully correct for {level}")
    return f"72 aggregates and {len(raw):,} pixel hashes reproduced"


def check_models(directory):
    details = read_json(directory / "analysis_details.json")
    config, _, summary = load_counts(directory, 100)
    require(details["run_id"] == config["run_id"] and details["trials"] == 100,
            "Analysis and main experiment run ID or trial count differ")
    exported = read_csv(directory / "analysis_summary.csv")
    table = read_csv(directory / "table_01_summary.csv")
    require([row["level"] for row in exported] == list(LEVELS) and len(table) == 4, "Missing analysis rows")
    supplemental = read_csv(directory / "local_validation_trials.csv")
    supplemental_keys = Counter()
    supplemental_correct = Counter()
    for row in supplemental:
        key = row["level"], int(row["seed"]), int(row["changed_modules"])
        supplemental_keys[key] += 1
        supplemental_correct[key[0], key[2]] += int(row["correct"])
        require(row["decode_status"] == decode_status(row["decoded_text"])
                and int(row["correct"]) == int(row["decoded_text"] == MESSAGE),
                "Supplementary decode status mismatch")
    expected_supplemental = set()
    for level, entry, table_entry in zip(LEVELS, exported, table):
        stored = details["levels"][level]
        rows = sorted((r for r in summary if r["level"] == level), key=lambda r: float(r["actual_pct"]))
        expected = analyze_piecewise(rows)
        segment = expected["segments"][expected["steepest_index"]]
        require(stored["piecewise"] == expected and 0 <= expected["mean_success_pct"] <= 100,
                f"Invalid piecewise model for {level}")
        expected_summary = {
            "area_0_to_10_actual": expected["area"], "mean_success_pct": expected["mean_success_pct"],
            "steepest_interval_start_pct": segment["start_pct"], "steepest_interval_end_pct": segment["end_pct"],
            "steepest_slope": segment["slope"],
        }
        require(all(isclose(float(entry[key]), val, abs_tol=1e-10) for key, val in expected_summary.items()),
                f"Exported summary mismatch for {level}")
        require(all(table_entry[key] == val for key, val in entry.items())
                and table_entry["run_id"] == config["run_id"], f"Summary table mismatch for {level}")
        points = select_local_points(rows, expected["steepest_index"])
        cubic = stored["cubic"]
        if not points:
            require(cubic is None, f"Unexpected cubic for {level}")
            continue
        expected_cubic = fit_local_cubic(points)
        require(all(cubic[key] == val for key, val in expected_cubic.items()), f"Cubic mismatch for {level}")
        validation = cubic["validation"]
        first_seed, last_seed = validation["first_seed"], validation["last_seed"]
        n = validation["trials_per_k"]
        require(first_seed == 2126 and n == 30 and last_seed - first_seed + 1 == n,
                f"Supplementary sample seeds mismatch for {level}")
        errors = []
        require([point["k"] for point in validation["points"]] == [k for k, _ in points],
                f"Supplementary grid mismatch for {level}")
        for point in validation["points"]:
            k = point["k"]
            expected_supplemental.update((level, seed, k) for seed in range(first_seed, last_seed + 1))
            estimate = evaluate(cubic["coefficients"], k - cubic["k0"])
            measured = 100 * supplemental_correct[level, k] / n
            error = abs(estimate - measured)
            require(isclose(point["prediction_pct"], estimate, abs_tol=1e-10)
                    and isclose(point["measured_pct"], measured, abs_tol=1e-10)
                    and isclose(point["absolute_error_pp"], error, abs_tol=1e-10),
                    f"Supplementary error mismatch for {level} at k={k}")
            errors.append(error)
        require(isclose(validation["mae_pp"], sum(errors) / len(errors), abs_tol=1e-10),
                f"Supplementary MAE mismatch for {level}")
    require(set(supplemental_keys) == expected_supplemental and all(n == 1 for n in supplemental_keys.values()),
            "Unexpected or duplicate supplementary attempts")
    return f"Four integrals and {len(supplemental):,} supplementary attempts checked"


def check_example():
    model = fit_local_cubic([(8, 100), (9, 80), (10, 40), (11, 10)])
    require(model["coefficients"] == [5, -25, 0, 100]
            and isclose(model["min_derivative_u"], 5 / 3)
            and isclose(model["local_mean_success_pct"], 58.75), "Pilot arithmetic check failed")
    return "5u³ − 25u² + 100, interior u=5/3, local mean 58.75%"


def check_figures(directory):
    for filename, description in zip(FIGURE_NAMES, DESCRIPTIONS):
        with Image.open(directory / "figures" / filename) as image:
            require(image.format == "PNG" and image.width > 0 and image.height > 0
                    and image.info.get("Description") == description and "%" in description,
                    f"Invalid PNG or units not documented: {filename}")
            image.verify()
    return "Four readable PNGs with explicit damage units and local validity caption"


def check_manuscript(directory, manuscript):
    from .report import REQUIRED_SECTIONS, TITLE

    report = manuscript.read_text(encoding="utf-8")
    config = read_json(directory / "config.json")
    models = read_json(directory / "analysis_details.json")["levels"]
    require(all(name in report for name in (TITLE, *REQUIRED_SECTIONS)), "Required report section missing")
    require(config["run_id"] in report and "7,200" in report and "20회 예비" in report
            and "trials.csv" in report and "https://" in report, "Missing sources or distinction between set sizes")
    require(all(f"{model['piecewise']['mean_success_pct']:.2f}%" in report for model in models.values()),
            "Stored integrals do not match manuscript")
    return "Eight sections, source URLs, run provenance and measured results"


def check_pdf(pdf, student_id, student_name, *, title=None):
    from pypdf import PdfReader

    from .report import TITLE

    reader = PdfReader(pdf)
    pages = len(reader.pages)
    require(1 <= pages <= 3, f"Unexpected PDF page count: {pages}")
    sizes = []
    for page in reader.pages:
        box = page.mediabox
        require(abs(float(box.width) - 595.276) < 0.5 and abs(float(box.height) - 841.890) < 0.5, "PDF page is not A4")

        def record_size(text, cm, _tm, _font, font_size):
            if text.strip():
                sizes.append(font_size * hypot(cm[0], cm[1]))

        page.extract_text(visitor_text=record_size)
    require(sizes and all(9.98 <= size <= 11.02 for size in sizes),
            f"Rendered text sizes out of 10–11pt range: {sorted({round(size, 3) for size in sizes})}")
    first = reader.pages[0].extract_text()
    require((title or TITLE) in first and "학번" in first and "이름" in first, "PDF title or identity labels missing")
    require(bool(student_id) == bool(student_name), "Provide both student ID and name to check the submission PDF")
    if student_id:
        compact = "".join(first.split())
        require(f"학번:{student_id}" in compact and f"이름:{''.join(student_name.split())}" in compact,
                "Submitted identity fields do not match the provided identity")
    return pages, min(sizes), max(sizes), bool(student_id)


def validate_run(output_dir, *, smoke_input=None, manuscript=None, pdf=None,
                 student_id=None, student_name=None, qa_out=None) -> dict:
    results = {}

    def record(key, status, detail):
        results[key] = {"status": status, "detail": detail}

    def check(key, function):
        try:
            detail = function()
        except Exception as error:
            record(key, "FAIL", f"{type(error).__name__}: {error}")
            return
        record(key, "PASS", detail)

    try:
        for key, (passed, detail) in preflight().items():
            record(key, "PASS" if passed else "FAIL", detail)
    except Exception as error:
        for i in range(1, 12):
            record(f"T{i:02}", "FAIL", f"Preflight could not complete: {error}")

    if smoke_input is None:
        candidate = output_dir.with_name("smoke")
        smoke_input = candidate if candidate.is_dir() else None
    if smoke_input is not None:
        check("T12", lambda: smoke_evidence(smoke_input))
    else:
        record("T12", "SKIP", "No smoke-run artifacts were provided")
    check("T13", lambda: main_evidence(output_dir))
    check("T14", lambda: check_group_totals(output_dir, *load_counts(output_dir, 100)))
    check("T15", lambda: check_models(output_dir))
    check("T16", check_example)
    check("T17", lambda: check_figures(output_dir))
    manuscript = manuscript or Path("reports/보고서_3쪽_원고.md")
    check("T18", lambda: check_manuscript(output_dir, manuscript))
    pdf = pdf or Path("reports/보고서_3쪽.pdf")
    try:
        count, smallest, largest, has_identity = check_pdf(pdf, student_id, student_name)
    except Exception as error:
        record("T19", "FAIL", f"{type(error).__name__}: {error}")
    else:
        detail = f"{count} A4 pages, rendered text {smallest:.1f}–{largest:.1f} pt, title and identity labels"
        record("T19", "PASS" if has_identity else "SKIP",
               detail + (", provided identity verified" if has_identity else ", identity verification skipped"))

    report = qa_out or output_dir / "qa_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# 자동 검증 T01–T19", "", "| ID | 결과 | 실행 근거 |", "|---|---|---|"]
    for key, item in results.items():
        status, detail = item["status"], item["detail"]
        print(f"{key} {status} {detail}")
        lines.append(f"| {key} | {status} | {detail.replace('|', '&#124;').replace(chr(10), ' ')} |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def smoke_evidence(directory):
    config, raw, summary = load_counts(directory, 2)
    check_group_totals(directory, config, raw, summary)
    return "144 attempts and 72 groups, including hashes and exact-match classifications"


def main_evidence(directory):
    load_counts(directory, 100)
    return "7,200 attempts and 72 groups with complete keys"
