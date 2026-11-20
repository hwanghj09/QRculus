"""T01–T20 against the archived experiment, models and report."""

from collections import Counter
from hashlib import sha256
from itertools import product
from math import isclose
from pathlib import Path

import numpy as np
from PIL import Image

from .config import LEVELS, read_json
from .experiment import read_csv
from .metrics import derivative, evaluate
from .module_mask import get_reserved_mask
from .pink_analysis import build_models, collect_data, fit_cubic
from .pink_experiment import REPO, TRIAL_FIELDS, ensure_result_destination, grid, originals_for, rgb_hash, sample_filename
from .pink_figures import DESCRIPTIONS, FIGURES
from .pink_masks import EDGE, MODULES, SYMBOL_END, actual_pct, adjacent, covered_count, mask_order, mask_seed, paint_cells
from .pink_payloads import CHARSET, MASK_TYPES, PCT, REPEATS, RGB, generate_payloads, load_pink_config, split_payloads
from .pink_report import SECTIONS, TITLE
from .validation import check_pdf, require

LEGACY_MANIFEST = REPO / "specs/pink_occlusion_v2/legacy_artifacts_sha256.json"


def payload_audit(out):
    recorded = read_csv(out / "payloads.csv")
    expected = generate_payloads()
    normalized = [{**row, "qr_version": int(row["qr_version"])} for row in recorded]
    require(normalized == expected and expected == generate_payloads(), "Payload sequence did not reproduce")
    labels = [p["payload"] for p in recorded]
    require(len(labels) == len(set(labels)) == 100 and all(len(label) == 6 and set(label) <= set(CHARSET)
                                                           for label in labels), "Payloads are not unique 6-character IDs")
    return expected


def mask_audit(out, payloads):
    records = read_csv(out / "raw/mask_orders.csv")
    expected_keys = set(product((p["payload_id"] for p in payloads), range(REPEATS), MASK_TYPES))
    orders = {}
    for row in records:
        pid, repeat, shape = key = row["payload_id"], int(row["repeat_id"]), row["mask_type"]
        require(key in expected_keys and key not in orders, f"Unexpected or duplicate mask key: {key}")
        seed = mask_seed(pid, repeat, shape)
        order = np.fromstring(row["cells_row_major_index"], sep=" ", dtype=np.intp)
        require(int(row["mask_seed"]) == seed and len(order) == MODULES, f"Incorrect seed or mask length: {key}")
        require(np.array_equal(order, mask_order(seed, shape)) and int(row["start_cell"]) == order[0],
                f"Wrong mask random sequence or initial cell: {key}")
        cells = set()
        for position, cell in enumerate(map(int, order)):
            require(0 <= cell < MODULES and cell not in cells
                    and (shape != "clustered" or position == 0 or any(n in cells for n in adjacent(cell))),
                    f"Invalid mask progression: {key} cell {cell}")
            cells.add(cell)
        previous = set()
        for target in PCT:
            k = covered_count(target)
            prefix = set(order[:k])
            require(len(prefix) == round(441 * target / 100) and previous <= prefix,
                    f"Incorrect prefix size or nesting: {key}, at {target}%")
            previous = prefix
        orders[key] = (order, seed)
    require(set(orders) == expected_keys, "Mask key grid incomplete")
    return orders


def check_sample(out, image, row):
    pid, level, shape, target = row["payload_id"], row["level"], row["mask_type"], int(row["target_pct"])
    if pid != "qr001" or int(row["repeat_id"]) != 0 or target not in (0, 5, 20, 40):
        return
    with Image.open(out / "samples" / sample_filename(pid, level, shape, target)) as png:
        saved = np.asarray(png)
        require(png.mode == "RGB" and png.size == (290, 290) and np.array_equal(saved, image),
                "Sample PNG differs from regenerated pixels")
        pink = np.all(saved == RGB, axis=-1)
        require(pink.sum() == int(row["covered_modules"]) * 100
                and all(np.all(border == 255) for border in
                        (saved[:EDGE], saved[SYMBOL_END:], saved[:, :EDGE], saved[:, SYMBOL_END:])),
                "Sample PNG contains wrong pink pixel count or altered quiet zone")


def raw_audit(out, payloads, orders):
    meta = read_json(out / "run_metadata.json")
    load_pink_config(out / "config.json")
    require(sha256((out / "config.json").read_bytes()).hexdigest() == meta["config_sha256"],
            "Configuration SHA-256 mismatch")
    require(meta["kind"] == "main" and meta["status"] == "complete" and 112_400 == meta["actual_calls"]
            and read_json(out / "environment.json") == meta["environment"], "Invalid run metadata or environment")
    raw = read_csv(out / "raw/trials.csv")
    require(len(raw) == 112_400 and tuple(raw[0]) == TRIAL_FIELDS and meta["baseline_calls"] == 400,
            "Wrong raw trial count or field schema")
    manifest = list(grid(PCT, REPEATS))
    reserved_mask = get_reserved_mask().ravel()
    baseline = read_csv(out / "preflight.csv")
    original_count = 0
    events = {}
    for event in read_csv(out / "raw/decoder_events.csv"):
        key = event["payload_id"], event["level"], event["mask_type"], event["repeat_id"], event["target_pct"]
        require(key not in events and event["detail"], "Duplicate or empty decoder event")
        events[key] = event["event"]
    for position, payload in enumerate(payloads):
        pid = payload["payload_id"]
        matrices, originals, hashes = originals_for(payload)
        for level, matrix in matrices.items():
            require(matrix.shape == (21, 21) and matrix.dtype == np.uint8 and np.isin(matrix, [0, 1]).all()
                    and originals[level].shape == (290, 290, 3) and originals[level].dtype == np.uint8,
                    "Original QR dimensions, type or modules are invalid")
        active_mask = None
        offset = position * len(manifest)
        for j, (shape, repeat, target, level) in enumerate(manifest):
            row = raw[offset + j]
            context = f"{pid}/{level}/{shape}/{repeat}/{target}"
            require(None not in row and all(value is not None for value in row.values())
                    and (row["payload_id"], row["payload"], row["run_id"], row["level"], row["mask_type"],
                         int(row["repeat_id"]), int(row["target_pct"])) ==
                        (pid, payload["payload"], meta["run_id"], level, shape, repeat, target),
                    f"Missing/duplicate row or malformed key: {context}")
            success, detected = int(row["success"]), int(row["detected"])
            text = row["decoded_text"]
            require(success == int(text == payload["payload"]) and detected in (0, 1)
                    and int(row["wrong_decode"]) == int(bool(text) and not success),
                    f"Invalid outcome indicators: {context}")
            event_key = pid, level, shape, row["repeat_id"], row["target_pct"]
            event = events.pop(event_key, None)
            if event:
                require((event == "degenerate_source_points" and detected == success == 0 and not text)
                        or (event == "non_utf8_payload" and int(row["wrong_decode"]) == 1),
                        f"Decoder diagnostic disagrees with trial: {context}")
            k = covered_count(target)
            require(int(row["covered_modules"]) == k and isclose(float(row["actual_pct"]), actual_pct(k), abs_tol=1e-10),
                    f"Incorrect covered count or actual area: {context}")
            require(row["original_pixel_sha256"] == hashes[level], f"Incorrect original SHA-256: {context}")
            if shape == "none":
                require(row == baseline[original_count] and success == detected == 1
                        and row["mask_seed"] == row["start_cell"] == "" and
                        row["covered_reserved"] == row["covered_black"] == row["covered_white"] == "0"
                        and row["damaged_pixel_sha256"] == hashes[level], f"Invalid 0% record: {context}")
                check_sample(out, originals[level], row)
                original_count += 1
                continue

            mask_key = pid, repeat, shape
            if mask_key != active_mask:
                order, seed = orders[mask_key]
                working = {lev: image.copy() for lev, image in originals.items()}
                last_k = dict.fromkeys(LEVELS, 0)
                active_mask = mask_key
            require(int(row["mask_seed"]) == seed and int(row["start_cell"]) == int(order[0]),
                    f"Different masks assigned across levels: {context}")
            cells = order[:k]
            black = int(matrices[level].ravel()[cells].sum())
            require(int(row["covered_reserved"]) == int(reserved_mask[cells].sum())
                    and int(row["covered_black"]) == black and int(row["covered_white"]) == k - black,
                    f"Covered function/black/white count mismatch: {context}")
            image = working[level]
            paint_cells(image, order[last_k[level]:k])
            last_k[level] = k
            require(row["damaged_pixel_sha256"] == rgb_hash(image), f"Damaged image SHA-256 mismatch: {context}")
            check_sample(out, image, row)
    require(original_count == len(baseline) == 400 and not events, "Unexpected baseline rows or orphan decoder events")


def compare_csv(path, expected):
    actual = read_csv(path)
    require(len(actual) == len(expected), f"Unexpected row count in {path.name}")
    for row, expected_row in zip(actual, expected):
        for key, expected_val in expected_row.items():
            if isinstance(expected_val, float):
                match = isclose(float(row[key]), expected_val, abs_tol=1e-9)
            else:
                match = row[key] == str(expected_val)
            require(match, f"Stored value differs from recalculation in {path.name}: {key}")


def split_audit(out, payloads):
    records = read_csv(out / "payload_split.csv")
    observed = {r["payload_id"]: r["split"] for r in records}
    require(len(records) == 100 and observed == split_payloads(payloads)
            and Counter(observed.values()) == {"train": 80, "validation": 20},
            "Missing, duplicate or invalid train/holdout assignment")


def audit_summaries(out, summaries, coverage):
    for name, expected, count in (
        ("summary.csv", summaries["all"], 100),
        ("training_summary.csv", summaries["train"], 80),
        ("validation_summary.csv", summaries["validation"], 20),
    ):
        require(len(expected) == 116, f"{name} has an incomplete 116-row grid")
        for row in expected:
            n = count if row["target_pct"] == 0 else count * 10
            require(row["trials"] == n and 0 <= row["success_rate_pct"] <= 100
                    and 0 <= row["detected_rate_pct"] <= 100
                    and row["successes"] + row["undetected_failure_count"] + row["detected_failure_count"] == n,
                    f"Invalid group denominator/outcome totals in {name}")
        compare_csv(out / "processed" / name, expected)
    compare_csv(out / "processed/finder_coverage.csv", coverage)


def audit_linear(models, summaries):
    for key, detail in models.items():
        level, shape = key.split("/")
        rows = sorted((r for r in summaries["train"] if r["level"] == level and r["mask_type"] in (shape, "none")),
                      key=lambda r: r["actual_pct"])
        line = detail["piecewise"]
        require(len(rows) == len(PCT) and len(line["segments"]) == len(PCT) - 1, f"Invalid model grid: {key}")
        area = 0.0
        for index, seg in enumerate(line["segments"]):
            x0, x1, y0, y1, slope = (seg[field] for field in ("x0", "x1", "y0", "y1", "slope"))
            width = x1 - x0
            require(isclose(x0, rows[index]["actual_pct"], abs_tol=1e-9)
                    and isclose(x1, rows[index + 1]["actual_pct"], abs_tol=1e-9)
                    and isclose(y0, rows[index]["success_rate_pct"], abs_tol=1e-9)
                    and isclose(y1, rows[index + 1]["success_rate_pct"], abs_tol=1e-9)
                    and isclose(y0 + slope * width, y1, abs_tol=1e-9), f"Linear interpolation failed: {key}")
            primitive = y0 * width + slope * width**2 / 2
            require(isclose(primitive, (y0 + y1) * width / 2, abs_tol=1e-9)
                    and isclose(primitive, seg["area"], abs_tol=1e-9), f"Integration verification failed: {key}")
            area += primitive
        require(isclose(area, line["area"], abs_tol=1e-9) and
                isclose(area / line["range_pct"][1], line["mean_success_pct"], abs_tol=1e-9),
                f"Integrated model average incorrect: {key}")


def audit_cubics(models):
    from fractions import Fraction

    fixture = [(Fraction(u), Fraction(100 - u**3)) for u in (0, 1, 2, 4)]
    known = fit_cubic(fixture)
    require(known["exact_coefficients"] == ["-1", "0", "0", "100"] and known["min_slope"] == -48,
            "Uneven-spacing hand-computed cubic test failed")
    for key, detail in models.items():
        cubic = detail["cubic"]
        coefficients = cubic["coefficients"]
        a, b, c, d = coefficients
        require(all(isclose(evaluate(coefficients, x - cubic["x0"]), y, abs_tol=1e-8) for x, y in cubic["points"]),
                f"Cubic interpolation mismatch: {key}")
        require(all(isclose(left, right, abs_tol=1e-10)
                    for left, right in zip(cubic["derivative_coefficients"], (3 * a, 2 * b, c), strict=True)),
                f"Incorrect cubic derivative: {key}")
        endpoints = [0, cubic["x1"] - cubic["x0"]]
        vertex = -b / (3 * a) if a else None
        candidates = [*endpoints, *([vertex] if vertex is not None and 0 < vertex < endpoints[1] else [])]
        expected_slope = min(derivative(coefficients, u) for u in candidates)
        require(isclose(cubic["min_slope"], expected_slope, abs_tol=1e-8)
                and (not cubic["valid"] or (cubic["model_min_pct"] >= -1e-9
                     and cubic["model_max_pct"] <= 100 + 1e-9 and cubic["max_derivative"] <= 1e-9)),
                f"Invalid cubic extrema or accepted range: {key}")


def figure_audit(out):
    for filename in FIGURES:
        with Image.open(out / "figures" / filename) as image:
            require(image.format == "PNG" and image.width > 600 and image.height > 300
                    and image.info.get("Description") == DESCRIPTIONS[filename],
                    f"Figure metadata or image is invalid: {filename}")
            image.verify()


def pdf_audit(pdf, manuscript, student_id, student_name):
    from pypdf import PdfReader

    baseline_text = manuscript.read_text(encoding="utf-8")
    require(all(section in baseline_text for section in (TITLE, *SECTIONS)) and
            "https://" in baseline_text and "112,400" in baseline_text, "Report sections or sources missing")
    count, small, large, has_identity = check_pdf(pdf, student_id, student_name, title=TITLE)
    text = "\n".join(page.extract_text() for page in PdfReader(pdf).pages)
    require(all(section in text for section in SECTIONS) and "P(u)" in text and "\ufffd" not in text,
            "PDF text extraction indicates missing Korean headings or formula")
    if not has_identity:
        require("학번: ________" in text and "이름: ________" in text, "Public PDF identity fields are not blank")
    return f"{count} A4 pages, embedded text {small:.1f}–{large:.1f} pt, headings, formulas and references"


def legacy_audit():
    files = read_json(LEGACY_MANIFEST)["files"]
    require(files and all(sha256((REPO / name).read_bytes()).hexdigest() == digest for name, digest in files.items()),
            "A legacy output was altered or removed")
    return f"{len(files)} original result and report files matched the Git baseline byte-for-byte"


def validate_pink(out, pdf, manuscript, student_id=None, student_name=None, qa_out=None):
    ensure_result_destination(out)
    status = {}

    def batch(tests, function):
        try:
            result = function()
        except Exception as error:
            for ident, _ in tests:
                status[ident] = ("FAIL", f"{type(error).__name__}: {error}")
            return None
        for ident, note in tests:
            status[ident] = ("PASS", result if isinstance(result, str) else note)
        return result

    def blocked(tests):
        for ident in tests:
            status[ident] = ("SKIP", "Required upstream data check failed")

    payloads = batch([("T01", "100 unique six-character payloads reproduced")], lambda: payload_audit(out))
    if payloads is None:
        blocked((*[f"T{i:02}" for i in range(2, 10)], "T11", "T12", "T13"))
    else:
        orders = batch([
            ("T03", "All 2,000 mask orders and 15 prefix sizes verified"),
            ("T05", "Unique scattered cells and 4-neighbor growth at each cluster step"),
            ("T06", "Nested masks; all level rows checked against shared seed and coordinate order in the raw audit"),
        ], lambda: mask_audit(out, payloads))
        if orders is None:
            blocked(("T02", "T04", "T07", "T08", "T09", "T11", "T12"))
        else:
            batch([
                ("T02", "400 generated binary QR matrices, RGB dimensions and correct baseline records"),
                ("T04", "Exact #FF69B4 pixel counts and untouched sample quiet zones"),
                ("T07", "All original and 112,400 final RGB SHA-256 values reproduced"),
                ("T08", "400 distinct exact decodes at 0%; baseline evidence equals raw rows"),
                ("T09", "112,400 complete trial keys and decoder diagnostics verified"),
                ("T11", "Actual coverage recomputed from 441 cells, valid per-condition rates checked in T10"),
                ("T12", "Exact-match success and independent outcome flags checked for every attempt"),
            ], lambda: raw_audit(out, payloads, orders))
            if status["T09"][0] != "PASS" and status["T06"][0] == "PASS":
                status["T06"] = ("FAIL", "Mask orders pass, but shared use in all trial rows could not be verified")
        batch([("T13", "80 training and 20 validation payload IDs, disjoint and reproducible")],
              lambda: split_audit(out, payloads))

    def summary_check():
        summaries, coverage = collect_data(out)
        audit_summaries(out, summaries, coverage)
        return summaries

    summaries = batch([("T10", "All 116 condition aggregates verified against raw measurements")], summary_check)
    if summaries is None:
        blocked(("T14", "T15", "T16", "T17"))
    else:
        def compared_models():
            models, overview, holdout, predictions = build_models(summaries)
            saved = read_json(out / "processed/models.json")
            require(models == saved["curves"] and saved["run_id"] == read_json(out / "run_metadata.json")["run_id"],
                    "Stored models do not match training/holdout computed from raw")
            audit_linear(models, summaries)
            compare_csv(out / "processed/model_summary.csv", overview)
            return models

        models = batch([("T14", "All eight linear models match the training knots"),
                        ("T15", "Primitive and trapezoid integrals agree for all intervals")], compared_models)
        if models is not None:
            batch([("T16", "Four-point cubic interpolation, derivative minimum and model rejection checked")],
                  lambda: audit_cubics(models))
        else:
            blocked(("T16",))

        def holdout_check():
            _, _, holdout, predictions = build_models(summaries)
            require(len(holdout) == 8 and len(predictions) == 120 and
                    all((row["cubic_candidate_prediction_pct"] != "") == bool(row["in_cubic_window"])
                        for row in predictions), "Holdout predictions missing or extrapolated")
            compare_csv(out / "processed/holdout_metrics.csv", holdout)
            compare_csv(out / "processed/holdout_predictions.csv", predictions)

        batch([("T17", "Eight curve MAEs and 120 prediction rows match held-out data; no extrapolation")], holdout_check)
    batch([("T18", "Five readable PNGs with units, provenance, sample counts and legends")], lambda: figure_audit(out))
    batch([("T19", "School report validated")], lambda: pdf_audit(pdf, manuscript, student_id, student_name))
    batch([("T20", "Legacy experiment artifacts unchanged")], legacy_audit)
    destination = qa_out or out / "validation_report.md"
    if destination.resolve() in {(REPO / name).resolve() for name in read_json(LEGACY_MANIFEST)["files"]}:
        raise ValueError("Cannot overwrite an original experiment artifact")
    lines = ["# QRculus 2.0 자동 검증", "", "| 항목 | 결과 | 근거 |", "|---|---|---|"]
    for key, (state, note) in sorted(status.items()):
        print(f"{key} {state} {note}")
        note = " ".join(note.splitlines()).replace("|", "&#124;")
        lines.append(f"| {key} | {state} | {note} |")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return all(result[0] == "PASS" for result in status.values())
