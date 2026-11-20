"""Execute OpenCV decoding attempts and save a reproducible run."""

import csv
import logging
import platform
import shutil
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import mkdtemp
from uuid import uuid4

import cv2

from .checks import preflight
from .config import (
    ELIGIBLE, HASH_RULE, LEVELS, RNG_RULE, SAMPLES, TARGETS, actual_pct,
    changed_count, validate_config, write_json,
)
from .damage import damage_with_order, permutation
from .decoder import decode_qr
from .module_mask import get_eligible_coords
from .qr_generator import make_qr, pixel_hash, render_qr, save_png

TRIAL_FIELDS = (
    "run_id", "level", "trial", "seed", "target_pct", "changed_modules",
    "eligible_modules", "actual_pct", "decoded_text", "correct",
    "decode_status", "original_sha256", "damaged_sha256",
)
SUMMARY_FIELDS = (
    "level", "target_pct", "changed_modules", "actual_pct", "correct_count",
    "trials", "success_pct", "first_fail_count",
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def environment():
    cpu = platform.processor() or platform.machine()
    if sys.platform == "linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "cpu": cpu,
        "opencv_runtime": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "library_versions": {
            name: version(name) for name in
            ("opencv-python", "numpy", "qrcode", "matplotlib", "pillow")
        },
    }


def capture_run(config, stage, log):
    config = {**config, "rng_rule": RNG_RULE, "hash_rule": HASH_RULE}
    write_json(stage / "config.json", config)
    write_json(stage / "environment.json", environment())
    images = stage / "images"
    images.mkdir()
    coords = get_eligible_coords()
    detector = cv2.QRCodeDetector()
    originals = {level: make_qr(level) for level in LEVELS}
    origin_hashes = {}
    matrices = {}
    for level, original in originals.items():
        pixels = render_qr(original)
        save_png(images / f"original_{level}.png", pixels)
        text, status = decode_qr(pixels, detector)
        origin_hashes[level] = pixel_hash(pixels)
        matrices[level] = {
            "matrix": ["".join(map(str, row)) for row in original],
            "matrix_sha256": pixel_hash(original),
            "pixel_sha256": origin_hashes[level],
        }
        if status != "correct":
            raise RuntimeError(f"Original {level} failed preflight: {status}, {text!r}")
    write_json(stage / "original_matrices.json", matrices)
    failures = [key for key, (ok, _) in preflight(config["seed_base"]).items() if not ok]
    if failures:
        raise RuntimeError(f"Preflight failed: {', '.join(failures)}")
    log.info("Preflight T01–T11 passed")
    counts = Counter()
    rows = []
    for level, original in originals.items():
        for trial in range(config["trials"]):
            seed = config["seed_base"] + trial
            order = permutation(len(coords), seed)
            for target in TARGETS:
                k = changed_count(target)
                pixels = render_qr(damage_with_order(original, coords, k, order))
                try:
                    text, status = decode_qr(pixels, detector)
                except Exception as error:
                    if rows:
                        write_csv(stage / "partial_trials.csv", TRIAL_FIELDS, rows)
                    raise RuntimeError(
                        f"Decode error at level={level}, trial={trial}, seed={seed}, target={target}"
                    ) from error
                correct = int(status == "correct")
                counts[level, target] += correct
                rows.append(dict(zip(TRIAL_FIELDS, (
                    config["run_id"], level, trial, seed, target, k, ELIGIBLE,
                    actual_pct(k), text, correct, status, origin_hashes[level], pixel_hash(pixels),
                ))))
                if trial == 0 and level == "H" and target in SAMPLES:
                    save_png(images / f"sample_H_{target:g}pct.png", pixels)
        log.info("%s completed, %d/%d pristine decodes", level, counts[level, 0], config["trials"])

    n = config["trials"]
    summary = [
        dict(zip(SUMMARY_FIELDS, (
            level, target, changed_count(target), actual_pct(changed_count(target)),
            counts[level, target], n, 100 * counts[level, target] / n, n - counts[level, target],
        )))
        for level in LEVELS for target in TARGETS
    ]
    write_csv(stage / "trials.csv", TRIAL_FIELDS, rows)
    write_csv(stage / "success_rate.csv", SUMMARY_FIELDS, summary)
    if any(counts[level, 0] != n for level in LEVELS):
        raise RuntimeError("Zero-damage results include decoding failures")
    log.info("Completed: %d attempts and %d condition summaries", len(rows), len(summary))


def promote_run(stage, out, overwrite):
    lock = out.with_name(f".{out.name}.qr-lock")
    lock.mkdir()
    try:
        previous = stage.parent / "previous"
        if out.exists():
            if not out.is_dir() or out.is_symlink():
                raise ValueError(f"Output must be a directory: {out}")
            if any(out.iterdir()) and not overwrite:
                raise FileExistsError(f"Output became nonempty during the experiment: {out}")
            out.rename(previous)
        try:
            stage.rename(out)
        except OSError:
            if previous.exists():
                previous.rename(out)
            raise
    finally:
        lock.rmdir()


def run_experiment(config) -> tuple[Path, Path]:
    validate_config(config)
    out = Path(config["out"]).absolute()
    if out.exists() and (not out.is_dir() or out.is_symlink()):
        raise ValueError(f"Output must be a directory: {out}")
    overwrite = config.get("overwrite", False)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} is not empty; use --overwrite or choose a new directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    # A sibling staging directory ensures renames stay on the same filesystem.
    stage_root = Path(mkdtemp(prefix=".qr-run-", dir=out.parent))
    stage = stage_root / "run"
    stage.mkdir()
    suffix = uuid4().hex[:12]
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + suffix
    saved_config = {key: val for key, val in config.items() if key not in ("out", "overwrite")}
    saved_config["run_id"] = run_id
    log = logging.getLogger(f"qr_calculus.run.{run_id}")
    log.setLevel(logging.INFO)
    log.propagate = False
    handler = logging.FileHandler(stage / "run.log", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    log.addHandler(handler)
    try:
        log.info("Started %s, %d trials per condition", run_id, config["trials"])
        capture_run(saved_config, stage, log)
    except Exception:
        log.exception("Run aborted")
        failed = out.with_name(f"{out.name}.failed-{suffix}")
        handler.close()
        stage.rename(failed)
        print(f"Failure evidence: {failed}", file=sys.stderr)
        raise
    finally:
        log.removeHandler(handler)
        handler.close()
        if not stage.exists():
            shutil.rmtree(stage_root)
    try:
        promote_run(stage, out, overwrite)
    except (OSError, ValueError):
        # Keep completed output if a concurrent run won the destination.
        recovered = out.with_name(f"{out.name}.recovered-{suffix}")
        stage.rename(recovered)
        print(f"Completed outputs recovered at {recovered}", file=sys.stderr)
        if (stage_root / "previous").exists():
            print(f"Previous output preserved at {stage_root / 'previous'}", file=sys.stderr)
        else:
            stage_root.rmdir()
        raise
    shutil.rmtree(stage_root)
    return out / "trials.csv", out / "success_rate.csv"
