"""Run the fixed QR occlusion study with per-attempt checkpoints."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from itertools import product
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sqlite3
import subprocess

import cv2
import numpy as np

from .config import LEVELS, read_json, write_json
from .decoder import decode_observation
from .experiment import environment, write_csv
from .module_mask import get_reserved_mask
from .pink_masks import actual_pct, covered_count, mask_order, mask_seed, original_rgb, paint_cells
from .pink_payloads import (
    MASK_TYPES, PAYLOAD_FIELDS, PAYLOAD_RULE, PCT, REPEATS, RGB, RGB_SHA,
    SEED, SELECT_RULE, SPLIT_RULE, generate_payloads, load_pink_config, make_pink_qr, split_payloads,
)
from .qr_generator import save_png

TRIAL_FIELDS = (
    "run_id", "payload_id", "payload", "level", "mask_type", "repeat_id",
    "target_pct", "covered_modules", "actual_pct", "mask_seed", "start_cell",
    "covered_reserved", "covered_black", "covered_white", "detected", "decoded_text",
    "success", "wrong_decode", "original_pixel_sha256", "damaged_pixel_sha256",
)
MASK_FIELDS = ("payload_id", "repeat_id", "mask_type", "mask_seed", "start_cell", "cells_row_major_index")
SMOKE_PCT = (0, 5, 20, 40)
SMOKE_REPEATS = 2
REPO = Path(__file__).resolve().parents[2]
LEGACY = (REPO / "results/pilot", REPO / "results/final", REPO / "reports")


def ensure_result_destination(path):
    resolved = path.resolve()
    if resolved == REPO or any(
        resolved == protected or protected in resolved.parents or resolved in protected.parents
        for protected in LEGACY
    ):
        raise ValueError("Choose a dedicated output directory such as results/pink_v2")
    if (resolved / "trials.csv").exists():
        raise ValueError("Output contains a legacy run")


def commit_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def rgb_hash(image):
    return sha256(memoryview(image)).hexdigest()


def runtime():
    cv2.setNumThreads(1)
    return environment()


def write_metadata(path, document):
    temporary = path.with_suffix(".tmp")
    write_json(temporary, document)
    temporary.replace(path)


@contextmanager
def run_lock(out):
    conn = sqlite3.connect(out / ".pink-checkpoints" / "run-lock.sqlite", timeout=0)
    try:
        try:
            conn.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as error:
            raise RuntimeError(f"Another process is using {out}") from error
        yield
    finally:
        conn.close()


def checkpoints(out):
    return out / ".pink-checkpoints"


def sample_filename(payload_id, level, mask_type, target):
    return f"{payload_id}_{level}_{mask_type}_{target:g}pct.png"


def grid(targets, repeats):
    yield from (("none", 0, 0, level) for level in LEVELS)
    for rep, shape, target, level in product(range(repeats), MASK_TYPES, targets[1:], LEVELS):
        yield shape, rep, target, level


def row_key(row):
    return row["mask_type"], row["repeat_id"], row["target_pct"], row["level"]


def read_checkpoint(path, expected, meta, payload, *, repair=False):
    if not path.is_file():
        return []
    rows = []
    with path.open("rb+" if repair else "rb") as stream:
        while True:
            offset = stream.tell()
            encoded = stream.readline()
            if not encoded:
                break
            if not encoded.endswith(b"\n") and repair:
                # The last write may have been interrupted. Prior complete rows remain intact.
                stream.truncate(offset)
                break
            row = json.loads(encoded)
            if (len(rows) >= len(expected) or row_key(row) != expected[len(rows)]
                    or row["run_id"] != meta["run_id"] or row["payload_id"] != payload["payload_id"]
                    or row["payload"] != payload["payload"] or tuple(row) != TRIAL_FIELDS):
                raise ValueError(f"Incorrect checkpoint row {len(rows) + 1}: {path}")
            rows.append(row)
    return rows


def append_row(handle, row):
    handle.write((json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n").encode("ascii"))


def trial_row(meta, payload, level, shape, repeat, target, seed, start,
              reserved, black, observation, original_sha, damaged_sha):
    k = covered_count(target)
    return {
        "run_id": meta["run_id"], "payload_id": payload["payload_id"], "payload": payload["payload"],
        "level": level, "mask_type": shape, "repeat_id": repeat, "target_pct": target,
        "covered_modules": k, "actual_pct": actual_pct(k), "mask_seed": seed,
        "start_cell": start, "covered_reserved": reserved, "covered_black": black,
        "covered_white": k - black, **observation,
        "original_pixel_sha256": original_sha, "damaged_pixel_sha256": damaged_sha,
    }


def originals_for(payload):
    modules = {level: make_pink_qr(level, payload["payload"]) for level in LEVELS}
    images = {level: original_rgb(matrix) for level, matrix in modules.items()}
    hashes = {level: rgb_hash(image) for level, image in images.items()}
    return modules, images, hashes


def audit_baseline(out, payloads, meta, expected):
    path = checkpoints(out)
    detector = cv2.QRCodeDetector()
    complete = []
    for payload in payloads:
        pid = payload["payload_id"]
        shard = path / f"{pid}.jsonl"
        rows = read_checkpoint(shard, expected, meta, payload, repair=True)
        if len(rows) < 4:
            _, images, hashes = originals_for(payload)
            with shard.open("ab", buffering=0) as handle:
                for level in LEVELS[len(rows):]:
                    image, digest = images[level], hashes[level]
                    # Black, white and gray are identical in BGR and RGB.
                    obs = decode_observation(image, payload["payload"], detector)
                    row = trial_row(meta, payload, level, "none", 0, 0, "", "",
                                    0, 0, obs, digest, digest)
                    append_row(handle, row)
                    rows.append(row)
        complete.extend(rows[:4])
    write_csv(out / "preflight.csv", TRIAL_FIELDS, complete)
    failures = [f"{r['payload_id']}/{r['level']}" for r in complete if not r["success"] or not r["detected"]]
    if failures:
        raise ValueError("Baseline failed for " + ", ".join(failures) + "; see preflight.csv")
    return complete


def process_payload(out, payload, meta):
    cv2.setNumThreads(1)
    manifest = list(grid(meta["targets"], meta["repeats"]))
    pid = payload["payload_id"]
    path = checkpoints(out) / f"{pid}.jsonl"
    previous_rows = read_checkpoint(path, manifest, meta, payload, repair=True)
    skip = len(previous_rows)
    if skip == len(manifest):
        return pid
    matrices, origins, original_sha = originals_for(payload)
    if any(original_sha[row["level"]] != row["original_pixel_sha256"] for row in previous_rows[:4]):
        raise ValueError(f"Restored original differs from checkpoint: {pid}")

    detector = cv2.QRCodeDetector()
    reserved_flat = get_reserved_mask().ravel()
    black_masks = {level: m.ravel() for level, m in matrices.items()}
    emit_samples = pid == "qr001"
    if emit_samples:
        for level in LEVELS:
            save_png(out / "samples" / sample_filename(pid, level, "none", 0), origins[level])

    index = 4
    with path.open("ab", buffering=0) as handle:
        for repeat_id, shape in product(range(meta["repeats"]), MASK_TYPES):
            end = index + (len(meta["targets"]) - 1) * len(LEVELS)
            if skip >= end:
                index = end
                continue
            seed = mask_seed(pid, repeat_id, shape)
            order = mask_order(seed, shape)
            start = int(order[0])
            images = {level: original.copy() for level, original in origins.items()}
            previous_k = 0
            for target in meta["targets"][1:]:
                k = covered_count(target)
                cells = order[:k]
                reserved = int(reserved_flat[cells].sum())
                for level in LEVELS:
                    rgb = images[level]
                    paint_cells(rgb, order[previous_k:k])
                    if index >= skip:
                        try:
                            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                            obs = decode_observation(bgr, payload["payload"], detector)
                        except (cv2.error, UnicodeError) as error:
                            raise RuntimeError(f"Decoder failure at {pid}/{level}/{shape}/{repeat_id}/{target}") from error
                        digest = rgb_hash(rgb)
                        if emit_samples and repeat_id == 0 and target in (5, 20, 40):
                            save_png(out / "samples" / sample_filename(pid, level, shape, target), rgb)
                        row = trial_row(
                            meta, payload, level, shape, repeat_id, target, seed, start, reserved,
                            int(black_masks[level][cells].sum()), obs, original_sha[level], digest,
                        )
                        append_row(handle, row)
                    index += 1
                previous_k = k
    return pid


def finalize(out, payloads, meta, manifest):
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    def verified_rows():
        for payload in payloads:
            rows = read_checkpoint(checkpoints(out) / f"{payload['payload_id']}.jsonl", manifest, meta, payload)
            if len(rows) != len(manifest):
                raise ValueError(f"Incomplete checkpoint: {payload['payload_id']}")
            yield from rows

    temp = raw / "trials.tmp"
    write_csv(temp, TRIAL_FIELDS, verified_rows())
    temp.replace(raw / "trials.csv")

    def mask_rows():
        for payload in payloads:
            pid = payload["payload_id"]
            for rep, shape in product(range(meta["repeats"]), MASK_TYPES):
                seed = mask_seed(pid, rep, shape)
                order = mask_order(seed, shape)
                yield dict(zip(MASK_FIELDS, (
                    pid, rep, shape, seed, int(order[0]), " ".join(map(str, order)),
                )))

    temp = raw / "mask_orders.tmp"
    write_csv(temp, MASK_FIELDS, mask_rows())
    temp.replace(raw / "mask_orders.csv")


def run_pink(config_path, out, mode, workers=2):
    ensure_result_destination(out)
    if workers < 1:
        raise ValueError("--workers must be a positive integer")
    config = load_pink_config(config_path)
    out.mkdir(parents=True, exist_ok=True)
    payloads = generate_payloads()
    smoke = mode == "smoke"
    if smoke:
        payloads = payloads[:3]
    expected_targets = list(SMOKE_PCT if smoke else PCT)
    expected_repeats = SMOKE_REPEATS if smoke else REPEATS
    manifest = list(grid(expected_targets, expected_repeats))
    if mode != "preflight" and (out / "raw" / "trials.csv").is_file() and not checkpoints(out).exists():
        raise ValueError("Completed archives have no checkpoints; choose a new output directory to repeat")
    checkpoints(out).mkdir(exist_ok=True)
    meta_path = out / "run_metadata.json"
    config_sha = sha256(config_path.read_bytes()).hexdigest()
    version_data = runtime()
    with run_lock(out):
        if meta_path.exists():
            meta = read_json(meta_path)
            if (meta["config_sha256"] != config_sha or meta["kind"] != ("smoke" if smoke else "main")
                    or meta["environment"] != version_data):
                raise ValueError("Resume requires the original config, environment, and run type")
        else:
            created = datetime.now(UTC).isoformat()
            meta = {
                "run_id": created + ("-smoke" if smoke else "-pink"),
                "kind": "smoke" if smoke else "main", "created_utc": created,
                "git_commit": commit_sha(), "config_sha256": config_sha,
                "seed_base": SEED, "payload_generator": PAYLOAD_RULE, "split_rule": SPLIT_RULE,
                "cubic_selection_rule": SELECT_RULE, "hash_rule": RGB_SHA,
                "targets": expected_targets, "repeats": expected_repeats,
                "expected_calls": len(payloads) * len(manifest), "environment": version_data,
                "status": "baseline_pending",
            }
            write_metadata(meta_path, meta)
            (out / "config.json").write_bytes(config_path.read_bytes())
            write_json(out / "environment.json", version_data)
            write_csv(out / "payloads.csv", PAYLOAD_FIELDS, payloads)
            if not smoke:
                split = split_payloads(payloads)
                write_csv(out / "payload_split.csv", ("payload_id", "split"),
                          ({"payload_id": p["payload_id"], "split": split[p["payload_id"]]} for p in payloads))
        if meta["status"] == "complete":
            print(f"{meta['expected_calls']:,} calls already archived at {out}", flush=True)
            return
        with (out / "run.log").open("a", encoding="utf-8", buffering=1) as log:
            def progress(message):
                print(message, flush=True)
                log.write(f"{datetime.now(UTC).isoformat()} {message}\n")

            try:
                progress(f"{mode}: run {meta['run_id']}, workers={workers}")
                baseline = audit_baseline(out, payloads, meta, manifest)
                meta["baseline_success_rate_pct"] = 100.0
                meta["baseline_calls"] = len(baseline)
                meta["opencv_version"] = cv2.__version__
                progress(f"Baseline verified: {len(baseline)} exact decodes")
                meta["status"] = "baseline_complete"
                write_metadata(meta_path, meta)
                if mode == "preflight":
                    return

                (out / "samples").mkdir(exist_ok=True)
                with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
                    # Keep only one task per worker queued to allow clean interruption.
                    remaining = iter(payloads)
                    pending = set()
                    for payload in remaining:
                        pending.add(pool.submit(process_payload, out, payload, meta))
                        if len(pending) == workers:
                            break
                    finished = 0
                    while pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            pid = future.result()
                            finished += 1
                            if finished == 1 or finished % 5 == 0 or finished == len(payloads):
                                progress(f"{finished}/{len(payloads)} payloads completed ({pid})")
                            payload = next(remaining, None)
                            if payload is not None:
                                pending.add(pool.submit(process_payload, out, payload, meta))

                finalize(out, payloads, meta, manifest)
                meta["actual_calls"] = meta["expected_calls"]
                meta["status"] = "complete"
                meta["completed_utc"] = datetime.now(UTC).isoformat()
                write_metadata(meta_path, meta)
                progress(f"Archived {meta['actual_calls']:,} calls")
            except (Exception, KeyboardInterrupt) as error:
                progress(f"Interrupted with checkpoints retained: {type(error).__name__}: {error}")
                raise
