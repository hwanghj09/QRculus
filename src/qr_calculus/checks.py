"""Preflight checks against generated QR matrices, pixels, and the decoder."""

import numpy as np

from .config import ELIGIBLE, LEVELS, MESSAGE, RESERVED, TARGETS, changed_count
from .damage import damage_qr
from .decoder import decode_qr, decode_status
from .module_mask import get_eligible_coords, get_reserved_mask
from .qr_generator import BOX, QUIET, make_qr, pixel_hash, render_qr

EDGE = BOX * QUIET


def valid_pixels(image):
    return (
        image.shape == (290, 290)
        and image.dtype == np.uint8
        and np.isin(image, [0, 255]).all()
        and all((edge == 255).all() for edge in
                (image[:EDGE], image[-EDGE:], image[:, :EDGE], image[:, -EDGE:]))
    )


def preflight(seed_base=2026):
    mask = get_reserved_mask()
    coords = get_eligible_coords()
    originals = [make_qr(level) for level in LEVELS]
    pixels = [render_qr(original) for original in originals]
    original_hashes = [pixel_hash(original) for original in originals]
    reserved_intact = True
    equal_across_levels = True
    nested = True
    counts_match = True
    for seed in range(seed_base, seed_base + 3):
        previous = set()
        for target in TARGETS:
            k = changed_count(target)
            changes_by_level = []
            for original, original_pixels in zip(originals, pixels):
                damaged = damage_qr(original, coords, k, seed)
                changed = damaged != original
                counts_match &= changed.sum() == k
                reserved_intact &= not changed[mask].any() and valid_pixels(render_qr(damaged))
                if k == 0:
                    counts_match &= np.array_equal(original_pixels, render_qr(damaged))
                changes_by_level.append(set(map(tuple, np.argwhere(changed))))
            first = changes_by_level[0]
            equal_across_levels &= all(positions == first for positions in changes_by_level[1:])
            nested &= previous <= first
            previous = first
    sample_a = render_qr(damage_qr(originals[0], coords, 4, seed_base))
    sample_b = render_qr(damage_qr(originals[0], coords, 4, seed_base))
    return {
        "T01": (mask.sum() == RESERVED and coords.shape == (ELIGIBLE, 2), "233 reserved / 208 eligible"),
        "T02": (all(o.shape == (21, 21) and o.dtype == np.uint8 and np.isin(o, [0, 1]).all()
                    for o in originals), "Four 21×21 binary matrices"),
        "T03": (all(map(valid_pixels, pixels)), "290×290 grayscale, 40px white borders"),
        "T04": (all(decode_qr(image) == (MESSAGE, "correct") for image in pixels),
                "Four original images decoded with OpenCV"),
        "T05": (counts_match and [changed_count(t) for t in (0, 2, 10)] == [0, 4, 21],
                "0%, 2%, 10% → 0, 4, 21; 0% pixels unchanged"),
        "T06": (reserved_intact, "Reserved modules and quiet zone preserved"),
        "T07": ([pixel_hash(o) for o in originals] == original_hashes, "Matrix hashes unchanged"),
        "T08": (pixel_hash(sample_a) == pixel_hash(sample_b), "Same seed yields same pixel hash"),
        "T09": (equal_across_levels, "Shared coordinates across levels"),
        "T10": (nested, "Nested damage across the target grid"),
        "T11": ([decode_status(s) for s in (MESSAGE, "", "test01", MESSAGE + " ")]
                == ["correct", "not_detected", "wrong_text", "wrong_text"], "Exact-match classification"),
    }
