"""Nested full-symbol occlusion and RGB rendering."""

from hashlib import sha256

import numpy as np

from .pink_payloads import RGB, SEED
from .qr_generator import BOX, QUIET, render_qr

SIDE = 21
MODULES = SIDE**2
EDGE = BOX * QUIET
SYMBOL_END = EDGE + BOX * SIDE


def covered_count(target):
    return round(MODULES * target / 100)


def actual_pct(k):
    return 100 * k / MODULES


def mask_seed(payload_id, repeat_id, mask_type):
    token = f"{SEED}|{payload_id}|{repeat_id}|{mask_type}".encode("ascii")
    return int.from_bytes(sha256(token).digest()[:8], "big", signed=False)


def adjacent(cell):
    y, x = divmod(cell, SIDE)
    if y:
        yield cell - SIDE
    if y + 1 < SIDE:
        yield cell + SIDE
    if x:
        yield cell - 1
    if x + 1 < SIDE:
        yield cell + 1


def mask_order(seed, mask_type):
    rng = np.random.default_rng(seed)
    if mask_type == "scattered":
        return rng.permutation(MODULES)
    if mask_type != "clustered":
        raise ValueError(f"Unknown occlusion shape: {mask_type}")
    start = int(rng.integers(MODULES))
    order = [start]
    seen = {start}
    frontier = list(adjacent(start))
    seen.update(frontier)
    while frontier:
        index = int(rng.integers(len(frontier)))
        cell = frontier[index]
        frontier[index] = frontier[-1]
        frontier.pop()
        order.append(cell)
        for neighbor in adjacent(cell):
            if neighbor not in seen:
                frontier.append(neighbor)
                seen.add(neighbor)
    return np.array(order, dtype=np.intp)


def mask_from_order(order, k):
    mask = np.zeros(MODULES, dtype=bool)
    mask[order[:k]] = True
    return mask.reshape(SIDE, SIDE)


def original_rgb(modules):
    return np.repeat(render_qr(modules)[..., np.newaxis], 3, axis=2)


def paint_cells(image, cells, *, color=RGB):
    for cell in cells:
        row, col = divmod(int(cell), SIDE)
        y, x = EDGE + row * BOX, EDGE + col * BOX
        image[y:y + BOX, x:x + BOX] = color


def cover_rgb(original, mask):
    image = original.copy()
    paint_cells(image, np.flatnonzero(mask))
    return image
