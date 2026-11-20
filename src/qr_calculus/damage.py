"""Seeded, nested XOR damage."""

import numpy as np


def permutation(size, seed):
    return np.random.default_rng(seed).permutation(size)


def damage_with_order(original, coords, k, order):
    if not 0 <= k <= len(coords):
        raise ValueError("Damage count is outside the eligible module count")
    damaged = original.copy()
    rows, columns = coords[order[:k]].T
    damaged[rows, columns] ^= 1
    return damaged


def damage_qr(original, coords, k, seed: int) -> np.ndarray:
    return damage_with_order(original, coords, k, permutation(len(coords), seed))
