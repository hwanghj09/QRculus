"""Reserved module map for version 1. Coordinates exclude the quiet zone."""

import numpy as np

from .config import ELIGIBLE, RESERVED


def get_reserved_mask():
    reserved = np.zeros((21, 21), dtype=bool)
    reserved[0:8, 0:8] = True
    reserved[0:8, 13:21] = True
    reserved[13:21, 0:8] = True

    reserved[8, 0:9] = True
    reserved[8, 13:21] = True
    reserved[0:9, 8] = True
    reserved[13:21, 8] = True

    reserved[6, :] = True
    reserved[:, 6] = True
    reserved[13, 8] = True
    return reserved


def get_eligible_coords():
    reserved = get_reserved_mask()
    coords = np.argwhere(~reserved)
    if coords.shape != (ELIGIBLE, 2) or reserved.sum() != RESERVED:
        raise RuntimeError("Invalid version-1 reservation map")
    return coords
