from hashlib import sha256
import unittest

import numpy as np

from qr_calculus.pink_masks import (
    EDGE, MODULES, SYMBOL_END, actual_pct, adjacent, cover_rgb, covered_count, mask_from_order,
    mask_order, mask_seed, original_rgb,
)
from qr_calculus.pink_payloads import MASK_TYPES, PCT, make_pink_qr


class PinkMaskTests(unittest.TestCase):
    def test_big_endian_seed_and_exact_nested_masks(self):
        for shape in MASK_TYPES:
            expected = int(sha256(f"2026|qr001|0|{shape}".encode()).hexdigest()[:16], 16)
            self.assertEqual(mask_seed("qr001", 0, shape), expected)
            order = mask_order(expected, shape)
            self.assertEqual(sorted(order), list(range(MODULES)))
            np.testing.assert_array_equal(order, mask_order(expected, shape))
            previous = np.zeros((21, 21), dtype=bool)
            for p in PCT:
                mask = mask_from_order(order, covered_count(p))
                self.assertEqual(mask.sum(), round(441 * p / 100))
                self.assertFalse((previous & ~mask).any())
                previous = mask

    def test_cluster_frontier_connects_every_prefix(self):
        order = mask_order(mask_seed("qr001", 9, "clustered"), "clustered")
        selected = {int(order[0])}
        for cell in map(int, order[1:]):
            self.assertTrue(any(n in selected for n in adjacent(cell)))
            selected.add(cell)

    def test_exact_rgb_draw_and_quiet_zone(self):
        original = original_rgb(make_pink_qr("H", "ABC123"))
        order = mask_order(mask_seed("qr001", 0, "scattered"), "scattered")
        k = covered_count(40)
        mask = mask_from_order(order, k)
        covered = cover_rgb(original, mask)
        self.assertAlmostEqual(actual_pct(k), 100 * round(441 * 0.4) / 441)
        np.testing.assert_array_equal(
            np.all(covered[EDGE:SYMBOL_END:10, EDGE:SYMBOL_END:10] == (255, 105, 180), axis=-1),
            mask,
        )
        self.assertEqual(np.all(covered == (255, 105, 180), axis=-1).sum(), k * 100)
        for current, before in zip(
            (covered[:EDGE], covered[SYMBOL_END:], covered[:, :EDGE], covered[:, SYMBOL_END:]),
            (original[:EDGE], original[SYMBOL_END:], original[:, :EDGE], original[:, SYMBOL_END:])
        ):
            np.testing.assert_array_equal(current, before)
        self.assertFalse(np.any(np.all(original == (255, 105, 180), axis=-1)))
