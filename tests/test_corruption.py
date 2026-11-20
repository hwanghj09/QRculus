import unittest

import numpy as np

from qr_calculus.config import LEVELS, TARGETS, changed_count
from qr_calculus.damage import damage_qr
from qr_calculus.module_mask import get_eligible_coords, get_reserved_mask
from qr_calculus.qr_generator import make_qr, pixel_hash, render_qr


class DamageTests(unittest.TestCase):
    def test_known_counts(self):
        self.assertEqual([changed_count(rate) for rate in (0, 2, 10)], [0, 4, 21])

    def test_shared_and_nested_damage(self):
        coords = get_eligible_coords()
        mask = get_reserved_mask()
        originals = [make_qr(level) for level in LEVELS]
        hashes = [pixel_hash(old) for old in originals]
        order = np.random.default_rng(2026).permutation(208)
        former = set()
        for target in TARGETS:
            k = changed_count(target)
            expected = set(map(tuple, coords[order[:k]]))
            self.assertTrue(former <= expected)
            former = expected
            for original in originals:
                damaged = damage_qr(original, coords, k, 2026)
                self.assertEqual(set(map(tuple, np.argwhere(damaged != original))), expected)
                np.testing.assert_array_equal(damaged[mask], original[mask])
        self.assertEqual([pixel_hash(old) for old in originals], hashes)

    def test_pixel_reproducibility(self):
        base = make_qr("Q")
        coords = get_eligible_coords()
        hashes = [pixel_hash(render_qr(damage_qr(base, coords, 4, seed))) for seed in (2026, 2026, 2027)]
        self.assertEqual(hashes[0], hashes[1])
        self.assertNotEqual(hashes[0], hashes[2])
