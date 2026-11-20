import unittest
from collections import Counter

import cv2
import numpy as np

from qr_calculus.config import LEVELS
from qr_calculus.decoder import decode_observation
from qr_calculus.pink_masks import original_rgb
from qr_calculus.pink_payloads import CHARSET, generate_payloads, make_pink_qr, split_payloads


class PinkPayloadTests(unittest.TestCase):
    def test_seed_reproduces_unique_payloads_and_80_20_split(self):
        rows = generate_payloads()
        self.assertEqual(rows, generate_payloads())
        labels = {r["payload"] for r in rows}
        self.assertEqual(len(rows), len(labels))
        self.assertEqual(len(labels), 100)
        self.assertTrue(all(len(label) == 6 and set(label) <= set(CHARSET) for label in labels))
        split = split_payloads(rows)
        self.assertEqual(split, split_payloads(reversed(rows)))
        self.assertEqual(Counter(split.values()), {"train": 80, "validation": 20})

    def test_explicit_alphanumeric_baselines(self):
        detector = cv2.QRCodeDetector()
        # Numeric-only data must also be encoded in the alphanumeric mode.
        for payload in ("123456", "ABC123", "ZZZZZZ"):
            for level in LEVELS:
                with self.subTest(payload=payload, level=level):
                    matrix = make_pink_qr(level, payload)
                    self.assertEqual(matrix.shape, (21, 21))
                    self.assertEqual(matrix.dtype, np.uint8)
                    pixels = original_rgb(matrix)
                    self.assertEqual(pixels.shape, (290, 290, 3))
                    self.assertEqual(decode_observation(pixels, payload, detector), {
                        "detected": 1, "decoded_text": payload, "success": 1, "wrong_decode": 0,
                    })
