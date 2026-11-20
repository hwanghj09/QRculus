import unittest
from unittest.mock import Mock

import numpy as np

from qr_calculus.checks import valid_pixels
from qr_calculus.config import LEVELS
from qr_calculus.decoder import decode_qr
from qr_calculus.qr_generator import make_qr, render_qr


class GenerationTests(unittest.TestCase):
    def test_all_originals_decode_exactly(self):
        for level in LEVELS:
            with self.subTest(level=level):
                matrix = make_qr(level)
                self.assertEqual(matrix.shape, (21, 21))
                self.assertEqual(matrix.dtype, np.uint8)
                self.assertTrue(np.isin(matrix, [0, 1]).all())
                image = render_qr(matrix)
                self.assertTrue(valid_pixels(image))
                np.testing.assert_array_equal(255 - image[40:250:10, 40:250:10], matrix * 255)
                self.assertEqual(decode_qr(image), ("TEST01", "correct"))

    def test_decoder_preserves_nonmatching_text(self):
        image = render_qr(make_qr("H"))
        detector = Mock()
        for text, status in (("", "not_detected"), ("TEST01 ", "wrong_text"), ("test01", "wrong_text")):
            with self.subTest(text=text):
                detector.detectAndDecode.return_value = (text, None, None)
                self.assertEqual(decode_qr(image, detector), (text, status))
