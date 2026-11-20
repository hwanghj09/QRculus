import unittest

import numpy as np
import qrcode
from qrcode import constants

from qr_calculus.module_mask import get_eligible_coords, get_reserved_mask


class MaskTests(unittest.TestCase):
    def test_mask_matches_qrcode_pre_data_function_layout(self):
        qr = qrcode.QRCode(version=1, error_correction=constants.ERROR_CORRECT_H)
        qr.modules_count = 21
        qr.modules = [[None] * 21 for _ in range(21)]
        for row, column in ((0, 0), (0, 14), (14, 0)):
            qr.setup_position_probe_pattern(row, column)
        qr.setup_timing_pattern()
        qr.setup_type_info(False, 1)
        expected = np.asarray([[cell is not None for cell in row] for row in qr.modules])
        np.testing.assert_array_equal(get_reserved_mask(), expected)
        np.testing.assert_array_equal(get_eligible_coords(), np.argwhere(~expected))
        self.assertEqual(int(expected.sum()), 233)
        self.assertEqual(get_eligible_coords().shape, (208, 2))
