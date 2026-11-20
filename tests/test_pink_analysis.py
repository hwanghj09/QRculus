from fractions import Fraction
import unittest

from qr_calculus.metrics import evaluate
from qr_calculus.pink_analysis import fit_cubic, piecewise_model, select_window


class PinkAnalysisTests(unittest.TestCase):
    def test_uneven_intervals_and_tie_handling(self):
        points = [(Fraction(x), Fraction(y)) for x, y in [(0, 100), (1, 90), (3, 70), (5, 30)]]
        model = piecewise_model(points)
        self.assertEqual(model["steepest_index"], 2)
        self.assertEqual(model["segments"][2]["slope"], -20.0)
        self.assertEqual(model["area"], 355.0)
        self.assertEqual(model["mean_success_pct"], 71.0)
        self.assertEqual(model["segments"][1]["area"], model["segments"][1]["trapezoid"])

    def test_balanced_window_near_interior_and_boundary(self):
        points = list(range(8))
        self.assertEqual(select_window(points, 4), (3, [3, 4, 5, 6]))
        self.assertEqual(select_window(points, 0), (0, [0, 1, 2, 3]))
        self.assertEqual(select_window(points, 6), (4, [4, 5, 6, 7]))

    def test_uneven_cubic_derivative_and_range(self):
        points = [(Fraction(10 + u), Fraction(100 - u**3)) for u in (0, 1, 2, 4)]
        model = fit_cubic(points)
        self.assertEqual(model["coefficients"], [-1, 0, 0, 100])
        self.assertEqual(model["derivative_coefficients"], [-3, 0, 0])
        self.assertEqual(model["min_slope_pct"], 14)
        self.assertEqual(model["min_slope"], -48)
        self.assertTrue(model["valid"])
        self.assertEqual([evaluate(model["coefficients"], x - 10) for x, _ in points], [100, 99, 92, 36])

    def test_reject_positive_overshoot(self):
        points = [(Fraction(x), Fraction(y)) for x, y in [(0, 100), (1, 100), (2, 20), (3, 0)]]
        model = fit_cubic(points)
        self.assertFalse(model["valid"])
        self.assertIn("success_outside_0_to_100", model["issues"])
        self.assertIn("increasing_within_decline_window", model["issues"])
