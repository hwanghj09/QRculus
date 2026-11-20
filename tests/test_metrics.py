import unittest

from qr_calculus.metrics import analyze_piecewise, evaluate, fit_local_cubic, select_local_points


class MetricsTests(unittest.TestCase):
    def test_nonuniform_integration_and_earliest_exact_tie(self):
        rows = [
            {"changed_modules": 0, "correct_count": 100, "trials": 100},
            {"changed_modules": 1, "correct_count": 90, "trials": 100},
            {"changed_modules": 3, "correct_count": 70, "trials": 100},
        ]
        model = analyze_piecewise(rows)
        self.assertAlmostEqual(model["mean_success_pct"], 85)
        self.assertEqual(model["steepest_index"], 0)
        self.assertAlmostEqual(model["segments"][1]["slope"], -20.8)

    def test_pilot_example(self):
        model = fit_local_cubic([(8, 100), (9, 80), (10, 40), (11, 10)])
        self.assertEqual(model["coefficients"], [5, -25, 0, 100])
        self.assertAlmostEqual(model["min_derivative_u"], 5 / 3)
        self.assertEqual(model["local_mean_success_pct"], 58.75)
        self.assertTrue(model["valid"])
        self.assertEqual([evaluate(model["coefficients"], u) for u in range(4)], [100, 80, 40, 10])

    def test_overshoot_and_lower_degree(self):
        overshooting = fit_local_cubic([(2, 100), (3, 100), (4, 22), (5, 0)])
        self.assertFalse(overshooting["valid"])
        self.assertGreater(overshooting["model_success_range_pct"][1], 100)
        linear = fit_local_cubic([(4, 80), (5, 70), (6, 60), (7, 50)])
        self.assertEqual(linear["degree"], 1)
        self.assertEqual(linear["min_derivative_u"], 0)

    def test_nonconsecutive_segment_has_no_candidate(self):
        rows = [{"changed_modules": k, "success_pct": 10} for k in (14, 15, 16, 17, 21)]
        self.assertIsNone(select_local_points(rows, 3))
