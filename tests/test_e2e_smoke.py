from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from qr_calculus.cli import main
from qr_calculus.config import load_config
from qr_calculus.decoder import decode_qr
from qr_calculus.experiment import run_experiment
from qr_calculus.figures import FIGURE_NAMES
from qr_calculus.validation import smoke_evidence


class SmokeTests(unittest.TestCase):
    def test_cli_experiment_analysis_and_overwrite_protection(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            out = Path(directory) / "smoke"
            command = ["experiment", "--trials", "2", "--out", str(out), "--seed-base", "2026"]
            self.assertEqual(main(command), 0)
            self.assertIn("144 attempts", smoke_evidence(out))
            before = (out / "trials.csv").read_bytes()
            self.assertEqual(main(command), 1)
            self.assertEqual((out / "trials.csv").read_bytes(), before)
            self.assertEqual(main(["analyze", "--input", str(out), "--out", str(out)]), 0)
            for name in FIGURE_NAMES:
                self.assertTrue((out / "figures" / name).is_file())

    def test_decode_exception_preserves_previous_results_and_logs_context(self):
        with TemporaryDirectory() as directory, redirect_stderr(StringIO()):
            out = Path(directory) / "run"
            out.mkdir()
            previous = out / "previous.txt"
            previous.write_text("previous run")
            calls = 0

            def failing_detector(image, detector):
                nonlocal calls
                calls += 1
                if calls == 6:
                    raise RuntimeError("decoder error for test")
                return decode_qr(image, detector)

            with patch("qr_calculus.experiment.decode_qr", side_effect=failing_detector):
                with self.assertRaisesRegex(RuntimeError, "trial=0, seed=2026, target=0.5"):
                    run_experiment({**load_config(trials=2), "out": out, "overwrite": True})
            self.assertEqual(previous.read_text(), "previous run")
            failed, = Path(directory).glob("run.failed-*")
            self.assertIn("decoder error for test", (failed / "run.log").read_text(encoding="utf-8"))
            self.assertEqual(len((failed / "partial_trials.csv").read_text(encoding="utf-8-sig").splitlines()), 2)
