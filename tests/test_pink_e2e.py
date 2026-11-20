from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from qr_calculus.config import read_json, write_json
from qr_calculus.decoder import decode_observation
from qr_calculus.experiment import read_csv
from qr_calculus.pink_experiment import ensure_result_destination, run_pink
from qr_calculus.pink_payloads import DEFAULT_CONFIG


class PinkEndToEndTests(unittest.TestCase):
    def test_smoke_and_resume_without_duplicate_baselines(self):
        with TemporaryDirectory() as stage, redirect_stdout(StringIO()):
            out = Path(stage)
            run_pink(DEFAULT_CONFIG, out, "smoke", workers=1)
            raw = out / "raw/trials.csv"
            initial = raw.read_bytes()
            rows = read_csv(raw)
            self.assertEqual(len(rows), 156)
            self.assertEqual(sum(r["mask_type"] == "none" for r in rows), 12)
            self.assertEqual(len(read_csv(out / "raw/mask_orders.csv")), 12)

            shard = out / ".pink-checkpoints/qr003.jsonl"
            lines = shard.read_bytes().splitlines(keepends=True)
            # Model a process interruption in the middle of the most recent row.
            shard.write_bytes(b"".join(lines[:-1]) + lines[-1][:len(lines[-1]) // 2])
            meta = read_json(out / "run_metadata.json")
            meta["status"] = "baseline_complete"
            write_json(out / "run_metadata.json", meta)
            run_pink(DEFAULT_CONFIG, out, "smoke", workers=1)
            self.assertEqual(raw.read_bytes(), initial)
            self.assertEqual(len(read_csv(out / "preflight.csv")), 12)

    def test_legacy_output_path_is_protected(self):
        with self.assertRaises(ValueError):
            ensure_result_destination(Path("results/final/attempt"))

    def test_detection_and_decoding_are_independent(self):
        detector = Mock()
        image = np.full((290, 290, 3), 255, dtype=np.uint8)
        rectangle = np.array([[[0, 0], [0, 10], [10, 10], [10, 0]]], dtype=np.float32)
        detector.detectAndDecode.return_value = "", rectangle, None
        self.assertEqual(decode_observation(image, "ABC123", detector), {
            "detected": 1, "decoded_text": "", "success": 0, "wrong_decode": 0,
        })
        detector.detectAndDecode.return_value = "XYZ123", rectangle, None
        self.assertEqual(decode_observation(image, "ABC123", detector)["wrong_decode"], 1)
        detector.detectAndDecode.side_effect = cv2.error("Invalid QR code source points; contourArea(src_points)")
        callback = Mock()
        self.assertEqual(decode_observation(image, "ABC123", detector, on_error=callback), {
            "detected": 0, "decoded_text": "", "success": 0, "wrong_decode": 0,
        })
        self.assertEqual(callback.call_args.args[0], "degenerate_source_points")
        detector.detectAndDecode.side_effect = cv2.error("Different decoder failure")
        with self.assertRaises(cv2.error):
            decode_observation(image, "ABC123", detector)
