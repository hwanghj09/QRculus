"""Fixed study configuration, deterministic payloads and QR encoding."""

from pathlib import Path

import numpy as np
from qrcode.util import MODE_ALPHA_NUM, QRData

from .config import LEVELS, read_json
from .qr_generator import make_qr

STUDY = "pink_occlusion_v2"
SEED = 2026
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
PCT = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 40)
MASK_TYPES = ("scattered", "clustered")
REPEATS = 10
PAYLOAD_COUNT = 100
TRAIN_COUNT = 80
RGB = (255, 105, 180)
RGB_SHA = "SHA-256 of contiguous 290×290×3 RGB uint8 pixels in C order"
DEFAULT_CONFIG = Path("configs/pink_occlusion_v2.json")
DEFAULT_RESULTS = Path("results/pink_v2")
PAYLOAD_FIELDS = ("payload_id", "payload", "encoding", "qr_version")
SELECT_RULE = "Earliest steepest interval; balance points on either side, then smallest x"
SPLIT_RULE = "PCG64 SeedSequence([2026, 2]); shuffle sorted payload IDs; first 80 train"
PAYLOAD_RULE = "PCG64 SeedSequence([2026, 1]); 6 independent charset indices; skip duplicate strings"


def load_pink_config(path):
    config = read_json(path)
    expected = {
        "study_id": STUDY, "seed_base": SEED,
        "payload": {
            "count": PAYLOAD_COUNT, "length": 6, "charset": CHARSET,
            "encoding_mode": "alphanumeric", "train_count": TRAIN_COUNT, "validation_count": 20,
        },
        "qr": {"version": 1, "mask_pattern": 1, "levels": list(LEVELS),
               "box_size": 10, "quiet_zone": 4, "fit": False},
        "occlusion": {
            "pixel_color_rgb": list(RGB), "opacity": 1.0, "region": "all_21x21_symbol_modules",
            "target_pcts": list(PCT), "mask_types": list(MASK_TYPES),
            "repetitions_per_nonzero_condition": REPEATS,
            "baseline_repetitions_per_payload_level": 1,
            "coordinate_order": "row-major",
            "rng_seed_derivation": "first_8_bytes_big_endian_sha256('2026|payload_id|repeat_id|mask_type')",
        },
        "decoder": "opencv_QRCodeDetector_detectAndDecode",
        "expected_total_decoder_calls": 112400,
        "preserve_legacy_results": True,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unexpected pink study setting: {key}")
    return config


def generate_payloads():
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 1]))
    seen = set()
    rows = []
    while len(rows) < PAYLOAD_COUNT:
        label = "".join(CHARSET[i] for i in rng.integers(len(CHARSET), size=6))
        if label not in seen:
            seen.add(label)
            rows.append({
                "payload_id": f"qr{len(rows) + 1:03d}", "payload": label,
                "encoding": "alphanumeric", "qr_version": 1,
            })
    return rows


def make_pink_qr(level, payload):
    return make_qr(level, QRData(payload, mode=MODE_ALPHA_NUM))


def split_payloads(rows):
    ids = sorted(row["payload_id"] for row in rows)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 2]))
    shuffled = rng.permutation(ids).tolist()
    return {payload_id: "train" if i < TRAIN_COUNT else "validation"
            for i, payload_id in enumerate(shuffled)}
