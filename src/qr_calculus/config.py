"""Fixed experimental design and execution parameters."""

import json
from importlib.resources import files
from pathlib import Path

DEFAULTS = json.loads(files("qr_calculus").joinpath("defaults.json").read_text(encoding="utf-8"))
LEVELS = tuple(DEFAULTS["levels"])
TARGETS = tuple(DEFAULTS["target_pcts"])
SAMPLES = tuple(DEFAULTS["sample_targets"])
MESSAGE = DEFAULTS["message"]
ELIGIBLE = DEFAULTS["eligible_modules"]
RESERVED = DEFAULTS["reserved_modules"]
RNG_RULE = "numpy.random.default_rng(seed_base + trial).permutation(208), row-major coordinates"
HASH_RULE = "SHA-256 of 290×290 uint8 pixels in C order"


def load_config(path=None, **overrides):
    if path is None and Path("config.json").is_file():
        path = Path("config.json")
    config = read_json(Path(path)) if path is not None else DEFAULTS.copy()
    config.update({key: value for key, value in overrides.items() if value is not None})
    validate_config(config)
    return config


def validate_config(config):
    for key, expected in DEFAULTS.items():
        if key not in ("trials", "seed_base") and config.get(key) != expected:
            raise ValueError(f"Fixed configuration {key} must be {expected!r}")
    for key, minimum in (("trials", 1), ("seed_base", 0)):
        if type(config.get(key)) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer ≥ {minimum}")


def changed_count(target_pct):
    return round(ELIGIBLE * target_pct / 100)


def actual_pct(count):
    return 100 * count / ELIGIBLE


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, document):
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
