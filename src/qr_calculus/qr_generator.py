"""Version-1 QR matrices, lossless rendering, and canonical byte hashes."""

from hashlib import sha256

import numpy as np
import qrcode
from PIL import Image
from qrcode import constants

from .config import DEFAULTS, MESSAGE

ERROR_LEVELS = {
    "L": constants.ERROR_CORRECT_L,
    "M": constants.ERROR_CORRECT_M,
    "Q": constants.ERROR_CORRECT_Q,
    "H": constants.ERROR_CORRECT_H,
}
QUIET = DEFAULTS["quiet_zone"]
BOX = DEFAULTS["box_size"]
SHAPE = (21, 21)


def make_qr(level: str, message: str = MESSAGE) -> np.ndarray:
    qr = qrcode.QRCode(
        version=1, error_correction=ERROR_LEVELS[level], box_size=BOX,
        border=QUIET, mask_pattern=1,
    )
    qr.add_data(message)
    qr.make(fit=False)
    modules = np.asarray(qr.modules, dtype=np.uint8)
    if modules.shape != SHAPE or np.any(modules > 1):
        raise ValueError("Expected a binary 21×21 QR matrix")
    return modules


def render_qr(modules: np.ndarray) -> np.ndarray:
    quiet_modules = np.pad(modules, QUIET, constant_values=0)
    pixels = np.repeat(np.repeat(quiet_modules, BOX, axis=0), BOX, axis=1)
    return (1 - pixels) * np.uint8(255)


def pixel_hash(array: np.ndarray) -> str:
    return sha256(array.tobytes(order="C")).hexdigest()


def save_png(path, image):
    Image.fromarray(image).save(path)
