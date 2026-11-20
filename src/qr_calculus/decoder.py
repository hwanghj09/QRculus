"""Exact-match decoding and classification."""

import cv2
import numpy as np

from .config import MESSAGE


def decode_status(text):
    if text == MESSAGE:
        return "correct"
    if text:
        return "wrong_text"
    return "not_detected"


def decode_qr(image: np.ndarray, detector=None) -> tuple[str, str]:
    if detector is None:
        detector = cv2.QRCodeDetector()
    text, _, _ = detector.detectAndDecode(image)
    return text, decode_status(text)


def decode_observation(image, payload, detector, *, on_error=None):
    """Classify valid geometry and string equality independently."""
    points = np.zeros((1, 4, 2), dtype=np.float32)
    try:
        text, points, _ = detector.detectAndDecode(image, points)
    except UnicodeDecodeError as error:
        # The binding may reject damaged, non-UTF-8 bytes even after OpenCV decodes them.
        text = error.object.decode("utf-8", errors="backslashreplace")
        if on_error is not None:
            on_error("non_utf8_payload", error.object.hex())
    except cv2.error as error:
        if "Invalid QR code source points" not in str(error) or "contourArea(src_points)" not in str(error):
            raise
        # This decoder exception explicitly identifies a degenerate, unusable contour.
        text, points = "", None
        if on_error is not None:
            on_error("degenerate_source_points", str(error).strip())
    detected = points is not None and (
        points.size == 8 and np.isfinite(points).all()
        and cv2.contourArea(points.reshape(4, 2).astype(np.float32, copy=False)) > 0
    )
    return {
        "detected": int(detected), "decoded_text": text, "success": int(text == payload),
        "wrong_decode": int(bool(text) and text != payload),
    }
