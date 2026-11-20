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


def decode_observation(image, payload, detector):
    """Classify a BGR OpenCV image, including whether a finite quadrilateral was found."""
    text, points, _ = detector.detectAndDecode(image)
    detected = points is not None and (
        points.size == 8 and np.isfinite(points).all()
        and cv2.contourArea(points.reshape(4, 2).astype(np.float32, copy=False)) > 0
    )
    return {
        "detected": int(detected), "decoded_text": text, "success": int(text == payload),
        "wrong_decode": int(bool(text) and text != payload),
    }
