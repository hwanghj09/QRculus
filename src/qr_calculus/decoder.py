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
