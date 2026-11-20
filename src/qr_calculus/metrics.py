"""Piecewise linear and local cubic interpolation with elementary calculus."""

from fractions import Fraction
from math import sqrt

from .config import ELIGIBLE, actual_pct


def analyze_piecewise(rows) -> dict:
    points = sorted(
        (Fraction(100 * int(row["changed_modules"]), ELIGIBLE),
         Fraction(100 * int(row["correct_count"]), int(row["trials"])))
        for row in rows
    )
    segments = []
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x1 <= x0:
            raise ValueError("Actual damage rates must be strictly increasing")
        slope = (y1 - y0) / (x1 - x0)
        segments.append({
            "start_pct": float(x0), "end_pct": float(x1),
            "start_success_pct": float(y0), "end_success_pct": float(y1),
            "slope": slope, "area": (x1 - x0) * (y0 + y1) / 2,
        })
    if not segments:
        raise ValueError("At least two distinct measured points are required")
    steepest_index = min(range(len(segments)), key=lambda i: segments[i]["slope"])
    area = sum(segment["area"] for segment in segments)
    return {
        "segments": [
            {key: float(value) for key, value in segment.items()} for segment in segments
        ],
        "steepest_index": steepest_index,
        "area": float(area),
        "mean_success_pct": float(area / (points[-1][0] - points[0][0])),
        "range_pct": [float(points[0][0]), float(points[-1][0])],
    }


def evaluate(coefficients, u):
    a, b, c, d = coefficients
    return ((a * u + b) * u + c) * u + d


def derivative(coefficients, u):
    a, b, c, _ = coefficients
    return (3 * a * u + 2 * b) * u + c


def stationary_points(a, b, c):
    """Real roots of 3a·u²+2b·u+c, handling lower-degree polynomials."""
    if a == 0:
        return [] if b == 0 else [-c / (2 * b)]
    discriminant = (2 * b) ** 2 - 12 * a * c
    if discriminant < 0:
        return []
    root = sqrt(discriminant)
    return [(-2 * b - root) / (6 * a), (-2 * b + root) / (6 * a)]


def equation(coefficients, symbol="u"):
    terms = []
    for coefficient, suffix in zip(coefficients, (f"{symbol}³", f"{symbol}²", symbol, "")):
        if coefficient == 0:
            continue
        magnitude = abs(Fraction(coefficient).limit_denominator(600))
        sign = "−" if coefficient < 0 else "+"
        factor = "" if magnitude == 1 and suffix else str(magnitude)
        if magnitude.denominator != 1 and suffix:
            factor = f"({factor})"
        terms.append((sign, factor + suffix))
    if not terms:
        return "0"
    first_sign, first_term = terms[0]
    return ("−" if first_sign == "−" else "") + first_term + "".join(
        f" {sign} {term}" for sign, term in terms[1:]
    )


def fit_local_cubic(four_points) -> dict:
    points = list(four_points)
    if len(points) != 4 or [k for k, _ in points] != list(range(points[0][0], points[0][0] + 4)):
        raise ValueError("Local interpolation requires four consecutive integer module counts")
    k0 = points[0][0]
    y0, y1, y2, y3 = [Fraction(str(y)) for _, y in points]
    # Subtract successive interpolation equations to solve for a, b, c, d.
    a = (y3 - 3 * y2 + 3 * y1 - y0) / 6
    b = (y2 - 2 * y1 + y0 - 6 * a) / 2
    c = y1 - y0 - a - b
    d = y0
    coefficients = a, b, c, d

    candidates = [Fraction(0), Fraction(3)]
    vertex = -b / (3 * a) if a else None
    if vertex is not None and 0 < vertex < 3:
        candidates.append(vertex)
    minimum_u = min(candidates, key=lambda u: derivative(coefficients, u))
    minimum_slope = derivative(coefficients, minimum_u)
    critical = [u for u in stationary_points(a, b, c) if 0 < u < 3]
    values = [evaluate(coefficients, u) for u in [0, *critical, 3]]
    increasing = any(y1 > y0 for (_, y0), (_, y1) in zip(points, points[1:]))
    max_slope = max(derivative(coefficients, u) for u in candidates)
    issues = []
    if min(values) < -1e-8 or max(values) > 100 + 1e-8:
        issues.append("유효 구간에서 성공률이 0~100% 범위를 벗어남")
    if increasing or max_slope > 1e-8:
        issues.append("유효 구간에 성공률이 증가하는 부분이 있음")
    if minimum_slope >= 0:
        issues.append("감소 구간 없음")

    integral = a * 3**4 / 4 + b * 3**3 / 3 + c * 3**2 / 2 + d * 3
    return {
        "points": [{"k": k, "success_pct": float(y)} for k, y in points],
        "k0": k0,
        "range_pct": [actual_pct(k0), actual_pct(k0 + 3)],
        "coefficients": [float(value) for value in coefficients],
        "exact_coefficients": [str(value) for value in coefficients],
        "derivative_coefficients": [float(3 * a), float(2 * b), float(c)],
        "equation": equation(coefficients),
        "derivative_equation": equation((0, 3 * a, 2 * b, c)),
        "degree": 3 if a else 2 if b else 1 if c else 0,
        "vertex_u": float(vertex) if vertex is not None else None,
        "min_derivative_u": float(minimum_u),
        "min_derivative": float(minimum_slope),
        "min_derivative_per_damage_pct": float(minimum_slope * ELIGIBLE / 100),
        "min_derivative_pct": actual_pct(k0 + float(minimum_u)),
        "interior_minimum": 0 < minimum_u < 3,
        "local_mean_success_pct": float(integral / 3),
        "model_success_range_pct": [float(min(values)), float(max(values))],
        "valid": not issues,
        "issues": issues,
    }


def select_local_points(rows, steepest_index):
    points = [(int(row["changed_modules"]), float(row["success_pct"])) for row in rows]
    low, high = [k for k, _ in points[steepest_index:steepest_index + 2]]
    windows = [points[i:i + 4] for i in range(len(points) - 3)]
    candidates = [
        window for window in windows
        if window[0][0] <= low and high <= window[-1][0]
        and window[-1][0] - window[0][0] == 3
    ]
    if not candidates:
        return None
    # Prefer the window whose center is closest to the steepest segment's center.
    return min(candidates, key=lambda window: (
        abs(window[0][0] + window[-1][0] - low - high), window[0][0],
    ))
