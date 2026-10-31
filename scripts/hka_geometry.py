"""Oriented HKA shared by the inference pipeline and validation audits.

Coordinates: x increases rightwards, y downwards. In a standard frontal
radiograph, patient right (Der) is on image left. Both axis vectors point
distally. Negative means varus and positive valgus on either limb.
"""

import math


def signed_hka_from_axes(points, side):
    """Return degrees from P0->P7 and P1->P2, independent of joint tangents.

    Raise ValueError for absent/nonfinite points, zero-length axes, an unknown
    side or axes that do not both point downwards in the stated convention.
    Mirrored/rotated images need explicit orientation handling before use.
    """
    if side not in ("Der", "Izq"):
        raise ValueError("side must be Der or Izq")
    try:
        p = {i: tuple(float(v) for v in points[i]) for i in (0, 7, 1, 2)}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("P0, P7, P1 and P2 are required") from exc
    if any(len(pt) != 2 or not all(math.isfinite(v) for v in pt) for pt in p.values()):
        raise ValueError("Points must be finite x/y pairs")
    fx, fy = p[7][0] - p[0][0], p[7][1] - p[0][1]
    tx, ty = p[2][0] - p[1][0], p[2][1] - p[1][1]
    if math.hypot(fx, fy) <= 1e-12 or math.hypot(tx, ty) <= 1e-12:
        raise ValueError("Degenerate axis")
    if fy <= 0 or ty <= 0:
        raise ValueError("Expected both axes to point distally/downwards")
    angle = math.degrees(math.atan2(fx * ty - fy * tx, fx * tx + fy * ty))
    if abs(angle) >= 90:
        raise ValueError("Axes diverge by at least 90 degrees; review geometry")
    return angle if side == "Der" else -angle
