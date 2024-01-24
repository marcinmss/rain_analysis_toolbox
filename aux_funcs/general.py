from numpy import ndarray, pi, exp

# from scipy.interpolate import CubicSpline


"""
Basic formula for the volume of an spherical drop using the diameter.
"""


def volume_drop(diameter: float) -> float:
    return diameter**3 * pi / 6


"""
Formula for the Hermitte tendency line
"""


def V_D_Lhermitte_1988(d_mm: float | ndarray) -> float | ndarray:
    d_m = d_mm * 1e-3
    return 9.25 * (1 - exp(-1 * (68000 * (d_m**2) + 488 * d_m)))
