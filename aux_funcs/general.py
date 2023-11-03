from numpy import pi, exp

# from scipy.interpolate import CubicSpline


"""
Basic formula for the volume of an spherical drop using the diameter.
"""


def volume_drop(diameter: float) -> float:
    return diameter**3 * pi / 6


"""
Formula for the Hermitte tendency line
"""


def V_D_Lhermitte_1988(d_mm: float) -> float:
    d_m = d_mm * 1e-3
    return 9.25 * (1 - exp(-1 * (68000 * (d_m**2) + 488 * d_m)))


"""
Spline aproximation for the Hemitte tendency line using the parcivel classes
"""


# def hermitte_line_parsivel_classes():
#     from parsivel.matrix_classes import CLASSES_DIAMETER_MIDDLE
#     from stereo3d.convert_to_parsivel import find_velocity_class
#
#     x = arange(1, 33, 1)
#     y = array(
#         [
#             find_velocity_class(V_D_Lhermitte_1988(diam))
#             for diam in CLASSES_DIAMETER_MIDDLE
#         ]
#     )
#     cs = CubicSpline(x, y)
#
#     pass
