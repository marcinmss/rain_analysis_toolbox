from math import log2
from typing import Tuple
from numpy import empty, ndarray
from multifractal_analysis.general import upscale
from multifractal_analysis.regression_solution import RegressionSolution
from multifractal_analysis.general import moment


"""
Function for getting the TM points
"""


def get_trace_moment_points(
    field_1d: ndarray, q: float = 1.0
) -> Tuple[ndarray, ndarray]:
    outer_scale = int(log2(field_1d.size)) + 1
    x, y = empty(outer_scale, dtype=float), empty(outer_scale, dtype=float)
    for i, (lamb, scalled_array) in enumerate(upscale(field_1d)):
        x[i], y[i] = log2(lamb), log2(moment(scalled_array, q))

    return (x, y)


def k_of_q(field_1d: ndarray, q: float = 1.5):
    x, y = get_trace_moment_points(field_1d, q)
    return RegressionSolution(x, y).angular_coef


"""
Calculate C1 and alpha
"""


def get_um_params_tm(field_1d: ndarray):
    h = 0.1
    kf = k_of_q(field_1d, 1.0 + h)
    k0 = k_of_q(field_1d, 1.0 - h)
    c1 = (kf - k0) / (2 * h)
    alpha = (kf + k0) / (h**2 * c1)
    return alpha, c1
