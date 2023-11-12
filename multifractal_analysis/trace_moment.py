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
    outer_scale = int(log2(field_1d.shape[0]))
    x, y = empty(outer_scale, dtype=float), empty(outer_scale, dtype=float)
    for i, (lamb, scalled_array) in enumerate(upscale(field_1d)):
        x[i], y[i] = log2(lamb), log2(moment(scalled_array, q))

    return (x, y)


def get_k_of_q(field_1d: ndarray, q: float = 1.0):
    x, y = get_trace_moment_points(field_1d, q)
    return RegressionSolution(x, y)


"""
Calculate C1 and alpha
"""


def get_um_params_tm(field_1d: ndarray):
    kf = get_k_of_q(field_1d, 1.05).angular_coef
    k0 = get_k_of_q(field_1d, 0.95).angular_coef
    c1 = (kf - k0) / 0.1
    alpha = (0.1 / 0.05**2) * (kf + k0) / (kf - k0)
    return alpha, c1
