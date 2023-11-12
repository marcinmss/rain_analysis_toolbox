from typing import Tuple
from numpy import exp, linspace, ndarray, log2, empty, where, quantile
from multifractal_analysis.general import moment, upscale
from multifractal_analysis.regression_solution import RegressionSolution

"""
Function for the moment used in DTM
"""


def double_moment(field: ndarray, q: float, eta: float) -> float:
    return moment(field**eta / moment(field, eta), q)


"""
Functions for finding K(q, eta)
"""


def get_kqeta_points(field: ndarray, q: float, eta: float) -> Tuple[ndarray, ndarray]:
    outer_scale = int(log2(field.shape[0]))
    x, y = empty(outer_scale, dtype=float), empty(outer_scale, dtype=float)
    for i, (lamb, scalled_array) in enumerate(upscale(field)):
        x[i], y[i] = log2(lamb), log2(double_moment(scalled_array, q, eta))

    return (x, y)


def get_kqeta(field: ndarray, q: float, eta: float) -> RegressionSolution:
    x, y = get_kqeta_points(field, q, eta)
    return RegressionSolution(x, y)


"""
Function for getting the K(q, eta) points agains eta for DTM analysis
"""


def get_eta_vs_kqeta_points(
    field: ndarray, q: float, lims: Tuple[float, float] = (-20.0, 5.0)
) -> Tuple[ndarray, ndarray]:
    N = 50
    x, y = empty(N, dtype=float), empty(N, dtype=float)
    for i, eta in enumerate(2**i for i in linspace(lims[0], lims[1], N)):
        kqeta = get_kqeta(field, q, eta).angular_coef
        x[i], y[i] = log2(eta), log2(kqeta)
    return (x, y)


"""
Function for calculating C1 or alpha using DTM analysis
"""


def get_alpha_c1_from_dtm(field: ndarray, q: float) -> Tuple[float, float]:
    x, y = get_eta_vs_kqeta_points(field, q)
    left, right = quantile(y, 0.45), quantile(y, 0.80)
    area = where([left < v < right for v in y])
    sol = RegressionSolution(x[area], y[area])

    alpha = sol.angular_coef
    b = sol.linear_coef
    c1 = exp(b) * (alpha - 1) / (q**alpha - 1)
    return (alpha, c1)
