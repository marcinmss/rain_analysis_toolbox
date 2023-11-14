from typing import Tuple
from numpy import array, linspace, ndarray, log2, log10, empty, argmax
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
    field: ndarray, q: float, lims: Tuple[float, float] = (-2.0, 1.0)
) -> Tuple[ndarray, ndarray]:
    N = 34
    x, y = empty(N, dtype=float), empty(N, dtype=float)
    for i, eta in enumerate(10**i for i in linspace(lims[0], lims[1], N)):
        kqeta = get_kqeta(field, q, eta).angular_coef
        x[i], y[i] = log10(eta), log10(kqeta)
    return (x, y)


"""
Function for calculating C1 or alpha using DTM analysis
"""


def get_alpha_c1_from_dtm(field: ndarray, q: float) -> Tuple[float, float]:
    x, y = get_eta_vs_kqeta_points(field, q)
    search_area = array([i for i in range(x.size) if -0.8 < x[i] < 0.0])
    assert len(search_area) > 5

    n = 5
    lines = [RegressionSolution(x[i : i + n], y[i : i + n]) for i in search_area]
    best_line = lines[argmax([line.angular_coef for line in lines])]

    alpha = best_line.angular_coef
    b = best_line.linear_coef
    c1 = 10**b * (alpha - 1) / (q**alpha - q)
    return (alpha, c1)
