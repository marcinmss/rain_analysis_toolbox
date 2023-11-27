from typing import Tuple
from numpy import log, ndarray, absolute, arange, zeros
from numpy.fft import fft
from multifractal_analysis.general import is_power_of_2, kq_theoretical
from multifractal_analysis.regression_solution import RegressionSolution
from multifractal_analysis.double_trace_moment import get_um_params_dtm


"""
Produces the points for the spectral analysis
"""


def get_spectral_analysis_points(field: ndarray) -> Tuple[ndarray, ndarray]:
    assert is_power_of_2(field.size), "The field needs to be a power of 2"

    n = field.shape[0] // 2
    n_samples = field.shape[1]
    E = zeros(n, dtype=float)
    for sample in range(n_samples):
        buf = absolute(fft(field[:, sample]))[0:n]
        E += (buf * buf / n_samples).flatten()
    k = arange(1.0001, E.size + 1.0001, 1)
    return (log(k), log(E))


"""
Calculate the beta and H coeficients
"""


def get_beta_and_h(field_1d: ndarray) -> Tuple[float, float]:
    assert is_power_of_2(field_1d.shape[0]), "The field needs to be a power of 2"
    x, y = get_spectral_analysis_points(field_1d)
    regression_line = RegressionSolution(x, y)
    beta = -regression_line.angular_coef
    alpha, c1 = get_um_params_dtm(field_1d, 1.5)
    k_2 = kq_theoretical(2, alpha, c1, 0)
    h = 0.5 * (beta - 1 + k_2)
    return (beta, h)
