from typing import Tuple
from numpy import log, ndarray, absolute, arange
from numpy.fft import fft
from multifractal_analysis.general import is_power_of_2
from multifractal_analysis.regression_solution import RegressionSolution


"""
Produces the points for the spectral analysis
"""


def get_spectral_analysis_points(field_1d: ndarray) -> Tuple[ndarray, ndarray]:
    assert is_power_of_2(field_1d.size), "The field needs to be a power of 2"

    n = field_1d.size // 2
    buf = absolute(fft(field_1d))[0:n]
    E = buf * buf
    k = arange(1, E.size + 1, 1)
    return (log(k), log(E))


"""
Calculate the beta and H coeficients
"""


def get_beta_and_h(field_1d: ndarray) -> Tuple[float, float]:
    assert is_power_of_2(field_1d.size), "The field needs to be a power of 2"
    x, y = get_spectral_analysis_points(field_1d)
    regression_line = RegressionSolution(x, y)
    beta = -regression_line.angular_coef
    # TODO: get the formula for the h coeficient
    return (beta, beta)
