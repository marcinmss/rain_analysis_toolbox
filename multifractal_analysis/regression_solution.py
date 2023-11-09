from dataclasses import dataclass
from sklearn.linear_model import LinearRegression

from numpy import ndarray


dataclass(slots=True)


class RegressionSolution:
    angular_coef: float
    linear_coef: float
    xpoints: ndarray
    ypoints: ndarray

    def __init__(self, x: ndarray, y: ndarray) -> None:
        model = LinearRegression()
        model.fit(x.reshape((-1, 1)), y.reshape((-1, 1)))
        self.angular_coef = model.coef_.flatten()[0]
        self.linear_coef = model.intercept_
        self.xpoints = x
        self.ypoints = y
