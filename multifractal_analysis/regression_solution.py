from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from numpy import ndarray


@dataclass(slots=True)
class RegressionSolution:
    angular_coef: float
    linear_coef: float
    xpoints: ndarray
    ypoints: ndarray
    r_square: float

    def __init__(self, x: ndarray, y: ndarray) -> None:
        x_true = x.reshape((-1, 1))
        y_true = y.reshape((-1, 1))
        model = LinearRegression()
        model.fit(x_true, y_true)
        self.angular_coef = model.coef_.flatten()[0]
        if isinstance(model.intercept_, ndarray):
            self.linear_coef = model.intercept_.flatten()[0]
        elif isinstance(model.intercept_, float):
            self.linear_coef = model.intercept_
        else:
            self.linear_coef = 0.0
        self.xpoints = x
        self.ypoints = y
        r_square = r2_score(y_true, model.predict(x_true))
        self.r_square = r_square if isinstance(r_square, float) else 0.0
