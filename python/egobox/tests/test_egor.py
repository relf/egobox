import unittest
import numpy as np
from egobox import (
    Optimizer,
    RegressionSpec,
    CorrelationSpec,
    InfillStrategy,
    InfillOptimizer,
)
import time
import logging


def xsinx(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    y = (x - 3.5) * np.sin((x - 3.5) / (np.pi))
    print(f"obj={y} at {x}")
    return y


def G24(point):
    """
    Function G24
    1 global optimum y_opt = -5.5080 at x_opt =(2.3295, 3.1785)
    """
    p = np.atleast_2d(point)
    return -p[:, 0] - p[:, 1]


# Constraints < 0
def G24_c1(point):
    p = np.atleast_2d(point)
    return (
        -2.0 * p[:, 0] ** 4.0
        + 8.0 * p[:, 0] ** 3.0
        - 8.0 * p[:, 0] ** 2.0
        + p[:, 1]
        - 2.0
    )


def G24_c2(point):
    p = np.atleast_2d(point)
    return (
        -4.0 * p[:, 0] ** 4.0
        + 32.0 * p[:, 0] ** 3.0
        - 88.0 * p[:, 0] ** 2.0
        + 96.0 * p[:, 0]
        + p[:, 1]
        - 36.0
    )


# Grouped evaluation
def g24(point):
    p = np.atleast_2d(point)
    res = np.array([G24(p), G24_c1(p), G24_c2(p)]).T
    print(res)
    return res


def six_humps(x):
    """
    Function Six-Hump Camel Back
    2 global optimum value =-1.0316 located at (0.089842, -0.712656) and  (-0.089842, 0.712656)
    https://www.sfu.ca/~ssurjano/camel6.html
    """
    x = np.atleast_2d(x)
    x1 = x[:, 0]
    x2 = x[:, 1]
    print(x)
    sum1 = (
        4 * x1 ** 2
        - 2.1 * x1 ** 4
        + 1.0 / 3.0 * x1 ** 6
        + x1 * x2
        - 4 * x2 ** 2
        + 4 * x2 ** 4
    )
    print(np.atleast_2d(sum1).T)
    return np.atleast_2d(sum1).T


class TestOptimizer(unittest.TestCase):
    def test_xsinx(self):
        logging.basicConfig(format="%(message)s")
        logging.getLogger().setLevel(logging.DEBUG)
        egor = Optimizer(np.array([[0.0, 25.0]]))
        res = egor.minimize(xsinx, n_eval=15)
        print(f"Optimization f={res.y_opt} at {res.x_opt}")

    def test_g24(self):
        egor = Optimizer(
            np.array([[0.0, 3.0], [0.0, 4.0]]),
            infill_optimizer=InfillOptimizer.COBYLA,
            infill_strategy=InfillStrategy.EI,
            regr_spec=RegressionSpec.CONSTANT,
            corr_spec=CorrelationSpec.SQUARED_EXPONENTIAL,
        )
        start = time.process_time()
        res = egor.minimize(g24, 2, n_eval=29)
        end = time.process_time()
        print(f"Optimization f={res.y_opt} at {res.x_opt} in {end-start}s")

    def test_six_humps(self):
        egor = Optimizer(
            np.array([[-3.0, 3.0], [-2.0, 2.0]]),
            infill_strategy=InfillStrategy.WB2,
            regr_spec=RegressionSpec.CONSTANT,
            corr_spec=CorrelationSpec.SQUARED_EXPONENTIAL,
        )
        start = time.process_time()
        res = egor.minimize(six_humps, n_eval=45)
        end = time.process_time()
        print(f"Optimization f={res.y_opt} at {res.x_opt} in {end-start}s")

    def test_constructor(self):
        self.assertRaises(TypeError, Optimizer)
        Optimizer(np.array([[0.0, 25.0]]), 22, n_doe=10)


if __name__ == "__main__":
    unittest.main()
