import unittest
import numpy as np
from egobox import SegoOptimizer, RegressionSpec


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
def f_g24(point):
    print(point)
    p = np.atleast_2d(point)
    res = np.array([G24(p), G24_c1(p), G24_c2(p)]).T
    print(res)
    return res


class TestSego(unittest.TestCase):
    def test_xsinx(self):
        sego = SegoOptimizer(np.array([[0.0, 25.0]]))
        res = sego.minimize(xsinx)
        print(f"Optimization f={res.y_opt} at {res.x_opt}")

    def test_g24(self):
        sego = SegoOptimizer(np.array([[0.0, 3.0], [0.0, 4.0]]))
        res = sego.minimize(f_g24, 2)
        print(f"Optimization f={res.y_opt} at {res.x_opt}")

    def test_constructor(self):
        self.assertRaises(TypeError, SegoOptimizer)
        SegoOptimizer(np.array([[0.0, 25.0]]), 22, n_doe=10)
        SegoOptimizer(
            np.array([[0.0, 25.0]]), 22, n_doe=10, regr_spec=RegressionSpec.ALL
        )


if __name__ == "__main__":
    unittest.main()
