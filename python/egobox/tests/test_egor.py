import unittest
import numpy as np
import egobox as egx
import time
import logging

logging.basicConfig(level=logging.INFO)


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
        egor = egx.Optimizer(xsinx, np.array([[0.0, 25.0]]), seed=42)
        res = egor.minimize(n_eval=20)
        print(f"Optimization f={res.y_opt} at {res.x_opt}")
        self.assertAlmostEqual(-15.125, res.y_opt[0], delta=1e-3)
        self.assertAlmostEqual(18.935, res.x_opt[0], delta=1e-3)

    def test_xsinx_with_initial_doe(self):
        xlimits = np.array([[0.0, 25.0]])
        doe = egx.lhs(xlimits, 10)
        egor = egx.Optimizer(xsinx, xlimits, doe=doe, seed=42)
        res = egor.minimize(n_eval=15)
        print(f"Optimization f={res.y_opt} at {res.x_opt}")
        self.assertAlmostEqual(-15.125, res.y_opt[0], delta=1e-3)
        self.assertAlmostEqual(18.935, res.x_opt[0], delta=1e-3)

        ydoe = xsinx(doe)
        doe = np.hstack((doe, ydoe))
        egor = egx.Optimizer(xsinx, xlimits, doe=doe)
        res = egor.minimize(n_eval=5)
        print(f"Optimization f={res.y_opt} at {res.x_opt}")
        self.assertAlmostEqual(-15.125, res.y_opt[0], delta=1e-2)
        self.assertAlmostEqual(18.935, res.x_opt[0], delta=1e-2)

    def test_g24(self):
        egor = egx.Optimizer(
            g24,
            np.array([[0.0, 3.0], [0.0, 4.0]]),
            n_cstr=2,
            infill_strategy=egx.InfillStrategy.EI,
            regr_spec=egx.RegressionSpec.CONSTANT,
            corr_spec=egx.CorrelationSpec.SQUARED_EXPONENTIAL,
            seed=42,
        )
        start = time.process_time()
        res = egor.minimize(n_eval=20)
        end = time.process_time()
        print(f"Optimization f={res.y_opt} at {res.x_opt} in {end-start}s")
        self.assertAlmostEqual(-5.5080, res.y_opt[0], delta=1e-2)
        self.assertAlmostEqual(2.3295, res.x_opt[0], delta=1e-2)
        self.assertAlmostEqual(3.1785, res.x_opt[1], delta=1e-2)

    def test_g24_kpls(self):
        egor = egx.Optimizer(
            g24,
            np.array([[0.0, 3.0], [0.0, 4.0]]),
            n_cstr=2,
            regr_spec=egx.RegressionSpec.CONSTANT,
            corr_spec=egx.CorrelationSpec.SQUARED_EXPONENTIAL
            | egx.CorrelationSpec.ABSOLUTE_EXPONENTIAL,
            kpls_dim=1,
            seed=0,
        )
        start = time.process_time()
        res = egor.minimize(n_eval=20)
        end = time.process_time()
        self.assertAlmostEqual(-5.5080, res.y_opt[0], delta=6e-2)
        print(f"Optimization f={res.y_opt} at {res.x_opt} in {end-start}s")

    def test_six_humps(self):
        egor = egx.Optimizer(
            six_humps,
            np.array([[-3.0, 3.0], [-2.0, 2.0]]),
            infill_strategy=egx.InfillStrategy.WB2,
            regr_spec=egx.RegressionSpec.CONSTANT,
            corr_spec=egx.CorrelationSpec.SQUARED_EXPONENTIAL,
            seed=42,
        )
        start = time.process_time()
        res = egor.minimize(n_eval=45)
        end = time.process_time()
        print(f"Optimization f={res.y_opt} at {res.x_opt} in {end-start}s")
        # 2 global optimum value =-1.0316 located at (0.089842, -0.712656) and  (-0.089842, 0.712656)
        self.assertAlmostEqual(-1.0316, res.y_opt[0], delta=1e-2)
        self.assertAlmostEqual(0.0898, res.x_opt[0], delta=1e-2)
        self.assertAlmostEqual(-0.7126, res.x_opt[1], delta=1e-2)

    def test_constructor(self):
        self.assertRaises(TypeError, egx.Optimizer)
        egx.Optimizer(xsinx, np.array([[0.0, 25.0]]), 22, n_doe=10)


if __name__ == "__main__":
    unittest.main()
