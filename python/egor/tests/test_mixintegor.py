import unittest
import numpy as np
import egor
import time
import logging

logging.basicConfig(level=logging.INFO)


def xsinx(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    y = (x - 3.5) * np.sin((x - 3.5) / (np.pi))
    print(f"obj={y} at {x}")
    return y


class TestMixintEgor(unittest.TestCase):
    def test_xsinx(self):
        xtypes = [egor.Vspec(egor.Vtype(egor.Vtype.INT), [0.0, 25.0])]

        egopt = egor.Optimizer(xsinx, xtypes, seed=42, n_doe=5)
        res = egopt.minimize(n_eval=10)
        print(f"Optimization f={res.y_opt} at {res.x_opt}")
        self.assertAlmostEqual(-15.125, res.y_opt[0], delta=5e-3)
        self.assertAlmostEqual(18.935, res.x_opt[0], delta=1e-1)


if __name__ == "__main__":
    unittest.main()
