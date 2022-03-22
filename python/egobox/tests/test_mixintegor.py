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


class TestMixintEgor(unittest.TestCase):
    def test_xsinx(self):
        xtypes = [egx.VarSpec(egx.VarType(egx.VarType.INT), [0.0, 25.0])]

        egor = egx.Optimizer(xsinx, np.array([[0.0, 25.0]]), xtypes, seed=42)
        res = egor.minimize(n_eval=20)
        # print(f"Optimization f={res.y_opt} at {res.x_opt}")
        # self.assertAlmostEqual(-15.125, res.y_opt[0], delta=1e-3)
        # self.assertAlmostEqual(18.935, res.x_opt[0], delta=1e-3)


if __name__ == "__main__":
    unittest.main()
