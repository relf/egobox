import unittest
import numpy as np
from egobox import SegoOptimizer


class TestSego(unittest.TestCase):
    @staticmethod
    def xsinx(x):
        x = np.atleast_2d(x)
        y = (x - 3.5) * np.sin((x - 3.5) / (np.pi))
        print(f"obj={y} at {x}")
        return y

    def test_egobox(self):
        ego = SegoOptimizer()
        print("start")
        res = ego.minimize(TestSego.xsinx)
        print(f"Optimization f={res.y_opt} at {res.x_opt}")


if __name__ == "__main__":
    unittest.main()
