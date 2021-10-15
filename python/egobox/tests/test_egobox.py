import unittest
import numpy as np
from egobox import Optimizer


class TestEgobox(unittest.TestCase):
    @staticmethod
    def xsinx(x):
        x = np.array(x)
        y = (x - 3.5) * np.sin((x - 3.5) / (np.pi))
        print(f"obj={y} at {x}")
        return float(y)

    def test_egobox(self):
        ego = Optimizer()
        print("start")
        res = ego.minimize(TestEgobox.xsinx)
        print(f"Optimization f={res.y_opt} at {res.x_opt}")


if __name__ == "__main__":
    unittest.main()
