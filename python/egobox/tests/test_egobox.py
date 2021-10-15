import unittest
import numpy as np
from egobox import *


class TestEgobox(unittest.TestCase):
    @staticmethod
    def xsinx(x):
        x = np.array(x)
        y = (x - 3.5) * np.sin((x - 3.5) / (np.pi))
        return float(y)

    def test_egobox(self):
        ego = Ego()
        print(ego.optimize(TestEgobox.xsinx).y_opt)
