import unittest
import numpy as np
import egobox as egx
import logging
from segomoe.cases.case_generator import _import_case

logging.basicConfig(level=logging.DEBUG)


def test_egor(case, options={}):
    options = {}
    case = _import_case("Mod_Branin", options)()
    xlimits = np.array([[v["lb"], v["ub"]] for v in case["vars"]])
    n_cstr = len(case["con"])
    fun = case["f_grouped"]
    f_grouped = lambda x: np.atleast_2d(
        np.array([fun(xi)[0] for xi in np.atleast_2d(x)])
    )
    return egx.Optimizer(fun=f_grouped, xlimits=xlimits, n_cstr=n_cstr), case["sol"]


class TestEgorOnSegomoeCases(unittest.TestCase):
    def test_branin(self):
        egor, expected = test_egor("Mod_Branin")
        res = egor.minimize(n_eval=30)
        self.assertAlmostEqual(expected["value"], res.y_opt[0], delta=1e-2)


if __name__ == "__main__":
    unittest.main()
