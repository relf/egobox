import os
import sys
import unittest
import numpy as np
import egobox as egx
import logging

SEGOMOE_NOT_INSTALLED = False
try:
    from segomoe.cases.case_generator import _import_case
except ImportError:
    SEGOMOE_NOT_INSTALLED = True

logging.basicConfig(level=logging.INFO)


def create_egor(case, **options):
    opts = {}
    case = _import_case(case, opts)()
    xlimits = np.array([[v["lb"], v["ub"]] for v in case["vars"]])
    n_cstr = len(case["con"])
    fun = case["f_grouped"]
    f_grouped = lambda x: np.atleast_2d(
        np.array([fun(xi)[0] for xi in np.atleast_2d(x)])
    )
    return (
        egx.Optimizer(fun=f_grouped, xlimits=xlimits, n_cstr=n_cstr, **options),
        case["sol"],
    )


class TestEgor(unittest.TestCase):
    @unittest.skipIf(SEGOMOE_NOT_INSTALLED, "SEGOMOE not installed")
    def test_branin(self):
        options = {"seed": 42, "n_doe": 8}
        egor, expected = create_egor("Mod_Branin", **options)
        res = egor.minimize(n_eval=50)
        self.assertAlmostEqual(expected["value"], res.y_opt[0], delta=expected["tol"])

    @unittest.skipIf(SEGOMOE_NOT_INSTALLED, "SEGOMOE not installed")
    def test_hesse(self):
        options = {
            "seed": 42,
            "kpls_dim": 3,
            "n_doe": 8,
            "regr_spec": 1,
            "corr_spec": 1,
        }
        egor, expected = create_egor("Hesse", **options)
        res = egor.minimize(n_eval=30)
        self.assertAlmostEqual(expected["value"], res.y_opt[0], delta=expected["tol"])

    @unittest.skipIf(SEGOMOE_NOT_INSTALLED, "SEGOMOE not installed")
    def test_mopta(self):
        options = {
            "seed": 42,
            "kpls_dim": 3,
            "n_doe": 20,
            "regr_spec": 1,
            "corr_spec": 1,
            "expected": egx.ExpectedOptimum(val=272.72, tol=1e-2),
            "outdir": "./out",
        }
        egor, expected = create_egor("Mopta_12D", **options)
        res = egor.minimize(n_eval=50)
        self.assertAlmostEqual(expected["value"], res.y_opt[0], delta=expected["tol"])


if __name__ == "__main__":
    unittest.main()
