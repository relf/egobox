import unittest
import numpy as np
import egobox as egx


class TestUtils(unittest.TestCase):
    def test_to_specs(self):
        actual = egx.to_specs([[0.0, 25.0]])
        expected = [egx.Vspec(egx.Vtype(egx.Vtype.FLOAT), [0.0, 25.0])]
        self.assertEqual(len(expected), len(actual))
        self.assertEqual(expected[0].vtype.id(), actual[0].vtype.id())
        self.assertEqual(expected[0].vlimits, actual[0].vlimits)

    def test_to_specs_empty(self):
        with self.assertRaises(ValueError):
            egx.to_specs([[]])

    def test_lhs(self):
        xspecs = egx.to_specs([[0.0, 25.0]])
        doe = egx.lhs(xspecs, 10)
        self.assertEqual(doe.shape, (10, 1))

    def test_mixint_lhs(self):
        xspecs = [
            egx.Vspec(egx.Vtype(egx.Vtype.FLOAT), [0.0, 25.0]),
            egx.Vspec(egx.Vtype(egx.Vtype.INT), [0, 25]),
        ]
        doe = egx.lhs(xspecs, 10)
        self.assertEqual(doe.shape, (10, 2))
