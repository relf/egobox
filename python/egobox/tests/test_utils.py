import unittest
import numpy as np
import egobox as egx


class TestUtils(unittest.TestCase):
    def test_to_specs(self):
        actual = egx.to_specs([[0.0, 25.0]])
        expected = [egx.XSpec(egx.XType.FLOAT, [0.0, 25.0])]
        self.assertEqual(len(expected), len(actual))
        self.assertEqual(expected[0].xtype, actual[0].xtype)
        self.assertEqual(expected[0].xlimits, actual[0].xlimits)

    def test_to_specs_empty(self):
        with self.assertRaises(ValueError):
            egx.to_specs([[]])

    def test_lhs(self):
        xspecs = egx.to_specs([[0.0, 25.0]])
        doe = egx.lhs(xspecs, 10)
        self.assertEqual(doe.shape, (10, 1))

    def test_mixint_lhs(self):
        xspecs = [
            egx.XSpec(egx.XType.FLOAT, [0.0, 25.0]),
            egx.XSpec(egx.XType.INT, [0, 25]),
        ]
        doe = egx.lhs(xspecs, 10)
        self.assertEqual(doe.shape, (10, 2))
