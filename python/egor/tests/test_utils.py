import unittest
import numpy as np
import egor


class TestUtils(unittest.TestCase):
    def test_to_specs(self):
        actual = egor.to_specs([[0.0, 25.0]])
        expected = [egor.Vspec(egor.Vtype(egor.Vtype.FLOAT), [0.0, 25.0])]
        self.assertEqual(len(expected), len(actual))
        self.assertEqual(expected[0].vtype.id(), actual[0].vtype.id())
        self.assertEqual(expected[0].vlimits, actual[0].vlimits)

    def test_to_specs_empty(self):
        with self.assertRaises(ValueError):
            egor.to_specs([[]])

    def test_lhs(self):
        xspecs = egor.to_specs([[0.0, 25.0]])
        doe = egor.lhs(xspecs, 10)
        self.assertEqual(doe.shape, (10, 1))

    def test_mixint_lhs(self):
        xspecs = [
            egor.Vspec(egor.Vtype(egor.Vtype.FLOAT), [0.0, 25.0]),
            egor.Vspec(egor.Vtype(egor.Vtype.INT), [0, 25]),
        ]
        doe = egor.lhs(xspecs, 10)
        self.assertEqual(doe.shape, (10, 2))
