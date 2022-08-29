import unittest
import numpy as np
import egobox as egx
import logging

logging.basicConfig(level=logging.INFO)


class TestGpMix(unittest.TestCase):
    def test_gpmix_kriging(self):
        xt = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]]).T
        yt = np.array([[0.0, 1.0, 1.5, 0.9, 1.0]]).T

        sm = egx.GpMix(
            regr_spec=egx.RegressionSpec.CONSTANT,
            corr_spec=egx.CorrelationSpec.SQUARED_EXPONENTIAL,
        )
        sm.set_training_values(xt, yt)
        sm.train()

        # should interpolate
        self.assertAlmostEqual(1.0, float(sm.predict_values(np.array([[1.0]]))))
        self.assertAlmostEqual(0.0, float(sm.predict_variances(np.array([[1.0]]))))

        # check a point not too far a training point
        self.assertAlmostEqual(
            1.1163, float(sm.predict_values(np.array([[1.1]]))), delta=1e-3
        )
        self.assertAlmostEqual(
            0.0, float(sm.predict_variances(np.array([[1.1]]))), delta=1e-3
        )


if __name__ == "__main__":
    unittest.main()
