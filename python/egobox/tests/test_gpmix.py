import os
import unittest
import numpy as np
import egobox as egx
import logging

logging.basicConfig(level=logging.INFO)


class TestGpMix(unittest.TestCase):
    def setUp(self):
        xt = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]]).T
        yt = np.array([[0.0, 1.0, 1.5, 0.9, 1.0]]).T

        self.gpmix = egx.GpMix()  # or egx.Gpx.builder()
        self.gpmix.set_training_values(xt, yt)
        self.gpx = self.gpmix.train()

    def test_gpx_kriging(self):
        gpx = self.gpx

        # should interpolate
        self.assertAlmostEqual(1.0, float(gpx.predict_values(np.array([[1.0]]))))
        self.assertAlmostEqual(0.0, float(gpx.predict_variances(np.array([[1.0]]))))

        # check a point not too far from a training point
        self.assertAlmostEqual(
            1.1163, float(gpx.predict_values(np.array([[1.1]]))), delta=1e-3
        )
        self.assertAlmostEqual(
            0.0, float(gpx.predict_variances(np.array([[1.1]]))), delta=1e-3
        )

    def test_gpx_save_load(self):
        filename = "gpdump.json"

        gpx = self.gpx

        if os.path.exists(filename):
            os.remove(filename)
        gpx.save(filename)
        gpx2 = egx.Gpx.load(filename)
        os.remove(filename)

        # should interpolate
        self.assertAlmostEqual(1.0, float(gpx2.predict_values(np.array([[1.0]]))))
        self.assertAlmostEqual(0.0, float(gpx2.predict_variances(np.array([[1.0]]))))

        # check a point not too far from a training point
        self.assertAlmostEqual(
            1.1163, float(gpx2.predict_values(np.array([[1.1]]))), delta=1e-3
        )
        self.assertAlmostEqual(
            0.0, float(gpx2.predict_variances(np.array([[1.1]]))), delta=1e-3
        )


if __name__ == "__main__":
    unittest.main()
