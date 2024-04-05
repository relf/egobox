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

        gpmix = egx.GpMix()  # or egx.Gpx.builder()
        self.gpx = gpmix.fit(xt, yt)

    def test_gpx_kriging(self):
        gpx = self.gpx

        print(f"gpx.theta = {gpx.thetas()}")
        print(f"gpx.variance= {gpx.variances()}")
        print(f"gpx.likelihood = {gpx.likelihoods()}")

        # should interpolate
        self.assertAlmostEqual(1.0, gpx.predict(np.array([[1.0]])).item())
        self.assertAlmostEqual(0.0, gpx.predict_var(np.array([[1.0]])).item())

        # check a point not too far from a training point
        self.assertAlmostEqual(
            1.1163, gpx.predict(np.array([[1.1]])).item(), delta=1e-3
        )
        self.assertAlmostEqual(
            0.0, gpx.predict_var(np.array([[1.1]])).item(), delta=1e-3
        )
        self.assertAlmostEqual(
            1.1204, gpx.predict_gradients(np.array([[1.1]])).item(), delta=1e-3
        )
        self.assertAlmostEqual(
            0.0092, gpx.predict_var_gradients(np.array([[1.1]])).item(), delta=1e-3
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
        self.assertAlmostEqual(1.0, gpx2.predict(np.array([[1.0]])).item())
        self.assertAlmostEqual(0.0, gpx2.predict_var(np.array([[1.0]])).item())

        # check a point not too far from a training point
        self.assertAlmostEqual(
            1.1163, gpx2.predict(np.array([[1.1]])).item(), delta=1e-3
        )
        self.assertAlmostEqual(
            0.0, gpx2.predict_var(np.array([[1.1]])).item(), delta=1e-3
        )


if __name__ == "__main__":
    unittest.main()
