import os
import unittest
import numpy as np
import egobox as egx
import logging

logging.basicConfig(level=logging.INFO)


def griewank(x):
    x = np.asarray(x)
    if x.ndim == 1 or max(x.shape) == 1:
        x = x.reshape((1, -1))
    # dim = x.shape[1]

    s, p = 0.0, 1.0
    for i, xi in enumerate(x.T):
        s += xi**2 / 4000.0
        p *= np.cos(xi / np.sqrt(i + 1))
    return s - p + 1.0


class TestGpMix(unittest.TestCase):
    def setUp(self):
        self.xt = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]]).T
        self.yt = np.array([[0.0, 1.0, 1.5, 0.9, 1.0]]).T

        self.gpx = egx.Gpx.builder().fit(self.xt, self.yt)

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
            0.0145, gpx.predict_var_gradients(np.array([[1.1]])).item(), delta=1e-3
        )

    def test_gpx_save_load(self):
        filename = "gpdump.json"
        filename_bin = "gpdump.bin"
        gpx = self.gpx

        for file in [filename, filename_bin]:
            if os.path.exists(file):
                os.remove(file)
            gpx.save(file)

            gpx2 = egx.Gpx.load(file)

            os.remove(file)

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

    def test_training_params(self):
        self.assertEqual(self.gpx.dims(), (1, 1))
        (xdata, ydata) = self.gpx.training_data()
        np.testing.assert_array_equal(xdata, self.xt)
        np.testing.assert_array_equal(np.atleast_2d(ydata).T, self.yt)

    def test_kpls_griewank(self):
        lb = -600
        ub = 600
        # n_dim = 100
        n_dim = 50
        xlimits = [[ub, lb]] * n_dim

        # LHS training point generation
        n_train = 100
        x_train = egx.lhs(xlimits, n_train)
        y_train = griewank(x_train)
        y_train = y_train.reshape((n_train, -1))  # reshape to 2D array

        # Random test point generation
        n_test = 5
        x_test = np.random.random_sample((n_test, n_dim))
        x_test = lb + (ub - lb) * x_test  # map generated samples to design space
        y_test = griewank(x_test)
        y_test = y_test.reshape((n_test, -1))  # reshape to 2D array

        # Surrogate model definition
        n_pls = 3
        builders = [
            egx.Gpx.builder(seed=42),
            egx.Gpx.builder(kpls_dim=n_pls, seed=42),
        ]

        # Surrogate model fit & error estimation
        for builder in builders:
            gpx = builder.fit(x_train, y_train)
            y_pred = gpx.predict(x_test)
            self.assertEqual(n_dim, gpx.dims()[0])
            error = np.linalg.norm(y_pred - y_test) / np.linalg.norm(y_test)
            self.assertAlmostEqual(0.0, error, delta=5e-1)
            print("   RMS error: " + str(error))

    def test_multi_outputs_exception(self):
        self.xt = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]]).T
        self.yt = np.array(
            [[0.0, 10.0], [1.0, -3.0], [1.5, 1.5], [0.9, 1.0], [1.0, 0.0]]
        )
        with self.assertRaises(BaseException):
            egx.Gpx.builder().fit(self.xt, self.yt)

    def test_1d_training_data(self):
        self.xt1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        self.yt1 = np.array([0.0, 1.0, 1.5, 0.9, 1.0])

        self.gpx = egx.Gpx.builder().fit(self.xt1, self.yt1)

    def test_fixed_theta_no_optim(self):
        print(f"gpx.theta = {self.gpx.thetas()}")
        self.assertNotEqual(0.314, self.gpx.thetas().item())
        self.gpx = egx.Gpx.builder(n_start=-1, theta_init=[0.314]).fit(self.xt, self.yt)
        print(f"gpx.theta = {self.gpx.thetas()}")
        self.assertEqual(0.314, self.gpx.thetas().item())


if __name__ == "__main__":
    unittest.main()
