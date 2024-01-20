import os
import unittest
import numpy as np
import egobox as egx
import logging

logging.basicConfig(level=logging.INFO)


def f_obj(x):
    return (
        np.sin(3 * np.pi * x)
        + 0.3 * np.cos(9 * np.pi * x)
        + 0.5 * np.sin(7 * np.pi * x)
    )


class TestSgp(unittest.TestCase):
    def setUp(self):
        # random generator for reproducibility
        rng = np.random.RandomState(0)

        # Generate training data
        nt = 200
        # Variance of the gaussian noise on our trainingg data
        eta2 = [0.01]
        gaussian_noise = rng.normal(loc=0.0, scale=np.sqrt(eta2), size=(nt, 1))
        xt = 2 * rng.rand(nt, 1) - 1
        yt = f_obj(xt) + gaussian_noise

        # Pick inducing points randomly in training data
        n_inducing = 30
        random_idx = rng.permutation(nt)[:n_inducing]
        Z = xt[random_idx].copy()

        sgp = SGP()
        sgp.set_training_values(xt, yt)
        sgp.set_inducing_inputs(Z=Z)
        # sgp.set_inducing_inputs()  # When Z not specified n_inducing points are picked randomly in traing data
        sgp.train()


if __name__ == "__main__":
    unittest.main()
