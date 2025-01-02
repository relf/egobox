import numpy as np
import egobox as egx

# To display optimization information (none by default)
import logging

logging.basicConfig(level=logging.INFO)

xspecs_g24 = egx.to_specs([[0.0, 3.0], [0.0, 4.0]])
n_cstr_g24 = 2


# Objective
def G24(point):
    """
    Function g24
    1 global optimum y_opt = -5.5080 at x_opt =(2.3295, 3.1785)
    """
    p = np.atleast_2d(point)
    return -p[:, 0] - p[:, 1]


# Constraints < 0
def G24_c1(point):
    p = np.atleast_2d(point)
    return (
        -2.0 * p[:, 0] ** 4.0
        + 8.0 * p[:, 0] ** 3.0
        - 8.0 * p[:, 0] ** 2.0
        + p[:, 1]
        - 2.0
    )


def G24_c2(point):
    p = np.atleast_2d(point)
    return (
        -4.0 * p[:, 0] ** 4.0
        + 32.0 * p[:, 0] ** 3.0
        - 88.0 * p[:, 0] ** 2.0
        + 96.0 * p[:, 0]
        + p[:, 1]
        - 36.0
    )


# Grouped evaluation
def g24(point):
    p = np.atleast_2d(point)
    return np.array([G24(p), G24_c1(p), G24_c2(p)]).T


egor = egx.Egor(
    g24,
    xspecs_g24,
    n_doe=10,
    n_cstr=n_cstr_g24,
    cstr_tol=1e-3,
    infill_strategy=egx.InfillStrategy.WB2,
    # expected=egx.ExpectedOptimum(val=-5.50, tol=1e-2),
    # outdir="./out",
    # hot_start=True
)  # see help(egor) for options

# Restrict regression and correlation models used
# egor = egx.Egor(g24, xlimits_g24, n_cstr=n_cstr_g24, n_doe=10,
#                      regr_spec=egx.RegressionSpec.LINEAR,
#                      corr_spec=egx.CorrelationSpec.MATERN32 | egx.CorrelationSpec.MATERN52)

res = egor.minimize(max_iters=30)
print(f"Optimization f={res.y_opt} at {res.x_opt}")
