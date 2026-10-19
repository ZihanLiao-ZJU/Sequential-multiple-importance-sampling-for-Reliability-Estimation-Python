import numpy as np

from ..Nataf import Nataf
from ..distributions import makedist


class Ex7:
    """Example 7: noisy performance function (6D), Pf_ref = 5.29e-4.

    g = x1 + 2*x2 + 2*x3 + x4 - 5*x5 - 5*x6 + 0.001 * sum_i sin(100*x_i), with
    independent lognormal inputs (mean/COV specification).

    Reference: S. Engelund, R. Rackwitz, A benchmark study on importance sampling
    techniques in structural reliability, Structural Safety 12 (1993) 255-276.
    """

    def __init__(self):
        self.Nfun = 1
        self.Ndim = 6
        m = np.array([120, 120, 120, 120, 50, 40], dtype=float)
        cov = np.array([8 / 120, 8 / 120, 8 / 120, 8 / 120, 10 / 50, 8 / 40])
        dist = []
        for i in range(6):
            sigma = np.log(1 + cov[i] ** 2) ** 0.5
            mu = np.log(m[i] / np.exp(sigma ** 2 / 2))
            dist.append(makedist("Lognormal", mu=mu, sigma=sigma))
        self.Dist = Nataf(dist, np.eye(6))

    def EvlLSF(self, theta):
        return (theta[0] + 2 * theta[1] + 2 * theta[2] + theta[3] - 5 * theta[4] - 5 * theta[5]
                + 0.001 * np.sum(np.sin(100 * theta), axis=0))
