import numpy as np

from ..Nataf import Nataf
from ..distributions import makedist


class Ex6:
    """Example 6: Brotone Bridge tower (5D), Pf_ref = 2.370e-4."""

    def __init__(self):
        self.Nfun = 1
        self.Ndim = 5
        mu = np.array([17.6, 4e7, 9.01e4, 2.12e3, 2.6e5])
        cov = np.array([0.10, 0.08, 0.10, 0.15, 0.15])
        sigma = cov * mu
        s2 = np.sqrt(np.log(1 + cov[1] ** 2))
        m2 = np.log(mu[1]) - 0.5 * s2 ** 2
        dist = [makedist("Normal", mu=mu[0], sigma=sigma[0]),
                makedist("Lognormal", mu=m2, sigma=s2),
                makedist("Normal", mu=mu[2], sigma=sigma[2]),
                makedist("Normal", mu=mu[3], sigma=sigma[3]),
                makedist("Normal", mu=mu[4], sigma=sigma[4])]
        self.Dist = Nataf(dist, np.eye(self.Ndim))

    def EvlLSF(self, X):
        H, l, W = 70.5, 47.4, 1170.0
        h = 0.4 * H
        X1, X2, X3, X4, X5 = X[0], X[1], X[2], X[3], X[4]
        Den = 3 * (5 * X1 * X2 - 2 * X3 * l ** 2)
        term1 = (5 * X3 * X4 * l ** 3) / Den
        term2 = (5 * W * X4 * l * h ** 2) / Den
        return X5 - term1 - term2 - X4 * l
