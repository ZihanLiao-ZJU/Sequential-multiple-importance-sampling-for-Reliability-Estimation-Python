import numpy as np

from ..Nataf import Nataf
from ..distributions import makedist


class Ex12:
    """Example 12: linear performance function with lognormal inputs (nD),
    g = (n + 3*sigma*sqrt(n)) - sum(x), X ~ Lognormal(mean 1.0, std 0.2); Pf_ref = 1.91e-3 (n = 50).

    Reference: C. Dang, J. Xu (2020), Reliability Engineering & System Safety 204, 107124.
    """

    Nfun = 1

    def __init__(self, Ndim):
        self.Ndim = Ndim
        mu, sigma = 1.0, 0.2
        sLN2 = np.log(1 + sigma ** 2 / mu ** 2)
        sLN = np.sqrt(sLN2)
        muLN = np.log(mu) - 0.5 * sLN2
        self.Dist = Nataf([makedist("Lognormal", mu=muLN, sigma=sLN)] * Ndim, np.eye(Ndim))

    def EvlLSF(self, theta):
        sigma = 0.2
        return (self.Ndim + 3 * sigma * np.sqrt(self.Ndim)) - np.sum(theta, axis=0)
