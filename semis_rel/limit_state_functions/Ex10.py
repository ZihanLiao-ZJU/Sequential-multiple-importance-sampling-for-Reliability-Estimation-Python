import numpy as np

from ..Nataf import Nataf
from ..distributions import makedist


class Ex10:
    """Example 10: linear hyperplane (nD), g = beta*sqrt(n) - sum(u); beta = 3, Pf_ref = 1.35e-3.

    Reference: S. Engelund, R. Rackwitz (1993), Structural Safety 12, 255-276.
    """

    Nfun = 1

    def __init__(self, Ndim):
        self.Ndim = Ndim
        self.Dist = Nataf([makedist("Normal", mu=0, sigma=1)] * Ndim, np.eye(Ndim))

    def EvlLSF(self, theta):
        beta = 3
        ndim = theta.shape[0]
        return beta * np.sqrt(ndim) - np.sum(theta, axis=0)
