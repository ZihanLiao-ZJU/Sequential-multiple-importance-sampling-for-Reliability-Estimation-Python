import numpy as np

from ..Nataf import Nataf
from ..distributions import makedist


class Ex11:
    """Example 11: nonlinear performance function (nD),
    g = beta0 - sum(u)/sqrt(n) - k*(u1 - u2)^2/4 with beta0 = 4, k = -10; Pf_ref = 4.75e-6.

    Reference: O. Ditlevsen, H.O. Madsen, Structural reliability methods, Wiley, 1996.
    """

    Nfun = 1

    def __init__(self, Ndim):
        self.Ndim = Ndim
        self.Dist = Nataf([makedist("Normal", mu=0, sigma=1)] * Ndim, np.eye(Ndim))

    def EvlLSF(self, theta):
        beta0 = 4
        k = -10
        return beta0 - np.sum(theta, axis=0) / np.sqrt(self.Ndim) - k * (theta[0] - theta[1]) ** 2 / 4
