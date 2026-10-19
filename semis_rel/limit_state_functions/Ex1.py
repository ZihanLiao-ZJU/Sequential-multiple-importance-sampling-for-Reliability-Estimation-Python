import numpy as np

from ..Nataf import Nataf
from ..distributions import makedist
from ._plot import plot_2d_lsf


class Ex1:
    """Example 1: piecewise-linear performance function (2D), Pf_ref = 3.20e-5."""

    def __init__(self):
        self.Parm = np.array([4, 1, 0.85, 0.1, 0.5, 0.1, 2.3, 1])
        self.Nfun = 1
        self.Ndim = 2
        self.Dist = Nataf([makedist("Normal", mu=0, sigma=1)] * 2, np.eye(2))

    def EvlLSF(self, theta):
        a1, a2, a3, a4, b1, b2, b3, b4 = self.Parm
        t1, t2 = theta[0], theta[1]
        L1 = np.where(t1 > 3.5, a1 - a2 * t1, a3 - a4 * t1)
        L2 = np.where(t2 > 2, b1 - b2 * t2, b3 - b4 * t2)
        return np.minimum(L1, L2)

    def PltFun(self):
        return plot_2d_lsf(self.EvlLSF, "Piecewise Linear")
