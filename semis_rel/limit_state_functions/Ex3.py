import numpy as np

from ..Nataf import Nataf
from ..distributions import makedist
from ._plot import plot_2d_lsf


class Ex3:
    """Example 3: linear-log performance function (2D), Pf_ref = 3.23e-5."""

    def __init__(self):
        self.Parm = np.array([5, 1, 1, 1, 2, 1, 4, 0.5])
        self.Nfun = 1
        self.Ndim = 2
        self.Dist = Nataf([makedist("Normal", mu=0, sigma=1)] * 2, np.eye(2))

    def EvlLSF(self, theta):
        a1, a2, b1, b2, b3, b4, b5, c = self.Parm
        L1 = a1 - a2 * theta[0]
        with np.errstate(over="ignore"):
            L2 = b1 / (b2 + np.exp(-b3 * (b4 * theta[1] + b5))) - c
        return np.minimum(L1, L2)

    def PltFun(self):
        return plot_2d_lsf(self.EvlLSF, "Linear Log")
