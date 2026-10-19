import numpy as np

from ..Nataf import Nataf
from ..distributions import makedist
from ._plot import plot_2d_lsf


class Ex2:
    """Example 2: changing topology (2D), Pf_ref = 1.13e-5."""

    def __init__(self):
        self.Parm = np.array([-2, 0, 2.5, 0.5, 4 / 9, 1 / 25, 1 / 4, 1 / 25, 30, 20, -5])
        self.Nfun = 1
        self.Ndim = 2
        self.Dist = Nataf([makedist("Normal", mu=0, sigma=1)] * 2, np.eye(2))

    def EvlLSF(self, theta):
        a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, d = self.Parm
        t1, t2 = theta[0], theta[1]
        return (d + c1 / ((b1 * (t1 - a1) ** 2 + b2 * (t2 - a2) ** 2) ** 2 + 1)
                + c2 / ((b3 * (t1 - a3) ** 2 + b4 * (t2 - a4) ** 2) ** 2 + 1))

    def PltFun(self):
        return plot_2d_lsf(self.EvlLSF, "Changing Topology")
