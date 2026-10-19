import numpy as np

from ..Nataf import Nataf
from ..distributions import makedist


class Ex5:
    """Example 5: parallel system (5D), Pf_ref = 2.13e-4."""

    def __init__(self):
        self.Nfun = 1
        self.Ndim = 5
        self.Dist = Nataf([makedist("Normal", mu=0, sigma=1)] * self.Ndim, np.eye(self.Ndim))

    def EvlLSF(self, u):
        g1 = 2.677 - u[0] - u[1]
        g2 = 2.500 - u[1] - u[2]
        g3 = 2.323 - u[2] - u[3]
        g4 = 2.250 - u[3] - u[4]
        return np.max(np.vstack([g1, g2, g3, g4]), axis=0)
