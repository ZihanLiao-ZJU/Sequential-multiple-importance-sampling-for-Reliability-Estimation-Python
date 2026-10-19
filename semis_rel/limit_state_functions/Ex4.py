import numpy as np

from ..Nataf import Nataf
from ..distributions import makedist


class Ex4:
    """Example 4: polynomial benchmark (3D), Pf_ref = 3.13e-4."""

    def __init__(self):
        self.Nfun = 1
        self.Ndim = 3
        self.Dist = Nataf([makedist("Normal", mu=0, sigma=1)] * 3, np.eye(3))

    def EvlLSF(self, theta):
        return 1 / 40 * theta[0] ** 4 + 2 * theta[1] ** 2 + theta[2] + 3
