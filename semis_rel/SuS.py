import copy

import numpy as np

from ._utils import flat
from .support_functions.U2X import U2X
from .support_functions.logGauss import logGauss


class SuS:
    """Subset simulation with Nataf transformation for reliability problems.

    SuS is a special, restricted case of the SeMIS framework (Section 3.2): no
    variance inflation (sigma = 1) and no surrogate truncation; the ISF is
    simply L_i(u) = I[g(u) < l_i].
    """

    def __init__(self, PF):
        self.TF = PF
        self.Nfun = PF.Nfun + 3
        self.Ndim = PF.Ndim
        self.Ncal = None
        self.X = None
        self.Y = None
        self.G = np.inf     # PF threshold l_i
        self.p = 0.1
        self.Nsam = 1000

    def EvlY(self, u):
        """Evaluate y = [log phi_n(u); g(u)]."""
        obj = copy.copy(self)
        u = flat(u)
        x = U2X(obj, u)
        g_val = np.asarray(obj.TF.EvlLSF(x), dtype=float).ravel()
        obj.X = u
        obj.Y = np.vstack([logGauss(u), g_val])
        obj.Ncal = u.shape[1]
        return obj

    def EvlLKF(self):
        """Intermediate likelihood log L_i = log I[g(u) < l_i]."""
        with np.errstate(divide="ignore"):
            return np.log((self.Y[1, :] < self.G).astype(float))

    def EvlPDF(self):
        """Intermediate prior PDF pi = phi_n(u) (no inflation, sigma = 1)."""
        return self.Y[0, :]

    def UpdObj(self, u, y):
        """Update the threshold l_i (Eqn. 24, SuS case). Returns (y, l, obj, N_cal)."""
        obj = copy.copy(self)
        g_val = flat(y)[3, :]
        l = max(np.quantile(g_val, obj.p, method="hazen"), 0)
        obj.G = l
        return y, l, obj, 0

    @property
    def FlgCvg(self):
        return bool(self.G <= 0)

    def EvlPf(self, inp):
        """Failure probability as the product of the level probabilities."""
        Nite = inp["Nite"]
        y = inp["y"]
        l = inp["g"]
        z_hat = np.ones(Nite + 1)
        for ite in range(Nite):
            ind_H = flat(y[ite])[3, :] < l[ite + 1]    # I[g < l_{ite+1}] (Eqn. 31)
            z_hat[ite + 1] = z_hat[ite] * np.mean(ind_H)
        return z_hat[Nite]
