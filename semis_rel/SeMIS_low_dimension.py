import copy

import numpy as np
from scipy.stats import norm

from ._utils import flat
from .support_functions.U2X import U2X
from .support_functions.logGauss import logGauss
from .surrogates import fitrgp, predict


class SeMIS_low_dimension:
    """SeMIS with Nataf transformation for structural reliability problems.

    Low-dimensional variant: Kriging-based hybrid truncation (Eqn. 25).
    ISD (Eqn. 21): q_i(u) = phi_n(u; sigma^2 I) * L_i(u) / z_i
      rPDF pi_i(u) = phi_n(u; sigma^2 I)                             (Eqn. 12)
      ISF  L_i(u)  = I[g(u) < l_i] * prod_m I[g_hat_m(u) <= l_hat_m]  (Eqn. 20)
    """

    def __init__(self, PF):
        self.TF = PF
        self.Nfun = PF.Nfun + 3
        self.Ndim = PF.Ndim
        self.X = None
        self.Y = None
        self.Nsam = 1000

        # approximate optimal variance-inflation factor (Eqn. 19):
        #   sigma_tilde* = sqrt(1 + beta * lambda(beta) / n)
        # with beta = 2.33 (target P_f ~ 1e-2) and lambda(beta) the inverse Mills ratio
        beta = 2.33
        lambda_beta = norm.pdf(beta) / (1 - norm.cdf(beta))
        sigma = np.sqrt((self.Ndim + beta * lambda_beta) / self.Ndim)

        L_pi = np.eye(self.Ndim) * sigma
        self.G = {
            "p": 0.1,            # intermediate level probability p (Section 4.1.2)
            "Ite": 0,            # current iteration counter
            "l": np.inf,         # exact PF threshold l_i (Eqn. 20)
            "g_hat": [],         # Kriging surrogates g_hat_m (Eqn. 25)
            "l_hat": np.zeros(0),  # surrogate thresholds l_hat_m (Eqn. 27)
            "kappa": 2.0,        # conservatism factor kappa (Eqn. 25)
            "L_pi": L_pi,        # rPDF (Eqn. 12): Cholesky factor (sigma*I)
            "Sigma": L_pi @ L_pi.T,   # covariance sigma^2 I
            "mu_pi": np.zeros((self.Ndim, 1)),
        }
        self.Ncal = 0

    def _screen(self, u):
        n_sam = u.shape[1]
        ind_scr = np.ones(n_sam, dtype=bool)
        for i in range(self.G["Ite"]):
            mu_hat, sd_hat = predict(self.G["g_hat"][i], u.T)
            g_hat = mu_hat - self.G["kappa"] * sd_hat
            ind_scr &= g_hat <= self.G["l_hat"][i]
        return ind_scr

    def EvlY(self, u):
        """Evaluate [log phi_n(u); g(u)]; g is only evaluated where the Kriging screening passes."""
        obj = copy.copy(self)
        u = flat(u)
        obj.X = u
        n_sam = u.shape[1]
        log_phi = logGauss(u)
        # screening condition defined by Kriging in U-space (Eqn. 25)
        ind_scr = obj._screen(u) if obj.G["Ite"] > 0 else np.ones(n_sam, dtype=bool)
        g_val = np.full(n_sam, np.inf)
        n_cal = 0
        if np.any(ind_scr):
            x = U2X(obj, u[:, ind_scr])
            g_val[ind_scr] = np.asarray(obj.TF.EvlLSF(x), dtype=float).ravel()
            n_cal = int(np.sum(ind_scr))
        obj.Y = np.vstack([log_phi, g_val])
        obj.Ncal = n_cal
        return obj

    def EvlLKF(self):
        """log ISF: log I[g(u) < l_i] + sum_m log I[g_hat_m(u) <= l_hat_m] (Eqn. 20)."""
        g_val = self.Y[1, :]
        l = self.G["l"]
        n_sam = g_val.size
        if np.isinf(l):
            return np.zeros(n_sam)
        ind_scr = self._screen(self.X) if self.G["Ite"] > 0 else np.ones(n_sam, dtype=bool)
        with np.errstate(divide="ignore"):
            return np.log(((g_val < l) & ind_scr).astype(float))

    def EvlPDF(self):
        """log rPDF pi_i(u) (Eqn. 12)."""
        return logGauss(self.X, self.G["mu_pi"], self.G["Sigma"])

    def UpdObj(self, u, y):
        """Update l_i (Eqn. 24), fit the Kriging surrogate g_hat_{i+1} (Eqn. 25) and
        set the conservative surrogate threshold l_hat (Eqn. 27).

        Returns (y, g, obj, N_cal).
        """
        obj = copy.copy(self)
        y2 = flat(y)
        g_val = y2[3, :]
        u2 = flat(u)
        g = dict(obj.G)

        l = max(np.quantile(g_val, g["p"], method="hazen"), 0)   # threshold l_i (Eqn. 24)
        ind_seed = g_val < l                                       # seed set S_{i+1}

        # fit Kriging with all truly evaluated samples
        ind_fit = np.isfinite(g_val)
        mdl = fitrgp(u2[:, ind_fit].T, g_val[ind_fit])

        # conservative Kriging score (Eqn. 25)
        mu_hat, sd_hat = predict(mdl, u2.T)
        g_hat = mu_hat - g["kappa"] * sd_hat

        # threshold chosen so that all current seeds are retained (Eqn. 27)
        l_hat_tmp = np.max(g_hat[ind_seed])

        g["g_hat"] = list(g["g_hat"]) + [mdl]
        g["l_hat"] = np.append(g["l_hat"], l_hat_tmp)
        g["l"] = l
        g["Ite"] = g["Ite"] + 1
        obj.G = g
        return y, g, obj, 0

    @property
    def FlgCvg(self):
        return bool(self.G["l"] <= 0)
