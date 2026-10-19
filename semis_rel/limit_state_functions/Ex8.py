import numpy as np

from ..Nataf import Nataf
from ..distributions import makedist


class Ex8:
    """Example 8: reinforced concrete beam (7D), Pf_ref = 3.20e-3.

    g(X) = X2*X3*X4 - X5*X3^2*X4^2 / (X6*X7) - X1, with mixed Gumbel (GEV, k = 0),
    normal and lognormal inputs.

    Reference: J. Gong, P. Yi, N. Zhao, Non-gradient-based algorithm for structural
    reliability analysis, J. Eng. Mech. 140 (2014) 04014029.
    """

    def __init__(self):
        self.Nfun = 1
        self.Ndim = 7
        # X1: Gumbel
        mu_mean, sig_std = 0.01, 0.003
        gamma = 0.5772156649015329
        beta = sig_std * np.sqrt(6) / np.pi
        mu_gev = mu_mean - gamma * beta
        # X4, X7: lognormal
        mu4, sd4 = 226e-6, 11.3e-6
        s4 = np.sqrt(np.log(1 + (sd4 / mu4) ** 2)); m4 = np.log(mu4) - 0.5 * s4 ** 2
        mu7, sd7 = 40.0, 6.0
        s7 = np.sqrt(np.log(1 + (sd7 / mu7) ** 2)); m7 = np.log(mu7) - 0.5 * s7 ** 2
        dist = [makedist("GeneralizedExtremeValue", k=0, sigma=beta, mu=mu_gev),
                makedist("Normal", mu=0.30, sigma=0.015),
                makedist("Normal", mu=360.0, sigma=36.0),
                makedist("Lognormal", mu=m4, sigma=s4),
                makedist("Normal", mu=0.5, sigma=0.05),
                makedist("Normal", mu=0.12, sigma=0.006),
                makedist("Lognormal", mu=m7, sigma=s7)]
        self.Dist = Nataf(dist, np.eye(self.Ndim))   # independent

    def EvlLSF(self, X):
        X1, X2, X3, X4, X5, X6, X7 = (X[i] for i in range(7))
        return X2 * X3 * X4 - (X5 * X3 ** 2 * X4 ** 2) / (X6 * X7) - X1
