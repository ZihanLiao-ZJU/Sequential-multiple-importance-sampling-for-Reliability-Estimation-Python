import numpy as np

from ..Nataf import Nataf
from ..distributions import makedist


class Ex9:
    """Example 9: two-degree-of-freedom damped oscillator (8D), Pf_ref = 3.78e-7.

    Reference: A. Der Kiureghian, M. De Stefano, Efficient algorithm for
    second-order reliability analysis, J. Eng. Mech. 117 (1991) 2904-2923.
    """

    Parm = np.array([3, 4, 4, 4], dtype=float)
    Nfun = 1
    Ndim = 8

    def __init__(self):
        m = np.array([1.5, 0.01, 1, 0.01, 0.05, 0.02, 27.5, 100])
        cov = np.array([0.1, 0.1, 0.2, 0.2, 0.4, 0.5, 0.1, 0.1])
        dist = []
        for i in range(self.Ndim):
            sigma = np.log(1 + cov[i] ** 2) ** 0.5
            mu = np.log(m[i] / np.exp(sigma ** 2 / 2))
            dist.append(makedist("Lognormal", mu=mu, sigma=sigma))
        self.Dist = Nataf(dist, np.eye(8))

    def EvlLSF(self, theta):
        a1, a2, a3, a4 = self.Parm
        omegas = np.sqrt(theta[3] / theta[1])
        omegap = np.sqrt(theta[2] / theta[0])
        omegaa = (omegas + omegap) / 2
        zetas = theta[5]
        zetap = theta[4]
        zetaa = (zetas + zetap) / 2
        Fs = theta[6]
        ks = theta[3]
        S0 = theta[7]
        gamma = theta[1] / theta[0]
        th = (omegap - omegas) / omegaa
        return Fs - a1 * ks * np.sqrt(
            np.pi * S0 / a2 / zetas / omegas ** 3
            * (zetaa * zetas * (zetap * omegap ** 3 + zetas * omegas ** 3) * omegap
               / (zetap * zetas * (a3 * zetaa ** 2 + th ** 2) + gamma * zetaa ** 2)
               / (a4 * zetaa * omegaa ** 4)))
