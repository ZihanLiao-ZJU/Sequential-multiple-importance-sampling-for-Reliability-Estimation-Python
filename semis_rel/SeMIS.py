"""
=======================================================================================
Sequential Multiple Importance Sampling (SeMIS) framework.

The algorithm sequentially constructs a series of intermediate importance sampling
distributions (ISDs) q_i(u) = pi_i(u) * L_i(u) / z_i (Eqn. 11/21) and generates samples
from each ISD to progressively approach the failure domain. At each iteration:

  1) Samples are generated from the current ISD q_i.
  2) The intermediate distribution parameters (l_i, l_hat_i, ...) are updated.
  3) Seed samples S_{i+1} are selected for the next iteration (Eqn. 24).
  4) New samples are produced from seeds using pMCMC (Eqn. 22-23).

The resulting sequence of ISDs and samples is used for multiple importance sampling
(MIS) estimation of the failure probability (Eqn. 9-10).

Developer : Zihan Liao, Weili Xia, Xiao He (Zhejiang University)
Supervisor: Binbin Li (Zhejiang University)
Initial Version: Jan 2023 | Last Modified: Jan 2026 | Python port: Oct 2026
=======================================================================================
"""
import copy

import numpy as np

from ._utils import mround
from .support_functions.DspRst import DspRst
from .support_functions.SamGen import SamGen
from .support_functions.SedSlt import SedSlt
from .support_functions.VabSrk import VabSrk


class SeMIS:
    """Sequential sampling framework (Algorithm 1).

    Properties
        IntBay  : intermediate Bayesian model defining the ISDs
        Sampler : sampling object supporting BayStc and SamGen
        NumSam  : number of samples per iteration N (scalar or vector)
        G       : predefined parameters of the intermediate ISDs
        Verbose : print the progress of each iteration (default True)
    """

    MaxIte = 10000

    def __init__(self, IntBay, Sampler):
        self.Sampler = Sampler
        self.IntBay = IntBay
        self.NumSam = IntBay.Nsam
        self.G = []
        self.X = []
        self.Y = []
        self.Verbose = True

    def RunIte(self):
        """Run the SeMIS iterations until l_{i+1} <= 0; returns a dict with all levels."""
        intBay = self.IntBay
        sampler = copy.copy(self.Sampler)
        max_ite = self.MaxIte
        NumSam = np.atleast_1d(self.NumSam)
        flg_N = NumSam.size > 1
        flg_predef = len(self.G) > 0

        N = np.zeros(max_ite + 1, dtype=int)
        Nc_Ns = np.zeros((max_ite + 1, 2))
        N_cal = np.zeros(max_ite + 1, dtype=int)
        u = [None] * (max_ite + 1)
        y = [None] * (max_ite + 1)
        g = [None] * (max_ite + 1)

        if flg_N:
            N[:NumSam.size] = NumSam
        else:
            N[:] = NumSam[0]
        Nc_Ns[0, :] = [N[0], 1]

        if flg_predef:
            n_predef = len(self.G)
            g[:n_predef] = list(self.G)
        else:
            n_predef = 0
            g[0] = intBay.G

        u_seed = y_seed = None
        flg_cvg = False
        for ite in range(1, max_ite + 1):
            i = ite - 1
            # === sample generation ===
            if np.all(Nc_Ns[i, :] > 0) and not np.isnan(Nc_Ns[i, 1]):
                if ite == 1:
                    u[i], y[i], N_cal[i], intBay = SamGen(intBay, int(Nc_Ns[i, 0]))
                else:
                    u[i], y[i], N_cal[i], intBay = sampler.SamGen(u_seed, y_seed, int(Nc_Ns[i, 1]))
            N[i] = int(np.prod(Nc_Ns[i, :]))

            # === update intermediate distribution ===
            dN_cal = 0
            if ite + 1 <= n_predef:
                intBay = copy.copy(intBay)
                intBay.G = g[i + 1]
            else:
                y[i], g[i + 1], intBay, dN_cal = intBay.UpdObj(u[i], y[i])
            sampler.BayStc = intBay
            N_cal[i] += dN_cal

            # === seed selection (Eqn. 24) ===
            u_seed, y_seed = SedSlt(intBay, u[i], y[i], int(N[i + 1]))
            Nc_Ns[i + 1, 0] = u_seed.shape[1]
            Nc_Ns[i + 1, 1] = mround(N[i + 1] / Nc_Ns[i + 1, 0]) if Nc_Ns[i + 1, 0] > 0 else np.inf

            if self.Verbose:
                DspRst(ite, N[i], Nc_Ns[i, :], N_cal[i])

            # === convergence check ===
            if Nc_Ns[i + 1, 0] <= 0 or ite + 1 == max_ite:
                flg_cvg = False
                break
            elif intBay.FlgCvg:
                flg_cvg = True
                break

        Nite = ite
        u, y, N, Nc_Ns, N_cal = VabSrk(Nite, u, y, N, Nc_Ns, N_cal)
        g = VabSrk(Nite + 1, g)
        return dict(x=u, y=y, g=g, Nsam=N, Nsze=Nc_Ns.astype(int), Ncal=N_cal,
                    Nite=Nite, flg_cvg=flg_cvg, intBay=intBay)
