import numpy as np

from .._utils import rand, randn
from .SamGen import _L_pi


class aCS:
    """Adaptive conditional sampling (pCN-type MCMC) with Nataf transformation.

    Constants
        SedSav : use the provided seeds at k = 1
        AcpOpt : target acceptance rate
        NumIte : number of adaptive rounds
        Lambda : initial proposal scaling
    """

    SedSav = False
    AcpOpt = 0.44
    NumIte = 5
    Lambda = 0.6

    def __init__(self, BayStc=None):
        self.BayStc = BayStc

    def SamGen(self, u_seed, y_seed, Ns):
        """Generate MCMC samples targeting q_i = pi_i * L_i / z_i (Eqn. 21):
        pCN proposal (Eqn. 22) with adaptive scaling on the acceptance rate.

        Returns u [Ndim x Nc x Ns], y [Nfun x Nc x Ns], N_cal and the updated model.
        """
        baystc = self.BayStc
        sedsav = self.SedSav
        Ndim = baystc.Ndim
        Nfun = baystc.Nfun
        n_round = self.NumIte
        lambda_pcn = self.Lambda
        acp_opt = self.AcpOpt
        Ns = int(Ns)

        # Cholesky factor of the rPDF covariance (Eqn. 12: sigma^2 I)
        L_pi = _L_pi(baystc)

        u_seed = np.array(u_seed, dtype=float, copy=True)
        y_seed = np.array(y_seed, dtype=float, copy=True)
        Nc = u_seed.shape[1]
        N_cal = 0

        u = np.zeros((Ndim, Nc, Ns))
        y = np.zeros((Nfun, Nc, Ns))

        # Gaussian innovations (Eqn. 22: v ~ N(0, sigma^2 I)) and MH uniforms
        v = np.einsum("ij,j...->i...", L_pi, randn(Ndim, Nc, Ns))
        u_mh = rand(1, Nc, Ns)

        # group chains for adaptive updates
        n_grp = max(int(np.ceil(Nc / n_round)), 1)
        n_round = int(np.ceil(Nc / n_grp))
        idx_grp = np.minimum(np.array([[1, n_grp]] * n_round) + np.arange(0, n_round * n_grp, n_grp)[:, None], Nc)

        sigma0 = 1.0
        for it in range(1, n_round + 1):
            # pCN parameters (Eqn. 22): xi = rho*u + sqrt(1-rho^2)*v
            sigma_pcn = min(lambda_pcn * sigma0, 1)
            rho = np.sqrt(1 - sigma_pcn ** 2)
            n_acp = 0
            dN_cal = 0
            idx_sam = np.arange(idx_grp[it - 1, 0] - 1, idx_grp[it - 1, 1])
            for k in range(Ns):
                if k == 0 and sedsav:
                    u[:, idx_sam, k] = u_seed[:, idx_sam]
                    y[:, idx_sam, k] = y_seed[:, idx_sam]
                else:
                    # pCN proposal (Eqn. 22)
                    u[:, idx_sam, k] = rho * u_seed[:, idx_sam] + sigma_pcn * v[:, idx_sam, k]
                    # evaluate target
                    baystc = baystc.EvlY(u[:, idx_sam, k])
                    y[:, idx_sam, k] = np.vstack([baystc.EvlLKF(), baystc.EvlPDF(), baystc.Y])
                    # MH accept/reject (Eqn. 23, log scale)
                    with np.errstate(invalid="ignore"):
                        idx_rej = np.exp(y[0, idx_sam, k] - y_seed[0, idx_sam]) < u_mh[0, idx_sam, k]
                    idx_sam_rej = idx_sam[idx_rej]
                    # revert rejected samples
                    u[:, idx_sam_rej, k] = u_seed[:, idx_sam_rej]
                    y[:, idx_sam_rej, k] = y_seed[:, idx_sam_rej]
                    n_acp += len(idx_sam) - len(idx_sam_rej)
                    dN_cal += baystc.Ncal
                    # update seeds
                    u_seed[:, idx_sam] = u[:, idx_sam, k]
                    y_seed[:, idx_sam] = y[:, idx_sam, k]
            acp = n_acp / len(idx_sam) / Ns
            lambda_pcn = np.exp(np.log(lambda_pcn) + (acp - acp_opt) / np.sqrt(it))
            N_cal += dN_cal
        return u, y, N_cal, baystc
