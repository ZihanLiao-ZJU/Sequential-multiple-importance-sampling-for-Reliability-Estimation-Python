import copy

import numpy as np

from ._utils import flat, nanmax, reshape_F, tail_shape
from .log_operations import logmean, logsum
from .support_functions.MisInt import MisInt
from .support_functions.MoveBlock import MoveBlock
from .support_functions.WgtMis_Blc import WgtMis_Blc

_REALMIN = np.finfo(float).tiny


class MIS_Rel:
    """Multiple-importance-sampling reliability estimator (Pf + CoV bounds).

    Failure-probability estimator (Eqn. 9-10, 33) and single-run uncertainty
    quantification (Eqn. 34-58 with Appendices A-C):
      P_hat_i = (1/N) sum_k W(U_k^(i); Z_hat)                     (Eqn. 33)
      Var(P_hat_i) ~= MCS noise + normalizing-constant noise
                     + cross-term noise                            (Eqn. 56)
      Var(P_hat_f) in [sum Var_lower, (sum sqrt(Var_upper))^2]     (Eqn. 49/58)
    """

    def __init__(self, inp):
        self.Nite = inp["Nite"]
        self.TF = inp["intBay"].TF
        self.intBay = inp["intBay"]
        self.X = inp["x"]
        self.Y = inp["y"]
        self.G = inp["g"]

        self.log_z, self.Y, self.log_H = self.normalizeY(self.X, self.Y, self.G)
        log_alpha, _ = WgtMis_Blc(self.intBay, self.X, self.Y, self.G, self.log_z)

        n_pf = self.TF.Nfun
        Pf_hat = np.zeros(n_pf)
        self.log_Pi_hat = [None] * n_pf
        for ipf in range(n_pf):
            log_W = []
            for ite in range(self.Nite):
                y_i = flat(self.Y[ite])
                idx_safe = y_i[3 + ipf, :] >= 0
                # log W part before the MIS denominator correction (Eqn. 8 numerator)
                lw = y_i[2, :] - y_i[0, :] - y_i[1, :]
                lw[idx_safe] = -np.inf               # I_F: g >= 0 -> weight 0
                log_W.append(reshape_F(lw, tail_shape(self.Y[ite])))
            log_Pi_hat = MisInt(log_alpha, log_W)   # Eqn. 10/33
            self.log_Pi_hat[ipf] = log_Pi_hat
            Pf_hat[ipf] = np.exp(logsum(log_Pi_hat, None))   # Eqn. 9
        self.Pf_hat = Pf_hat

    def estimateCov(self):
        """Lower and upper bounds of CoV(Pf) from a single run (Eqn. 50-58).

        The single-run variance is decomposed into the three terms of Eqn. 55-57:
          Var(P_i)_lw = MCS noise + normalizing-constant noise (lower)
          Var(P_i)_up = MCS noise + normalizing-constant noise (upper) + cross-term noise
        """
        n_pf = self.TF.Nfun
        M = self.Nite - 1

        # Step 1. incremental level probabilities H_m and their variances
        H_hat_mean = np.zeros(M)
        var_H = np.zeros(M)
        for m in range(1, M + 1):
            H_hat_mean[m - 1] = np.exp(self.log_z[m] - self.log_z[m - 1])
            if m == 1:
                h1 = H_hat_mean[0]
                Nc, Ns = tail_shape(self.Y[0])
                var_H[0] = h1 * (1 - h1) / (Nc * Ns)
            else:
                var_H[m - 1], _ = MoveBlock(self.log_H[m - 1])

        # delta_Hm^2 = Var(H_hat_m) / H_hat_m^2   (Eqn. 34)
        delta_H_sq = np.maximum(var_H / np.maximum(H_hat_mean ** 2, _REALMIN), 0)
        delta_H = np.sqrt(delta_H_sq)

        # Step 2. bounds of the relative covariance C^Z (Eqn. 38)
        Cov_Z_lw = np.zeros((M, M))
        Cov_Z_up = np.zeros((M, M))
        for j in range(1, M + 1):
            for l in range(1, M + 1):
                Cov_Z_lw[j - 1, l - 1] = np.sum(delta_H_sq[:min(j, l)])
                Cov_Z_up[j - 1, l - 1] = np.sum(delta_H[:j]) * np.sum(delta_H[:l])

        # Step 3. normalizing constants: z_0 = 1, z_i = exp(log_z(i+1)) (Eqn. 28)
        z_hat = np.concatenate([[1.0], np.exp(self.log_z[1:])])
        zz = np.outer(z_hat[1:], z_hat[1:])

        # Step 4. bound for |E[Z_hat_j - z_j]| (Eqn. 37 upper)
        bias_Z_upper = np.zeros(M)
        for j in range(1, M + 1):
            dj = delta_H[:j]
            pair_sum = 0.5 * (np.sum(dj) ** 2 - np.sum(dj ** 2))
            bias_Z_upper[j - 1] = z_hat[j] * max(pair_sum, 0)

        # Step 5. MIS denominator part of the current run
        log_alpha, log_alpha_full = WgtMis_Blc(self.intBay, self.X, self.Y, self.G, self.log_z)

        CoV_lw = np.zeros(n_pf)
        CoV_up = np.zeros(n_pf)
        # Step 6. loop over performance functions
        for ipf in range(n_pf):
            Pf_hat_i = max(self.Pf_hat[ipf], _REALMIN)
            var_mcmc = np.zeros(self.Nite)
            var_Z_lw = np.zeros(self.Nite)
            var_Z_up = np.zeros(self.Nite)
            cross_i = np.zeros(self.Nite)          # cross-term noise (Eqn. 55)

            for iLvl in range(self.Nite):
                y_i = flat(self.Y[iLvl])
                sz = tail_shape(self.Y[iLvl])
                idx_safe = y_i[3 + ipf, :] >= 0    # failure indicator I_F = 1 when g(u) < 0
                lw = y_i[2, :] - y_i[0, :] - y_i[1, :]
                lw[idx_safe] = -np.inf
                log_W = reshape_F(lw, sz)
                # final empirical MIS log weight: the sequence used in P_hat_i = mean(W_i)
                log_W_emp = log_alpha[iLvl] + log_W

                # MCS noise: Var_U(P_hat_i | Z_hat) (Eqn. 52/55); i.i.d. at the initial level (Eqn. B.4)
                if iLvl == 0:
                    W0 = np.exp(log_W_emp.ravel(order="F"))
                    var_mcmc[0] = np.var(W0, ddof=1) / W0.size
                else:
                    var_mcmc[iLvl], _ = MoveBlock(log_W_emp)

                W = np.exp(log_W_emp.ravel(order="F"))
                W_mean = np.mean(W)

                if M > 0:
                    # normalizing-constant noise (Eqn. 51), strict (C.9) plug-in estimate of d_ij
                    D_ij = np.zeros(M)
                    for j in range(1, M + 1):
                        alpha_j = np.exp(log_alpha_full[iLvl][j, :])
                        D_ij[j - 1] = np.mean(W * alpha_j / z_hat[j])
                    var_Z_lw[iLvl] = max(D_ij @ (zz * Cov_Z_lw) @ D_ij, 0)
                    var_Z_up[iLvl] = max(D_ij @ (zz * Cov_Z_up) @ D_ij, 0)

                    # cross-term noise: plug-in F_ij (Eqn. 54/55, C.9)
                    ss = np.sum((W - W_mean) ** 2)
                    scale = var_mcmc[iLvl] / max(ss, _REALMIN)
                    F_ij = np.zeros(M)
                    for j in range(1, M + 1):
                        alpha_j = np.exp(log_alpha_full[iLvl][j, :])
                        dW_j = W * alpha_j / z_hat[j]
                        D_hat = np.mean(dW_j)
                        F_ij[j - 1] = scale * np.sum(2 * (W - W_mean) * (dW_j - D_hat))
                    cross_i[iLvl] = np.sum(F_ij * bias_Z_upper)

            var_Phat_lw = np.maximum(var_mcmc + var_Z_lw, 0)
            var_Phat_up = np.maximum(var_mcmc + var_Z_up + cross_i, 0)
            var_Pf_lw = np.sum(var_Phat_lw)
            var_Pf_up = np.sum(np.sqrt(var_Phat_up)) ** 2
            CoV_lw[ipf] = np.sqrt(max(var_Pf_lw, 0)) / Pf_hat_i
            CoV_up[ipf] = np.sqrt(max(var_Pf_up, 0)) / Pf_hat_i
        return CoV_lw, CoV_up

    def normalizeY(self, x, y, g):
        """Sequential estimation of the ISD normalizing constants (Eqn. 28-31):
        log_z(i+1) = log_z(i) + logmean(log H_hat_{i+1}), with log H_hat_{i+1} = log L_{i+1}(U^(i))
        (Eqn. 31); y_norm stores log L_i shifted by -log z_hat_i (Eqn. 32)."""
        log_z = np.zeros(self.Nite)
        y_norm = [np.array(v, copy=True) for v in y]
        log_H = [None] * (self.Nite - 1)
        intBay = copy.copy(self.intBay)
        for ite in range(self.Nite - 1):
            intBay.G = g[ite + 1]
            intBay.X = flat(x[ite])
            yi = flat(y[ite])
            intBay.Y = yi[2:, :]
            log_pi = intBay.EvlPDF()      # log pi_{ite+1}(u)
            log_L = intBay.EvlLKF()       # log L_{ite+1}(u) (Eqn. 20)
            with np.errstate(invalid="ignore"):
                lh = log_L + log_pi - yi[0, :] - yi[1, :]
                lh = lh - nanmax(lh)      # numeric stabilization
            log_z[ite + 1] = log_z[ite] + np.real(logmean(lh, None))
            y_norm[ite + 1][0] = y_norm[ite + 1][0] - log_z[ite + 1]   # Eqn. 32
            log_H[ite] = reshape_F(lh, tail_shape(y[ite]))
        return log_z, y_norm, log_H
