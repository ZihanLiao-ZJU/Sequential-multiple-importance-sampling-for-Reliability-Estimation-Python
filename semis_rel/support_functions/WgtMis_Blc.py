import copy

import numpy as np

from .._utils import flat, reshape_F, tail_shape
from ..log_operations import logsum


def WgtMis_Blc(intBay, u, y, g, log_z):
    """Weight of multiple importance sampling (balance heuristic, Eqn. 6), log scale.

    Returns
        log_alpha      : list, log alpha_ite(u) reshaped to [Nc, Ns]
        log_alpha_full : list, Nite x Nsam matrix; row j = log alpha_j(u), the
                         balance-heuristic weight of the j-th ISD at every sample
                         of level ite (used by the (C.9) estimates)
    """
    beta = 1
    Nite = len(y)
    Nc_Ns = [tail_shape(v) for v in y]
    N = np.array([a * b for a, b in Nc_Ns], dtype=float)
    intBay = copy.copy(intBay)
    log_alpha = [None] * Nite
    log_alpha_full = [None] * Nite
    for ite in range(Nite):
        ui = flat(u[ite])
        yi = flat(y[ite])
        log_q = np.zeros((Nite, yi.shape[1]))
        for ite1 in range(Nite):
            intBay.G = g[ite1]
            intBay.X = ui
            intBay.Y = yi[2:, :]
            log_q[ite1, :] = intBay.EvlLKF() + intBay.EvlPDF() - log_z[ite1]
        # balance heuristic (Eqn. 6): full matrix, rows = ISDs
        logN = np.log(N)[:, None]
        mat = beta * logN + beta * log_q - logsum(beta * log_q + beta * logN, 0)
        log_alpha_full[ite] = np.real(mat)
        log_alpha[ite] = reshape_F(np.real(mat[ite, :]), Nc_Ns[ite])
    return log_alpha, log_alpha_full
