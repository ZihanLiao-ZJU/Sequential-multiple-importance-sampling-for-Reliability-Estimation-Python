import numpy as np

from ..log_operations import logmean


def MisInt(log_alpha, log_W):
    """Component-wise MIS estimator in log scale (Eqn. 10 / 33):
    log_Pi_hat(ite) = log P_hat_ite = log[(1/N) sum_k W(U_k^(ite))]."""
    return np.array([np.real(logmean(a + w, None)) for a, w in zip(log_alpha, log_W)])
