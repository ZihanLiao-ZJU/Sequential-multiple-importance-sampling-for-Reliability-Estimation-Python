import numpy as np

from ..log_operations import logmean


def MoveBlock(log_w):
    """Moving-block variance estimator (Appendix B, Eqns. B.5-B.10).

    Estimates the variance of a chain-wise mean estimator from Nc Markov chains:
    var_est = R_eff * sigma_C^2 / Nc (Eqn. B.8). ``log_w`` is a [Nc x Ns] array
    (or a list of such arrays). Returns (var_low, var_up).
    """
    items = log_w if isinstance(log_w, (list, tuple)) else [log_w]
    var_est = np.zeros(len(items))
    for ite, lw in enumerate(items):
        lw = np.asarray(lw)
        if lw.ndim == 1:
            lw = lw[:, None]
        log_mean = logmean(lw, None)
        w = np.exp(lw)
        C_hat = np.mean(w, axis=1)                  # chain-wise sub-estimators (Eqn. B.5)
        Nc = C_hat.size
        var_C = np.var(C_hat, ddof=1) if Nc > 1 else 0.0   # sigma_C^2 (Eqn. B.6)
        nb = int(np.floor(Nc / 4 + 0.5))
        Reff = np.zeros(max(nb - 1, 0))
        with np.errstate(invalid="ignore", divide="ignore"):
            for b in range(2, nb + 1):
                N_block = Nc - b + 1
                S_block = np.array([np.mean(C_hat[i:i + b]) for i in range(N_block)])   # (Eqn. B.7)
                var_S = np.var(S_block, ddof=1) * (N_block - 1) / N_block              # (Eqn. B.10)
                Reff[b - 2] = b * var_S / var_C                                         # (Eqn. B.9)
        Reff_max = np.nanmax(np.concatenate([Reff, [1.0]]))
        var_est[ite] = 0.0 if log_mean == -np.inf else Reff_max * var_C / Nc         # (Eqn. B.8)
    return np.sum(var_est), np.sum(np.sqrt(var_est)) ** 2
