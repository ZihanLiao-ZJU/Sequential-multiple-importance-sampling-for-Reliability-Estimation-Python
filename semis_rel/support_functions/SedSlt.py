import numpy as np

from .._utils import flat, mround, randi, randperm, randsample
from .ResRat import ResRat
from .UpdY import UpdY


def SedSlt(IntBay, u, y, N):
    """Select the seeds for the next iteration (Eqn. 24):
    S_{i+1} = {k : g(U_k^(i)) < l_{i+1}}, implemented by weighted resampling
    with weight proportional to the new ISF L_{i+1}.

    Returns (u_seed, y_seed).
    """
    y2 = flat(y)
    u2 = flat(u)
    n_sam = y2.shape[1]
    log_L_new, _ = ResRat(IntBay, u2, y2, IntBay.G)
    w_seed = np.exp(log_L_new).ravel(order="F")
    with np.errstate(invalid="ignore", divide="ignore"):
        Nc_eff = mround(np.sum(w_seed) ** 2 / np.sum(w_seed ** 2))
    if not (Nc_eff > 0):
        return np.zeros((u2.shape[0], 0)), np.zeros((y2.shape[0], 0))
    Nc_eff = int(Nc_eff)
    if np.all((w_seed == 0) | (w_seed == 1)):
        idx_can = np.nonzero(w_seed)[0]
    else:
        idx_can = randsample(n_sam, Nc_eff, w_seed)
    u_can = u2[:, idx_can]
    y_can = y2[:, idx_can]
    # numbers of required seeds
    Ns = mround(N / np.arange(1, N + 1))
    Nc_seed = mround(N / Ns)
    Nc_seed = int(Nc_seed[np.nonzero(Nc_eff - Nc_seed >= 0)[0][-1]])
    if Nc_eff >= Nc_seed:
        idx_seed = randperm(Nc_eff, Nc_seed)
    else:
        idx_seed = randi(Nc_eff, Nc_seed)
    u_seed = u_can[:, idx_seed]
    y_seed = y_can[:, idx_seed]
    if y_seed.size:
        y_seed = UpdY(IntBay, u_seed, y_seed)
    return u_seed, y_seed
