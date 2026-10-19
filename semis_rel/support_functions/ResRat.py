import copy

import numpy as np

from .._utils import flat, nanmax, reshape_F, tail_shape


def ResRat(intBay, u, y, g):
    """Selection weight (log scale) of existing samples under the NEW ISF L_new:

        log_L_new = log L_new(u) + log pi_new(u) - log L_old(u) - log pi_old(u)

    Since the rPDF pi is unchanged across iterations, this reduces to
    log L_new(u) (samples lie in the support of L_old).
    Returns (log_L_new [Nc x Ns], log_max).
    """
    u2 = flat(u)
    y2 = flat(y)
    intBay = copy.copy(intBay)
    intBay.G = g
    intBay.X = u2
    intBay.Y = y2[2:, :]
    log_L = intBay.EvlLKF()
    log_pi = intBay.EvlPDF()
    with np.errstate(invalid="ignore"):
        tmp = log_L + log_pi - y2[0, :] - y2[1, :]
        log_max = nanmax(tmp)
        tmp = tmp - log_max
    tmp[np.isnan(tmp)] = -np.inf
    return reshape_F(tmp, tail_shape(y)), log_max
