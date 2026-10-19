import copy

import numpy as np

from .._utils import flat


def UpdY(IntBay, u, y):
    """Recompute the intermediate rows of the function list:

    y = [log L_i (ISF); log pi_i (rPDF); log phi_n; g (Nfun rows)]
    """
    y = flat(y)[2:, :]
    IntBay = copy.copy(IntBay)
    IntBay.X = u
    IntBay.Y = y
    log_pi = IntBay.EvlPDF()
    log_L = IntBay.EvlLKF()
    return np.vstack([log_L, log_pi, y])
