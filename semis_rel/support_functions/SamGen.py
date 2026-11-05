import numpy as np

from .._utils import randn


def _L_pi(baystc):
    G = baystc.G
    if isinstance(G, dict) and "L_pi" in G:
        return G["L_pi"]
    return np.eye(baystc.Ndim)


def SamGen(baystc, N):
    """Generate i.i.d. samples from the initial ISD q_0 = phi_n(.; sigma^2 I)
    (Eqn. 21 with i = 0: no truncation), via u = L_pi * randn.

    Returns u, y = [log L; log pi; log phi_n; g], N_cal and the updated model.
    """
    u = _L_pi(baystc) @ randn(baystc.Ndim, N)
    baystc = baystc.EvlY(u)
    y = np.vstack([baystc.EvlLKF(), baystc.EvlPDF(), baystc.Y])
    return u, y, baystc.Ncal, baystc
