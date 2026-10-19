import numpy as np


def plot_2d_lsf(EvlLSF, title=None, lim=8.0, step=0.02):
    """Contour plot of a 2D performance function with its limit state g = 0 (matplotlib)."""
    import matplotlib.pyplot as plt
    x_ref = np.arange(-lim, lim + step / 2, step)
    xx, yy = np.meshgrid(x_ref, x_ref)
    zz = np.vstack([EvlLSF(np.vstack([xx[i], yy[i]])) for i in range(xx.shape[0])])
    fig, ax = plt.subplots()
    cs = ax.contourf(xx, yy, zz, levels=20)
    ax.contour(xx, yy, zz, levels=[0], colors="red", linewidths=1.5)
    fig.colorbar(cs, ax=ax)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    if title:
        ax.set_title(title)
    plt.show()
    return fig
