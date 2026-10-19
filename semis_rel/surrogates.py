"""
Surrogate models used to screen samples before calling the performance function.

``fitrgp``      : Gaussian process regression in the configuration used by the
                  MATLAB code (fitrgp with an ARD squared exponential kernel,
                  constant basis, standardized predictors, exact fit), together
                  with ``predict`` returning the mean and the standard
                  deviation of a new response.
``plsregress``  : partial least squares regression with one or more
                  components, returning the coefficient vector [b; a] of
                  MATLAB's ``plsregress`` (SIMPLS).
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, lapack, solve_triangular
from scipy.optimize import minimize


def _chol(K):
    """Lower Cholesky factor, adding a small diagonal jitter if K is numerically singular."""
    jitter = 0.0
    scale = np.mean(np.diag(K))
    for _ in range(10):
        try:
            return cho_factor(K + jitter * np.eye(K.shape[0]), lower=True)
        except np.linalg.LinAlgError:
            jitter = scale * 1e-10 if jitter == 0 else jitter * 10
    return cho_factor(K + jitter * np.eye(K.shape[0]), lower=True)


def _aligned(n):
    """Empty float array of length n starting on a 64-byte boundary."""
    buf = np.empty(n + 8)
    off = (-buf.ctypes.data % 64) // 8
    return buf[off:off + n]


def _aligned_copy(a):
    out = _aligned(a.size)
    out[:] = a
    return out


class GPModel:
    """Fitted Gaussian process (the counterpart of MATLAB's RegressionGP)."""

    def __init__(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        self.mu_x = X.mean(axis=0)
        sd = X.std(axis=0, ddof=1) if n > 1 else np.ones(d)
        sd[~(sd > 0)] = 1.0
        self.sd_x = sd
        Z = (X - self.mu_x) / sd
        self.Z = Z
        self.y = y
        sy = np.std(y, ddof=1) if n > 1 else 1.0
        if not sy > 0:
            sy = 1.0
        self.sigma_lb = 1e-2 * sy             # fitrgp default SigmaLowerBound
        # fitrgp default initial values: length scales = std of the (standardized)
        # predictors, signal and noise standard deviations = std(y)/sqrt(2)
        theta0 = np.concatenate([np.log(np.ones(d)), [np.log(sy / np.sqrt(2))],
                                 [np.log(max(sy / np.sqrt(2) - self.sigma_lb, 1e-12))]])
        self._D2 = [np.subtract.outer(Z[:, j], Z[:, j]) ** 2 for j in range(d)]   # d x [n, n]
        res = minimize(self._nll, theta0, jac=True, method="L-BFGS-B",
                       bounds=[(-10, 10)] * d + [(-20, 20), (-30, 20)])
        self._set(res.x)
        del self._D2

    def _unpack(self, theta):
        d = self.Z.shape[1]
        ell = np.exp(theta[:d])
        sf = np.exp(theta[d])
        sn = self.sigma_lb + np.exp(theta[d + 1])
        return ell, sf, sn

    def _nll(self, theta):
        ell, sf, sn = self._unpack(theta)
        n = self.y.size
        S = np.zeros((n, n))
        for j, D2j in enumerate(self._D2):
            S += D2j / ell[j] ** 2
        E = np.exp(-0.5 * S)
        K = sf ** 2 * E + sn ** 2 * np.eye(n)
        cf = _chol(K)
        H = np.ones((n, 1))
        KiH = cho_solve(cf, H)
        Kiy = cho_solve(cf, self.y)
        beta = (H.T @ Kiy) / (H.T @ KiH)
        r = self.y - beta[0]
        alpha = cho_solve(cf, r)
        logdet = 2 * np.sum(np.log(np.diag(cf[0])))
        nll = 0.5 * r @ alpha + 0.5 * logdet + 0.5 * n * np.log(2 * np.pi)
        # gradient: 0.5 * tr((K^-1 - alpha alpha') dK)
        Kinv, info = lapack.dpotri(cf[0], lower=1)
        Kinv = np.tril(Kinv) + np.tril(Kinv, -1).T
        W = Kinv - np.outer(alpha, alpha)
        WE = W * E
        d = self.Z.shape[1]
        grad = np.zeros_like(theta)
        for j in range(d):
            grad[j] = 0.5 * sf ** 2 * np.sum(WE * self._D2[j]) / ell[j] ** 2
        grad[d] = sf ** 2 * np.sum(WE)
        grad[d + 1] = 0.5 * np.trace(W) * 2 * sn * (sn - self.sigma_lb)
        return nll, grad

    def _set(self, theta):
        ell, sf, sn = self._unpack(theta)
        self.KernelParameters = np.concatenate([ell * self.sd_x, [sf]])  # length scales in original units
        self.Sigma = sn
        self._ell, self._sf, self._sn = ell, sf, sn
        n = self.y.size
        S = np.zeros((n, n))
        for j in range(self.Z.shape[1]):
            S += np.subtract.outer(self.Z[:, j], self.Z[:, j]) ** 2 / ell[j] ** 2
        K = sf ** 2 * np.exp(-0.5 * S) + sn ** 2 * np.eye(n)
        self._L = _chol(K)[0]
        self._L = np.tril(self._L)
        H = np.ones((n, 1))
        KiH = cho_solve((self._L, True), H)
        Kiy = cho_solve((self._L, True), self.y)
        self.Beta = float(((H.T @ Kiy) / (H.T @ KiH)).item())
        self._alpha = cho_solve((self._L, True), self.y - self.Beta)
        # [L^-T, alpha]: k*' [L^-T, alpha] gives v' = k*' L^-T and k*' alpha in one product
        n = self._L.shape[0]
        M = np.empty((n, n + 1))
        M[:, :n] = solve_triangular(self._L, np.eye(n), lower=True).T
        M[:, n] = self._alpha
        self._M = M
        self._ZT = [_aligned_copy(self.Z[:, j]) for j in range(self.Z.shape[1])]

    def predict(self, Xnew):
        """Mean and standard deviation of a new response at Xnew [m x d].

        Every query point is computed on its own, in buffers with a fixed memory
        alignment, so its prediction is bit-for-bit the same whichever other
        points are predicted with it. The screening g_hat <= l_hat compares
        predictions made in different calls with l_hat, the maximum over the
        seeds, so the seed attaining it must reproduce exactly that value; with
        vectorized (SIMD) arithmetic the last bits of a result can otherwise
        depend on the batch size and on the position of a row in memory.
        """
        Zn = (np.atleast_2d(np.asarray(Xnew, dtype=float)) - self.mu_x) / self.sd_x
        m = Zn.shape[0]
        n = self._M.shape[0]
        d2 = _aligned(n)
        t = _aligned(n)
        r = _aligned(n + 1)
        mu = np.empty(m)
        var = np.empty(m)
        for i in range(m):
            d2.fill(0.0)
            for j, zj in enumerate(self._ZT):
                np.subtract(Zn[i, j], zj, out=t)
                np.divide(t, self._ell[j], out=t)
                np.multiply(t, t, out=t)
                np.add(d2, t, out=d2)
            np.multiply(d2, -0.5, out=d2)
            np.exp(d2, out=d2)
            np.multiply(d2, self._sf ** 2, out=d2)      # k* (covariances with the training points)
            np.dot(d2, self._M, out=r)
            mu[i] = self.Beta + r[n]
            var[i] = self._sf ** 2 - np.dot(r[:n], r[:n]) + self._sn ** 2
        return mu, np.sqrt(np.maximum(var, 0))


def fitrgp(X, y):
    """Fit a GP with ARD squared exponential kernel, constant basis and
    standardized predictors (exact fit and prediction)."""
    return GPModel(X, y)


def predict(mdl, Xnew):
    """MATLAB-style ``[mu, sd] = predict(mdl, Xnew)``."""
    return mdl.predict(Xnew)


def plsregress(X, Y, ncomp=1):
    """PLS regression coefficients BETA = [b; a] (SIMPLS), shape [(p+1) x m]."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    mx, my = X.mean(axis=0), Y.mean(axis=0)
    X0, Y0 = X - mx, Y - my
    n, p = X0.shape
    R = np.zeros((p, ncomp))
    Q = np.zeros((Y.shape[1], ncomp))
    V = np.zeros((p, ncomp))
    Cov = X0.T @ Y0
    for i in range(ncomp):
        u_, s_, vt_ = np.linalg.svd(Cov, full_matrices=False)
        ri = u_[:, 0]
        ti = X0 @ ri
        normti = np.linalg.norm(ti)
        ti = ti / normti
        R[:, i] = ri / normti
        Q[:, i] = Y0.T @ ti
        pi = X0.T @ ti
        vi = pi
        if i > 0:
            vi = vi - V[:, :i] @ (V[:, :i].T @ pi)
            vi = vi - V[:, :i] @ (V[:, :i].T @ vi)
        vi = vi / np.linalg.norm(vi)
        V[:, i] = vi
        Cov = Cov - np.outer(vi, vi @ Cov)
        if i > 0:
            Cov = Cov - V[:, :i] @ (V[:, :i].T @ Cov)
    B = R @ Q.T
    return np.vstack([my - mx @ B, B])
