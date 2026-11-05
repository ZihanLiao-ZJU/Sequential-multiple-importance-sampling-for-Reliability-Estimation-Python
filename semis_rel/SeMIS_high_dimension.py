from .SeMCS import state_property
import copy

import numpy as np
from scipy.stats import norm

from ._utils import flat
from .support_functions.logGauss import logGauss
from .surrogates import plsregress


def _pls_predict(beta, u):
    """g_hat = beta.' * [1; u] (Eqn. 26), accumulated term by term in a fixed order.

    A BLAS product can round differently depending on the memory layout of u,
    which would let the seed that defines l_hat = max g_hat fail its own
    screening g_hat <= l_hat. The fixed summation order keeps the prediction
    identical wherever it is evaluated.
    """
    g_hat = np.repeat(beta[0:1, :].T, u.shape[1], axis=1)
    for j in range(u.shape[0]):
        g_hat = g_hat + beta[j + 1, :][:, None] * u[j, :][None, :]
    return g_hat


from .SeMCS import prescribed_state
from .support_functions.SedSlt import SedSlt

class SeMIS_high_dimension:
    Nsam = state_property("Nsam",1000)
    """SeMIS with Nataf transformation for structural reliability problems.

    High-dimensional variant: PLS-based hybrid truncation (Eqn. 26).
    ISD (Eqn. 21): q_i(u) = phi_n(u; sigma^2 I) * L_i(u) / z_i
      rPDF pi_i(u) = phi_n(u; sigma^2 I)                             (Eqn. 12)
      ISF  L_i(u)  = I[g(u) < l_i] * prod_m I[g_hat_m(u) <= l_hat_m]  (Eqn. 20)
    """

    def __init__(self, PF):
        self.TF = PF
        self.Nfun = PF.Nfun + 3
        self.Ndim = PF.Ndim
        self.X = None
        self.Y = None
        self.Nsam = 1000

        # approximate optimal variance-inflation factor (Eqn. 19)
        beta = 2.33
        lambda_beta = norm.pdf(beta) / (1 - norm.cdf(beta))
        sigma = np.sqrt((self.Ndim + beta * lambda_beta) / self.Ndim)

        L_pi = np.eye(self.Ndim) * sigma
        self.G = {
            "Nsam": 1000, "variance_design_beta": beta,
            "p": 0.1,                              # intermediate level probability p
            "Ite": 0,                              # current iteration counter
            "l": np.inf,                           # exact PF threshold l_i (Eqn. 20)
            "pls_beta": np.zeros((self.Ndim + 1, 0)),  # PLS coefficients [b; a] (Eqn. 26), one column per iteration
            "l_hat": np.zeros((0, 1)),             # surrogate thresholds l_hat_m (Eqn. 27)
            "L_pi": L_pi,
            "Sigma": L_pi @ L_pi.T,
            "mu_pi": np.zeros((self.Ndim, 1)),
        }
        self.Ncal = 0

    def EvlY(self, u):
        """Evaluate [log phi_n(u); g(u)]; g is only evaluated where the PLS screening passes."""
        obj = copy.copy(self)
        u = flat(u)
        obj.X = u
        n_sam = u.shape[1]
        log_phi = logGauss(u)
        # screening by the PLS hyperplanes g_hat_m(u) = a_m' u + b_m (Eqn. 26)
        if obj.G["Ite"] > 0:
            g_hat = _pls_predict(obj.G["pls_beta"], u)
            ind_scr = np.all(g_hat <= obj.G["l_hat"], axis=0)
        else:
            ind_scr = np.ones(n_sam, dtype=bool)
        g_val = np.full(n_sam, np.inf)
        n_cal = 0
        if np.any(ind_scr):
            x = obj.TF.Dist.U2X(u[:, ind_scr])
            g_val[ind_scr] = np.asarray(obj.TF.EvlLSF(x), dtype=float).ravel()
            n_cal = int(np.sum(ind_scr))
        obj.Y = np.vstack([log_phi, g_val])
        obj.Ncal = n_cal
        return obj

    def EvlLKF(self):
        """log ISF (Eqn. 20) with the PLS screening (Eqn. 26)."""
        g_val = self.Y[1, :]
        l = self.G["l"]
        n_sam = g_val.size
        if np.isinf(l):
            return np.zeros(n_sam)
        g_hat = _pls_predict(self.G["pls_beta"], self.X)
        with np.errstate(divide="ignore"):
            return np.log(((g_val < l) & np.all(g_hat <= self.G["l_hat"], axis=0)).astype(float))

    def EvlPDF(self):
        """log rPDF pi_i(u) (Eqn. 12)."""
        return logGauss(self.X, self.G["mu_pi"], self.G["Sigma"])

    def UpdObj(self, u, y, p=None):
        """Update l_i (Eqn. 24), fit the PLS hyperplane on the seeds (Eqn. 26) and
        set the conservative threshold l_hat (Eqn. 27). Returns (y, g, obj, N_cal)."""
        obj = copy.copy(self)
        y2 = flat(y)
        g_val = y2[3, :]
        u2 = flat(u)
        g = dict(obj.G)
        if p is not None:
            g["p"] = p
        n_sam = u2.shape[1]

        l = max(np.quantile(g_val, g["p"], method="hazen"), 0)   # threshold l_i (Eqn. 24)
        ind_seed = g_val < l                                       # seed set S_{i+1}

        pls_beta = plsregress(u2[:, ind_seed].T, g_val[ind_seed], 1)   # PLS coefficients (Eqn. 26)
        g_hat = _pls_predict(pls_beta, u2)
        l_hat_tmp = np.max(g_hat[:, ind_seed], axis=1, keepdims=True)  # conservative threshold (Eqn. 27)

        g["pls_beta"] = np.concatenate([g["pls_beta"], pls_beta], axis=1)
        g["l_hat"] = np.concatenate([g["l_hat"], l_hat_tmp], axis=0)
        g["l"] = l
        g["Ite"] = g["Ite"] + 1
        obj.G = g
        return y, g, obj, 0

    @property
    def FlgCvg(self):
        return bool(self.G["l"] <= 0)

    def run_defaults(self):
        return dict(Verbose=False)

    def start_run(self,owner,sampler):
        owner.IntBay.G=dict(owner.IntBay.G,p=owner.ConPrb)
        counts=np.atleast_1d(owner.NumSam).astype(int)
        if np.any(counts<1) or owner.MaxIte<2:
            raise ValueError("NumSam must be positive and MaxIte at least 2")
        model=copy.copy(owner.IntBay)
        if len(owner.G):
            specified=copy.deepcopy(owner.G[0])
            initial_mapping=isinstance(specified,dict)
            if not isinstance(specified,dict): specified=dict(parameters=specified)
            model.G=prescribed_state(model.G,specified,initial=initial_mapping)
        sampler=copy.copy(sampler);sampler.BayStc=model
        return dict(owner=owner,model=model,sampler=sampler,limit=owner.MaxIte-1,
                    counts=counts,nc=int(counts[0]),steps=1,xseed=None,yseed=None,
                    x=[],y=[],g=[copy.deepcopy(model.G)],Nsam=[],Nsze=[],Ncal=[],
                    selected=[],sigma=[],complete=False)

    def sample_level(self,c,level):
        if level==0: x,y,calls,model=c["sampler"].SamGen(c["nc"],c["steps"])
        else: x,y,calls,model=c["sampler"].SamGen(c["xseed"],c["yseed"],c["steps"])
    
        c["model"]=model
        r=dict(x=x,y=y,calls=calls,count=c["nc"]*c["steps"],shape=(c["nc"],c["steps"]))
        c["x"].append(x);c["y"].append(y);c["Nsam"].append(r["count"])
        c["Nsze"].append(r["shape"]);c["Ncal"].append(calls)
        c["sigma"].append(getattr(c["sampler"],"sigma",None))
        return r

    def advance_level(self,c,r,level):
        model=c["model"]
        if level+1<len(c["owner"].G):
            model=copy.copy(model); previous=model.G
            specified=copy.deepcopy(c["owner"].G[level+1])
            if not isinstance(specified,dict): specified=dict(parameters=specified)
            model.G=prescribed_state(previous,specified)
        else:
            y,_,model,extra=model.UpdObj(r["x"],r["y"],c["owner"].ConPrb)
            c["y"][-1]=y;r["y"]=y;c["Ncal"][-1]+=extra
        c["model"]=model;c["sampler"].BayStc=model
        c["g"].append(copy.deepcopy(model.G))

    def select_run_seeds(self,c,r,level):
        counts=c["counts"]
        if counts.size>1 and level+1>=counts.size:
            raise ValueError("NumSam schedule exhausted before seed selection")
        target=int(counts[0] if counts.size==1 else counts[level+1])
        xs,ys=SedSlt(c["model"],r["x"],r["y"],target)
        indices=None
        c["xseed"],c["yseed"]=xs,ys;c["selected"].append(indices)
        c["nc"]=xs.shape[1]
        c["steps"]=int(np.floor(target/c["nc"]+0.5)) if c["nc"] else 0
        if c["owner"].Verbose:
            print(f"level {level+1}: samples={r['count']}, evaluations={c['Ncal'][-1]}")

    def run_complete(self,c,r,phase):
        if phase!="after_seeds": return False
        if c["nc"]<=0 or len(c["x"])==c["limit"]: return True
        c["complete"]=bool(c["model"].FlgCvg)
        return c["complete"]

    def finish_run(self,c):
        return dict(x=c["x"],y=c["y"],g=c["g"],Nsam=np.asarray(c["Nsam"]),
                    Nsze=np.asarray(c["Nsze"]),Ncal=np.asarray(c["Ncal"]),
                    selected=c["selected"],sigma=c["sigma"],flg_cvg=c["complete"],
                    terminal=bool(c["model"].FlgCvg),Nite=len(c["x"]),intBay=c["model"])
