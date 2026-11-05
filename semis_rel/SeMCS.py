"""Method-specific sequential policies preserving the source random order."""
import copy
import numpy as np
from semcs_core import Method
from semcs_core.engine import prescribed_state
from .support_functions.SedSlt import SedSlt

class ReliabilityMethod(Method):
    def start(self,owner,rng):
        counts=np.atleast_1d(owner.NumSam).astype(int)
        if np.any(counts<1) or owner.MaxIte<2:
            raise ValueError("NumSam must be positive and MaxIte at least 2")
        model=copy.copy(owner.IntBay)
        if len(owner.G):
            specified=copy.deepcopy(owner.G[0])
            initial_mapping=isinstance(specified,dict)
            if not isinstance(specified,dict): specified=dict(parameters=specified)
            model.G=prescribed_state(model.G,specified,initial=initial_mapping)
        sampler=copy.copy(owner.Sampler);sampler.BayStc=model
        return dict(owner=owner,model=model,sampler=sampler,limit=owner.MaxIte-1,
                    counts=counts,nc=int(counts[0]),steps=1,xseed=None,yseed=None,
                    x=[],y=[],g=[copy.deepcopy(model.G)],Nsam=[],Nsze=[],Ncal=[],
                    selected=[],sigma=[],complete=False)
    def sample(self,c,level):
        if level==0: x,y,calls,model=c["sampler"].SamGen(c["nc"],c["steps"])
        else: x,y,calls,model=c["sampler"].SamGen(c["xseed"],c["yseed"],c["steps"])
        
        c["model"]=model
        r=dict(x=x,y=y,calls=calls,count=c["nc"]*c["steps"],shape=(c["nc"],c["steps"]))
        c["x"].append(x);c["y"].append(y);c["Nsam"].append(r["count"])
        c["Nsze"].append(r["shape"]);c["Ncal"].append(calls)
        c["sigma"].append(getattr(c["sampler"],"sigma",None))
        return r
    def advance(self,c,r,level):
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
    def select_seeds(self,c,r,level):
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
    def is_complete(self,c,r,phase):
        if phase!="after_seeds": return False
        if c["nc"]<=0 or len(c["x"])==c["limit"]: return True
        c["complete"]=bool(c["model"].FlgCvg)
        return c["complete"]
    def finish(self,c):
        return dict(x=c["x"],y=c["y"],g=c["g"],Nsam=np.asarray(c["Nsam"]),
                    Nsze=np.asarray(c["Nsze"]),Ncal=np.asarray(c["Ncal"]),
                    selected=c["selected"],sigma=c["sigma"],flg_cvg=c["complete"],
                    terminal=bool(c["model"].FlgCvg),Nite=len(c["x"]),intBay=c["model"])


from semcs_core import LegacyFacade

class SeMCS(LegacyFacade):
    """Sequential Monte Carlo framework using the shared engine.

    A bound sampler can also be passed alone, preserving the unified entry API.
    """
    default_verbose = False

    def __init__(self, IntBay, Sampler=None):
        if Sampler is None:
            Sampler = IntBay
        else:
            Sampler.BayStc = IntBay
        super().__init__(Sampler, ReliabilityMethod())

