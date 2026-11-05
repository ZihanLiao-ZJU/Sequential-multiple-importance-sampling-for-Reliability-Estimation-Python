"""One orchestration loop; every algorithm owns its lifecycle policies.

Methods implement start/sample/advance/select_seeds/is_complete/finish.
The engine never identifies an algorithm, changes its probability law, or draws
random numbers. Frozen level records describe the distribution actually sampled.
"""
import copy
from types import MappingProxyType
import numpy as np

def frozen(value):
    if isinstance(value, np.ndarray):
        value = value.copy()
        value.setflags(write=False)
        return value
    if isinstance(value, dict):
        return MappingProxyType({k: frozen(v) for k,v in value.items()})
    if isinstance(value, (list,tuple)):
        return tuple(frozen(v) for v in value)
    return copy.deepcopy(value)

def state_property(name, default=None):
    """Compatibility attribute backed solely by the method's G mapping."""
    def get(self): return self.G.get(name, default)
    def set_(self,value):
        state=dict(getattr(self,"G",{})); state[name]=value; self.G=state
    return property(get,set_)

def prescribed_state(current, specified, *, initial=False):
    """Apply method parameters while retaining run and kernel configuration.

    Explicit schedules historically replaced the method's parameter payload,
    while options lived on the run/sampler objects. Those options now live in
    G and must survive every prescribed transition. Live adaptation wins at
    later levels, as in the source loops; the first state uses its prescribed
    adaptation keys or the configured kernel defaults if they are absent.
    """
    requested=copy.deepcopy(specified)
    state=copy.deepcopy(current)
    state.update(requested)
    for key in ("run","p","Nsam","sampler","sampler_name"):
        if key in current: state[key]=copy.deepcopy(current[key])
    for key in ("lambda","HMC_time"):
        if initial:
            if key not in requested: state.pop(key,None)
        elif key in current:
            state[key]=copy.deepcopy(current[key])
    return state

class Method:
    """Policy protocol: algorithm-specific methods live with their repository."""
    def snapshot(self, context, record):
        return frozen(dict(G=context["model"].G, x=record["x"], y=record["y"],
                           Ncal=record["calls"], Nsam=record["count"],
                           chain_shape=record["shape"]))
    def is_complete(self, context, record, phase): return False

class Engine:
    def __init__(self, method): self.method=method
    def run(self, owner, rng=None):
        method=self.method
        context=method.start(owner,rng)
        levels=[]
        for level in range(context["limit"]):
            record=method.sample(context,level)
            # Capture sampler adaptation as part of the sampled method state.
            levels.append(method.snapshot(context,record))
            method.advance(context,record,level)
            if method.is_complete(context,record,"before_seeds"): break
            method.select_seeds(context,record,level)
            if method.is_complete(context,record,"after_seeds"): break
        output=method.finish(context)
        output["levels"]=tuple(levels)
        output["engine"]="semcs_core.Engine"
        return output

class LegacyFacade:
    """Old constructors/properties without a private sequential run loop."""
    default_samples=1000
    default_verbose=False
    default_max_iterations=10000
    def __init__(self, Sampler, method):
        if Sampler is None or Sampler.BayStc is None:
            raise ValueError("Sampler with BayStc is required")
        self.Sampler=Sampler
        self.IntBay=Sampler.BayStc
        self._method=method
        self.IntBay.G=dict(self.IntBay.G)
        self.IntBay.G.setdefault("run",{})
        self.NumSam=getattr(self.IntBay,"Nsam",self.default_samples)
        self.ConPrb=self.IntBay.G.get("p",0.1)
        self.Verbose=self.default_verbose
    def _get(self,key,default): return self.IntBay.G.get("run",{}).get(key,default)
    def _set(self,key,value):
        g=dict(self.IntBay.G); run=dict(g.get("run",{})); run[key]=value
        g["run"]=run; self.IntBay.G=g
    @property
    def NumSam(self): return self._get("samples",self.default_samples)
    @NumSam.setter
    def NumSam(self,value): self._set("samples",value)
    @property
    def ConPrb(self): return self.IntBay.G.get("p",0.1)
    @ConPrb.setter
    def ConPrb(self,value): self.IntBay.G=dict(self.IntBay.G,p=value)
    @property
    def MaxIte(self): return self._get("max_iterations",self.default_max_iterations)
    @MaxIte.setter
    def MaxIte(self,value): self._set("max_iterations",value)
    @property
    def Verbose(self): return self._get("verbose",self.default_verbose)
    @Verbose.setter
    def Verbose(self,value): self._set("verbose",bool(value))
    @property
    def G(self): return self._get("schedule",[])
    @G.setter
    def G(self,value): self._set("schedule",value)
    def RunIte(self): return Engine(self._method).run(self)


"""Method-specific sequential policies preserving the source random order."""
import copy
import numpy as np
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

