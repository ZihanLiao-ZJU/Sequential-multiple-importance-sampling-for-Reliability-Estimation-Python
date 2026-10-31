"""Sequential Monte Carlo driver shared by the reliability toolboxes."""
import copy
import numpy as np
from .support_functions.SedSlt import SedSlt


class SeMCS:
    """Iterate sampling, intermediate-state updates and seed selection.

    Sampler.BayStc is the distribution model. NumSam may be a scalar or a
    per-level vector; G/H optionally specify intermediate parameters.
    """
    MaxIte = 10000

    def __init__(self, Sampler):
        self.Sampler = Sampler
        self.IntBay = Sampler.BayStc
        self.NumSam = self.IntBay.Nsam
        self.ConPrb = (self.IntBay.G["p"] if isinstance(self.IntBay.G, dict)
                       else self.IntBay.p)
        self.G = []
        self.H = []
        self.Verbose = False

    def RunIte(self):
        model = self.IntBay
        sampler = copy.copy(self.Sampler)
        counts = np.atleast_1d(self.NumSam).astype(int)
        if np.any(counts < 1) or self.MaxIte < 2:
            raise ValueError("NumSam must be positive and MaxIte at least 2")
        xs, ys, ns, sizes, calls = [], [], [], [], []
        selected, scales = [], []
        gs = [copy.deepcopy(self.G[0] if len(self.G) else model.G)]
        hs = [copy.deepcopy(self.H[0] if len(self.H) else getattr(model, "H", None))]
        model = copy.copy(model)
        model.G = (gs[0] if isinstance(gs[0], dict)
                   else dict(model.G, parameters=gs[0]))
        gs[0] = copy.deepcopy(model.G)
        if hs[0] is not None:
            model.H = hs[0]
        sampler.BayStc = model
        nc, steps = int(counts[0]), 1
        xseed = yseed = None
        complete = False
        for level in range(self.MaxIte - 1):
            if level == 0:
                x, y, ncal, model = sampler.SamGen(nc, steps)
            else:
                x, y, ncal, model = sampler.SamGen(xseed, yseed, steps)
            xs.append(x)
            ys.append(y)
            ns.append(nc * steps)
            sizes.append((nc, steps))
            calls.append(ncal)
            scales.append(getattr(sampler, "sigma", None))
            if level + 1 < len(self.G):
                model = copy.copy(model)
                previous_state = model.G
                specified = copy.deepcopy(self.G[level + 1])
                model.G = (dict(previous_state, **specified) if isinstance(specified, dict)
                           else dict(previous_state, parameters=specified))
                for key in ("lambda", "HMC_time"):
                    if key in previous_state:
                        model.G[key] = previous_state[key]
                if level + 1 < len(self.H):
                    model.H = self.H[level + 1]
            else:
                y, _, model, extra = model.UpdObj(x, y, self.ConPrb)
                ys[-1] = y
                calls[-1] += extra
            sampler.BayStc = model
            gs.append(copy.deepcopy(model.G))
            hs.append(copy.deepcopy(getattr(model, "H", None)))
            if counts.size > 1 and level + 1 >= counts.size:
                raise ValueError("NumSam schedule exhausted before seed selection")
            target = int(counts[0] if counts.size == 1 else counts[level + 1])
            xseed, yseed = SedSlt(model, x, y, target)
            indices = None
            selected.append(indices)
            nc = xseed.shape[1]
            steps = int(np.floor(target / nc + 0.5)) if nc else 0
            if self.Verbose:
                print(f"level {level + 1}: samples={ns[-1]}, evaluations={calls[-1]}")
            if nc <= 0 or level + 2 == self.MaxIte:
                break
            if model.FlgCvg:
                complete = True
                break
        return dict(x=xs, y=ys, g=gs, h=hs, Nsam=np.asarray(ns),
                    Nsze=np.asarray(sizes), Ncal=np.asarray(calls),
                    selected=selected, sigma=scales, flg_cvg=complete,
                    terminal=bool(model.FlgCvg), Nite=len(xs), intBay=model)
