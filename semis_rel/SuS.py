"""Subset Simulation intermediate distribution shared by both reliability toolboxes.

The driver chooses coordinates and the threshold convention explicitly. This
class does not run iterations; SeMCS owns sampling and stopping.
Both coordinate selectors store Gaussian u; they choose PF.Dist or PF.Rdis
to map u to physical performance-function inputs.
"""
from .SeMCS import state_property
import copy
import numpy as np


from .SeMCS import prescribed_state
from .support_functions.SedSlt import SedSlt

class SuS:
    p = state_property("p",0.1)
    Nsam = state_property("Nsam",1000)
    coordinate_system = state_property("coordinate_system","standard_normal")
    threshold_rule = state_property("threshold_rule","hazen")
    clip_threshold = state_property("clip_threshold",True)
    def __init__(self, PF, coordinate_system="standard_normal",
                 threshold_rule="hazen", clip_threshold=True):
        if coordinate_system not in ("standard_normal", "physical"):
            raise ValueError("coordinate_system must be standard_normal or physical")
        if threshold_rule not in ("hazen", "midpoint"):
            raise ValueError("threshold_rule must be hazen or midpoint")
        self.TF = PF
        self.Nfun = getattr(PF, "Nfun", getattr(PF, "nfun", 1)) + 3
        self.Ndim = PF.Ndim
        self.Ncal = self.X = self.Y = None
        self.G = dict(parameters=np.array(np.inf), p=0.1, L=np.eye(self.Ndim))
        self.G["active"] = np.array(np.nan)
        self.p = 0.1
        self.Nsam = 1000
        self.coordinate_system = coordinate_system
        self.threshold_rule = threshold_rule
        self.clip_threshold = bool(clip_threshold)

    def EvlY(self, u):
        obj = copy.copy(self)
        u = np.asarray(u)
        u = u.reshape(u.shape[0], -1, order="F")
        if self.coordinate_system == "standard_normal":
            x = self.TF.Dist.U2X(u)
            log_pdf = -0.5 * (self.Ndim * np.log(2 * np.pi) + np.sum(u * u, axis=0))
        else:
            x = self.TF.Rdis.U2X(u) if hasattr(self.TF, "Rdis") else u
            log_pdf = -0.5 * (self.Ndim * np.log(2 * np.pi) + np.sum(u * u, axis=0))
        values = self.TF.EvlLSF(x) if hasattr(self.TF, "EvlLSF") else self.TF.evaluate(x)
        values = np.asarray(values, dtype=float)
        nf = self.Nfun - 3
        if nf == 1:
            values = values.reshape(1, -1)
        elif u.shape[1] == 1 and values.ndim == 1:
            values = values[:, None]
        if values.shape != (nf, u.shape[1]):
            raise ValueError("Performance function must return (Nfun, Nsam)")
        obj.X = u
        obj.Y = np.vstack([log_pdf, values])
        obj.Ncal = u.shape[1]
        return obj

    def EvlLKF(self):
        with np.errstate(divide="ignore"):
            return np.log((self.Y[1, :] < self.G["parameters"]).astype(float))

    def EvlPDF(self):
        return self.Y[0, :]

    def UpdObj(self, u, y, p=None):
        obj = copy.copy(self)
        if p is not None:
            obj.p = p
        values = np.asarray(y).reshape(np.shape(y)[0], -1, order="F")[3, :]
        if self.threshold_rule == "midpoint":
            ordered = np.sort(values)
            k = int(np.floor(len(ordered) * obj.p + 0.5))
            if k < 1 or k >= len(ordered):
                raise IndexError("Midpoint index outside sample size")
            threshold = (ordered[k - 1] + ordered[k]) / 2
        else:
            threshold = np.quantile(values, obj.p, method="hazen")
        if self.clip_threshold:
            threshold = max(threshold, 0)
        obj.G = dict(obj.G, parameters=np.array(threshold), p=obj.p)
        return y, obj.G, obj, 0

    @property
    def FlgCvg(self):
        return bool(self.G["parameters"] <= 0)

    def run_defaults(self):
        return dict(Verbose=False)

    def start_run(self, owner, sampler):
        owner.IntBay.G = dict(owner.IntBay.G, p=owner.ConPrb)
        counts = np.atleast_1d(owner.NumSam).astype(int)
        if np.any(counts < 1) or owner.MaxIte < 2:
            raise ValueError("NumSam must be positive and MaxIte at least 2")
        model = copy.copy(owner.IntBay)
        if len(owner.G):
            specified = copy.deepcopy(owner.G[0])
            initial_mapping = isinstance(specified, dict)
            if not isinstance(specified, dict):
                specified = dict(parameters=specified)
            model.G = prescribed_state(model.G, specified, initial=initial_mapping)
        sampler = copy.copy(sampler)
        sampler.BayStc = model
        return dict(
            owner=owner,
            model=model,
            sampler=sampler,
            limit=owner.MaxIte - 1,
            counts=counts,
            nc=int(counts[0]),
            steps=1,
            xseed=None,
            yseed=None,
            x=[],
            y=[],
            g=[copy.deepcopy(model.G)],
            Nsam=[],
            Nsze=[],
            Ncal=[],
            selected=[],
            sigma=[],
            complete=False,
        )

    def advance_level(self, context, record, level):
        context["sigma"].append(getattr(context["sampler"], "sigma", None))

        model = context["model"]
        if level + 1 < len(context["owner"].G):
            model = copy.copy(model)
            previous = model.G
            specified = copy.deepcopy(context["owner"].G[level + 1])
            if not isinstance(specified, dict):
                specified = dict(parameters=specified)
            model.G = prescribed_state(previous, specified)
        else:
            y, _, model, extra = model.UpdObj(record["x"], record["y"], context["owner"].ConPrb)
            context["y"][-1] = y
            record["y"] = y
            context["Ncal"][-1] += extra
        context["model"] = model
        context["sampler"].BayStc = model
        context["g"].append(copy.deepcopy(model.G))

    def select_run_seeds(self, context, record, level):
        counts = context["counts"]
        if counts.size > 1 and level + 1 >= counts.size:
            raise ValueError("NumSam schedule exhausted before seed selection")
        target = int(counts[0] if counts.size == 1 else counts[level + 1])
        xs, ys = SedSlt(context["model"], record["x"], record["y"], target)
        indices = None
        context["xseed"], context["yseed"] = xs, ys
        context["selected"].append(indices)
        context["nc"] = xs.shape[1]
        context["steps"] = int(np.floor(target / context["nc"] + 0.5)) if context["nc"] else 0
        if context["owner"].Verbose:
            print(f"level {level+1}: samples={record['count']}, evaluations={context['Ncal'][-1]}")

    def run_complete(self, context, record, phase):
        if phase != "after_seeds":
            return False
        if context["nc"] <= 0 or len(context["x"]) == context["limit"]:
            return True
        context["complete"] = bool(context["model"].FlgCvg)
        return context["complete"]

    def finish_run(self, context):
        return dict(
            x=context["x"],
            y=context["y"],
            g=context["g"],
            Nsam=np.asarray(context["Nsam"]),
            Nsze=np.asarray(context["Nsze"]),
            Ncal=np.asarray(context["Ncal"]),
            selected=context["selected"],
            sigma=context["sigma"],
            flg_cvg=context["complete"],
            terminal=bool(context["model"].FlgCvg),
            Nite=len(context["x"]),
            intBay=context["model"],
        )

