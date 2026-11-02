"""Subset Simulation intermediate distribution shared by both reliability toolboxes.

The driver chooses coordinates and the threshold convention explicitly. This
class does not run iterations; SeMCS owns sampling and stopping.
Both coordinate selectors store Gaussian u; they choose PF.Dist or PF.Rdis
to map u to physical performance-function inputs.
"""
from semcs_core import state_property
import copy
import numpy as np


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

