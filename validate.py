"""Validate the Python port against the original MATLAB implementation.

Part 1 - replay (exact): validation/reference/ holds random number streams and
results recorded by running the ORIGINAL MATLAB code once per case
(validation/matlab/record_reference.m). The Python code is driven by the same
random numbers and must reproduce the MATLAB levels, thresholds, number of
performance-function calls, failure probability, single-run CoV bounds and
final samples to round-off. The recorded cases cover the PLS-based SeMIS
(Ex10-Ex12, n = 50) and the SuS baseline (Ex1-Ex9).

Part 2 - statistics: repeated Python runs of SeMIS on the bundled examples;
the mean failure probability is compared with the reference value of the paper.

    python validate.py                   # both parts
    python validate.py --replay          # part 1 only
    python validate.py --nrun 50         # part 2 with 50 runs per example
    python validate.py --examples Ex1 Ex10
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "validation"))

import semis_rel as sr  # noqa: E402
from semis_rel import _utils  # noqa: E402
from replay import ReplayRNG  # noqa: E402

HIGH_DIM = ("Ex10", "Ex11", "Ex12")


def make_example(name, ndim=50):
    cls = getattr(sr, name)
    return cls(ndim) if name in HIGH_DIM else cls()


def make_intbay(method, Tfun):
    return {"low": sr.SeMIS_low_dimension, "high": sr.SeMIS_high_dimension, "sus": sr.SuS}[method](Tfun)


def run_once(method, Tfun, nsam=1000, cov=False):
    intbay = make_intbay(method, Tfun)
    semcs = sr.SeMIS(intbay, sr.aCS())
    semcs.NumSam = nsam
    semcs.Verbose = False
    out = semcs.RunIte()
    if method == "sus":
        return out, out["intBay"].EvlPf(out), np.nan, np.nan
    mis = sr.MIS_Rel(out)
    cv_lw, cv_up = mis.estimateCov() if cov else (np.nan, np.nan)
    return out, mis.Pf_hat, cv_lw, cv_up


def thresholds(method, g):
    return np.array([gi if method == "sus" else gi["l"] for gi in g], dtype=float).ravel()


def rel(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(np.isfinite(b), np.abs(a - b) / np.maximum(np.abs(b), 1e-300), 0)
    if not np.array_equal(np.isinf(a), np.isinf(b)):
        return np.inf
    return float(np.max(d)) if d.size else 0.0


def replay():
    print("Part 1: replay of MATLAB random streams (exact comparison)")
    print(f"{'case':16s} {'levels':>6s} {'Ncal':>6s} {'Pf (Python)':>12s} {'Pf (MATLAB)':>12s} {'rel|dPf|':>9s} "
          f"{'rel|dl|':>8s} {'rel|dCoV|':>9s} {'max|du|':>8s}  result")
    ok_all = True
    for fn in sorted(glob.glob(os.path.join(HERE, "validation", "reference", "rec_*.mat"))):
        tag = os.path.basename(fn)[4:-4]
        method, ex, ndim, _ = tag.split("_")
        rng = ReplayRNG(fn)
        _utils.set_rng(rng)
        try:
            out, pf, cv_lw, cv_up = run_once(method, make_example(ex, int(ndim)), cov=method != "sus")
        finally:
            _utils.set_rng(None)
        r = loadmat(os.path.join(HERE, "validation", "reference", f"res_{tag}.mat"))
        pf_m = r["Pf"].ravel()
        dpf = rel(pf, pf_m)
        dl = rel(thresholds(method, out["g"]), r["l"])
        dcv = 0.0 if method == "sus" else rel(np.r_[cv_lw, cv_up], np.r_[r["cv_lw"].ravel(), r["cv_up"].ravel()])
        u = _utils.flat(out["x"][-1])
        um = _utils.flat(r["u_last"])
        du = float(np.max(np.abs(u - um))) if u.shape == um.shape else np.inf
        ncal = int(np.sum(out["Ncal"]))
        ok = (out["Nite"] == r["Nite"].item() and ncal == r["Ncal"].item() and dpf < 1e-8 and dl < 1e-8
              and dcv < 1e-6 and du < 1e-8 and rng.done)
        ok_all &= ok
        name = f"{method} {ex}" + (f"({ndim})" if ex in HIGH_DIM else "")
        print(f"{name:16s} {out['Nite']:6d} {ncal:6d} {np.ravel(pf)[0]:12.4e} {pf_m[0]:12.4e} {dpf:9.1e} "
              f"{dl:8.1e} {dcv:9.1e} {du:8.1e}  {'PASS' if ok else 'FAIL'}")
    return ok_all


def statistics(nrun, seed, examples):
    print(f"\nPart 2: {nrun} independent Python SeMIS runs per example (N = 1000 per level)")
    print(f"{'example':10s} {'method':6s} {'mean Pf':>10s} {'Pf_ref':>10s} {'mean/ref':>8s} {'CoV':>6s} "
          f"{'|bias|/s.e.':>11s} {'Ncal':>7s} {'time/run':>8s}  result")
    np.random.seed(seed)
    ok_all = True
    for ex in examples:
        method = "high" if ex in HIGH_DIM else "low"
        T = make_example(ex)
        ref = sr.PF_REF[ex]
        pf = np.zeros(nrun)
        ncal = np.zeros(nrun)
        t0 = time.time()
        for i in range(nrun):
            out, p, _, _ = run_once(method, T)
            pf[i] = np.ravel(p)[0]
            ncal[i] = np.sum(out["Ncal"])
        dt = (time.time() - t0) / nrun
        m, s = np.mean(pf), np.std(pf, ddof=1)
        t = abs(m - ref) / (s / np.sqrt(nrun))
        # pass: within 4 standard errors of the reference, or mean within 5 % of it
        ok = t < 4 or abs(m / ref - 1) < 0.05
        ok_all &= ok
        print(f"{ex:10s} {method:6s} {m:10.3e} {ref:10.3e} {m / ref:8.3f} {s / m:6.3f} {t:11.2f} "
              f"{np.mean(ncal):7.0f} {dt:7.1f}s  {'PASS' if ok else 'CHECK'}")
    return ok_all


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", action="store_true", help="only run the replay comparison")
    ap.add_argument("--nrun", type=int, default=20, help="runs per example in part 2 (default 20)")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--examples", nargs="+", default=["Ex1", "Ex2", "Ex3", "Ex4", "Ex10", "Ex11", "Ex12"])
    a = ap.parse_args()
    ok = replay()
    if not a.replay:
        ok &= statistics(a.nrun, a.seed, a.examples)
    print("\nAll checks passed." if ok else "\nSome checks did not pass.")
    sys.exit(0 if ok else 1)
