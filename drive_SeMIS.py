"""SeMIS driver.

Python port of drive_SeMIS.m: runs Sequential Multiple Importance Sampling
(SeMIS) on a selected performance-function example with independent runs and
estimates the failure probability P_hat_f (Eqn. 9-10, 33).

    python drive_SeMIS.py                        # Example 1, 100 runs (as in MATLAB)
    python drive_SeMIS.py --example Ex10 --nrun 20
    python drive_SeMIS.py --cov                  # also the single-run CoV bounds (Eqn. 34-58)
    python drive_SeMIS.py --workers 4            # independent runs in parallel processes

The MATLAB driver runs in a parfor loop; here --workers > 1 uses a process pool.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from semis_rel import (PF_REF, MIS_Rel, SeMIS, SeMIS_high_dimension, SeMIS_low_dimension, aCS,  # noqa: F401
                       Ex1, Ex2, Ex3, Ex4, Ex5, Ex6, Ex7, Ex8, Ex9, Ex10, Ex11, Ex12)

HIGH_DIM = ("Ex10", "Ex11", "Ex12")


def build(example, ndim):
    # -------- Custom example ------------------------------------------------
    # Low-dimensional examples (paper Table 1, n < 20):
    #   Ex1   piecewise-linear (2D),       Pf_ref = 3.20e-5
    #   Ex2   changing topology (2D),      Pf_ref = 1.13e-5
    #   Ex3   linear-log (2D),             Pf_ref = 3.23e-5
    #   Ex4   polynomial benchmark (3D),   Pf_ref = 3.13e-4
    #   Ex5   parallel system (5D),        Pf_ref = 2.13e-4
    #   Ex6   Brotone Bridge tower (5D),   Pf_ref = 2.370e-4
    #   Ex7   noisy PF (6D),               Pf_ref = 5.29e-4
    #   Ex8   RC beam (7D),                Pf_ref = 3.20e-3
    #   Ex9   2-DOF oscillator (8D),       Pf_ref = 3.78e-7
    # High-dimensional examples (paper Table 4, n >= 20):
    #   Ex10(Ndim)  linear hyperplane,     Pf_ref = 1.35e-3 (Ndim = 50)
    #   Ex11(Ndim)  nonlinear PF,          Pf_ref = 4.75e-6 (Ndim = 50)
    #   Ex12(Ndim)  linear (lognormal),    Pf_ref = 1.91e-3 (Ndim = 50)
    Tfun = globals()[example](ndim) if example in HIGH_DIM else globals()[example]()

    # -------- Intermediate Bayesian model (ISD definition) ------------------
    # Low-dimensional problems (n < 20):   Kriging-based hybrid truncation (Eqn. 25)
    # High-dimensional problems (n >= 20): PLS-based hybrid truncation (Eqn. 26)
    IntBay = SeMIS_high_dimension(Tfun) if Tfun.Ndim >= 20 else SeMIS_low_dimension(Tfun)

    # -------- Sampler and SeMIS settings ------------------------------------
    sampler = aCS()                    # pMCMC sampler (Eqn. 22-23)
    semcs = SeMIS(IntBay, sampler)
    semcs.NumSam = 1000                # sample size per level (N)
    return Tfun, semcs


def one_run(args):
    example, ndim, seed, cov = args
    np.random.seed(seed)
    _, semcs = build(example, ndim)
    semcs.Verbose = False
    out_semcs = semcs.RunIte()         # sequential sampling (Algorithm 1)
    mis = MIS_Rel(out_semcs)           # MIS estimator (Eqn. 9-10, 33)
    cv_lw, cv_up = mis.estimateCov() if cov else (np.nan, np.nan)   # single-run CoV bounds (Eqn. 34-58)
    return mis.Pf_hat, np.sum(out_semcs["Ncal"]), cv_lw, cv_up


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--example", default="Ex1", help="Ex1 ... Ex12 (default Ex1)")
    parser.add_argument("--ndim", type=int, default=50, help="dimension of Ex10-Ex12 (default 50)")
    parser.add_argument("--nrun", type=int, default=100, help="number of independent runs (default 100)")
    parser.add_argument("--cov", action="store_true", help="estimate the single-run CoV bounds")
    parser.add_argument("--workers", type=int, default=1, help="parallel worker processes (default 1)")
    parser.add_argument("--seed", type=int, default=0, help="base random seed; run i uses seed + i")
    args = parser.parse_args()

    Tfun, _ = build(args.example, args.ndim)
    Pf_ref = PF_REF.get(args.example) if args.example not in HIGH_DIM or args.ndim == 50 else None
    n_pf = Tfun.Nfun
    N_run = args.nrun

    # -------- independent runs ----------------------------------------------
    jobs = [(args.example, args.ndim, args.seed + irun, args.cov) for irun in range(N_run)]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            res = list(ex.map(one_run, jobs))
    else:
        res = []
        for irun, job in enumerate(jobs):
            res.append(one_run(job))
            print(f"run {irun + 1:4d}/{N_run}: P_hat_f = {np.ravel(res[-1][0])[0]:.4e}, N_cal = {res[-1][1]:.0f}")

    Pf_hat = np.array([r[0] for r in res]).reshape(N_run, n_pf).T     # P_hat_f per run (Eqn. 9)
    N_cal = np.array([r[1] for r in res])                            # PF evaluations per run
    cv_lw_all = np.array([r[2] for r in res]).reshape(N_run, -1).T     # estimated CoV lower bounds
    cv_up_all = np.array([r[3] for r in res]).reshape(N_run, -1).T     # estimated CoV upper bounds

    # -------- Output --------------------------------------------------------
    Pf_mean = np.mean(Pf_hat, axis=1)                                # mean of P_hat_f over runs
    Pf_std = np.std(Pf_hat, axis=1, ddof=1) if N_run > 1 else np.zeros(n_pf)   # std over runs
    cov_emp = Pf_std / Pf_mean                                       # empirical (sampled) CoV
    N_cal_mean = np.mean(N_cal)

    print("================================================")
    print(f"SeMIS results over {N_run} independent runs ({args.example})")
    for i in range(n_pf):
        print(f"  P_hat_f       : {Pf_mean[i]:.4e} +/- {Pf_std[i]:.4e}")
        if Pf_ref:
            print(f"  P_hat_f/Pf_ref: {Pf_mean[i] / Pf_ref:.4f} +/- {Pf_std[i] / Pf_ref:.4f}")
        print(f"  CoV (empirical) : {cov_emp[i]:.4f}")
        if args.cov:
            print(f"  CoV (estimated) : [{np.mean(cv_lw_all[i]):.4f}, {np.mean(cv_up_all[i]):.4f}]")
    print(f"  N_cal (mean)    : {N_cal_mean:.1f}")
    print("================================================")
