"""Edit the settings below, then run this script from the repository root."""
import numpy as np

from semis_rel import (PF_REF, MIS, SeMCS, SeMIS_high_dimension, SeMIS_low_dimension, aCS, CW_aMH, HMC, ES,  # noqa: F401
                       Ex1, Ex2, Ex3, Ex4, Ex5, Ex6, Ex7, Ex8, Ex9, Ex10, Ex11, Ex12)

def summarize(example, algorithm, N, repeats, records):
    estimates = np.asarray([row["probability"] for row in records])
    if estimates.ndim == 1:
        estimates = estimates[:, None]
    mean = estimates.mean(axis=0)
    std = (estimates.std(axis=0, ddof=1) if repeats > 1
           else np.full(estimates.shape[1], np.nan))
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = std / mean
    return dict(example=example, algorithm=algorithm, N=N, repeats=repeats,
                mean=mean.tolist(), standard_deviation=std.tolist(), cov=cov.tolist(),
                mean_joint_calls=float(np.mean([r["joint_calls"] for r in records])),
                unfinished=sum(not r["completed"] for r in records), records=records)


# %% Problem and sampling settings
Tfun = Ex1()
# Tfun = Ex10(50)            # Examples 10–12 accept a dimension
N = 1000                    # Constant samples per level
Nrun = 100                  # Set to 1 for a single run
SamplerClass = aCS          # CW_aMH, aCS, HMC or ES
cov = False                 # Optional single-run CoV bounds
example = type(Tfun).__name__

# %% Model and sampler
model = SeMIS_high_dimension(Tfun) if Tfun.Ndim >= 20 else SeMIS_low_dimension(Tfun)
simulation = SeMCS(SamplerClass(model))
simulation.NumSam = N

# %% Run simulations
records = []
for i in range(Nrun):
    out = simulation.RunIte()
    estimator = MIS(out)
    row = dict(probability=estimator.Pf_hat.tolist(),
               joint_calls=int(np.sum(out["Ncal"])),
               layer_joint_calls=[out["Ncal"].tolist()],
               actual_level_samples=[out["Nsam"].tolist()],
               completed=bool(out["flg_cvg"]))
    if cov:
        lower, upper = estimator.estimateCov()
        row["cov_lower"] = np.atleast_1d(lower).tolist()
        row["cov_upper"] = np.atleast_1d(upper).tolist()
    records.append(row)
results = summarize(example, "SeMIS", N, Nrun, records)

Pf_ref = PF_REF.get(example) if not isinstance(Tfun, (Ex10, Ex11, Ex12)) or Tfun.Ndim == 50 else None
print(f"SeMIS results over {Nrun} independent runs ({example})")
for j, mean in enumerate(results["mean"]):
    print(f"  P_hat_f: {mean:.4e}; CoV: {results['cov'][j]:.4f}")
    if Pf_ref:
        print(f"  P_hat_f/Pf_ref: {mean / Pf_ref:.4f}")
print(f"  N_cal (mean): {results['mean_joint_calls']:.1f}")
