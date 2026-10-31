"""Summaries of independent runs without discarding unfinished repetitions."""
import numpy as np


def summarize(example, algorithm, N, repeats, seed, records):
    estimates = np.asarray([row["probability"] for row in records])
    if estimates.ndim == 1:
        estimates = estimates[:, None]
    mean = estimates.mean(axis=0)
    std = (estimates.std(axis=0, ddof=1) if repeats > 1
           else np.full(estimates.shape[1], np.nan))
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = std / mean
    return dict(example=example, algorithm=algorithm, N=N, repeats=repeats, seed=seed,
                mean=mean.tolist(), standard_deviation=std.tolist(), cov=cov.tolist(),
                mean_joint_calls=float(np.mean([r["joint_calls"] for r in records])),
                unfinished=sum(not r["completed"] for r in records), records=records)
