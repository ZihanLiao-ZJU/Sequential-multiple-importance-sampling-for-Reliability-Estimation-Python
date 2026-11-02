"""Drive classical SuS through SeMCS using the reliability examples."""
import argparse
import numpy as np
from drive_SeMIS import HIGH_DIM
from semis_rel.support_functions.results import summarize
import semis_rel as sr


def estimate_pf(result, clip_threshold=True):
    """Classical product estimator, using zero for an unclipped final target."""
    probability = 1.0
    for i, y in enumerate(result["y"]):
        values = np.asarray(y).reshape(np.shape(y)[0], -1, order="F")[3, :]
        threshold = (0 if i == len(result["y"]) - 1 and not clip_threshold
                     else result["g"][i + 1]["parameters"])
        probability *= np.mean(values < threshold)
    return probability


def drive_SuS(example="Ex1", N=1000, repeats=100, seed=0, ndim=2, sampler_name="aCS"):
    if isinstance(example, int):
        example = f"Ex{example}"
    if repeats < 1 or N < 20:
        raise ValueError("repeats must be positive and N at least 20")
    records = []
    for i in range(repeats):
        np.random.seed(seed + i)
        cls = getattr(sr, example)
        pf = cls(ndim) if example in HIGH_DIM else cls()
        model = sr.SuS(pf, "standard_normal", "hazen", True)
        sampler = getattr(sr, sampler_name)(model)
        engine = sr.SeMCS(sampler)
        engine.NumSam = N
        engine.Verbose = False
        out = engine.RunIte()
        records.append(dict(seed=seed + i, probability=[float(estimate_pf(out, model.clip_threshold))],
                            joint_calls=int(np.sum(out["Ncal"])),
                            layer_joint_calls=[out["Ncal"].tolist()],
                            actual_level_samples=[out["Nsam"].tolist()],
                            completed=bool(out["flg_cvg"])))
    return summarize(example, "SuS", N, repeats, seed, records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", default="Ex1")
    parser.add_argument("--ndim", type=int, default=2)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampler", choices=["CW_aMH", "aCS", "HMC", "ES"], default="aCS")
    args = parser.parse_args()
    results = drive_SuS(args.example, args.samples, args.repeats, args.seed, args.ndim, args.sampler)
    print("Mean failure probability:", results["mean"])
    print("Mean joint evaluations:", results["mean_joint_calls"])
