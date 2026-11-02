# SeMIS reliability estimation — Python

## Overview
Sequential multiple importance sampling (SeMIS) estimates small failure probabilities using adaptive intermediate distributions and a balance-heuristic MIS estimator. A separate SuS driver uses the classical product estimator. The default sampler is `aCS`; the four available kernels are a unified implementation and do not imply that every kernel/example combination was reported in the cited paper.

## Structure
- `drive_SeMIS.py` / `drive_SuS.py` — independent-run example drivers.
- `semis_rel/SeMCS.py` — iteration, distribution updates and seed selection.
- `semis_rel/SeMIS_low_dimension.py` / `SeMIS_high_dimension.py` — GP/PLS-based intermediate distributions and convergence; not iteration drivers.
- `semis_rel/SuS.py` — SuS intermediate distribution; `semis_rel/MIS.py` — standalone MIS postprocessor.
- `semis_rel/limit_state_functions/` — bundled Ex1–Ex12 models.
- `semis_rel/support_functions/` — `CW_aMH`, `aCS`, `HMC`, `ES`, shared sampler contract and numerical helpers.

Shared components are vendored inside this repository: no other toolbox checkout is required. Retain the included helper files and license when copying the toolbox.

## Quick start
Run from this repository's root. The explicit small runs below are smoke examples, not the paper's full repetition protocol.

```bash
python -m pip install -r requirements.txt
python drive_SeMIS.py --example Ex10 --ndim 20 --samples 200 --repeats 2 --seed 2026
python drive_SuS.py --example Ex10 --ndim 20 --samples 200 --repeats 2 --seed 2026
```

The driver returns or prints failure-probability estimates and actual model-call counts. The Python function API returns per-run `completed` flags and an `unfinished` count; the CLI is a compact summary. Inspect these flags before treating a run as completed; an estimate alone does not prove convergence.

## Examples and samplers
| Example | Model / dimension | Driver selector |
|---|---|---|
| Ex1 | Piecewise-linear / 2D | `Ex1` |
| Ex2 | Changing topology / 2D | `Ex2` |
| Ex3 | Linear-log / 2D | `Ex3` |
| Ex4 | Polynomial benchmark / 3D | `Ex4` |
| Ex5 | Parallel system / 5D | `Ex5` |
| Ex6 | Brotone Bridge tower / 5D | `Ex6` |
| Ex7 | Noisy performance function / 6D | `Ex7` |
| Ex8 | Reinforced-concrete beam / 7D | `Ex8` |
| Ex9 | Two-DOF damped oscillator / 8D | `Ex9` |
| Ex10 | Linear standard-normal model / configurable n | `Ex10` |
| Ex11 | Nonlinear standard-normal model / configurable n | `Ex11` |
| Ex12 | Linear lognormal model / configurable n | `Ex12` |

- Available samplers: `CW_aMH`, `aCS`, `HMC`, `ES`; default `aCS`.
- Restored defaults: Ex1, 1,000 samples per level, 100 independent repetitions, base seed 0, conditional probability 0.1, and dimension 2 unless explicitly supplied for Ex10–Ex12. Each repetition uses base seed plus its zero-based index.
- Ex1–Ex9 retain their fixed dimensions. The shared sampler contract uses the model-defined dimension rather than a hard-coded 2D state. Ex10–Ex12 accept the dimension argument; the driver selects GP below 20 dimensions and PLS otherwise. With dimension omitted they therefore use the 2D GP path, not a 50D PLS case. Pass `20`, `50`, `100`, or `1000` explicitly when selecting those high-dimensional paper conditions.
- Select a kernel explicitly:

```python
from drive_SeMIS import drive_SeMIS
results = drive_SeMIS("Ex10", 200, 2, 2026, 20, sampler_name="ES")
```

## Dependencies
Python 3.9 or later; NumPy >= 1.22 and SciPy >= 1.8. Matplotlib >= 3.5 is only needed for optional plots (it is listed in `requirements.txt`). The custom GP/PLS implementation uses NumPy/SciPy, not scikit-learn.

## Method & validation notes
- **Implementation:** SeMIS preserves its existing GP/PLS screening and MIS estimator. SuS retains standard-normal coordinates, Hazen quantiles and thresholds clipped at zero. Failure is strictly `g < 0`. Different toolboxes may select different SuS policies explicitly; common code does not silently unify them.
- **Sampler policy:** aCS starts with unit base sigma and carries its adapted `lambda` in `G` into the next level. Componentwise ES uses an initial angle in `[0, pi]`. HMC uses exact Gaussian-reference Hamiltonian proposals with a likelihood acceptance correction, not generic physical-space gradient HMC. Nataf maps evaluations to physical variables. These are current unified kernels, not a claim of untouched original-paper sampling code.
- **Outputs:** Empirical SD uses independent repeats and sample normalization; CoV is SD divided by the mean. Model-call counts count actual evaluated sample points, not nominal sample totals or screened-out proposals. Optional single-run CoV bounds are a different quantity from empirical repeat-to-repeat CoV.
- **Nataf compatibility:** The author implementation's positive-interval inversion is retained for nonstandard marginal pairs, including its limitations for negative correlations. Normal/lognormal direct transformations and the author's GEV CDF clipping are retained. This release does not silently extend that numerical method.
- **Validation scope:** Focused same-seed current-to-cleanup regression and standalone entrypoint checks are tracked separately from the original-paper comparisons. Python low-dimensional runs use the custom GP; native MATLAB low-dimensional GP results were not validated in the available environment. Interface/trajectory/accounting checks do not establish exact reproduction of every published statistic.
- **Historical scope:** Earlier framework cleanup separated SeMCS, distribution models and MIS. The user's later MATLAB source also changed optional single-run CoV bounds from stochastic normalizer perturbations to analytic bounds. The later author MATLAB implementation is the non-MCMC baseline, not the earliest stochastic-bound version. The recent imposed seed default is reverted; the dimension default is aligned to the author script's value 2. An unqualified claim that all changes since the earliest upload were MCMC-only would be incorrect.
- **Limitations:** The custom SciPy GP follows the same general ARD squared-exponential/constant-mean model family but optimizer, noise/jitter and normalization details do not establish exact equivalence to MATLAB `fitrgp`. Ex8's public reference remains `3.20e-3`, whereas the inspected author-proof table reports `3.01e-3`; neither the model nor its reference was changed to force agreement. Ex10's public `3*sqrt(n)-sum(X)` is a positive scaling of the proof's `3-sum(X)/sqrt(n)`: the zero-threshold failure event agrees, not necessarily adaptive trajectories or costs. The author proof is not a verified final publisher version.

## Reference & license
Weili Xia, Zihan Liao, and Binbin Li. **Sequential multiple importance sampling for robust and efficient (possibly high-dimensional) reliability estimation.** *Mechanical Systems and Signal Processing*, 259 (2026), 114798. [doi:10.1016/j.ymssp.2026.114798](https://doi.org/10.1016/j.ymssp.2026.114798).

Licensed under the [MIT License](LICENSE). The original copyright notice is retained in `LICENSE`.
