# SeMIS: Sequential Multiple Importance Sampling for Reliability Estimation (Python)

Python implementation of **Sequential Multiple Importance Sampling (SeMIS)** for estimating small failure probabilities, including in high-dimensional problems. This is a line-by-line port of the MATLAB code in [Sequential-multiple-importance-sampling-for-Reliability-Estimation](https://github.com/ZihanLiao-ZJU/Sequential-multiple-importance-sampling-for-Reliability-Estimation), which accompanies the paper:

> **Sequential multiple importance sampling for robust and efficient (possibly high-dimensional) reliability estimation.**
> *Mechanical Systems and Signal Processing*, 259 (2026) 114798. doi: [10.1016/j.ymssp.2026.114798](https://doi.org/10.1016/j.ymssp.2026.114798)

## Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Repository structure](#repository-structure)
- [Quick start](#quick-start)
- [Built-in examples](#built-in-examples)
- [Using a custom performance function](#using-a-custom-performance-function)
- [Outputs](#outputs)
- [Optional single-run CoV bounds](#optional-single-run-cov-bounds)
- [Code-to-paper mapping](#code-to-paper-mapping)
- [Validation against MATLAB](#validation-against-matlab)
- [Differences from the MATLAB version](#differences-from-the-matlab-version)
- [Citation](#citation)
- [License](#license)

## Overview

SeMIS estimates the failure probability

$$p_f = P\{g(\mathbf X) < 0\} = \int \phi_n(\mathbf u)\, I_{\mathcal F}(\mathbf u)\, \mathrm d\mathbf u$$

by constructing a sequence of intermediate importance sampling distributions (ISDs) that move progressively toward the failure domain:

1. The intermediate distributions are defined adaptively, using a Kriging-based truncation for low-dimensional problems or a PLS-based truncation for high-dimensional problems.
2. Samples are drawn from each intermediate distribution with a preconditioned Crank–Nicolson (pCN) MCMC sampler with adaptive step size.
3. All samples are combined with the balance-heuristic multiple importance sampling (MIS) estimator to obtain the failure probability.

As an option, lower and upper bounds of the coefficient of variation (CoV) can be estimated from a single run. All probability weights are computed in the log domain for numerical stability.

The class structure, method names and variable names follow the MATLAB code, so the two versions can be read side by side.

## Requirements

- Python 3.9 or later
- NumPy and SciPy (the Kriging model `fitrgp` and `plsregress` are implemented in `semis_rel/surrogates.py`)
- Matplotlib (optional, only for plotting)

```bash
pip install -r requirements.txt
```

The Gaussian process fit solves dense linear systems with up to 1000 points. If NumPy uses a multithreaded BLAS and several runs are executed in parallel, setting `OPENBLAS_NUM_THREADS=1` (or `OMP_NUM_THREADS=1`) avoids oversubscribing the CPU.

## Repository structure

```
.
├── drive_SeMIS.py               # main driver: selected example, 100 independent runs
├── validate.py                  # comparison with the MATLAB implementation
├── semis_rel/
│   ├── SeMIS.py                 # sequential sampling framework (Algorithm 1)
│   ├── SeMIS_low_dimension.py   # intermediate distributions, Kriging-based truncation (Eqn. 25)
│   ├── SeMIS_high_dimension.py  # intermediate distributions, PLS-based truncation (Eqn. 26)
│   ├── MIS_Rel.py               # MIS estimator and optional single-run CoV bounds (Eqn. 34–58)
│   ├── SuS.py                   # subset simulation, used as a comparison baseline
│   ├── surrogates.py            # Gaussian process regression (fitrgp/predict) and PLS regression (plsregress)
│   ├── Nataf.py, distributions.py
│   ├── limit_state_functions/   # built-in performance functions Ex1 ... Ex12
│   ├── support_functions/       # sampler (aCS), MIS weights, seed selection, ...
│   └── log_operations.py        # log-domain arithmetic (logsum, logplus, logmean, ...)
├── validation/                  # MATLAB random streams and results used by validate.py
├── LICENSE
└── README.md
```

## Quick start

Run the driver from the repository root:

```bash
python drive_SeMIS.py
```

By default this runs Example 1 (`Ex1`) with 100 independent runs and prints the estimated failure probability, its empirical CoV, and the average number of performance-function evaluations. The MATLAB driver uses `parfor`; here `--workers 4` runs the independent runs in 4 processes. Other options: `--example Ex10` selects an example, `--nrun` sets the number of runs, `--cov` adds the single-run CoV bounds and `--seed` fixes the random seeds.

The same steps from Python code:

```python
from semis_rel import Ex1, SeMIS_low_dimension, SeMIS, aCS, MIS_Rel

Tfun = Ex1()
IntBay = SeMIS_low_dimension(Tfun)     # SeMIS_high_dimension(Tfun) for n >= 20
semcs = SeMIS(IntBay, aCS())
semcs.NumSam = 1000                    # sample size per level (N)

out_semcs = semcs.RunIte()             # sequential sampling (Algorithm 1)
mis = MIS_Rel(out_semcs)               # MIS estimator (Eqn. 9-10, 33)
print(mis.Pf_hat, out_semcs["Ncal"].sum())
cv_lw, cv_up = mis.estimateCov()       # optional single-run CoV bounds (Eqn. 34-58)
```

## Built-in examples

Select an example with `--example`, or by editing `build` in `drive_SeMIS.py`. The examples are numbered as in Tables 1 and 4 of the paper; `semis_rel.PF_REF` holds the reference values.

| Example | Description | Dimension | `Pf_ref` |
|---|---|---|---|
| `Ex1` | piecewise-linear | 2 | 3.20e-5 |
| `Ex2` | changing topology | 2 | 1.13e-5 |
| `Ex3` | linear-log | 2 | 3.23e-5 |
| `Ex4` | polynomial benchmark | 3 | 3.13e-4 |
| `Ex5` | parallel system | 5 | 2.13e-4 |
| `Ex6` | Brotone Bridge tower | 5 | 2.370e-4 |
| `Ex7` | noisy performance function | 6 | 5.29e-4 |
| `Ex8` | RC beam | 7 | 3.20e-3 |
| `Ex9` | 2-DOF oscillator | 8 | 3.78e-7 |
| `Ex10(50)` | linear hyperplane | 50 | 1.35e-3 |
| `Ex11(50)` | nonlinear performance function | 50 | 4.75e-6 |
| `Ex12(50)` | linear, lognormal inputs | 50 | 1.91e-3 |

The driver uses `SeMIS_high_dimension` (PLS-based truncation) for n ≥ 20, i.e. for `Ex10`–`Ex12`, and `SeMIS_low_dimension` (Kriging-based truncation) otherwise; `--ndim` sets the dimension of `Ex10`–`Ex12`.

## Using a custom performance function

1. Copy one of `semis_rel/limit_state_functions/Ex*.py` (each is a class with the same interface) and rename it, e.g. `MyProblem`.
2. Define:
   - `Dist`: probabilistic model of the input vector `X`, a `Nataf` object built from a list of marginal distributions (`makedist`) and their correlation matrix;
   - `Ndim`, `Nfun`: input dimension and number of performance functions;
   - `EvlLSF(self, x)`: evaluates the performance function on the samples `x` (`Ndim × Nsam`, in the physical space given by `Dist`), where `g < 0` is the failure domain; for a single performance function, return a vector of length `Nsam` (SuS accepts either a row or column vector). `SuS` additionally accepts an `(Nfun, Nsam)` matrix, one performance function per row; its first row controls the sampling thresholds, while `MIS_Rel` retains all rows for estimation.
3. Run SeMIS with `IntBay = SeMIS_low_dimension(MyProblem())` for n < 20 or `IntBay = SeMIS_high_dimension(MyProblem())` for n ≥ 20, as in the Python example above.

Minimal example: a linear limit state in two standard normal variables, with `Pf = Φ(−3.5)`.

```python
import numpy as np
from semis_rel import Nataf, makedist


class MyProblem:
    def __init__(self):
        self.Nfun = 1
        self.Ndim = 2
        self.Dist = Nataf([makedist("Normal", mu=0, sigma=1)] * 2, np.eye(2))

    def EvlLSF(self, x):
        # x : [Ndim x Nsam] samples, g : [Nsam] performance function (g < 0: failure)
        return 3.5 - (x[0] + x[1]) / np.sqrt(2)
```

## Outputs

The driver prints the following quantities, averaged over the independent runs:

| Output | Meaning |
|---|---|
| `P_hat_f` | estimated failure probability (Eqn. 9–10, 33) |
| `P_hat_f/Pf_ref` | estimate normalized by the reference value (cf. Tables 2 and 5 of the paper) |
| `CoV (empirical)` | coefficient of variation across the independent runs |
| `N_cal (mean)` | average number of performance-function evaluations per run |

`SeMIS.RunIte()` returns a dictionary with the samples `x`, function values `y` and ISD parameters `g` of every level, the number of performance-function calls per level `Ncal` and the number of levels `Nite`. Set `semcs.Verbose = False` to silence the progress printed at every level.

## Optional single-run CoV bounds

The paper (Eqn. 34–58, Appendices A–C) derives analytical bounds of the CoV from a single run, using a three-term decomposition (MCS noise, normalizing-constant noise and cross-term noise). This is disabled by default. To enable it, run the driver with `--cov`, or call

```python
cv_lw, cv_up = mis.estimateCov()   # single-run CoV bounds (Eqn. 34–58)
```

## Code-to-paper mapping

| Step | Code | Paper |
|---|---|---|
| Sequence of intermediate distributions | `SeMIS.RunIte` / `UpdObj` | Eqn. (11)–(21), (24)–(27) |
| pCN-MCMC sampling with adaptive step | `support_functions/aCS.py` | Eqn. (22)–(23), Algorithm 2 |
| Normalizing constants of the ISDs | `MIS_Rel.normalizeY` | Eqn. (28)–(32) |
| Balance-heuristic MIS estimator | `MIS_Rel` / `WgtMis_Blc` / `MisInt` | Eqn. (4)–(10), (33) |
| CoV bounds (optional) | `MIS_Rel.estimateCov` | Eqn. (34)–(58), App. A–C |

## Validation against MATLAB

```bash
python validate.py
```

1. **Reference replay (Octave, not native MATLAB).** `validation/reference/` contains recorded random draws and results from the MATLAB-source implementation running under GNU Octave 8.4 with compatibility shims, for PLS-based SeMIS on `Ex10`–`Ex12` (n = 50, two seeds each) and SuS on `Ex1`–`Ex9`. `validate.py` compares levels, thresholds, performance-function calls, failure probability, single-run CoV bounds and final samples. Every one of the 15 expected recording/result pairs is required: missing or empty files, NaNs and mismatched infinities fail validation. This is a reference replay, **not native MATLAB dynamic-equivalence testing**. Recorded `randsample` indices do not independently validate the weights passed to resampling; equal NumPy/MATLAB seeds also do not imply equal random streams.
2. **Statistics.** Independent Python runs of SeMIS per example (`--nrun`, default 20; `--examples` to choose); the mean failure probability is compared with the reference value of the paper.

Previously recorded replay results (`python validate.py --replay`; rerun locally to verify your environment):

| Method | Cases | Replay vs Octave reference |
|---|---|---|
| SeMIS, PLS-based truncation | `Ex10`, `Ex11`, `Ex12` (n = 50), seeds 1 and 2 | identical (levels, thresholds, N_cal, P_hat_f, CoV bounds, samples) |
| SuS baseline | `Ex1` … `Ex9` | identical (levels, thresholds, N_cal, P_hat_f, samples) |

Previously reported statistics of 20 independent Python SeMIS runs per example (N = 1000 per level; Kriging-based truncation for `Ex1`–`Ex9`, PLS-based for `Ex10`–`Ex12`):

| Example | Mean P_hat_f | Pf_ref | Mean / Pf_ref | Empirical CoV | Estimated CoV bounds (mean) | N_cal (mean) |
|---|---|---|---|---|---|---|
| `Ex1` | 3.58e-5 | 3.20e-5 | 1.12 | 0.19 | [0.25, 0.28] | 1498 |
| `Ex2` | 1.22e-5 | 1.13e-5 | 1.08 | 0.28 | [0.25, 0.29] | 1425 |
| `Ex3` | 3.09e-5 | 3.23e-5 | 0.96 | 0.22 | [0.19, 0.22] | 1491 |
| `Ex4` | 3.08e-4 | 3.13e-4 | 0.99 | 0.23 | [0.22, 0.30] | 1690 |
| `Ex5` | 2.25e-4 | 2.13e-4 | 1.06 | 0.29 | [0.24, 0.33] | 2091 |
| `Ex6` | 2.39e-4 | 2.370e-4 | 1.01 | 0.18 | [0.20, 0.26] | 1699 |
| `Ex7` | 5.42e-4 | 5.29e-4 | 1.03 | 0.21 | [0.20, 0.25] | 1535 |
| `Ex8` | 3.09e-3 | 3.20e-3 | 0.97 | 0.20 | [0.18, 0.21] | 1490 |
| `Ex9` | 3.68e-7 | 3.78e-7 | 0.97 | 0.36 | [0.34, 0.68] | 2930 |
| `Ex10(50)` | 1.40e-3 | 1.35e-3 | 1.04 | 0.15 | [0.20, 0.29] | 2975 |
| `Ex11(50)` | 4.59e-6 | 4.75e-6 | 0.97 | 0.45 | [0.37, 0.79] | 5299 |
| `Ex12(50)` | 1.90e-3 | 1.91e-3 | 1.00 | 0.15 | [0.18, 0.26] | 2979 |

All means are within 3 standard errors of the reference value (`Ex1`: 2.6, all others below 1.2).

The Kriging-based variant cannot be replayed exactly: MATLAB's `fitrgp` estimates the hyperparameters with its own optimizer, and the Python Gaussian process (same model: ARD squared exponential kernel, constant basis, standardized predictors, exact fit and prediction, noise standard deviation bounded below by 1e-2·std(y)) reaches slightly different hyperparameters. The Python optimizer uses bounded L-BFGS-B rather than MATLAB's default `fitrgp` optimizer, so the model specification is similar but optimizer behavior is not `fitrgp` parity. Statistical comparisons with the paper's reference probabilities do not establish native MATLAB equivalence.

The reference streams were recorded by running the original MATLAB code under GNU Octave 8.4 (no MATLAB licence was available) with the scripts in `validation/matlab/`: they shadow `randn`, `rand`, `randperm`, `randi` and `randsample` to log every draw, and `validation/matlab/octave/` provides Octave stand-ins for `makedist`, `pagemtimes` and the `"all"` option of `max`, `min` and `sum`. Octave compares complex numbers by their modulus, so for the recording the sign tests in `logsum`, `logplus` and `logminus` were applied to `real(...)`, which is what MATLAB compares, and the empty array that `Ex7` and `Ex9` grow their marginals from was created as `struct([])`. MATLAB does not need any of these changes. To record them with MATLAB instead (this also allows `'low'`, which needs `fitrgp`):

```matlab
>> cd validation/matlab
>> record_reference('/path/to/Sequential-multiple-importance-sampling-for-Reliability-Estimation', 'high', 'Ex10', 50, 1, '../reference')
```

### Regression tests and known limitations

```bash
python -m unittest discover -s tests -v
python validate.py --replay
```

The unit suite covers frozen SciPy distribution wrappers, signed `logcumsum` output storage, positive/negative Nataf correlations, multi-performance SuS output shapes, fail-closed validation and each available reference recording. Reference tests explicitly skip missing local fixtures; the standalone validator instead fails unless **all 15** cases are present. The suite retains one expected failure documenting an inherited `logsum` limitation: an all-negative signed-log slice (including a negative first cumulative prefix) returns `-inf`. Complex-buffer preservation fixes mixed-sign cumulative results but does not repair that upstream signed-arithmetic algorithm. No native MATLAB runtime was used for these regression tests.

## Differences from the MATLAB version

- Frozen `Distribution(name, scipy_frozen)` wrappers use their inverse CDF when explicit `mu`/`sigma`/`k` attributes are absent; `makedist` parameterized fast paths are retained.
- Numerical Nataf correlation inversion now searches a negative interval for negative target correlations and preserves exact zero pairs. This deliberately repairs an inherited positive-only interval, rather than claiming bit-for-bit source parity for those inputs.
- `SuS.EvlY` preserves `(Nfun, Nsam)` multi-performance matrices. The first performance function still determines SuS thresholds, as in MATLAB; `MIS_Rel` can estimate the retained functions separately.

- `fitrgp`/`predict` and `plsregress` of the Statistics and Machine Learning Toolbox are replaced by `semis_rel/surrogates.py`. `plsregress` gives the same coefficients as MATLAB's SIMPLS algorithm; for `fitrgp`, see [Validation against MATLAB](#validation-against-matlab).
- The Kriging predictions are computed one query point at a time, in buffers with a fixed memory alignment, so the prediction at a sample is bit-for-bit the same whichever other samples are predicted with it. The screening `g_hat <= l_hat` compares predictions from different calls with `l_hat`, the maximum over the seeds; when the seed attaining the maximum lost this comparison by round-off (a difference of about 1e-14), its chain started outside the support of the ISD, the normalizing constant became NaN and the run returned P_hat_f = 0. With batched predictions, this happened in 1 to 8 of 20 runs for each of `Ex4`–`Ex7` and `Ex9`. The MATLAB code compares predictions in the same way, so it may be exposed to the same round-off effect.
- `RunIte` returns a dictionary instead of a struct, and the ISD parameters `G` of the Kriging and PLS variants are dictionaries.
- Arrays keep the MATLAB layout (`Ndim × Nc × Ns`, flattened in column-major order); Python indices start at 0.
- Random numbers are drawn through `semis_rel._utils` (`randn`, `rand`, `randperm`, `randi`, `randsample`), backed by NumPy; `np.random.seed` fixes a run.
- `drive_SeMIS.py` takes the example and options on the command line and runs the independent runs in parallel processes with `--workers` (MATLAB: `parfor`).

## Citation

If you use this code in your research, please cite:

> Xia, W., Liao, Z., & Li, B. (2026). Sequential multiple importance sampling for robust and efficient (possibly high-dimensional) reliability estimation. *Mechanical Systems and Signal Processing*, 259, 114798. https://doi.org/10.1016/j.ymssp.2026.114798

```bibtex
@article{xia2026sequential,
  title   = {Sequential multiple importance sampling for robust and efficient (possibly high-dimensional) reliability estimation},
  author  = {Xia, Weili and Liao, Zihan and Li, Binbin},
  journal = {Mechanical Systems and Signal Processing},
  volume  = {259},
  pages   = {114798},
  year    = {2026},
  doi     = {10.1016/j.ymssp.2026.114798}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
