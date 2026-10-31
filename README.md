# SeMIS: Sequential Multiple Importance Sampling for Reliability Estimation (Python)

Python implementation of **Sequential Multiple Importance Sampling (SeMIS)** for estimating small failure probabilities, including in high-dimensional problems. This is a Python port of the MATLAB code in [Sequential-multiple-importance-sampling-for-Reliability-Estimation](https://github.com/ZihanLiao-ZJU/Sequential-multiple-importance-sampling-for-Reliability-Estimation), which accompanies the paper:

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
- [Implementation notes](#implementation-notes)
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
├── drive_SuS.py                # SuS experiment driver
├── semis_rel/
│   ├── SeMCS.py                 # sequential sampling framework (Algorithm 1)
│   ├── SeMIS_low_dimension.py   # intermediate distributions, Kriging-based truncation (Eqn. 25)
│   ├── SeMIS_high_dimension.py  # intermediate distributions, PLS-based truncation (Eqn. 26)
│   ├── MIS.py               # MIS estimator and optional single-run CoV bounds (Eqn. 34–58)
│   ├── SuS.py                   # subset simulation, used as a comparison baseline
│   ├── surrogates.py            # Gaussian process regression (fitrgp/predict) and PLS regression (plsregress)
│   ├── Nataf.py, distributions.py
│   ├── limit_state_functions/   # built-in performance functions Ex1 ... Ex12
│   ├── support_functions/       # sampler (aCS), MIS weights, seed selection, ...
│   ├── operations_on_logarithmic_values/  # one .py file per MATLAB log function
├── LICENSE
└── README.md
```

The logarithmic helpers use matching MATLAB/Python filenames: `logsum`,
`logmean`, `logplus`, `logminus`, `logmultiply`, `logcumsum`, and `logstd`.
Their Python implementations are in `operations_on_logarithmic_values/`.


## Quick start

Run the driver from the repository root:

```bash
python drive_SeMIS.py
```

By default this runs Example 1 (`Ex1`) with 100 independent runs and prints the estimated failure probability, its empirical CoV, and the average number of performance-function evaluations. The drivers run repetitions serially by default; `--workers 4` runs the Python independent runs in 4 processes. Other options: `--example Ex10` selects an example, `--repeats` sets the number of runs, `--cov` adds the single-run CoV bounds and `--seed` fixes the random seeds.

The same steps from Python code:

```python
from semis_rel import Ex1, SeMIS_low_dimension, SeMCS, aCS, MIS

Tfun = Ex1()
IntBay = SeMIS_low_dimension(Tfun)     # SeMIS_high_dimension(Tfun) for n >= 20
sampler = aCS(IntBay)
semcs = SeMCS(sampler)
semcs.NumSam = 1000                    # sample size per level (N)

out_semcs = semcs.RunIte()             # sequential sampling (Algorithm 1)
mis = MIS(out_semcs)               # MIS estimator (Eqn. 9-10, 33)
print(mis.Pf_hat, out_semcs["Ncal"].sum())
cv_lw, cv_up = mis.estimateCov()       # optional single-run CoV bounds (Eqn. 34-58)
```

The same shared structure is used by the GSS-MIS toolbox: `SeMCS` owns iteration;
`SeMIS_low_dimension`, `SeMIS_high_dimension`, and `SuS` only define intermediate
distributions and convergence; `MIS` estimates probabilities after sampling.
The `SeMCS.py` and `SuS.py` implementations are identical in the two Python toolboxes.

```python
from drive_SeMIS import drive_SeMIS
from drive_SuS import drive_SuS
result = drive_SeMIS(example="Ex10", N=1000, repeats=10, seed=2026, ndim=50)
baseline = drive_SuS(example="Ex10", N=1000, repeats=10, seed=2026, ndim=50)
print(result["mean"], result["cov"], result["mean_joint_calls"])
```

Drivers share `(example, N, repeats, seed)` and return summary dictionaries with
`mean`, `standard_deviation`, `cov`, `mean_joint_calls`, `unfinished`, and `records`.
`ndim` selects the dimension of Ex10–Ex12; `cov=True` adds the SeMIS single-run
bounds to each record. CLI options include `--samples`, `--repeats`, and `--seed`.

SuS explicitly uses `SuS(PF, "standard_normal", "hazen", True)`. The GSS driver
selects physical coordinates and its midpoint/unclipped convention using the same
class. The product estimator belongs to `drive_SuS`. `NumSam` accepts a scalar or
per-level vector; finite schedules must include each required next level and
raise a clear error if exhausted. `G` optionally supplies predefined distributions.

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
   - `EvlLSF(self, x)`: evaluates the performance function on the samples `x` (`Ndim × Nsam`, in the physical space given by `Dist`), where `g < 0` is the failure domain; for a single performance function, return a vector of length `Nsam` (SuS accepts either a row or column vector). `SuS` additionally accepts an `(Nfun, Nsam)` matrix, one performance function per row; its first row controls the sampling thresholds, while `MIS` retains all rows for estimation.
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

`SeMCS.RunIte()` returns a dictionary with the samples `x`, function values `y` and ISD parameters `g` of every level, the number of performance-function calls per level `Ncal` and the number of levels `Nite`. Set `semcs.Verbose = False` to silence the progress printed at every level.

## Optional single-run CoV bounds

The paper (Eqn. 34–58, Appendices A–C) derives analytical bounds of the CoV from a single run, using a three-term decomposition (MCS noise, normalizing-constant noise and cross-term noise). This is disabled by default. To enable it, run the driver with `--cov`, or call

```python
cv_lw, cv_up = mis.estimateCov()   # single-run CoV bounds (Eqn. 34–58)
```

## Code-to-paper mapping

| Step | Code | Paper |
|---|---|---|
| Sequence of intermediate distributions | `SeMCS.RunIte` / `UpdObj` | Eqn. (11)–(21), (24)–(27) |
| pCN-MCMC sampling with adaptive step | `support_functions/aCS.py` | Eqn. (22)–(23), Algorithm 2 |
| Normalizing constants of the ISDs | `MIS.normalizeY` | Eqn. (28)–(32) |
| Balance-heuristic MIS estimator | `MIS` / `WgtMis_Blc` / `MisInt` | Eqn. (4)–(10), (33) |
| CoV bounds (optional) | `MIS.estimateCov` | Eqn. (34)–(58), App. A–C |


## Implementation notes

- Frozen SciPy distribution wrappers use their inverse CDF when explicit distribution parameters are unavailable.
- Nataf correlation conversion supports positive, negative and zero target correlations.
- `SuS.EvlY` preserves `(Nfun, Nsam)` performance-function matrices. The first function determines the SuS sampling thresholds; `MIS` retains all rows for estimation.
- Gaussian process regression and PLS regression are implemented in `semis_rel/surrogates.py`. The Gaussian process uses an ARD squared-exponential kernel, a constant basis, standardized predictors, exact fitting and prediction, and bounded L-BFGS-B optimization. The noise standard deviation is bounded below by `1e-2 * std(y)`.
- Kriging predictions are evaluated one query point at a time with fixed memory alignment to keep support checks consistent across calls.
- `RunIte` returns a dictionary; the Kriging and PLS ISD parameters `G` are dictionaries.
- Arrays use the `Ndim × Nc × Ns` layout and column-major flattening. Python indices start at 0.
- Random draws use NumPy through `semis_rel._utils`; `np.random.seed` sets the random seed.
- `drive_SeMIS.py` accepts command-line options and uses independent processes with `--workers`.
- Signed-log arithmetic does not support slices containing only negative terms, including all-negative cumulative prefixes.

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


## Unified MCMC interface

Each standalone toolbox includes `CW_aMH`, `aCS`, `HMC`, and `ES`. The original
outer algorithm and default sampler remain unchanged; choose the sampler in the
driver (Python CLI: `--sampler`). Example: `drive_SeMIS("Ex10", N=1000, repeats=1, ndim=50, sampler_name="HMC")`.
All four expose `BayStc` and the same call:
`[u, y, Ncal, BayStc] = SamGen(u_seed, y_seed, Ns)` (Python returns a tuple).
`SamGen(Nchains, Nsteps)` initializes from the Gaussian reference. Arrays are
`[Ndim, Nchains, Nsteps]`; packed outputs are `[log_likelihood; log_reference; Y]`.
Dimension comes from the model, not from the two-dimensional examples.

Distribution state `G` carries model parameters and sampler adaptation. Legacy
numeric thresholds are stored in `G.parameters` (Python `G["parameters"]`);
existing named distribution parameters retain their names. aCS uses initial
`sigma0=1`, `lambda0=0.6`, five adaptation groups, and target acceptance 0.44.
The final `G.lambda` of one level initializes the next, rather than resetting.
ES uses the coordinatewise initial-angle `[0, pi]` version, with no stored seed
at the start of a conditional chain. HMC uses analytic Gaussian rotation in
Gaussian/Nataf reference coordinates, followed by the likelihood-factor MH test;
it is not a generic physical-coordinate gradient-HMC implementation. `G.L` or
`G.L_pi` is the Gaussian reference Cholesky factor, not the covariance matrix.
HMC's final trajectory time is likewise carried in `G.HMC_time`.

Custom nonnormal physical models must be mapped into Gaussian/Nataf coordinates
before these kernels are used. Arbitrary dimensional arrays are supported; the
supplied benchmark formulas retain their documented dimensions. Interface and
smoke checks do not establish statistical reproduction of every paper result.
