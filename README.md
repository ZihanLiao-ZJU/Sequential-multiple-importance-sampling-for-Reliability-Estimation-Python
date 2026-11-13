# SeMIS: Sequential Multiple Importance Sampling for Reliability Estimation (Python)

Python implementation of **Sequential Multiple Importance Sampling (SeMIS)** for estimating small failure probabilities, including in high-dimensional problems. This is a line-by-line port of the MATLAB code in [Sequential-multiple-importance-sampling-for-Reliability-Estimation](https://github.com/ZihanLiao-ZJU/Sequential-multiple-importance-sampling-for-Reliability-Estimation), which accompanies the paper:

> Xia, W., Liao, Z., & Li, B. (2026). Sequential multiple importance sampling for robust and efficient (possibly high-dimensional) reliability estimation. *Mechanical Systems and Signal Processing, 259*, Article 114798. [https://doi.org/10.1016/j.ymssp.2026.114798](https://doi.org/10.1016/j.ymssp.2026.114798)

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

```text
.
├── LICENSE
├── README.md
├── drive_SeMIS.py
├── requirements.txt
└── semis_rel/
    ├── MIS.py
    ├── Nataf.py
    ├── SeMCS.py
    ├── SeMIS_high_dimension.py
    ├── SeMIS_low_dimension.py
    ├── SuS.py
    ├── __init__.py
    ├── _utils.py
    ├── distributions.py
    ├── limit_state_functions/
    │   ├── Ex1.py
    │   ├── Ex10.py
    │   ├── Ex11.py
    │   ├── Ex12.py
    │   ├── Ex2.py
    │   ├── Ex3.py
    │   ├── Ex4.py
    │   ├── Ex5.py
    │   ├── Ex6.py
    │   ├── Ex7.py
    │   ├── Ex8.py
    │   ├── Ex9.py
    │   ├── __init__.py
    │   └── _plot.py
    ├── mcmc_samplers/
    │   ├── CW_aMH.py
    │   ├── ES.py
    │   ├── HMC.py
    │   ├── LICENSE
    │   ├── MCMCBase.py
    │   ├── __init__.py
    │   └── aCS.py
    ├── operations_on_logarithmic_values/
    │   └── __init__.py
    ├── support_functions/
    │   ├── MisInt.py
    │   ├── MoveBlock.py
    │   ├── ResRat.py
    │   ├── SedSlt.py
    │   ├── UpdY.py
    │   ├── WgtMis_Blc.py
    │   ├── __init__.py
    │   └── logGauss.py
    └── surrogates.py
```

## Quick start

Edit `Tfun`, `N`, `Nrun` and `SamplerClass` inside `drive_SeMIS.py`. Select `SamplerClass = aCS` (the default), or choose another supplied sampler: `CW_aMH`, `aCS`, `HMC` or `ES`.

The script defaults to Example 1, 1000 samples per level and 100 runs. Pass the dimension to the constructor for Examples 10–12, for example `Ex10(50)`. For a single run, set `Nrun = 1` inside the script.

Run the chosen script from the repository root:

```bash
python drive_SeMIS.py
```

Use multiple runs to estimate statistics across independent runs.

## Built-in examples

Construct the selected performance function in the driver, for example `Tfun = Ex1()` or `Tfun = Ex10(50)`. The examples are numbered as in Tables 1 and 4 of the paper; `semis_rel.PF_REF` holds the reference values.

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

The driver uses `SeMIS_high_dimension` (PLS-based truncation) for n ≥ 20 and `SeMIS_low_dimension` (Kriging-based truncation) for smaller dimensions.

## Using a custom performance function

1. Copy one of `semis_rel/limit_state_functions/Ex*.py` (each is a class with the same interface) and rename it, e.g. `MyProblem`.
2. Define:
   - `Dist`: probabilistic model of the input vector `X`, a `Nataf` object built from a list of marginal distributions (`makedist`) and their correlation matrix;
   - `Ndim`, `Nfun`: input dimension and number of performance functions;
   - `EvlLSF(self, x)`: evaluates the performance function on the samples `x` (`Ndim × Nsam`, in the physical space given by `Dist`), where `g < 0` is the failure domain; for a single performance function, return a vector of length `Nsam` (SuS accepts either a row or column vector). `SuS` additionally accepts an `(Nfun, Nsam)` matrix, one performance function per row; its first row controls the sampling thresholds, while `MIS` retains all rows for estimation.
3. Import your problem class and set `Tfun = MyProblem()` in `drive_SeMIS.py`. The script selects the intermediate model from `Tfun.Ndim`.

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

The raw sampling history `g` contains `Nite` sampled states followed by one final candidate state. `Nsze` records the chain and step counts at each sampled level.

The driver prints the following quantities, averaged over the independent runs:

| Output | Meaning |
|---|---|
| `P_hat_f` | estimated failure probability (Eqn. 9–10, 33) |
| `P_hat_f/Pf_ref` | estimate normalized by the reference value (cf. Tables 2 and 5 of the paper) |
| `CoV (empirical)` | coefficient of variation across the independent runs |
| `N_cal (mean)` | average number of performance-function evaluations per run |

`SeMCS.RunIte()` returns a dictionary with the samples `x`, function values `y` and ISD parameters `g` of every level, the number of performance-function calls per level `Ncal` and the number of levels `Nite`. Set `semcs.Verbose = False` to silence the progress printed at every level.

## Optional single-run CoV bounds

The paper (Eqn. 34–58, Appendices A–C) derives analytical bounds of the CoV from a single run, using a three-term decomposition (MCS noise, normalizing-constant noise and cross-term noise). This is disabled by default. To enable it, set `cov = True` inside the driver, or call

```python
cv_lw, cv_up = mis.estimateCov()   # single-run CoV bounds (Eqn. 34–58)
```

## Code-to-paper mapping

| Step | Code | Paper |
|---|---|---|
| Sequence of intermediate distributions | `SeMCS.RunIte` / `UpdObj` | Eqn. (11)–(21), (24)–(27) |
| pCN-MCMC sampling with adaptive step | `mcmc_samplers/aCS.py` | Eqn. (22)–(23), Algorithm 2 |
| Normalizing constants of the ISDs | `MIS.normalizeY` | Eqn. (28)–(32) |
| Balance-heuristic MIS estimator | `MIS` / `WgtMis_Blc` / `MisInt` | Eqn. (4)–(10), (33) |
| CoV bounds (optional) | `MIS.estimateCov` | Eqn. (34)–(58), App. A–C |

Signed-log arithmetic retains an inherited limitation: an all-negative signed-log slice, including a negative first cumulative prefix, is not handled correctly by `logsum`. Preserving complex `logcumsum` outputs retains mixed-sign prefixes but does not change that reduction algorithm.

## Differences from the MATLAB version

- Frozen `Distribution(name, scipy_frozen)` wrappers use their inverse CDF when explicit `mu`/`sigma`/`k` attributes are absent; `makedist` parameterized fast paths are retained.
- Numerical Nataf correlation inversion now searches a negative interval for negative target correlations and preserves exact zero pairs. This deliberately repairs an inherited positive-only interval, rather than claiming bit-for-bit source parity for those inputs.
- `SuS.EvlY` preserves `(Nfun, Nsam)` multi-performance matrices. The first performance function still determines SuS thresholds, as in MATLAB; `MIS` can estimate the retained functions separately.

- `fitrgp`/`predict` and `plsregress` of the Statistics and Machine Learning Toolbox are replaced by `semis_rel/surrogates.py`. The PLS implementation follows SIMPLS. The Gaussian process uses an ARD squared exponential kernel, constant basis, standardized predictors, exact fit and prediction, and noise standard deviation bounded below by 1e-2·std(y). Its bounded L-BFGS-B optimizer differs from MATLAB's default `fitrgp` optimizer and can yield different hyperparameters.
- The Kriging predictions are computed one query point at a time, in buffers with a fixed memory alignment, so the prediction at a sample does not depend on the surrounding query batch. This matters when `g_hat <= l_hat` compares predictions from separate calls at a selected seed attaining the maximum: round-off can otherwise place that seed outside the ISD support.
- `RunIte` returns a dictionary instead of a struct, and the ISD parameters `G` of the Kriging and PLS variants are dictionaries.
- Arrays keep the MATLAB layout (`Ndim × Nc × Ns`, flattened in column-major order); Python indices start at 0.

## Citation

If you use this toolbox in your research, please cite:

> Xia, W., Liao, Z., & Li, B. (2026). Sequential multiple importance sampling for robust and efficient (possibly high-dimensional) reliability estimation. *Mechanical Systems and Signal Processing, 259*, Article 114798. [https://doi.org/10.1016/j.ymssp.2026.114798](https://doi.org/10.1016/j.ymssp.2026.114798)

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

