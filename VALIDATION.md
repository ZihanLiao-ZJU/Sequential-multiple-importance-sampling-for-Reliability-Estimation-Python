# Shared-core migration validation (2026-11-02)

Each configuration below has 100 independent runs of both the previous public implementation and this shared-core implementation. Paired seeds, settings and numerical-library versions are identical. Every pair has exactly equal samples, cached model values, scientific G history, estimates, model-call counts, sample counts and sampling RNG state. The refactor adds no model calls and changes none of these 400 sampled histories.

The default aCS uses unit base scale and carries lambda between levels. The following source-paper comparison is a separate statistical check; E = mean(Ncal) × [sample SD / sample mean]². Lower E is better. The source values are from the author proof, Tables 2 and 5, using 1000 repetitions. It reports rounded statistics, so reproducing every printed digit is not expected.

| Case | Reference probability | Mean (100 runs) | Sample SD | Mean Ncal | E | Paper E | E / paper E |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ex1 (2D) | 3.195788e-05 | 3.302598e-05 | 7.982958e-06 | 1496.24 | 87.421 | 95.579 | 0.91465 |
| Ex4 (3D) | 0.0003042914 | 0.0003021066 | 6.399141e-05 | 1757.25 | 78.842 | 88.477 | 0.89110 |
| Ex10 (20D) | 0.001349898 | 0.001415408 | 0.0002692437 | 2937.17 | 106.28 | 139.49 | 0.76192 |
| Ex10 (100D) | 0.001349898 | 0.001330752 | 0.0002952354 | 2989.35 | 147.14 | 145.2 | 1.01333 |

These paper-versus-current differences already occur in the previous public implementation; they are not performance gains caused by sharing Engine. Three tested configurations improve this point efficiency metric; the 100D configuration is 1.33% higher, within the predeclared 10% allowance. This result supports this implementation and these settings, rather than a universal ranking of samplers.

For Ex4, independent conditional Gaussian integration gives 0.00030429140759982183 (recommended numerical tolerance 2e-16, not a certified full error bound). The printed source reference 0.000313 is retained separately; it is not the rounding of this independent result. The reference correction changes no sampled estimates, SD, model calls or efficiency statistics. Ex10 uses the public positively scaled form of the printed limit-state function; its failure event is the same. Paper and current sampler settings are documented separately and are not claimed to be historically identical.

The 20D mean has an unadjusted 95% mean-interval flag against the independent truth. Across the independently referenced responses in the full five-family validation, none of the unadjusted flags survives Holm adjustment. The paired migration difference is exactly zero; this does not prove that the original adaptive estimator is unbiased at finite N.

Primary cohort seeds are 2026110200–2026110299. Server numerical environment: Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1, single numerical-library thread. Cross-language outputs require a separate statistical assessment.

Source: Xia, Liao and Li, *Sequential multiple importance sampling for robust and efficient (possibly high-dimensional) reliability estimation*, DOI [10.1016/j.ymssp.2026.114798](https://doi.org/10.1016/j.ymssp.2026.114798). This release preserves the repository’s original benchmark models and license.

The repository also retains its SuS method and alternative kernels; smaller exact trajectory/API checks cover those paths. The 100-run table above evaluates the SeMIS driver with aCS and is not a claim that all optional kernel/example combinations reproduce published results.
