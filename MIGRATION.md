# Canonical shared core

This repository maintains the canonical source of semcs_core.Engine,
MCMCBase and the four Gaussian MCMC kernels. Its semcs-core distribution
installs both semcs_core and the semis_rel reliability toolbox.

Install this checkout with:

    python -m pip install .

The other four algorithm repositories depend on a fixed Git commit of this
canonical repository through their package metadata. They import these
implementations; they do not maintain another engine or copies of the kernels.
The legacy semis_rel.SeMCS and sampler import paths remain compatibility
entry points.

Method configuration and evolution state live in G: thresholds, level
probability, Gaussian parameters, run budgets, sampler options and adapted
lambda/HMC time. Historical configuration attributes are G-backed views.
The former H active-target mask is G["active"].

Each method retains its own advance, seed-selection, chain-plan and completion
rules. The common Engine only orchestrates these policies and records frozen
sampled-level G/data history. Native estimators remain separate. The default
aCS uses unit base scale and carries lambda between levels; historical author
settings must be labeled explicitly rather than silently replacing that default.

The semis_rel methods retain their GP/PLS screening and native MIS estimator.
Its classical SuS method retains its own threshold/seed and product-estimator
policies. Architecture equivalence and source-paper performance comparisons
are distinct validation questions; consult the validation evidence associated
with the release rather than treating a common engine as proof of paper parity.
