# semcs_core

One sequential orchestration loop and one implementation of each Gaussian
MCMC kernel serve all five Python algorithm families.

Engine.run calls the repository-owned Method lifecycle:
start, sample, snapshot, advance, is_complete(before_seeds),
select_seeds, is_complete(after_seeds), finish. The core never chooses a
threshold, consumes random numbers, changes a seed law, or computes an estimate.

Method configuration and evolution live in its G dictionary. The old attribute
names are views backed by G. Sampler configuration lives in
G["sampler"][kernel_name]; adapted lambda, HMC_time and CW_width also live in G.
The sampler object retains only the model reference, RNG, and pending settings
until a model is attached.

Each level snapshot has read-only copies of G, x, y, Ncal, Nsam, and chain_shape.
The legacy result fields remain detached copies for existing native estimators.
The sampled G snapshot precedes advance; the legacy g list includes the next
parameters where the original estimator expects them. ReSuS retains only the
sampled states and separately records terminal_G.

Completion timing is method-owned: source Bayesian/reliability/GSS algorithms
select seeds before their historical completion test, preserving RNG order;
ReSuS tests the candidate next threshold before selecting/sampling it. Its
native product is evaluated by its repository postprocessor.

The public package distribution is semcs-core. Its repository also contains
semis_rel. Four consumer repositories pin this canonical repository at a
reviewed commit; no consumer contains an independent Engine or kernel.

