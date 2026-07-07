"""Deterministic time-stepping models.

Direct next-step predictors (``x_{n+1} = network(x_n, context, params)``)
trained with plain MSE. They reuse the same architectures and batch layout as
the stochastic interpolant models and serve as deterministic baselines.
"""

from scisi.deterministic_models.deterministic_model import DeterministicModel
