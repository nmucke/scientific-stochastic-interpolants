# Scientific Stochastic Interpolants

> **Warning**: This repository is under active development. APIs, configurations, and features may change without notice.

A PyTorch-based framework for training and sampling from stochastic interpolants for scientific computing applications. The framework implements **Follmer Stochastic Interpolants**, **Flow Matching**, and **Diffusion Models** for modeling spatiotemporal dynamics in fluid mechanics, weather prediction, and atmospheric simulations.

## Overview

Stochastic interpolants provide a principled approach to learning generative models by constructing a continuous-time stochastic process that interpolates between a base (noise) distribution and a target (data) distribution. This framework applies that idea to scientific computing, where the goal is to generate physically realistic trajectories of PDE solutions.

The project supports:
- **Training** generative models on scientific simulation data
- **Sampling** trajectories from trained models via SDE/ODE integration
- **Posterior sampling** with observational constraints (Bayesian data assimilation)
- **Visualization** and spectral evaluation of generated fields

## Mathematical Background

### Follmer Stochastic Interpolants

Given a base distribution $\rho_0$ and a target distribution $\rho_1$, a stochastic interpolant defines an intermediate state:

$$X_t = \alpha(t) X_0 + \beta(t) X_1 + \gamma(t) W_t$$

where $\alpha(t), \beta(t)$ are deterministic interpolation coefficients, $\gamma(t)$ controls stochastic noise, and $W_t$ is a Wiener process. The framework learns a drift function $b(x, t)$ such that the SDE:

$$dX_t = b(X_t, t) \, dt + \sigma(t) \, dW_t$$

transports samples from $\rho_0$ to $\rho_1$.

### Interpolation Schemes

| Scheme | $\alpha(t)$ | $\beta(t)$ | Stochastic |
|--------|-------------|------------|------------|
| Linear Deterministic | $1 - t$ | $t$ | No |
| Quadratic Deterministic | $1 - t$ | $t^2$ | No |
| Linear Stochastic | $1 - t$ | $t$ | $\gamma(t) = c(1-t)\sqrt{t}$ |
| Quadratic Stochastic | $1 - t$ | $t^2$ | $\gamma(t) = c(1-t)\sqrt{t}$ |

### Posterior Sampling

For Bayesian inverse problems with observations $y = \mathcal{H}(x) + \eta$, the posterior drift incorporates a likelihood correction:

$$b_{\text{posterior}}(x, t) = b_{\text{prior}}(x, t) + \sigma^2(t) \nabla_x \log p(y \mid x)$$

This enables data assimilation by guiding the generative process toward states consistent with observations.

## Key Features

- **Three generative model paradigms**: Stochastic Interpolants, Flow Matching, and Diffusion Models
- **Multiple architectures**: U-Net (with ConvNext blocks and spatial attention) and PDE Transformer
- **Flexible interpolation**: Linear and quadratic schemes, deterministic and stochastic variants
- **Posterior sampling**: Gaussian likelihood models, observation operators, ensemble-based uncertainty quantification
- **Hydra configuration**: Fully configurable experiments via YAML files
- **Experiment tracking**: TrackIO integration for logging metrics and managing checkpoints
- **Mixed precision training**: AMP support with configurable warmup
- **Evaluation metrics**: LSiM (learned perceptual similarity), enstrophy spectrum, RMSE
- **Visualization**: MP4 animations, field plots, spectral analysis, point distributions

## Installation

### Prerequisites

- **Python**: 3.13 (required)
- **CUDA**: 12.6 (for GPU acceleration)
- **uv**: Package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/))

### Setup with uv

```bash
git clone <repository-url>
cd scientific-stochastic-interpolants
uv sync
```

For GPU support:

```bash
uv sync --extra dev-gpu
```

For CPU-only:

```bash
uv sync --extra dev-cpu
```

### Alternative Installation (pip)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

## Project Structure

```
scientific-stochastic-interpolants/
├── src/scisi/                       # Main package
│   ├── architectures/               # Neural network architectures
│   │   ├── u_net.py                 #   U-Net with ConvNext blocks
│   │   ├── pde_transformer.py       #   PDE Transformer wrapper
│   │   ├── attention.py             #   Spatial & rotary attention
│   │   ├── conv_next.py             #   ConvNext blocks
│   │   ├── embeddings.py            #   Fourier & conditional embeddings
│   │   └── rotary_positional_embedding.py
│   ├── models/                      # Core generative models
│   │   ├── follmer_stochastic_interpolant.py
│   │   ├── flow_matching_model.py
│   │   ├── diffusion_model.py
│   │   ├── flow_map_model.py        #   Deterministic flow map
│   │   ├── follmer_ito_map_model.py #   Stochastic Itô map
│   │   ├── interpolations.py        #   Interpolation schemes
│   │   ├── base_model.py            #   Abstract base class
│   │   └── base_map_model.py        #   Abstract base class for maps
│   ├── data/                        # Dataset classes & loaders
│   │   ├── datasets.py              #   All dataset implementations
│   │   └── load_data.py
│   ├── training/                    # Training loop & utilities
│   │   ├── trainer.py               #   Base trainer: loop, checkpoints, resume
│   │   ├── flow_trainer.py          #   Flow models (FM, SI, diffusion)
│   │   ├── map_trainer.py           #   Map models (self-distillation)
│   │   ├── loss_functions.py        #   Latitude-weighted MSE, etc.
│   │   └── gradient_clipping.py     #   EMA-based gradient clipping
│   ├── sampling/                    # SDE/ODE solvers
│   │   ├── sde_solvers.py           #   Euler-Maruyama, Heun
│   │   ├── ode_solvers.py           #   Euler step (flow matching)
│   │   └── brownian_path.py         #   Shared Brownian paths for Itô maps
│   ├── posterior_models/            # Bayesian posterior sampling
│   │   ├── base_posterior.py
│   │   ├── stochastic_interpolant_posterior.py
│   │   ├── flow_matching_posterior.py
│   │   └── diffusion_posterior.py
│   ├── likelihood_models/           # Likelihood & observation operators
│   │   ├── gaussian_likelihood.py   #   Multiple Gaussian variants
│   │   ├── observation_operators.py #   Grid & random observations
│   │   └── guidance.py              #   Classifier-free guidance
│   ├── preprocessing/               # Data normalization
│   ├── metrics/                     # LSiM, enstrophy spectrum
│   ├── plotting/                    # Animation & field visualization
│   │   └── cases.py                 #   Figures for each test case
│   ├── particle_filter/             # Particle filtering methods
│   ├── utils/                       # Device utilities
│   └── bin/                         # Entry point scripts
│       ├── main_train.py
│       ├── main_test.py
│       ├── main_test_knmi.py
│       ├── main_posterior.py
│       └── ...
├── scripts/                         # train.py and test.py for all models
├── config/                          # Hydra configs: one file per case, plus
│   ├── data/                        #   datasets per case
│   ├── models/                      #   model types
│   ├── training/                    #   flow and map training
│   ├── posterior/                   #   data assimilation
│   └── test.yaml                    #   settings for scripts/test.py
├── external_libs/                   # Workspace member packages
│   ├── torch_cfd_lib/              #   PyTorch CFD utilities
│   └── jax_cfd_lib/                #   JAX CFD utilities
├── paper_scripts/                   # Scripts for reproducing paper results
├── checkpoints/                     # Saved model weights
├── data/                            # Dataset storage
├── figures/                         # Output visualizations
└── outputs/                         # Training outputs
```

## Quick Start

### Training

Pick a case with `--config-name` and a model with `models=<name>`
(`stochastic_interpolant`, `flow_matching`, `diffusion`, `flow_map`, `ito_map`).
Run from the repository root:

```bash
uv run scripts/train.py --config-name stochastic_navier_stokes models=flow_matching
uv run scripts/train.py --config-name udales models=stochastic_interpolant trainer.num_epochs=500
```

Each run is saved to `checkpoints/<project>/<run name>/` (`model.pth`, `last.pth`,
`history.json`, `config.yaml`). Continue an interrupted run with
`trainer.resume_from=checkpoints/<project>/<run name>/last.pth`.

### Sampling from a Trained Model

```bash
uv run scripts/test.py project=stochastic_navier_stokes name=<run name>
```

This loads a checkpoint, generates an ensemble forecast, logs LSiM and RMSE, and
saves the case's animations and plots to `figures/<project>/<run name>/`.
Settings are in `config/test.yaml`.

### Posterior Sampling

```bash
uv run python -m scisi.bin.main_posterior
```

Performs Bayesian inference by incorporating sparse observations into the generative sampling process, comparing prior and posterior predictions.

## Supported Datasets

| Dataset | Description | Channels | Spatial Size | Config |
|---------|-------------|----------|-------------|--------|
| **Stochastic Navier-Stokes** | Turbulent vorticity fields | 1 | 128 x 128 | `stochastic_navier_stokes*.yaml` |
| **KNMI Weather** | KNMI meteorological data | 1 | Variable | `knmi*.yaml` |
| **UDALES** | Large eddy simulation (u, v, w, T) | 4 | 256 x 256 | `udales*.yaml` |
| **Xie & Castro** | Atmospheric boundary layer | 1 | 128 x 128 | `data/xie_and_castro.yaml` (no case file) |
| **Weather (Global)** | Global weather reanalysis | 1 | 721 x 1440 | `weather*.yaml` |

Data is expected in `.npz` format under the `data/` directory.

## Generative Model Variants

### Follmer Stochastic Interpolant

The primary model. Learns a drift function for an SDE that transports noise to data. Supports stochastic interpolation with configurable noise levels via `gamma_multiplier`.

### Flow Matching

Deterministic transport via an ODE. Uses `LinearDeterministicInterpolation` and the Euler solver. No stochastic noise during sampling.

### Diffusion Model

Standard score-based diffusion. Learns the score function and converts it to a velocity for sampling.

All three share the same architecture backends (U-Net, PDE Transformer) and can be swapped via configuration.

## Model Architectures

### U-Net

Multi-scale encoder-decoder with skip connections. Features include:
- ConvNext blocks for feature processing
- Spatial attention with rotary positional embeddings (configurable per layer)
- Time embedding via sinusoidal positional encoding
- Field history concatenation for temporal conditioning
- Circular padding for periodic boundary conditions

### PDE Transformer

Patch-based transformer architecture designed for structured grid data with periodic boundaries. Available in three sizes (PDE-S, PDE-B, PDE-L) via the external `pdetransformer` package.

## Configuration

The project uses [Hydra](https://hydra.cc/). Configs in `config/` are split into
groups that a case file combines:

| Location | Contents |
|----------|----------|
| `data/<case>.yaml` | Datasets, preprocessing, field history length, mask |
| `models/<model>.yaml` | Model type; wraps the case's `backbone` |
| `training/default.yaml` | `FlowTrainer`, optimizer, scheduler, loss |
| `training/map.yaml` | `MapTrainer` for `flow_map` and `ito_map` (selected automatically) |
| `<case>.yaml` | Chooses the defaults and defines the backbone and case-specific settings |

```yaml
# config/stochastic_navier_stokes.yaml
defaults:
  - data: stochastic_navier_stokes
  - training: default
  - models: stochastic_interpolant
  - _self_

experiment_tracking:
  project: stochastic_navier_stokes

backbone:
  _target_: scisi.architectures.u_net.UNet
  in_channels: 1
  out_channels: 1
  hidden_channels: [16, 32, 64, 128]
  len_field_history: ${len_field_history}
  ...
```

Any value can be overridden on the command line, e.g. `backbone.dropout_rate=0`
or `trainer.early_stopping.patience=25`.

## Posterior Sampling & Data Assimilation

The framework supports Bayesian inverse problems where partial observations constrain the generative process.

**Likelihood models:**
- `InterpolantGaussianLikelihood` — the closed-form observation-interpolant likelihood (the method)
- `FlowdasGaussianLikelihood` — FlowDAS baseline (paper Algorithm-2 importance-weighted residual guidance)
- `GuidanceGaussianLikelihood` / `FIGGaussianLikelihood` — Guided FM baselines (FIG and OT-ODE modes)
- `DFlowSGLDLikelihood` — D-Flow SGLD baseline (thin H/R holder)
- `SDALikelihood` — SDA baseline (single-window adaptation)

**Posterior samplers (paper data-assimilation experiments).** The unified family —
**SI-SDE**, **FM-ODE**, **FM-SDE** (shown as "FM-SDE (DM)") — shares one
observation-interpolant likelihood and differs only in the diffusion `g_τ`.
Baselines: FlowDAS, Guided FM (FIG), Guided FM (OT-ODE), D-Flow SGLD
(`DFlowPosterior`), SDA, SURGE (`SurgePosterior`), plus the classical filters
(EnKF, LETKF, particle filter, ensemble score filter) for cases with a true solver.

**Observation operators:**
- `LinearObservationOperator` — supports grid-based, random sparse, and block-average super-resolution patterns

Posterior sampling produces an ensemble of trajectories, enabling uncertainty quantification via ensemble spread.

> The full paper-experiment harness, method lineup, and run status live in
> `paper_experiments/` (see `paper_experiments/results/README.md` for the layout,
> `paper_experiments/RUN_STATUS.md` for status, and `paper_experiments/DESIGN_NOTES.md`
> for the code↔math mapping and design rationale).

## Evaluation Metrics

- **LSiM** — Learned perceptual similarity metric for scientific fields
- **Enstrophy spectrum** — FFT-based energy spectrum of vorticity fields
- **Enstrophy error** — RMS difference in spectral space between predicted and true fields
- **RMSE** — Per-timestep root mean squared error

## Development

### Code Quality

```bash
# Format code
black src/

# Sort imports
isort src/

# Type checking
mypy src/

# Run tests
pytest
```

### Pre-commit Hooks

```bash
pre-commit install
```

Hooks run Black, isort, MyPy, and standard checks (trailing whitespace, YAML validation, large file detection) on every commit.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{scientific_stochastic_interpolants,
  title={Scientific Stochastic Interpolants},
  author={Nikolaj T. M\"{u}cke},
  year={2025},
  url={https://github.com/nmucke/scientific-stochastic-interpolants}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

- Create an issue on GitHub
- Contact: nmucke@gmail.com

## Conditional flow and Itô maps

`BaseMapModel` provides the map equivalent of `BaseModel`, with the same field
history, conditioning, ensemble sampling, and physical trajectory conventions.
`FlowMapModel` starts each forecast from Gaussian noise. `FollmerItoMapModel`
starts from the last observed or generated state and conditions on a shared
Brownian path. Both support choosing the number of map evaluations at inference.

`MapConditionedNetwork` adds interval and Brownian inputs to the UNet and can
initialize from an existing drift checkpoint. `MapTrainer` combines diagonal
regression with an off-diagonal self-distillation loss against an EMA copy of the
model (`trainer.off_diagonal`, currently `progressive_self_distillation`). The
off-diagonal loss can be delayed (`trainer.diagonal_epochs`) and ramped in
(`trainer.warmup_epochs`). Map models currently require the UNet backbone and an
identically-one spatial mask.

```bash
# Train the Föllmer Itô map or the deterministic flow map.
uv run scripts/train.py --config-name stochastic_navier_stokes models=ito_map
uv run scripts/train.py --config-name stochastic_navier_stokes models=flow_map

# Warm-start from a trained drift model.
uv run scripts/train.py --config-name stochastic_navier_stokes models=flow_map \
  teacher_checkpoint_dir=checkpoints/stochastic_navier_stokes/flow_matching

# Forecast and plot with a trained map (map steps per physical step: num_map_steps).
uv run scripts/test.py project=stochastic_navier_stokes name=<run name> num_map_steps=1
```
