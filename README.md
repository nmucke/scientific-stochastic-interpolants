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
- **Multiple architectures**: U-Net (with ConvNext blocks and spatial attention), PDE Transformer, and Aurora weather model wrapper
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
│   │   ├── aurora.py                #   Aurora weather model wrapper
│   │   ├── attention.py             #   Spatial & rotary attention
│   │   ├── conv_next.py             #   ConvNext blocks
│   │   ├── embeddings.py            #   Fourier & conditional embeddings
│   │   └── rotary_positional_embedding.py
│   ├── models/                      # Core generative models
│   │   ├── follmer_stochastic_interpolant.py
│   │   ├── flow_matching_model.py
│   │   ├── diffusion_model.py
│   │   ├── interpolations.py        #   Interpolation schemes
│   │   └── base_model.py            #   Abstract base class
│   ├── data/                        # Dataset classes & loaders
│   │   ├── datasets.py              #   All dataset implementations
│   │   └── load_data.py
│   ├── training/                    # Training loop & utilities
│   │   ├── trainer.py               #   Trainer with early stopping
│   │   ├── loss_functions.py        #   Latitude-weighted MSE, etc.
│   │   └── gradient_clipping.py     #   EMA-based gradient clipping
│   ├── sampling/                    # SDE/ODE solvers
│   │   ├── sde_solvers.py           #   Euler-Maruyama, Heun
│   │   └── ode_solvers.py           #   Euler step (flow matching)
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
│   ├── particle_filter/             # Particle filtering methods
│   ├── utils/                       # Device utilities
│   └── bin/                         # Entry point scripts
│       ├── main_train.py
│       ├── main_test.py
│       ├── main_test_knmi.py
│       ├── main_posterior.py
│       └── ...
├── config/                          # Hydra YAML configs (24 files)
├── external_libs/                   # Workspace member packages
│   ├── torch_cfd_lib/              #   PyTorch CFD utilities
│   ├── jax_cfd_lib/                #   JAX CFD utilities
│   └── aurora_lib/                  #   Aurora weather model integration
├── paper_scripts/                   # Scripts for reproducing paper results
├── checkpoints/                     # Saved model weights
├── data/                            # Dataset storage
├── figures/                         # Output visualizations
└── outputs/                         # Training outputs
```

## Quick Start

### Training

```bash
cd src/scisi/bin
python main_train.py
```

Override configuration via Hydra:

```bash
python main_train.py model.drift_model.hidden_channels=[64,128,256] trainer.num_epochs=500
```

Use a different config file:

```bash
python main_train.py --config-name knmi_pde_transformer.yaml
```

### Sampling from a Trained Model

```bash
python main_test.py
```

This loads a checkpoint, generates ensemble trajectory predictions, computes evaluation metrics, and saves animations and plots to `figures/`.

### Posterior Sampling

```bash
python main_posterior.py
```

Performs Bayesian inference by incorporating sparse observations into the generative sampling process, comparing prior and posterior predictions.

## Supported Datasets

| Dataset | Description | Channels | Spatial Size | Config |
|---------|-------------|----------|-------------|--------|
| **Stochastic Navier-Stokes** | Turbulent vorticity fields | 1 | 128 x 128 | `stochastic_navier_stokes*.yaml` |
| **KNMI Weather** | KNMI meteorological data | 1 | Variable | `knmi*.yaml` |
| **UDALES** | Large eddy simulation (u, v, w, T) | 4 | 256 x 256 | `udales*.yaml` |
| **Aurora** | Global weather forecasting | Multiple | Global grid | `aurora.yaml` |
| **Xie & Castro** | Atmospheric boundary layer | 1 | 128 x 128 | `xie_and_castro.yaml` |
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

### Aurora Wrapper

Integration with Microsoft's pre-trained Aurora weather forecasting model for global weather prediction tasks.

## Configuration

The project uses [Hydra](https://hydra.cc/) for hierarchical configuration management. All configs live in `config/` and follow this structure:

```yaml
experiment_tracking:
  project: "stochastic_navier_stokes"

preprocesser:
  _target_: scisi.preprocessing.preprocessor.Preprocesser
  mean: [0.0]
  std: [1.0]

model:
  _target_: scisi.models.follmer_stochastic_interpolant.FollmerStochasticInterpolant
  interpolation:
    _target_: scisi.models.interpolations.QuadraticStochasticInterpolation
    gamma_multiplier: 1.0
    wiener_process: true
  drift_model:
    _target_: scisi.architectures.u_net.UNet
    in_channels: 1
    out_channels: 1
    hidden_channels: [8, 16, 32, 64]
    len_field_history: 5

optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-4

scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: 1000

trainer:
  num_epochs: 1000
  early_stopping:
    patience: 25
  mixed_precision_warmup: 50
```

24 configuration files cover all combinations of datasets, architectures, and model types.

## Posterior Sampling & Data Assimilation

The framework supports Bayesian inverse problems where partial observations constrain the generative process.

**Likelihood models:**
- `InterpolantGaussianLikelihood` — point-wise Gaussian noise
- `FlowdasGaussianLikelihood` — flow-based data assimilation
- `KalmanInterpolantGaussianLikelihood` — Kalman-style ensemble covariance
- `SpatialInterpolantGaussianLikelihood` — spatially-correlated likelihood

**Observation operators:**
- `LinearObservationOperator` — supports grid-based (regular skip) and random observation patterns

Posterior sampling produces an ensemble of trajectories, enabling uncertainty quantification via ensemble spread.

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
