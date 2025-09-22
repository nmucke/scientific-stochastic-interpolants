# Scientific Stochastic Interpolants

A PyTorch-based framework for training and sampling from stochastic interpolants for scientific computing applications, with a focus on partial differential equations (PDEs) and weather modeling.

## Overview

This project implements **Follmer Stochastic Interpolants** for modeling and generating scientific data, particularly for:
- **Stochastic Navier-Stokes equations** - Fluid dynamics simulations
- **Weather modeling** - KNMI (Royal Netherlands Meteorological Institute) data
- **General PDE modeling** - Using transformer-based architectures

The framework provides tools for:
- Training stochastic interpolant models
- Sampling trajectories from trained models
- Posterior sampling with observational constraints
- Visualization and animation of results

## Key Features

- **Multiple Architecture Support**: U-Net and PDE Transformer architectures
- **Flexible Interpolation Schemes**: Linear, quadratic, and custom stochastic interpolations
- **Posterior Sampling**: Bayesian inference with observational constraints
- **Experiment Tracking**: Integration with TrackIO for experiment management
- **Mixed Precision Training**: Support for efficient GPU training
- **Comprehensive Visualization**: Animation and plotting utilities

## Installation

### Prerequisites

- **Python**: 3.12 (required)
- **CUDA**: 12.6 (for GPU acceleration)
- **Pixi**: Package manager for dependency management

### Setup

1. **Install Pixi** (if not already installed):
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd scientific-stochastic-interpolants
   ```

3. **Install dependencies using Pixi**:
   ```bash
   pixi install
   ```

4. **Activate the environment**:
   ```bash
   pixi shell
   ```

### Alternative Installation (without Pixi)

If you prefer not to use Pixi, you can install dependencies manually:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install hydra-core>=1.3.2
pip install trackio>=0.3.3
pip install pdetransformer>=0.1.0
pip install h5py>=3.14.0
pip install matplotlib>=3.10.5
pip install tqdm
pip install einops
```

## Project Structure

```
scientific-stochastic-interpolants/
├── src/scisi/                    # Main package
│   ├── architectures/           # Model architectures
│   │   ├── u_net.py            # U-Net implementation
│   │   ├── pde_transformer.py  # PDE Transformer wrapper
│   │   └── attention.py        # Attention mechanisms
│   ├── bin/                     # Main scripts
│   │   ├── main_train.py       # Training script
│   │   ├── main_test.py        # Testing script
│   │   ├── main_test_knmi.py   # KNMI-specific testing
│   │   └── main_posterior.py   # Posterior sampling
│   ├── models/                  # Core models
│   │   ├── follmer_stochastic_interpolant.py
│   │   └── interpolations.py   # Interpolation schemes
│   ├── data/                    # Data handling
│   │   ├── datasets.py         # Dataset classes
│   │   └── load_data.py        # Data loading utilities
│   ├── training/               # Training utilities
│   │   ├── trainer.py          # Training loop
│   │   └── loss_functions.py   # Loss functions
│   ├── sampling/               # SDE solvers
│   │   └── sde_solvers.py      # Euler-Maruyama, Heun methods
│   ├── posterior_models/       # Posterior sampling
│   ├── likelihood_models/      # Likelihood models
│   ├── preprocessing/         # Data preprocessing
│   └── plotting/              # Visualization
├── config/                     # Configuration files
│   ├── stochastic_navier_stokes.yaml
│   ├── knmi.yaml
│   └── knmi_pde_transformer.yaml
├── checkpoints/               # Model checkpoints
├── data/                      # Data directory
├── figures/                   # Output figures
└── outputs/                   # Training outputs
```

## Quick Start

### 1. Training a Model

Train a stochastic interpolant model on stochastic Navier-Stokes data:

```bash
cd src/scisi/bin
python main_train.py
```

The training script uses Hydra configuration management. You can override settings:

```bash
python main_train.py model.drift_model.hidden_channels=[64,128,256] trainer.num_epochs=500
```

### 2. Testing a Trained Model

Generate samples from a trained model:

```bash
python main_test.py
```

This will:
- Load a pre-trained model from checkpoints
- Generate trajectory samples
- Create animations and plots
- Save results to the `figures/` directory

### 3. Posterior Sampling

Perform Bayesian inference with observational constraints:

```bash
python main_posterior.py
```

This demonstrates how to:
- Incorporate observational data
- Sample from the posterior distribution
- Compare prior vs posterior predictions

## Configuration

The project uses Hydra for configuration management. Key configuration files:

### Model Configuration

```yaml
model:
  _target_: scisi.models.follmer_stochastic_interpolant.FollmerStochasticInterpolant
  interpolation:
    _target_: scisi.models.interpolations.QuadraticStochasticInterpolation
    gamma_multiplier: 0.1
    wiener_process: true
  drift_model:
    _target_: scisi.architectures.u_net.UNet
    in_channels: 1
    out_channels: 1
    hidden_channels: [32, 64, 128, 256]
    cond_dim: 1
    len_field_history: 5
```

### Training Configuration

```yaml
trainer:
  _target_: scisi.training.trainer.Trainer
  num_epochs: 1000
  device: cuda
  early_stopping:
    _target_: scisi.training.trainer.EarlyStopping
    patience: 25
  max_grad_norm: 5.0
  mixed_precision_warmup: 50
```

## Available Datasets

### 1. Stochastic Navier-Stokes

- **Purpose**: Fluid dynamics simulations
- **Data Format**: `.npz` files with trajectory data
- **Configuration**: `stochastic_navier_stokes.yaml`

### 2. KNMI Weather Data

- **Purpose**: Weather modeling and prediction
- **Data Format**: `.npz` files with meteorological data
- **Configuration**: `knmi.yaml`, `knmi_pde_transformer.yaml`

## Model Architectures

### 1. U-Net Architecture

- **Use Case**: General-purpose spatial modeling
- **Features**: Skip connections, attention mechanisms
- **Configuration**: Available in all config files

### 2. PDE Transformer

- **Use Case**: PDE-specific modeling with transformer attention
- **Features**: Patch-based processing, periodic boundaries
- **Configuration**: `*_pde_transformer.yaml` files

## Interpolation Schemes

### 1. Linear Deterministic Interpolation
```python
interpolation:
  _target_: scisi.models.interpolations.LinearDeterministicInterpolation
```

### 2. Quadratic Stochastic Interpolation
```python
interpolation:
  _target_: scisi.models.interpolations.QuadraticStochasticInterpolation
  gamma_multiplier: 0.1
  wiener_process: true
```

## Training Features

### Mixed Precision Training
- Automatic mixed precision (AMP) support
- Configurable warmup period
- Memory-efficient training

### Early Stopping
- Configurable patience
- Best model checkpointing
- Validation loss monitoring

### Experiment Tracking
- TrackIO integration
- Automatic logging of metrics
- Model checkpoint management

## Sampling Methods

### SDE Solvers

1. **Euler-Maruyama**: Basic stochastic differential equation solver
2. **Heun Method**: Higher-order accuracy solver

### Sampling Parameters

```python
NUM_STEPS = 100              # Number of SDE integration steps
NUM_PHYSICAL_STEPS = 50       # Physical time steps to predict
BATCH_SIZE = 1               # Batch size for sampling
MIXED_PRECISION = False      # Use mixed precision for sampling
```

## Visualization

### Animation Generation
```python
create_animation_from_tensors(
    [true_trajectory, predicted_trajectory],
    fps=10,
    file_name="prediction.mp4",
    colormaps="viridis",
    titles=["True", "Predicted"]
)
```

### Plotting Utilities
- Trajectory comparison plots
- Error analysis visualizations
- RMSE over time plots

## Development

### Code Quality

The project includes several development tools:

```bash
# Format code
pixi run format

# Type checking
pixi run type-check

# Sort imports
pixi run sort-imports

# Run tests
pixi run py.test
```

### Pre-commit Hooks

Install pre-commit hooks for automatic code formatting:

```bash
pixi run pre-commit install
```

## Examples

### Example 1: Training on Custom Data

1. **Prepare your data** in `.npz` format
2. **Create a dataset class** inheriting from `torch.utils.data.Dataset`
3. **Update configuration** to point to your data
4. **Run training**:

```bash
python main_train.py train_data.dataset.paths=[/path/to/your/data]
```

### Example 2: Custom Architecture

Create a custom drift model by implementing the required interface:

```python
class CustomDriftModel(nn.Module):
    def forward(self, x, t, field_history=None, field_cond=None):
        # Your implementation
        return drift_output
```

### Example 3: Posterior Sampling with Custom Observations

```python
# Define observation operator
obs_operator = CustomObservationOperator()

# Create likelihood model
likelihood_model = GaussianLikelihood(obs_operator)

# Sample from posterior
posterior_samples = posterior_model.sample_trajectory(
    base=initial_state,
    observations=obs_data,
    num_steps=100
)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Use mixed precision training
   - Reduce model size

2. **Slow Training**:
   - Enable flash attention
   - Use multiple workers for data loading
   - Check GPU utilization

3. **Configuration Errors**:
   - Verify Hydra configuration syntax
   - Check file paths in config files
   - Ensure all required parameters are specified

### Performance Tips

- Use `save_in_memory=True` for small datasets
- Enable `use_existing_cache=True` for repeated experiments
- Adjust `num_workers` based on your system
- Use mixed precision for large models

## Citation

If you use this code in your research, please cite:

```bibtex
@software{scientific_stochastic_interpolants,
  title={Scientific Stochastic Interpolants},
  author={Nikolaj T. Mücke},
  year={2024},
  url={https://github.com/your-repo/scientific-stochastic-interpolants}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For questions and support:
- Create an issue on GitHub
- Contact: nmucke@gmail.com

## Acknowledgments

- Based on the Follmer Stochastic Interpolants framework
- Uses the PDE Transformer architecture from the research community
- Integrates with TrackIO for experiment tracking
