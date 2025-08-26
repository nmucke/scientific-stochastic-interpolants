import hydra
from scisinterpolant.data.load_data import load_stochastic_navier_stokes
import pdb


@hydra.main(
    config_path="../../../config",
    config_name="stochastic_navier_stokes.yaml",
    version_base=None,
)
def main(cfg):
    data = load_stochastic_navier_stokes(cfg.data)

if __name__ == "__main__":
    main()