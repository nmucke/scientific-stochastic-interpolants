"""Load data from the data directory."""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_stochastic_navier_stokes(paths: str, files: str) -> np.ndarray:
    """
    Load stochastic Navier-Stokes data from the data directory.

    Args:
        paths: List of paths to the data.
        files: List of files to load.

    Returns:
        The data.
    """
    logger.info(f"Loading data from {paths[0]}/{files[0]}")
    data = np.load(f"{paths[0]}/{files[0]}")
    return data['state']