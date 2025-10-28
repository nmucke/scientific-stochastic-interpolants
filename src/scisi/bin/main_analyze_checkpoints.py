import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def get_best_trackio_run(
    project: str, metric_name: str = "val_loss", maximize: bool = False
) -> Optional[str]:
    """
    Fetches the best run name from the local checkpoints directory for a given project.

    Since metrics are not stored locally in checkpoint directories, this function returns
    the most recently created checkpoint directory as a fallback.

    Args:
        project (str): The project name in the checkpoints directory.
        metric_name (str): The metric to compare runs by (default: "val_loss") - not used in current implementation.
        maximize (bool): Whether to maximize (True) or minimize (False) the metric - not used in current implementation.

    Returns:
        Optional[str]: The name of the most recent run, or None if no runs found.
    """
    try:
        # Define the checkpoints directory path
        checkpoints_dir = Path("checkpoints") / project

        if not checkpoints_dir.exists():
            logger.warning(f"No checkpoints directory found for project '{project}'")
            return None

        # Get all run directories
        run_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir()]

        if not run_dirs:
            logger.warning(f"No runs found for project '{project}'")
            return None

        # Since metrics are not stored locally, we'll use the most recently modified directory
        # as a proxy for the "best" run (assuming more recent runs are better)
        most_recent_run = max(run_dirs, key=lambda x: x.stat().st_mtime)

        logger.info(f"Found {len(run_dirs)} runs for project '{project}'")
        logger.info(f"Most recent run: {most_recent_run.name}")
        logger.warning(
            f"Note: Metrics are not available locally. Returning most recent run."
        )

        return most_recent_run.name

    except Exception as e:
        logger.error(f"Failed to get best run for project '{project}': {e}")
        return None


def get_all_trackio_runs(project: str) -> List[str]:
    """
    Get all available run names for a given project.

    Args:
        project (str): The project name in the checkpoints directory.

    Returns:
        List[str]: List of run names, sorted by modification time (newest first).
    """
    try:
        checkpoints_dir = Path("checkpoints") / project

        if not checkpoints_dir.exists():
            logger.warning(f"No checkpoints directory found for project '{project}'")
            return []

        run_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
        run_names = [d.name for d in run_dirs]

        # Sort by modification time (newest first)
        run_dirs_with_times = [(d, d.stat().st_mtime) for d in run_dirs]
        run_dirs_with_times.sort(key=lambda x: x[1], reverse=True)
        sorted_run_names = [d.name for d, _ in run_dirs_with_times]

        logger.info(
            f"Found {len(sorted_run_names)} runs for project '{project}': {sorted_run_names}"
        )

        return sorted_run_names

    except Exception as e:
        logger.error(f"Failed to get runs for project '{project}': {e}")
        return []


# Example usage:
if __name__ == "__main__":
    PROJECT = "stochastic_navier_stokes"

    # Get the best (most recent) run
    best_run = get_best_trackio_run(PROJECT, metric_name="val_loss", maximize=False)
    print(f"Best run for project '{PROJECT}': {best_run}")

    # Get all runs
    all_runs = get_all_trackio_runs(PROJECT)
    print(f"All runs for project '{PROJECT}': {all_runs}")
