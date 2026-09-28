"""Locations shared by the relocated paper experiment entry points.

Research assets belong to the repository; generated artifacts belong to this
experiment tree. Explicit relative output paths remain relative to the launch
working directory, including when Hydra changes directory.
"""
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/scisi").is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot locate repository containing src/scisi from {start}")


REPO_ROOT = find_repo_root(EXPERIMENT_ROOT)
RESULTS_ROOT = EXPERIMENT_ROOT / "results"
FIGURES_ROOT = EXPERIMENT_ROOT / "figures"


def repository_path(value: str | Path) -> Path:
    """Resolve checkpoint-stored asset paths against the research repository."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def output_path(value: str | Path) -> Path:
    """Resolve a user-supplied output/input override against the launch directory."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import get_original_cwd

    launch_dir = Path(get_original_cwd()) if HydraConfig.initialized() else Path.cwd()
    return launch_dir / path


def resolve_checkpoint_paths(config):
    """Copy a checkpoint config and anchor its dataset/mask/cache paths.

    Filenames and trajectory selections are deliberately left unchanged. The
    original on-disk checkpoint configuration is never rewritten.
    """
    from omegaconf import DictConfig, OmegaConf

    resolved = OmegaConf.create(OmegaConf.to_container(config, resolve=True))

    def visit(node):
        if not isinstance(node, DictConfig):
            return
        for key in node:
            value = node[key]
            if value is None:
                continue
            if key == "paths":
                node[key] = [str(repository_path(path)) for path in value]
            elif key in {"mask_path", "cache_dir"}:
                node[key] = str(repository_path(value))
            elif isinstance(value, DictConfig):
                visit(value)

    visit(resolved)
    return resolved
