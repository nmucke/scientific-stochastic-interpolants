"""Train a Lorenz 96 prior with the shared trainer, offline.

    python cases/lorenz96/train.py --config lorenz96_stochastic_interpolant --name si
    python cases/lorenz96/train.py --config lorenz96_flow_matching --name fm

Trailing ``key=value`` arguments override the config. Checkpoints land in
``checkpoints/lorenz96/<name>/`` at the repository root.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parents[1]))

from cases.lorenz96._l96_system import resolve_device  # noqa: E402
from common.paths import REPO_ROOT  # noqa: E402
from common.seeding import derive_seed  # noqa: E402
from scisi.architectures.architecture_utils import count_model_parameters  # noqa: E402


class LocalTracker:
    """The part of the tracker interface the trainer uses, writing to a local file."""

    def __init__(self, project: str, name: str, config: dict) -> None:
        self.project, self.name, self.config = project, name, config
        self.directory = Path("checkpoints") / project / name
        self.directory.mkdir(parents=True, exist_ok=True)
        self.url = str(self.directory / "metrics.jsonl")
        self._start = time.time()
        Path(self.url).write_text("")

    def log(self, metrics: dict) -> None:
        row = {key: float(value) for key, value in metrics.items()}
        row["elapsed"] = time.time() - self._start
        with open(self.url, "a") as handle:
            handle.write(json.dumps(row) + "\n")

    def finish(self) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(args.threads)
    torch.manual_seed(derive_seed("lorenz96-training", args.name, args.seed))
    os.chdir(REPO_ROOT)  # the trainer writes to a relative checkpoints/ directory

    cfg = OmegaConf.merge(
        OmegaConf.load(Path("config") / f"{args.config}.yaml"),
        OmegaConf.from_dotlist(args.overrides),
    )
    cfg.trainer.device = resolve_device(cfg.trainer.device)
    tracker = LocalTracker(
        cfg.experiment_tracking.project,
        args.name,
        OmegaConf.to_container(cfg, resolve=True),
    )
    train_loader = hydra.utils.instantiate(cfg.train_data)
    val_loader = hydra.utils.instantiate(cfg.val_data)
    model = hydra.utils.instantiate(cfg.model)
    logging.info("parameters: %.3fM", count_model_parameters(model) / 1e6)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        model=model,
        optimizer=optimizer,
        scheduler=hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer),
        loss_fn=hydra.utils.instantiate(cfg.loss_fn),
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        tracker=tracker,
    )
    trainer.train()
    tracker.finish()


if __name__ == "__main__":
    main()
