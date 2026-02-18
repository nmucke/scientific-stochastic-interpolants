import logging
import os
import pdb
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from aurora_lib.batch_adapter import BatchAdapter
from aurora_lib.load_data import load_batch

from scisi.data.load_data import load_stochastic_navier_stokes
from scisi.preprocessing.preprocessor import Preprocesser

logger = logging.getLogger(__name__)


class StochasticNavierStokesDataset(torch.utils.data.Dataset):
    """Dataset for the stochastic Navier-Stokes data."""

    def __init__(
        self,
        paths: str,
        files: str,
        len_field_history: int,
        tajectories_ids: list[int] | str | None = None,
        preprocesser: Preprocesser | None = None,
        train_or_test: str = "train",
    ) -> None:
        """
        Initialize the dataset.

        Args:
            paths: List of paths to the data.
            files: List of files to load.
            len_field_history: Length of the history.
            tajectories_ids: List of trajectories ids to load. (a,b) to load trajectories from a to b.
            preprocesser: Preprocesser to use.
            train_or_test: Train or test.
        """
        self.data = load_stochastic_navier_stokes(paths, files)
        self.data = torch.from_numpy(self.data)
        self.data = torch.permute(
            self.data, (0, 2, 3, 1)
        )  # [num_trajectories, num_channels, height, width, num_steps]
        self.data = self.data.unsqueeze(1)
        self.len_field_history = len_field_history
        self.preprocesser = preprocesser
        self.train_or_test = train_or_test

        if isinstance(tajectories_ids, list):
            self.data = self.data[tajectories_ids]
        elif isinstance(tajectories_ids, str):
            # Convert string tuple '(a,b)' to tuple of ints
            tajectories_ids = tuple(  # type: ignore[assignment]
                int(x) for x in tajectories_ids.strip("()").split(",")
            )
            self.data = self.data[tajectories_ids[0] : tajectories_ids[1]]  # type: ignore[misc]

        self.num_trajectories = self.data.shape[0]
        self.num_channels = self.data.shape[1]
        self.height = self.data.shape[2]
        self.width = self.data.shape[3]
        self.num_steps = self.data.shape[-1]

        if self.train_or_test == "train":
            self.data = self._prepare_data_windows()

    def _prepare_data_windows(self) -> torch.Tensor:
        """
        Prepare the data.

        Returns:
            torch.Tensor: Data windows. [num_trajectories * (num_steps - len_field_history - 1), num_channels, height, width, len_field_history + 1]
        """

        logger.info(
            f"Preparing data windows for {self.num_trajectories} "
            f"trajectories with {self.num_steps} steps and "
            f"{self.len_field_history} history."
        )

        data = torch.zeros(
            self.num_trajectories * (self.num_steps - self.len_field_history - 1),
            self.num_channels,
            self.height,
            self.width,
            self.len_field_history + 1,
        )
        for trajectory in range(self.num_trajectories):
            for step in range(self.num_steps - self.len_field_history - 1):
                window = self.data[
                    trajectory, :, :, :, step : step + self.len_field_history + 1
                ]
                data[
                    trajectory * (self.num_steps - self.len_field_history - 1) + step
                ] = window

        return data

    def __len__(self) -> int:
        """Get the length of the dataset."""
        if self.train_or_test == "train":
            return int(
                self.num_trajectories * (self.num_steps - self.len_field_history - 1)
            )
        else:
            return int(self.num_trajectories)

    def _prepare_train_sample(self, sample: torch.Tensor) -> dict:
        """
        Prepare the train sample.

        Returns:
            dict: Sample.
                'field_history': Field history. [B, C, H, W, L]
                'base': Base. [B, C, H, W]
                'target': Target. [B, C, H, W]
        """

        field_history = sample[:, :, :, :-1]
        base = sample[:, :, :, -2]
        target = sample[:, :, :, -1]

        if self.preprocesser is not None:
            sample = self.preprocesser.transform(
                base=base,
                target=target,
                field_history=field_history,
            )

        return {
            "field_history": sample["field_history"],
            "base": sample["base"],
            "target": sample["target"],
        }

    def __getitem__(self, idx: int) -> dict:
        """Get the item at the given index."""

        sample = self.data[idx]

        if self.train_or_test == "train":
            return self._prepare_train_sample(sample)
        else:
            return {
                "x": sample,
            }


class KNMIDataset(torch.utils.data.Dataset):
    """Dataset for the KNMIData."""

    def __init__(
        self,
        paths: list[str],
        files: list[str],
        len_field_history: int,
        starting_time: int | None = None,
        ending_time: int | None = None,
        preprocesser: Preprocesser | None = None,
        train_or_test: str = "train",
        save_in_memory: bool = True,
        cache_dir: str | None = None,
        use_exisiting_cache: bool = False,
    ) -> None:
        """
        Initialize the dataset.
        """

        self.paths = paths
        self.files = files
        self.starting_time = starting_time
        self.ending_time = ending_time
        self.len_field_history = len_field_history
        self.preprocesser = preprocesser
        self.train_or_test = train_or_test
        self.save_in_memory = save_in_memory
        self.cache_dir = cache_dir
        self.use_exisiting_cache = use_exisiting_cache

        self.num_trajectories = len(self.files)

        if self.train_or_test == "train":
            self.data, self.field_cond, self.pars_cond = self._prepare_data_windows()
        else:
            self.data = []
            self.field_cond = []
            self.pars_cond = []

            for trajectory in range(self.num_trajectories):
                tas, ym, time, self.lat, self.lon = self._load_file(
                    self.paths[trajectory], self.files[trajectory]
                )

                self.data.append(tas)
                self.field_cond.append(ym)
                self.pars_cond.append(time)

            self.data = torch.concat(self.data, dim=0)
            self.field_cond = torch.concat(self.field_cond, dim=0)
            self.pars_cond = torch.concat(self.pars_cond, dim=0)

            del tas, ym, time

    def _load_file(
        self, path: str, file: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load the file."""

        time = []
        tas = []
        ym = []
        data = np.load(os.path.join(path, file))
        if self.starting_time is not None:
            data["time"] = data["time"][self.starting_time : self.ending_time]
            data["tas"] = data["tas"][self.starting_time : self.ending_time]
            data["ym"] = data["ym"][self.starting_time : self.ending_time]
        time.append(data["time"] % 365)  # days
        tas.append(data["tas"])  # temperature at surface
        ym.append(data["ym"])  # yearly mean temperature per grid cell

        lat = torch.from_numpy(data["lat"])
        lon = torch.from_numpy(data["lon"])

        # Stack the data
        time = np.stack(time, axis=0)
        tas = np.stack(tas, axis=0)
        ym = np.stack(ym, axis=0)

        # Convert to torch.Tensor
        time = torch.from_numpy(time)
        tas = torch.from_numpy(tas)
        ym = torch.from_numpy(ym)

        # Permute the data to have time as the last dimension
        tas = tas.permute(0, 2, 3, 1)  # type: ignore[attr-defined]
        ym = ym.permute(0, 2, 3, 1)  # type: ignore[attr-defined]

        # Unsqueeze the data to have channels as the second dimension
        tas = tas.unsqueeze(1)  # type: ignore[attr-defined]
        ym = ym.unsqueeze(1)  # type: ignore[attr-defined]

        time = time.float()  # type: ignore[attr-defined]
        lat = lat.float()
        lon = lon.float()

        return tas, ym, time, lat, lon

    def _prepare_data_windows(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare the data.

        Returns:
            torch.Tensor: Data windows. [num_trajectories * (num_steps - len_field_history - 1), num_channels, height, width, len_field_history + 1]
        """

        logger.info(
            f"Preparing data windows for {self.num_trajectories} "
            f"and {self.len_field_history} history."
        )

        if self.save_in_memory:
            data = []
            field_cond = []
            pars_cond = []

        if self.use_exisiting_cache and (not self.save_in_memory):
            logger.info(f"Using existing cache directory {self.cache_dir}...")

            tas, ym, time, self.lat, self.lon = self._load_file(
                self.paths[0], self.files[0]
            )

            self.num_channels = tas.shape[1]
            self.height = tas.shape[2]
            self.width = tas.shape[3]
            self.num_steps = tas.shape[-1]

            del tas, ym, time
        else:
            if not self.save_in_memory:

                logger.info(
                    f"Saving data windows to cache directory {self.cache_dir}..."
                )
                # Create cache folder for storing data windows
                os.makedirs(self.cache_dir, exist_ok=True)  # type: ignore[arg-type]
                counter = 0

            for trajectory in range(self.num_trajectories):

                tas, ym, time, self.lat, self.lon = self._load_file(
                    self.paths[0], self.files[trajectory]
                )

                self.num_channels = tas.shape[1]
                self.height = tas.shape[2]
                self.width = tas.shape[3]
                self.num_steps = tas.shape[-1]

                tas = tas.squeeze(0)
                ym = ym.squeeze(0)
                time = time.squeeze(0)

                for step in range(self.num_steps - self.len_field_history - 1):
                    data_window = tas[:, :, :, step : step + self.len_field_history + 1]
                    field_cond_window = ym[:, :, :, step + self.len_field_history + 1]
                    pars_cond_window = time[step + self.len_field_history + 1]

                    if self.save_in_memory:
                        data.append(data_window)
                        field_cond.append(field_cond_window)
                        pars_cond.append(pars_cond_window)
                    else:
                        np.savez(
                            os.path.join(self.cache_dir, f"sample_{counter}.npz"),  # type: ignore[arg-type]
                            data=data_window.numpy(),
                            field_cond=field_cond_window.numpy(),
                            pars_cond=pars_cond_window.numpy(),
                        )
                        counter += 1
                logger.info(f"Saved trajectory {trajectory} to cache directory.")

        if self.save_in_memory:
            data = torch.stack(data)
            field_cond = torch.stack(field_cond)
            pars_cond = torch.stack(pars_cond)
            return data, field_cond, pars_cond
        else:
            return None, None, None

    def _prepare_train_sample(
        self, sample: torch.Tensor, field_cond: torch.Tensor, pars_cond: torch.Tensor
    ) -> dict:
        """
        Prepare the train sample.

        Returns:
            dict: Sample.
                'base': Base. [B, C, H, W]
                'field_cond': Field conditional. [B, C, H, W]
                'pars_cond': Pars conditional. [B, D]
        """
        target = sample[:, :, :, -1]
        base = sample[:, :, :, -2]
        field_history = sample[:, :, :, :-1]

        if self.preprocesser is not None:
            sample = self.preprocesser.transform(
                base=base,
                target=target,
                field_cond=field_cond,
                field_history=field_history,
            )

        return {
            "base": sample["base"],
            "field_history": sample["field_history"],
            "field_cond": sample["field_cond"],
            "pars_cond": pars_cond,
            "target": sample["target"],
        }

    def __len__(self) -> int:
        """Get the length of the dataset."""
        if self.train_or_test == "train":
            return int(
                self.num_trajectories * (self.num_steps - self.len_field_history - 1)
            )
        else:
            return int(self.num_trajectories)

    def __getitem__(self, idx: int) -> dict:
        """Get the item at the given index."""

        if self.save_in_memory:
            sample = self.data[idx]
            field_cond = self.field_cond[idx]
            pars_cond = self.pars_cond[idx]
        else:
            data = np.load(os.path.join(self.cache_dir, f"sample_{idx}.npz"))  # type: ignore[arg-type]
            sample = data["data"]
            field_cond = data["field_cond"]
            pars_cond = data["pars_cond"]

            sample = torch.from_numpy(sample)
            field_cond = torch.from_numpy(field_cond)
            pars_cond = torch.from_numpy(pars_cond)
            pars_cond = pars_cond.unsqueeze(0)
            pars_cond = pars_cond.float()

        if self.train_or_test == "train":
            return self._prepare_train_sample(sample, field_cond, pars_cond)
        else:
            return {
                "x": sample,
                "field_cond": field_cond,
                "pars_cond": pars_cond,
            }


class WeatherDataset(torch.utils.data.Dataset):
    """Dataset for the KNMIData."""

    def __init__(
        self,
        paths: list[str],
        files: list[str],
        len_field_history: int,
        starting_time: int | None = None,
        ending_time: int | None = None,
        preprocesser: Preprocesser | None = None,
        train_or_test: str = "train",
        save_in_memory: bool = True,
        cache_dir: str | None = None,
        use_exisiting_cache: bool = False,
    ) -> None:
        """
        Initialize the dataset.
        """

        self.paths = paths
        self.files = files
        self.starting_time = starting_time
        self.ending_time = ending_time

        data = self._load_file(self.paths[0], self.files)

        self.num_trajectories = data.shape[0]
        self.num_channels = data.shape[1]
        self.height = data.shape[2]
        self.width = data.shape[3]
        self.num_steps = data.shape[-1]
        self.len_field_history = len_field_history
        self.preprocesser = preprocesser
        self.train_or_test = train_or_test
        self.save_in_memory = save_in_memory
        self.cache_dir = cache_dir
        self.use_exisiting_cache = use_exisiting_cache

        if self.train_or_test == "train":
            self.data = self._prepare_data_windows(data)
        else:
            self.data = data

        del data

    def _load_file(self, path: str, files: list[str]) -> torch.Tensor:
        """Load the files."""

        data = []
        for file in files:
            sample = np.load(os.path.join(path, file))["vil"]
            sample = torch.from_numpy(sample)
            sample = sample.unsqueeze(1)
            sample = sample.float()
            data.append(sample)

        return torch.concat(data, dim=0)

    def _prepare_data_windows(
        self,
        data: torch.Tensor,
    ) -> torch.Tensor | None:
        """
        Prepare the data.

        Returns:
            torch.Tensor: Data windows. [num_trajectories * (num_steps - len_field_history - 1), num_channels, height, width, len_field_history + 1]
        """

        logger.info(
            f"Preparing data windows for {self.num_trajectories} "
            f"trajectories with {self.num_steps} steps and "
            f"{self.len_field_history} history."
        )

        if self.save_in_memory:
            data_windows = torch.zeros(
                self.num_trajectories * (self.num_steps - self.len_field_history - 1),
                self.num_channels,
                self.height,
                self.width,
                self.len_field_history + 1,
            )

        if self.use_exisiting_cache and (not self.save_in_memory):
            logger.info(f"Using existing cache directory {self.cache_dir}...")
        else:
            if not self.save_in_memory:
                logger.info(
                    f"Saving data windows to cache directory {self.cache_dir}..."
                )
                # Create cache folder for storing data windows
                os.makedirs(self.cache_dir, exist_ok=True)  # type: ignore[arg-type]

            counter = 0
            for trajectory in range(self.num_trajectories):
                for step in range(self.num_steps - self.len_field_history - 1):
                    window = data[
                        trajectory, :, :, :, step : step + self.len_field_history + 1
                    ]

                    if self.save_in_memory:
                        data_windows[counter] = window
                    else:
                        np.savez(
                            os.path.join(self.cache_dir, f"sample_{counter}.npz"),  # type: ignore[arg-type]
                            data=window.numpy(),
                        )

                    counter += 1

        if self.save_in_memory:
            return data_windows
        else:
            return None

    def _prepare_train_sample(
        self,
        sample: torch.Tensor,
    ) -> dict:
        """
        Prepare the train sample.

        Returns:
            dict: Sample.
                'base': Base. [B, C, H, W]
        """
        target = sample[:, :, :, -1]
        base = sample[:, :, :, -2]
        field_history = sample[:, :, :, :-1]

        if self.preprocesser is not None:
            sample = self.preprocesser.transform(
                base=base,
                target=target,
                field_history=field_history,
            )

        return {
            "base": sample["base"],
            "field_history": sample["field_history"],
            "target": sample["target"],
        }

    def __len__(self) -> int:
        """Get the length of the dataset."""
        if self.train_or_test == "train":
            return int(
                self.num_trajectories * (self.num_steps - self.len_field_history - 1)
            )
        else:
            return int(self.num_trajectories)

    def __getitem__(self, idx: int) -> dict:
        """Get the item at the given index."""

        if self.save_in_memory:
            sample = self.data[idx]  # type: ignore[index]
        else:
            sample = np.load(os.path.join(self.cache_dir, f"sample_{idx}.npz"))["data"]  # type: ignore[arg-type]

            sample = torch.from_numpy(sample)
        if self.train_or_test == "train":
            return self._prepare_train_sample(sample)
        else:
            return {
                "x": sample,
            }


class AuroraDataset(torch.utils.data.Dataset):
    """Dataset for the Aurora data."""

    def __init__(
        self,
        datetimes: list[str],
        cache_dir: str,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            paths: List of paths to the data.
        """
        self.datetimes = [datetime.strptime(dt, "%Y-%m-%d") for dt in datetimes]

        self.cache_dir = cache_dir

        logger.info(f"Loading initial aurora batch...")
        aurora_batch = load_batch(day=self.datetimes[0], cache_path=self.cache_dir)
        self.batch_adapter = BatchAdapter(
            aurora_batch.metadata, aurora_batch.static_vars
        )

        self.lat = aurora_batch.metadata.lat

        del aurora_batch

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.datetimes)

    def __getitem__(self, idx: int) -> dict:
        """Get the item at the given index."""

        aurora_batch = load_batch(day=self.datetimes[idx], cache_path=self.cache_dir)

        x, field_history = self.batch_adapter.aurora_to_scisi(aurora_batch)

        return {
            "base": x.squeeze(0),
            "field_history": field_history.squeeze(0),
            "target": x.squeeze(0),
        }


class UDalesDataset(torch.utils.data.Dataset):
    """Dataset for the stochastic Navier-Stokes data."""

    def __init__(
        self,
        paths: str,
        files: str,
        len_field_history: int,
        preprocesser: Preprocesser | None = None,
        train_or_test: str = "train",
        starting_time: int = 0,
        ending_time: int = 200,
        skip_steps: int = 1,
        save_in_memory: bool = True,
        cache_dir: str | None = None,
        use_exisiting_cache: bool = False,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            paths: List of paths to the data.
            files: String of files to load. (e.g. "(1,10)" to load files 1 to 10)
            len_field_history: Length of the history.
            preprocesser: Preprocesser to use.
            train_or_test: Train or test.
            starting_time: Starting time.
            ending_time: Ending time.
            skip_steps: Skip steps.
            save_in_memory: Save in memory.
            cache_dir: Cache directory.
            use_exisiting_cache: Use existing cache.
        """
        self.paths = paths

        # Convert string tuple '(a,b)' to tuple of ints
        ids = tuple(int(x) for x in files.strip("()").split(","))
        self.files = [f"sim_{i}.nc" for i in range(ids[0], ids[1] + 1)]
        self.files = [f"{paths[0]}/{file}" for file in self.files]
        self.starting_time = starting_time
        self.ending_time = ending_time
        self.skip_steps = skip_steps
        self.num_steps = (ending_time - starting_time) // skip_steps + 1
        self.len_field_history = len_field_history
        self.preprocesser = preprocesser
        self.train_or_test = train_or_test
        self.save_in_memory = save_in_memory
        self.cache_dir = cache_dir
        self.use_exisiting_cache = use_exisiting_cache

        self.num_trajectories = len(self.files)

        self.mask = np.load(f"{paths[0]}/mask.npz")["mask"]
        self.mask = torch.from_numpy(self.mask).unsqueeze(0)
        self.mask = self.mask.to(dtype=torch.float32)

        self.height = 128
        self.width = 128
        self.num_channels = 5

        if self.train_or_test == "train":
            if (not self.use_exisiting_cache) and (not self.save_in_memory):
                self._prepare_data_windows()
        else:
            self.data = torch.stack(
                [self._load_file(file) for file in self.files], dim=0
            )

    def _load_file(self, file: str) -> torch.Tensor:
        """Load the file."""
        data = xr.open_dataset(file)

        u = torch.from_numpy(data.u.values)
        v = torch.from_numpy(data.v.values)
        w = torch.from_numpy(data.w.values)
        thl = torch.from_numpy(data.thl.values)
        # qt = torch.from_numpy(data.qt.values)

        torch_data = torch.stack(
            [u, v, w, thl], dim=0
        )  # [num_channels, num_steps, height, width]

        torch_data = torch_data.permute(
            0, 2, 3, 1
        )  # [num_channels, height, width, num_steps]
        return torch_data[..., self.starting_time : self.ending_time : self.skip_steps]

    def _prepare_data_windows(self) -> None:
        """Prepare the data windows."""

        logger.info(
            f"Preparing data windows for {self.num_trajectories} "
            f"trajectories with {self.num_steps} steps and "
            f"{self.len_field_history} history."
        )

        if self.save_in_memory:
            logger.info(f"Saving data windows to memory...")
            self.data = torch.zeros(
                self.num_trajectories * (self.num_steps - self.len_field_history - 1),
                self.num_channels,
                self.height,
                self.width,
                self.len_field_history + 1,
            )
        else:
            logger.info(f"Saving data windows to cache directory {self.cache_dir}...")
            # Create cache folder for storing data windows
            os.makedirs(self.cache_dir, exist_ok=True)  # type: ignore[arg-type]

        counter = 0
        for file in self.files:
            sample = self._load_file(file)

            for step in range(self.num_steps - self.len_field_history - 1):
                window = sample[:, :, :, step : step + self.len_field_history + 1]
                if self.save_in_memory:
                    self.data[counter] = window
                else:
                    np.savez(
                        os.path.join(self.cache_dir, f"sample_{counter}.npz"),  # type: ignore[arg-type]
                        data=window.numpy(),
                    )
                counter += 1

    def __len__(self) -> int:
        """Get the length of the dataset."""
        if self.train_or_test == "train":
            return int(
                self.num_trajectories * (self.num_steps - self.len_field_history - 1)
            )
        else:
            return int(self.num_trajectories)

    def _prepare_train_sample(self, sample: torch.Tensor) -> dict:
        """
        Prepare the train sample.

        Returns:
            dict: Sample.
                'field_history': Field history. [B, C, H, W, L]
                'base': Base. [B, C, H, W]
                'target': Target. [B, C, H, W]
        """

        field_history = sample[:, :, :, :-1]
        base = sample[:, :, :, -2]
        target = sample[:, :, :, -1]

        if self.preprocesser is not None:
            sample = self.preprocesser.transform(
                base=base,
                target=target,
                field_history=field_history,
            )

        return {
            "field_history": sample["field_history"],
            "base": sample["base"],
            "target": sample["target"],
            "field_cond": self.mask,
        }

    def __getitem__(self, idx: int) -> dict:
        """Get the item at the given index."""

        if self.save_in_memory:
            sample = self.data[idx]
        else:
            sample = np.load(os.path.join(self.cache_dir, f"sample_{idx}.npz"))["data"]  # type: ignore[arg-type]

            sample = torch.from_numpy(sample)

        if self.train_or_test == "train":
            return self._prepare_train_sample(sample)
        else:
            return {
                "x": sample,
                "field_cond": self.mask.unsqueeze(-1).repeat(1, 1, 1, sample.shape[-1]),
            }


class XieAndCastroDataset(torch.utils.data.Dataset):
    """Dataset for the stochastic Navier-Stokes data."""

    def __init__(
        self,
        paths: str,
        files: str,
        len_field_history: int,
        preprocesser: Preprocesser | None = None,
        train_or_test: str = "train",
        starting_time: int = 0,
        ending_time: int = 200,
        skip_steps: int = 1,
        save_in_memory: bool = True,
        cache_dir: str | None = None,
        use_exisiting_cache: bool = False,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            paths: List of paths to the data.
            files: String of files to load. (e.g. "(1,10)" to load files 1 to 10)
            len_field_history: Length of the history.
            preprocesser: Preprocesser to use.
            train_or_test: Train or test.
            starting_time: Starting time.
            ending_time: Ending time.
            skip_steps: Skip steps.
            save_in_memory: Save in memory.
            cache_dir: Cache directory.
            use_exisiting_cache: Use existing cache.
        """
        self.paths = paths

        # Convert string tuple '(a,b)' to tuple of ints
        ids = tuple(int(x) for x in files.strip("()").split(","))
        self.files = [f"sim_{i}.nc" for i in range(ids[0], ids[1] + 1)]
        self.files = [f"{paths[0]}/{file}" for file in self.files]
        self.starting_time = starting_time
        self.ending_time = ending_time
        self.skip_steps = skip_steps
        self.num_steps = (ending_time - starting_time) // skip_steps + 1
        self.len_field_history = len_field_history
        self.preprocesser = preprocesser
        self.train_or_test = train_or_test
        self.save_in_memory = save_in_memory
        self.cache_dir = cache_dir
        self.use_exisiting_cache = use_exisiting_cache

        self.num_trajectories = len(self.files)

        self.mask = np.load(f"{paths[0]}/mask.npz")["mask"]
        self.mask = torch.from_numpy(self.mask).unsqueeze(0)
        self.mask = self.mask.to(dtype=torch.float32)

        self.height = 128
        self.width = 128
        self.num_channels = 5

        if self.train_or_test == "train":
            if (not self.use_exisiting_cache) and (not self.save_in_memory):
                self._prepare_data_windows()
        else:
            self.data = torch.stack(
                [self._load_file(file) for file in self.files], dim=0
            )

    def _load_file(self, file: str) -> torch.Tensor:
        """Load the file."""
        data = xr.open_dataset(file, engine="netcdf4")

        u = torch.from_numpy(data.u.values)
        v = torch.from_numpy(data.v.values)
        w = torch.from_numpy(data.w.values)
        pres = torch.from_numpy(data.pres.values)
        # qt = torch.from_numpy(data.qt.values)

        torch_data = torch.stack(
            [u, v, w, pres], dim=0
        )  # [num_channels, num_steps, height, width]

        torch_data = torch_data.permute(
            0, 2, 3, 1
        )  # [num_channels, height, width, num_steps]
        return torch_data[..., self.starting_time : self.ending_time : self.skip_steps]

    def _prepare_data_windows(self) -> None:
        """Prepare the data windows."""

        logger.info(
            f"Preparing data windows for {self.num_trajectories} "
            f"trajectories with {self.num_steps} steps and "
            f"{self.len_field_history} history."
        )

        if self.save_in_memory:
            logger.info(f"Saving data windows to memory...")
            self.data = torch.zeros(
                self.num_trajectories * (self.num_steps - self.len_field_history - 1),
                self.num_channels,
                self.height,
                self.width,
                self.len_field_history + 1,
            )
        else:
            logger.info(f"Saving data windows to cache directory {self.cache_dir}...")
            # Create cache folder for storing data windows
            os.makedirs(self.cache_dir, exist_ok=True)  # type: ignore[arg-type]

        counter = 0
        for file in self.files:
            sample = self._load_file(file)

            for step in range(self.num_steps - self.len_field_history - 1):
                window = sample[:, :, :, step : step + self.len_field_history + 1]
                if self.save_in_memory:
                    self.data[counter] = window
                else:
                    np.savez(
                        os.path.join(self.cache_dir, f"sample_{counter}.npz"),  # type: ignore[arg-type]
                        data=window.numpy(),
                    )
                counter += 1

    def __len__(self) -> int:
        """Get the length of the dataset."""
        if self.train_or_test == "train":
            return int(
                self.num_trajectories * (self.num_steps - self.len_field_history - 1)
            )
        else:
            return int(self.num_trajectories)

    def _prepare_train_sample(self, sample: torch.Tensor) -> dict:
        """
        Prepare the train sample.

        Returns:
            dict: Sample.
                'field_history': Field history. [B, C, H, W, L]
                'base': Base. [B, C, H, W]
                'target': Target. [B, C, H, W]
        """

        field_history = sample[:, :, :, :-1]
        base = sample[:, :, :, -2]
        target = sample[:, :, :, -1]

        if self.preprocesser is not None:
            sample = self.preprocesser.transform(
                base=base,
                target=target,
                field_history=field_history,
            )

        return {
            "field_history": sample["field_history"],
            "base": sample["base"],
            "target": sample["target"],
            "field_cond": self.mask,
        }

    def __getitem__(self, idx: int) -> dict:
        """Get the item at the given index."""

        if self.save_in_memory:
            sample = self.data[idx]
        else:
            sample = np.load(os.path.join(self.cache_dir, f"sample_{idx}.npz"))["data"]  # type: ignore[arg-type]

            sample = torch.from_numpy(sample)

        if self.train_or_test == "train":
            return self._prepare_train_sample(sample)
        else:
            return {
                "x": sample,
                "field_cond": self.mask.unsqueeze(-1).repeat(1, 1, 1, sample.shape[-1]),
            }
