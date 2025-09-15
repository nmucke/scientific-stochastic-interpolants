import torch
from scisi.preprocessing.preprocessor import Preprocesser
from scisi.data.load_data import load_stochastic_navier_stokes
import logging
import pdb
import os
import numpy as np

logger = logging.getLogger(__name__)

class StochasticNavierStokesDataset(torch.utils.data.Dataset):
    """Dataset for the stochastic Navier-Stokes data."""
    def __init__(
        self,
        paths: str,
        files: str,
        len_field_history: int,
        tajectories_ids: list[int] | tuple[int] = None,
        preprocesser: Preprocesser | None = None,
        train_or_test: str = "train",
    ):
        """
        Initialize the dataset.

        Args:
            paths: List of paths to the data.
            files: List of files to load.
            len_field_history: Length of the history.
        """
        self.data = load_stochastic_navier_stokes(paths, files)
        self.data = torch.from_numpy(self.data)
        self.data = torch.permute(self.data, (0, 2, 3, 1)) # [num_trajectories, num_channels, height, width, num_steps]
        self.data = self.data.unsqueeze(1)
        self.len_field_history = len_field_history
        self.preprocesser = preprocesser
        self.train_or_test = train_or_test

        if isinstance(tajectories_ids, list):
            self.data = self.data[tajectories_ids]
        elif isinstance(tajectories_ids, str):
            # Convert string tuple '(a,b)' to tuple of ints
            tajectories_ids = tuple(int(x) for x in tajectories_ids.strip('()').split(','))
            self.data = self.data[tajectories_ids[0]:tajectories_ids[1]]

        self.num_trajectories = self.data.shape[0]
        self.num_channels = self.data.shape[1]
        self.height = self.data.shape[2]
        self.width = self.data.shape[3]
        self.num_steps = self.data.shape[-1]

        if self.train_or_test == "train":
            self.data = self._prepare_data_windows()

    def _prepare_data_windows(self):
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
            self.len_field_history + 1
        )
        for trajectory in range(self.num_trajectories):
            for step in range(self.num_steps - self.len_field_history - 1):
                window = self.data[trajectory, :, :, :, step:step + self.len_field_history + 1]
                data[trajectory * (self.num_steps - self.len_field_history - 1) + step] = window

        return data

    def __len__(self):
        """Get the length of the dataset."""
        if self.train_or_test == "train":
            return self.num_trajectories * (self.num_steps - self.len_field_history - 1)
        else:
            return self.num_trajectories

    def _prepare_train_sample(self, sample):
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
            'field_history': sample["field_history"],
            'base': sample["base"],
            'target': sample["target"],
        }

    def __getitem__(self, idx):
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
        starting_time: int = None,
        ending_time: int = None,
        preprocesser: Preprocesser | None = None, 
        train_or_test: str = "train",
        save_in_memory: bool = True,
        cache_dir: str = None,
        use_exisiting_cache: bool = False,
    ) -> None:
        """
        Initialize the dataset.
        """

        self.paths = paths
        self.files = files
        self.starting_time = starting_time
        self.ending_time = ending_time

        tas, ym, time, self.lat, self.lon = self._load_file(self.paths[0], self.files)

        self.num_trajectories = tas.shape[0]
        self.num_channels = tas.shape[1]
        self.height = tas.shape[2]
        self.width = tas.shape[3]
        self.num_steps = tas.shape[-1]
        self.len_field_history = len_field_history
        self.preprocesser = preprocesser
        self.train_or_test = train_or_test
        self.save_in_memory = save_in_memory
        self.cache_dir = cache_dir
        self.use_exisiting_cache = use_exisiting_cache

        if self.train_or_test == "train":
            self.data, self.field_cond, self.pars_cond = self._prepare_data_windows(tas, ym, time)
        else:
            self.data = tas
            self.field_cond = ym
            self.pars_cond = time

        del tas, ym, time

    def _load_file(self, path: str, files: list[str]) -> torch.Tensor:
        """Load the file."""

        time = []
        tas = []
        ym = []
        for file in files:
            data = np.load(os.path.join(path, file))
            if self.starting_time is not None:
                data["time"] = data["time"][self.starting_time:self.ending_time]
                data["tas"] = data["tas"][self.starting_time:self.ending_time]
                data["ym"] = data["ym"][self.starting_time:self.ending_time]
            time.append(data["time"] % 365) # days
            tas.append(data["tas"]) # temperature at surface
            ym.append(data["ym"]) # yearly mean temperature per grid cell

        lat = data["lat"]
        lon = data["lon"]

        # Stack the data
        time = np.stack(time, axis=0)
        tas = np.stack(tas, axis=0)
        ym = np.stack(ym, axis=0)

        # Convert to torch.Tensor
        time = torch.from_numpy(time)
        tas = torch.from_numpy(tas)
        ym = torch.from_numpy(ym)

        # Permute the data to have time as the last dimension
        tas = tas.permute(0, 2, 3, 1)
        ym = ym.permute(0, 2, 3, 1)

        # Unsqueeze the data to have channels as the second dimension
        tas = tas.unsqueeze(1)
        ym = ym.unsqueeze(1)

        time = time.float()

        return tas, ym, time, lat, lon

    
    def _prepare_data_windows(
        self,
        tas: torch.Tensor,
        ym: torch.Tensor,
        time: torch.Tensor,
    ):
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
            data = torch.zeros(
                self.num_trajectories * (self.num_steps - self.len_field_history - 1), 
                self.num_channels, 
                self.height, 
                self.width,
                self.len_field_history + 1
            )
            field_cond = torch.zeros(
                self.num_trajectories * (self.num_steps - self.len_field_history - 1), 
                self.num_channels, 
                self.height, 
                self.width,
            )
            pars_cond = torch.zeros(
                self.num_trajectories * (self.num_steps - self.len_field_history - 1), 
                1, 
            )

        if self.use_exisiting_cache and (not self.save_in_memory):
            logger.info(f"Using existing cache directory {self.cache_dir}...")
        else:
            if not self.save_in_memory:
                logger.info(f"Saving data windows to cache directory {self.cache_dir}...")
                # Create cache folder for storing data windows
                os.makedirs(self.cache_dir, exist_ok=True)
                counter = 0

            for trajectory in range(self.num_trajectories):
                for step in range(self.num_steps - self.len_field_history - 1):
                    data_window = tas[trajectory, :, :, :, step:step + self.len_field_history + 1]
                    field_cond_window = ym[trajectory, :, :, :, step + self.len_field_history + 1]
                    pars_cond_window = time[trajectory, step + self.len_field_history + 1]

                    if self.save_in_memory:
                        data[trajectory * (self.num_steps - self.len_field_history - 1) + step] = data_window
                        field_cond[trajectory * (self.num_steps - self.len_field_history - 1) + step] = field_cond_window
                        pars_cond[trajectory * (self.num_steps - self.len_field_history - 1) + step] = pars_cond_window
                    else:
                        np.savez(os.path.join(self.cache_dir, f"data_{counter}.npz"), data_window.numpy())
                        np.savez(os.path.join(self.cache_dir, f"field_cond_{counter}.npz"), field_cond_window.numpy())
                        np.savez(os.path.join(self.cache_dir, f"pars_cond_{counter}.npz"), pars_cond_window.numpy())
                        counter += 1

        if self.save_in_memory:
            return data, field_cond, pars_cond
        else:
            return None, None, None


    def _prepare_train_sample(
        self, 
        sample: torch.Tensor, 
        field_cond: torch.Tensor, 
        pars_cond: torch.Tensor
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
            'base': sample["base"],
            'field_history': sample["field_history"],
            'field_cond': sample["field_cond"],
            'pars_cond': pars_cond,
            'target': sample["target"],
        }

    def __len__(self):
        """Get the length of the dataset."""
        if self.train_or_test == "train":
            return self.num_trajectories * (self.num_steps - self.len_field_history - 1)
        else:
            return self.num_trajectories


    def __getitem__(self, idx: int):
        """Get the item at the given index."""

        if self.save_in_memory:
            sample = self.data[idx]
            field_cond = self.field_cond[idx]
            pars_cond = self.pars_cond[idx] / 365.0
        else:
            sample = np.load(os.path.join(self.cache_dir, f"data_{idx}.npz"))["arr_0"]
            field_cond = np.load(os.path.join(self.cache_dir, f"field_cond_{idx}.npz"))["arr_0"]
            pars_cond = np.load(os.path.join(self.cache_dir, f"pars_cond_{idx}.npz"))["arr_0"]

            sample = torch.from_numpy(sample)
            field_cond = torch.from_numpy(field_cond)
            pars_cond = torch.from_numpy(pars_cond) / 365.0
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
