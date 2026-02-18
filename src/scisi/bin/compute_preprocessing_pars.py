import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import xarray as xr

dir = "data/udales"

u_mean = 0
v_mean = 0
w_mean = 0
thl_mean = 0

u_std = 0
v_std = 0
w_std = 0
thl_std = 0

mask = np.load(f"{dir}/mask.npz")["mask"]
mask_ids = np.where(mask == 1)

files = [f"{dir}/sim_{i}.nc" for i in range(0, 178)]
for file in tqdm.tqdm(files):
    data = xr.open_dataset(file, engine="netcdf4")
    u = data.u.values
    v = data.v.values
    w = data.w.values
    thl = data.thl.values

    num_steps = u.shape[0]

    u = u[:, mask_ids[0], mask_ids[1]]
    v = v[:, mask_ids[0], mask_ids[1]]
    w = w[:, mask_ids[0], mask_ids[1]]
    thl = thl[:, mask_ids[0], mask_ids[1]]

    u_std += u.std()
    v_std += v.std()
    w_std += w.std()
    thl_std += thl.std()

    u_mean += u.mean()
    v_mean += v.mean()
    w_mean += w.mean()
    thl_mean += thl.mean()

u_mean = u_mean / len(files)  # type: ignore[assignment]
v_mean = v_mean / len(files)  # type: ignore[assignment]
w_mean = w_mean / len(files)  # type: ignore[assignment]
thl_mean = thl_mean / len(files)  # type: ignore[assignment]

u_std = u_std / len(files)  # type: ignore[assignment]
v_std = v_std / len(files)  # type: ignore[assignment]
w_std = w_std / len(files)  # type: ignore[assignment]
thl_std = thl_std / len(files)  # type: ignore[assignment]

print(f"u_mean: {u_mean}")
print(f"v_mean: {v_mean}")
print(f"w_mean: {w_mean}")
print(f"thl_mean: {thl_mean}")

print(f"u_std: {u_std}")
print(f"v_std: {v_std}")
print(f"w_std: {w_std}")
print(f"thl_std: {thl_std}")
