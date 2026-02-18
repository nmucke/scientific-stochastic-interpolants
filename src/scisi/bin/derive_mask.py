import pdb

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

data = xr.open_dataset("data/xie_and_castro/sim_1.nc", engine="netcdf4")

u = data.u.values
v = data.v.values
w = data.w.values
pres = data.pres.values


# thl = data.thl.values
# qt = data.qt.values

u_mean = u.mean(axis=0)

# Find indices along axis 1 and 2 where u is zero for all indices 0
zero_indices = np.where(np.all(np.isclose(u, 0, atol=1e-2), axis=0))

mask = np.ones((128, 128))
mask[zero_indices] = 0

np.savez("data/xie_and_castro/mask.npz", mask=mask)

plt.imshow(mask)
plt.colorbar()
plt.show()
