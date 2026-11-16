import sys, os
sys.path.insert(0, os.getcwd())
import torch
print('torch',torch.__version__,torch.version.cuda,'cuda',torch.cuda.is_available(),flush=True)
print(torch.cuda.get_device_name(0), [round(x/2**30,2) for x in torch.cuda.mem_get_info()],flush=True)
x=torch.randn(2048,2048,device='cuda'); print('torch matmul',(x@x).abs().sum().item(),flush=True)
from cases.navier_stokes import enkf_baseline
import jax, jax.numpy as jnp
print('jax',jax.__version__,jax.devices(),flush=True)
print('jax matmul',float(jnp.abs(jnp.ones((1024,1024))@jnp.ones((1024,1024))).sum()),flush=True)
import jax_cfd.base.grids as g, jax_cfd_lib
print('OK all imports',flush=True)
