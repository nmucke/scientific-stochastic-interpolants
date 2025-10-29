import numpy as np
import torch

from scisi.metrics.LSIM.distance_model import DistanceModel

lsim_model = DistanceModel(
    baseType="lsim", isTrain=False, useGPU=torch.cuda.is_available()
)
lsim_model.load("src/scisi/metrics/LSIM/LSiM.pth")


def LSiM_distance(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # https://github.com/tum-pbs/LSIM
    # Expected input sizes: [1, 3, 256, 256], [3, 256, 256]  or [256,256]
    assert A.shape == B.shape, "Input shapes must match!"
    global lsim_model

    if len(A.shape) == 4:
        A = A[0]
        B = B[0]

    if A.shape[0] == 3:
        return np.mean(
            [
                LSiM_distance(A[0], B[0]),
                LSiM_distance(A[1], B[1]),
                LSiM_distance(A[2], B[2]),
            ]
        )

    if len(A.shape) == 2:
        A = A.unsqueeze(-1)

    if len(B.shape) == 2:
        B = B.unsqueeze(-1)

    A = A.cpu() if type(A) is torch.Tensor else A
    B = B.cpu() if type(B) is torch.Tensor else B

    dist = lsim_model.computeDistance(A, B)

    return dist[0]
