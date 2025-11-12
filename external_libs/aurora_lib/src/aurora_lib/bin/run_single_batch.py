import pdb

import matplotlib.pyplot as plt
import numpy as np
import torch
from aurora import AuroraSmallPretrained
from aurora_lib.batch_adapter import BatchAdapter
from aurora_lib.load_data import load_batch
from aurora_lib.model_wrapper import AuroraModelWrapper


def main() -> None:
    batch = load_batch()
    print("Batch loaded")

    model = AuroraModelWrapper(autocast=False)
    print("Model loaded")

    with torch.no_grad():
        pred1 = model.forward(batch, pseudo_time=torch.tensor([1.0]))
    print("Prediction made")

    msl = pred1.surf_vars["msl"].cpu().numpy()

    batch_adapter = BatchAdapter(batch.metadata, batch.static_vars)

    _, field_history = batch_adapter.aurora_to_scisi(batch)
    batch1 = batch_adapter.scisi_to_aurora(field_history)

    for key in batch1.surf_vars.keys():
        print(
            key,
            (
                batch1.surf_vars[key].cpu().numpy() - batch.surf_vars[key].cpu().numpy()
            ).max(),
        )

    for key in batch1.atmos_vars.keys():
        print(
            key,
            (
                batch1.atmos_vars[key].cpu().numpy()
                - batch.atmos_vars[key].cpu().numpy()
            ).max(),
        )

    with torch.no_grad():
        pred2 = model.forward(batch1, pseudo_time=torch.tensor([1.0]))

    msl2 = pred2.surf_vars["msl"].cpu().numpy()

    print((msl - msl2).max())

    plt.figure()
    plt.imshow(msl[0, 0, :, :] - msl2[0, 0, :, :])
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
