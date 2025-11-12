import pdb
from datetime import datetime

import torch
from aurora import AuroraSmallPretrained, Batch, Metadata

model = AuroraSmallPretrained()
model.load_checkpoint()

batch = Batch(
    # surface variables (b, t, h, w)
    surf_vars={k: torch.randn(1, 2, 721, 1440) for k in ("2t", "10u", "10v", "msl")},
    # static variables (h, w)
    static_vars={k: torch.randn(721, 1440) for k in ("lsm", "z", "slt")},
    # atmospheric variables (b, t, c, h, w)
    atmos_vars={k: torch.randn(1, 2, 4, 721, 1440) for k in ("z", "u", "v", "t", "q")},
    # metadata
    metadata=Metadata(
        lat=torch.linspace(90, -90, 721),
        lon=torch.linspace(0, 360, 1440 + 1)[:-1],
        time=(datetime(2020, 6, 1, 12, 0),),  # time
        atmos_levels=(100, 250, 500, 850),  # atmospheric levels
    ),
)

prediction = model.forward(batch)

print(prediction.surf_vars["2t"].shape)
