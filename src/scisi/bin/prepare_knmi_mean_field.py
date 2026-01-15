import os

import numpy as np


def main() -> None:
    """Prepare the pixel-wise mean field of the KNMI data."""

    mean_fields = []
    for file in os.listdir("data/knmi/mean_fields"):
        data = np.load(f"data/knmi/mean_fields/{file}")["tasdoy"]

        mean_fields.append(data)

    mean_field = np.array(mean_fields)

    np.savez(
        "data/knmi/mean_field.npz",
        mean=mean_field.mean(axis=(0, 1)),
        std=mean_field.std(axis=(0, 1)),
    )


if __name__ == "__main__":
    main()
