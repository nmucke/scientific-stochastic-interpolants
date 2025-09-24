from typing import Optional

import matplotlib.pyplot as plt
import torch


def plot_fields(
    fields: list[list[torch.Tensor]],
    titles: list[list[str]],
    vmin: Optional[float],
    vmax: Optional[float],
    figsize: Optional[tuple],
    figure_path: str,
    imshow_kwargs: dict = {},
) -> None:

    num_rows = len(fields)
    num_cols = len(fields[0])

    plt.figure(figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            plt.subplot(num_rows, num_cols, i * num_cols + j + 1)
            plt.imshow(fields[i][j], vmin=vmin, vmax=vmax, **imshow_kwargs)
            plt.title(titles[i][j])
    plt.savefig(f"{figure_path}")
    plt.show()
