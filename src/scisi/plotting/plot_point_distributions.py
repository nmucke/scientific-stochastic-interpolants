import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_point_distributions(
    true_field: torch.Tensor,
    predicted_fields: torch.Tensor,
    points: list[tuple[int, int]] = [(32, 32), (64, 64), (96, 96)],
    figure_path: str = "figures/point_distributions.png",
    show: bool = False,
) -> None:
    plt.figure(figsize=(5 * len(points), 5))
    plt.subplot(1, len(points) + 1, 1)
    plt.imshow(
        true_field[:, :, -1],
        vmin=np.min(true_field[:, :, -1].numpy()),
        vmax=np.max(true_field[:, :, -1].numpy()),
        origin="lower",
    )
    for point in points:
        plt.scatter(point[1], point[0], color="red", marker="x", s=100)
    for i, point in enumerate(points):
        flattened_true = true_field[point[0], point[1]].flatten()
        flattened_predicted = predicted_fields[:, point[0], point[1]].flatten()
        plt.subplot(1, len(points) + 1, i + 2)
        plt.hist(flattened_true, bins=50, label="True", density=True, alpha=0.5)
        plt.hist(
            flattened_predicted, bins=50, label="Ensemble", density=True, alpha=0.5
        )
        plt.title(f"Point ({point[0]}, {point[1]})")
        plt.legend()
    plt.savefig(f"{figure_path}")
    if show:
        plt.show()
    else:
        plt.close()