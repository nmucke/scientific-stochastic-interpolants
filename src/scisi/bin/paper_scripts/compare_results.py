import pdb

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def main() -> None:
    """Main function."""

    # data = np.load("paper_results/enkf_ns/state_grid_8_time_2.npz")
    data = np.load("paper_results/enkf_ns/state_grid_skip_4_time_46.npz")

    data = data["arr_0"]
    data = data[:, ::2, ::2]

    plt.figure()

    for i, (x, y) in enumerate([(32, 32), (64, 64), (96, 96)]):
        plt.subplot(1, 3, i + 1)
        middle_point = data[:, x, y]
        plt.hist(middle_point, bins=100, label="Prior", density=True, alpha=0.5)

        xx = np.linspace(np.min(middle_point), np.max(middle_point), 100)
        kde = stats.gaussian_kde(middle_point)
        plt.plot(xx, kde(xx), label="KDE", linewidth=4)
        plt.legend()
        plt.title("Point distribution")
        plt.xlabel("Vorticity")
        plt.ylabel("Frequency")
    plt.show()


if __name__ == "__main__":
    main()
