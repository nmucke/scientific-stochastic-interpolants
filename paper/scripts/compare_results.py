import pdb

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def main() -> None:
    """Main function."""

    # Prepare figure for animation
    fig, ax = plt.subplots(figsize=(10, 6))

    true_trajectory = np.load("trajectory.npz")["trajectory"]
    true_trajectory = true_trajectory[:, ::2, ::2]
    true_trajectory = true_trajectory[100:]
    
        
    # Animation update function
    # def animate(frame):
    #     ax.clear()
        
    #     i = frame + 5  # frames start from 0, so add 5 to get the time index
    #     # data = np.load("paper_results/enkf_ns/state_grid_8_time_2.npz")
    #     data = np.load(f"paper_results/enkf_ns/state_grid_skip_4_time_{i}.npz")

    #     data = data["arr_0"]
    #     data = data[:, ::2, ::2]

    #     middle_point = data[:, 64, 64]
    #     true_point = true_trajectory[i, 64, 64]
    #     ax.hist(middle_point, bins=100, label="Prior", density=True, alpha=0.5)
    #     ax.axvline(true_point, color="red", label="True", linewidth=4)

    #     xx = np.linspace(np.min(middle_point), np.max(middle_point), 100)
    #     kde = stats.gaussian_kde(middle_point)
    #     ax.plot(xx, kde(xx), label="KDE", linewidth=4)
    #     ax.legend()
    #     ax.set_title(f"Point distribution at time step {i}")
    #     ax.set_xlabel("Vorticity")
    #     ax.set_ylabel("Frequency")
        
    #     return ax
    
    # # Create animation
    # anim = animation.FuncAnimation(
    #     fig, animate, frames=range(90 - 5), interval=10, repeat=True
    # )
    
    # Save animation
    # anim.save("paper_results/enkf_ns/point_distribution_animation.mp4", writer="ffmpeg", fps=10)
    # print("Animation saved to paper_results/enkf_ns/point_distribution_animation.mp4")
    
    # Optionally display the animation
    # plt.show()


    plt.figure()

    # Load data for multi-point visualization (using last time step)
    data = np.load("paper_results/enkf_ns/state_grid_skip_4_time_89.npz")
    data = data["arr_0"]
    data = data[:, ::2, ::2]

    for i, (x, y) in enumerate[tuple[int, int]]([(32, 32), (64, 64), (96, 96)]):
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
    plt.savefig("figures/point_distribution_multi_point.png")
    # plt.show()
    plt.close()


if __name__ == "__main__":
    main()
