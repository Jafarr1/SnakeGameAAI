import matplotlib.pyplot as plt
import numpy as np

from dqn_train import train_dqn
from ga_train import train_ga


def moving_average(data, window=50):

    if len(data) < window:
        return data

    return np.convolve(
        data,
        np.ones(window) / window,
        mode='valid'
    )


def run_comparison():

    print("--- Træner Deep Q-Learning (DQN) ---")

    dqn_data = train_dqn(
        record_every=0
    )

    print("\n--- Træner Genetic Algorithm (GA) ---")

    ga_data = train_ga(
        record_every=0
    )

    # -------------------------
    # Udpak DQN data
    # -------------------------

    dqn_times = [d[0] for d in dqn_data]
    dqn_scores = [d[1] for d in dqn_data]

    # -------------------------
    # Udpak GA data
    # -------------------------

    ga_times = [g[0] for g in ga_data]
    ga_scores = [g[1] for g in ga_data]

    # -------------------------
    # Moving average
    # -------------------------

    dqn_window = 50
    ga_window = 5

    dqn_avg_scores = moving_average(
        dqn_scores,
        dqn_window
    )

    ga_avg_scores = moving_average(
        ga_scores,
        ga_window
    )

    dqn_times_smoothed = dqn_times[dqn_window - 1:]
    ga_times_smoothed = ga_times[ga_window - 1:]

    # -------------------------
    # Plot
    # -------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        dqn_times_smoothed,
        dqn_avg_scores,
        label="DQN",
        linewidth=2
    )

    plt.plot(
        ga_times_smoothed,
        ga_avg_scores,
        label="GA",
        linewidth=2
    )

    plt.title(
        "Learning Performance Over Time"
    )

    plt.xlabel(
        "Training Time (Seconds)"
    )

    plt.ylabel(
        "Average Score"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    # -------------------------
    # Gem graf
    # -------------------------

    plt.savefig(
        "comparison_plot.png",
        dpi=300
    )

    print(
        "\nGraf gemt som "
        "'comparison_plot.png'"
    )

    plt.show()


if __name__ == "__main__":

    run_comparison()