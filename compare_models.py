import matplotlib.pyplot as plt
import numpy as np
from dqn_train import train_dqn
from ga_train import train_ga


def smooth(scores, window=50):
    if len(scores) < window:
        return scores, list(range(len(scores)))
    avg = np.convolve(scores, np.ones(window) / window, mode='valid')
    return avg, list(range(len(avg)))


def run_comparison(max_minutes: int = 60):
    print("--- Træner Deep Q-Learning (DQN) ---")
    dqn_data = train_dqn(max_minutes=max_minutes, record_every=0)

    print("\n--- Træner Genetic Algorithm (GA) ---")
    ga_data = train_ga(max_minutes=max_minutes)

    # --- Udpak data ---
    dqn_times  = [d[0] for d in dqn_data]
    dqn_scores = [d[1] for d in dqn_data]

    ga_times  = [g[0] for g in ga_data]
    ga_scores = [g[1] for g in ga_data]

    # Glidende gennemsnit for DQN
    dqn_avg, dqn_idx = smooth(dqn_scores, window=50)
    dqn_times_smoothed = [dqn_times[i] for i in dqn_idx]

    # --- Opsæt graf ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Graf: Score over TID
    ax1.plot(dqn_times_smoothed, dqn_avg, label="DQN (Glidende Gns.)", color="blue")
    ax1.plot(ga_times, ga_scores, label="GA (Bedste i generationen)", color="orange", marker="o", markersize=4)
    ax1.set_title("Læring over Tid (Sekunder)")
    ax1.set_xlabel("Træningstid (Sekunder)")
    ax1.set_ylabel("Point (Antal mad spist)")
    ax1.legend()
    ax1.grid(True)

    plt.tight_layout()
    plt.savefig("dual_comparison_plot.png", dpi=300)
    print("\nGraf gemt som 'dual_comparison_plot.png'")
    plt.show()


if __name__ == "__main__":
    run_comparison(max_minutes=60)
