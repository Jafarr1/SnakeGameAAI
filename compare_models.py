import matplotlib.pyplot as plt
import numpy as np

from dqn_train import train_dqn
from ga_train import train_ga


CHECKPOINTS_MIN = [5, 10, 20, 60]
CHECKPOINTS_SEC = [t * 60 for t in CHECKPOINTS_MIN]


def moving_average(data, window):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def score_at_checkpoint(times, scores, target_sec):
    for i in range(len(times) - 1, -1, -1):
        if times[i] <= target_sec:
            return scores[i]
    return scores[0]


def run_comparison():

    print("--- Træner Deep Q-Learning (DQN) ---")
    dqn_data = train_dqn(record_every=0)

    print("\n--- Træner Genetic Algorithm (GA) ---")
    ga_data = train_ga(record_every=0)

    dqn_times = [d[0] / 60 for d in dqn_data]   # sekunder → minutter
    dqn_scores = [d[1] for d in dqn_data]

    ga_times = [g[0] / 60 for g in ga_data]
    ga_scores = [g[1] for g in ga_data]

    # DQN: window proportinonalt med datamængden (~1% af episoder)
    dqn_window = max(10, len(dqn_scores) // 100)
    ga_window = max(3, len(ga_scores) // 20)

    dqn_avg = moving_average(dqn_scores, dqn_window)
    ga_avg = moving_average(ga_scores, ga_window)

    dqn_times_sm = dqn_times[dqn_window - 1:]
    ga_times_sm = ga_times[ga_window - 1:]

    # -------------------------
    # Checkpoint-scores til tabel
    # -------------------------

    print("\n--- Gennemsnitsscore ved checkpoints ---")
    print(f"{'Tid':>8} | {'DQN avg':>10} | {'GA avg':>10}")
    print("-" * 34)

    dqn_cp = []
    ga_cp = []
    for cp_sec, cp_min in zip(CHECKPOINTS_SEC, CHECKPOINTS_MIN):
        cp_min_f = cp_sec / 60
        d = score_at_checkpoint(dqn_times_sm, dqn_avg, cp_min_f)
        g = score_at_checkpoint(ga_times_sm, ga_avg, cp_min_f)
        dqn_cp.append(d)
        ga_cp.append(g)
        print(f"{cp_min:>5} min | {d:>10.1f} | {g:>10.1f}")

    # -------------------------
    # Plot
    # -------------------------

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(dqn_times_sm, dqn_avg, label="DQN (Deep Q-Learning)", linewidth=2, color="steelblue")
    ax.plot(ga_times_sm, ga_avg, label="GA (Genetic Algorithm)", linewidth=2, color="darkorange")

    # Checkpoint-linjer og annotations
    for cp_min, d, g in zip(CHECKPOINTS_MIN, dqn_cp, ga_cp):
        ax.axvline(x=cp_min, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.annotate(f"{d:.0f}", xy=(cp_min, d), xytext=(cp_min + 0.3, d + 1),
                    color="steelblue", fontsize=8, fontweight="bold")
        ax.annotate(f"{g:.0f}", xy=(cp_min, g), xytext=(cp_min + 0.3, g - 2.5),
                    color="darkorange", fontsize=8, fontweight="bold")

    ax.set_title("Learning Performance Over Time\n(DQN vs Genetic Algorithm — Snake Game)", fontsize=13)
    ax.set_xlabel("Training Time (Minutes)")
    ax.set_ylabel("Average Score")
    ax.set_xticks(CHECKPOINTS_MIN)
    ax.set_xticklabels([f"{m} min" for m in CHECKPOINTS_MIN])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()

    fig.savefig("comparison_plot.png", dpi=300)
    print("\nGraf gemt som 'comparison_plot.png'")
    plt.show()


if __name__ == "__main__":
    run_comparison()
