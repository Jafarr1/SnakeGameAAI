import matplotlib.pyplot as plt
import numpy as np
from dqn_train import train_dqn
from ga_train import train_ga

def run_comparison():
    # Kør begge modeller
    dqn_episodes = 2000
    ga_generations = 100
    ga_pop_size = 50  # Din population_size i GAConfig er 50

    print("--- Træner Deep Q-Learning (DQN) ---")
    dqn_data = train_dqn(episodes=dqn_episodes, record_every=0)
    
    print("\n--- Træner Genetic Algorithm (GA) ---")
    ga_data = train_ga(generations=ga_generations, record_every=0)

    # --- Udpak data ---
    dqn_times = [d[0] for d in dqn_data]
    dqn_scores = [d[1] for d in dqn_data]
    dqn_games = list(range(1, dqn_episodes + 1))
    
    ga_times = [g[0] for g in ga_data]
    ga_scores = [g[1] for g in ga_data]
    ga_games = [i * ga_pop_size for i in range(1, ga_generations + 1)] # 50, 100, 150...

    # Lav et glidende gennemsnit (moving average) for DQN
    window = 50
    dqn_avg_scores = np.convolve(dqn_scores, np.ones(window)/window, mode='valid')
    dqn_times_smoothed = dqn_times[window-1:]
    dqn_games_smoothed = dqn_games[window-1:]

    # --- Opsæt graf med 2 undergrafer (Subplots) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Graf: Score baseret på TID (Jeres RQ)
    ax1.plot(dqn_times_smoothed, dqn_avg_scores, label="DQN (Glidende Gns.)", color="blue")
    ax1.plot(ga_times, ga_scores, label="GA (Bedste i generationen)", color="orange", marker="o", markersize=4)
    ax1.set_title("Læring over Tid (Sekunder) - RQ Svar")
    ax1.set_xlabel("Træningstid (Sekunder)")
    ax1.set_ylabel("Point (Antal mad spist)")
    ax1.legend()
    ax1.grid(True)

    # 2. Graf: Score baseret på ANTAL SPIL (Erfaring) - Giver en pænere kurve for GA
    ax2.plot(dqn_games_smoothed, dqn_avg_scores, label="DQN (Glidende Gns.)", color="blue")
    ax2.plot(ga_games, ga_scores, label="GA (Bedste i generationen)", color="orange", marker="o", markersize=4)
    ax2.set_title("Læring pr. Spillet Spil (Erfaring)")
    ax2.set_xlabel("Antal Spil")
    ax2.set_ylabel("Point (Antal mad spist)")
    ax2.legend()
    ax2.grid(True)

    # Afslut og gem
    plt.tight_layout()
    plt.savefig("dual_comparison_plot.png", dpi=300)
    print("\nGraf gemt som 'dual_comparison_plot.png' - Dobbelt graf klar til rapporten!")
    plt.show()

if __name__ == "__main__":
    run_comparison()