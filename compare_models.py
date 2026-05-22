import matplotlib.pyplot as plt
from dqn_train import train_dqn
from ga_train import train_ga

def run_comparison():
    # Sæt iterationer. Af hensyn til hastighed, lad os teste 200.
    iterations = 500

    print("--- Træner Deep Q-Learning (DQN) ---")
    dqn_scores = train_dqn(episodes=iterations, record_every=0)

    print("\n--- Træner Genetic Algorithm (GA) ---")
    ga_scores = train_ga(generations=iterations, record_every=0)

    # Tegn grafen
    plt.figure(figsize=(10, 6))
    
    # Plottes som linjer med små markører
    plt.plot(dqn_scores, label="DQN (Score pr. Episode)", alpha=0.7)
    plt.plot(ga_scores, label="GA (Bedste Score pr. Generation)", alpha=0.7)
    
    plt.title("Sammenligning: DQN vs. Genetic Algorithm")
    plt.xlabel("Iterationer (Episoder for DQN / Generationer for GA)")
    plt.ylabel("Score (Antal mad spist)")
    plt.legend()
    plt.grid(True)
    
    # Gem graf til rapporten og vis den på skærmen
    plt.savefig("comparison_plot.png")
    print("\nGraf gemt som 'comparison_plot.png'")
    plt.show()

if __name__ == "__main__":
    run_comparison()