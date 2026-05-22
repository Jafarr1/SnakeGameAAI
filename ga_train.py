import os
import random
from dataclasses import dataclass

import numpy as np

from env_video_recorder import EnvVideoRecorder
from snake_env import ACTION_LEFT, ACTION_RIGHT, ACTION_STRAIGHT, SnakeEnv


@dataclass
class GAConfig:
    population_size: int = 50
    elite_size: int = 10
    mutation_rate: float = 0.1
    mutation_strength: float = 0.2
    generations: int = 50
    max_steps: int = 1000


class Genome:
    def __init__(self, input_size: int = 11, hidden_size: int = 16, output_size: int = 3):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros(output_size)

    def forward(self, state: np.ndarray) -> np.ndarray:
        x = np.maximum(0, np.dot(state, self.w1) + self.b1)
        return np.dot(x, self.w2) + self.b2

    def act(self, state: np.ndarray) -> int:
        logits = self.forward(state)
        return int(np.argmax(logits))

    def copy(self):
        clone = Genome()
        clone.w1 = np.copy(self.w1)
        clone.b1 = np.copy(self.b1)
        clone.w2 = np.copy(self.w2)
        clone.b2 = np.copy(self.b2)
        return clone


def evaluate(genome: Genome, env: SnakeEnv, max_steps: int) -> tuple[float, int, float]:
    state = env.reset()
    score = 0
    steps = 0
    total_reward = 0.0

    for _ in range(max_steps):
        action = genome.act(state)
        state, reward, done, _, info = env.step(action)
        score = info.get("score", 0)
        total_reward += reward
        steps += 1
        if done:
            break

    fitness = total_reward
    return fitness, score, total_reward


def crossover(parent_a: Genome, parent_b: Genome) -> Genome:
    child = parent_a.copy()

    mask_w1 = np.random.rand(*child.w1.shape) < 0.5
    mask_b1 = np.random.rand(*child.b1.shape) < 0.5
    mask_w2 = np.random.rand(*child.w2.shape) < 0.5
    mask_b2 = np.random.rand(*child.b2.shape) < 0.5

    child.w1 = np.where(mask_w1, parent_a.w1, parent_b.w1)
    child.b1 = np.where(mask_b1, parent_a.b1, parent_b.b1)
    child.w2 = np.where(mask_w2, parent_a.w2, parent_b.w2)
    child.b2 = np.where(mask_b2, parent_a.b2, parent_b.b2)

    return child


def mutate(genome: Genome, rate: float, strength: float):
    for param in [genome.w1, genome.b1, genome.w2, genome.b2]:
        mask = np.random.rand(*param.shape) < rate
        param += mask * np.random.randn(*param.shape) * strength


def record_episode(genome: Genome, filename: str, *, max_steps: int = 1000):
    env = SnakeEnv(render=True, speed=20)
    recorder = EnvVideoRecorder(env)

    state = recorder.reset()
    for _ in range(max_steps):
        recorder.render()
        action = genome.act(state)
        state, reward, done, _, info = recorder.step(action)
        if done:
            break

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    recorder.save(filename)
    return info.get("score", 0)


def train_ga(generations: int = 50, record_every: int = 10, recordings_dir: str = "recordings"):
    config = GAConfig(generations=generations)
    env = SnakeEnv(render=False, speed=0)

    population = [Genome() for _ in range(config.population_size)]

    for generation in range(1, config.generations + 1):
        scored = []
        for genome in population:
            fitness, score, total_reward = evaluate(genome, env, config.max_steps)
            scored.append((fitness, score, total_reward, genome))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_fitness, best_score, best_reward, best_genome = scored[0]

        print(
            f"Generation {generation} | Best score {best_score} | "
            f"Reward {best_reward:.2f} | Fitness {best_fitness:.2f}"
        )

        if record_every and generation % record_every == 0:
            filename = os.path.join(recordings_dir, f"ga_generation_{generation}.mp4")
            record_score = record_episode(best_genome, filename)
            print(f"Recorded GA generation {generation} (score {record_score}) -> {filename}")

        elites = [genome.copy() for _, _, _, genome in scored[: config.elite_size]]
        next_population = elites[:]

        while len(next_population) < config.population_size:
            parent_a, parent_b = random.sample(elites, 2)
            child = crossover(parent_a, parent_b)
            mutate(child, config.mutation_rate, config.mutation_strength)
            next_population.append(child)

        population = next_population


if __name__ == "__main__":
    train_ga()
