import os
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env_video_recorder import EnvVideoRecorder
from snake_env import ACTION_LEFT, ACTION_RIGHT, ACTION_STRAIGHT, SnakeEnv


@dataclass
class DQNConfig:
    max_memory: int = 100_000
    batch_size: int = 1000
    lr: float = 0.001
    gamma: float = 0.9
    hidden_size: int = 128
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995


class LinearQNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)


class QTrainer:
    def __init__(self, model: nn.Module, lr: float, gamma: float):
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone().detach()

        for idx in range(len(done)):
            q_new = reward[idx]
            if not done[idx]:
                q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][action[idx]] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()


class DQNAgent:
    def __init__(self, config: DQNConfig):
        self.n_games = 0
        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.gamma = config.gamma
        self.memory = deque(maxlen=config.max_memory)
        self.model = LinearQNet(11, config.hidden_size, 3)
        self.trainer = QTrainer(self.model, lr=config.lr, gamma=config.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self, batch_size: int):
        if len(self.memory) > batch_size:
            mini_sample = random.sample(self.memory, batch_size)
        else:
            mini_sample = list(self.memory)

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.choice([ACTION_STRAIGHT, ACTION_RIGHT, ACTION_LEFT])

        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        return int(torch.argmax(prediction).item())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def record_episode(model: LinearQNet, filename: str, *, max_steps: int = 1000):
    env = SnakeEnv(render=True, speed=20)
    recorder = EnvVideoRecorder(env)

    state = recorder.reset()
    for _ in range(max_steps):
        recorder.render()
        state0 = torch.tensor(state, dtype=torch.float)
        action = int(torch.argmax(model(state0)).item())
        state, reward, done, _, info = recorder.step(action)
        if done:
            break

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    recorder.save(filename)
    return info.get("score", 0)


def train_dqn(episodes: int = 50, record_every: int = 50, recordings_dir: str = "recordings"):
    config = DQNConfig()
    env = SnakeEnv(render=False, speed=0)
    agent = DQNAgent(config)

    best_score = 0
    scores_history = []


    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _, info = env.step(action)
            agent.train_short_memory(state, action, reward, next_state, done)
            agent.remember(state, action, reward, next_state, done)
            state = next_state

        agent.n_games += 1
        agent.train_long_memory(config.batch_size)
        agent.update_epsilon()

        score = info.get("score", 0)
        best_score = max(best_score, score)
        scores_history.append(score)

        print(f"Episode {episode} | Score {score} | Best {best_score} | Epsilon {agent.epsilon:.3f}")

        if record_every and episode % record_every == 0:
            filename = os.path.join(recordings_dir, f"dqn_episode_{episode}.mp4")
            record_score = record_episode(agent.model, filename)
            print(f"Recorded DQN episode {episode} (score {record_score}) -> {filename}")

    return scores_history



if __name__ == "__main__":
    train_dqn(episodes=50, record_every=0)