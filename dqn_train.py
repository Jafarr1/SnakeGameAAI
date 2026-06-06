import os
import random
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env_video_recorder import EnvVideoRecorder
from snake_env import ACTION_LEFT, ACTION_RIGHT, ACTION_STRAIGHT, SnakeEnv


CHECKPOINT_SECONDS = [300, 600, 1200, 3600]  # 5, 10, 20, 60 min


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
    target_update_freq: int = 100


class LinearQNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)


class QTrainer:
    def __init__(self, model: nn.Module, target_model: nn.Module, lr: float, gamma: float):
        self.model = model
        self.target_model = target_model
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
                q_new = reward[idx] + self.gamma * torch.max(self.target_model(next_state[idx]))
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
        self.target_model = LinearQNet(11, config.hidden_size, 3)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.trainer = QTrainer(self.model, self.target_model, lr=config.lr, gamma=config.gamma)

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

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


def record_episode(model: LinearQNet, filename: str, *, max_steps: int = 10000, n_attempts: int = 5):
    import imageio
    best_score = -1
    best_frames = []

    for _ in range(n_attempts):
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

        score = info.get("score", 0)
        if score > best_score:
            best_score = score
            best_frames = list(recorder._frame_buffer)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with imageio.get_writer(filename, fps=30) as writer:
        for frame in best_frames:
            writer.append_data(frame)

    return best_score


def train_dqn(record_every: int = 50, recordings_dir: str = "recordings"):

    config = DQNConfig()
    env = SnakeEnv(render=False, speed=0)
    agent = DQNAgent(config)

    best_score = 0
    scores_history = []

    max_training_time = 3600  # 60 minutter
    start_time = time.time()

    episode = 0
    checkpoints_hit = set()

    while time.time() - start_time < max_training_time:

        episode += 1

        state = env.reset()
        done = False

        while not done:

            action = agent.get_action(state)

            next_state, reward, done, _, info = env.step(action)

            agent.train_short_memory(
                state,
                action,
                reward,
                next_state,
                done
            )

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done
            )

            state = next_state

        agent.n_games += 1

        agent.train_long_memory(config.batch_size)

        agent.update_epsilon()

        if agent.n_games % config.target_update_freq == 0:
            agent.update_target_network()

        score = info.get("score", 0)

        best_score = max(best_score, score)

        elapsed_time = time.time() - start_time

        scores_history.append((elapsed_time, score))

        for checkpoint in CHECKPOINT_SECONDS:
            if elapsed_time >= checkpoint and checkpoint not in checkpoints_hit:
                checkpoints_hit.add(checkpoint)
                minutes = checkpoint // 60
                filename = os.path.join(recordings_dir, f"dqn_{minutes}min.mp4")
                model_path = os.path.join(recordings_dir, f"dqn_{minutes}min.pth")
                torch.save(agent.model.state_dict(), model_path)
                rec_score = record_episode(agent.model, filename)
                print(f"[DQN] Checkpoint {minutes} min — model gemt: {model_path} | video score: {rec_score}")

        if episode % 10 == 0:

            print(
                f"DQN Episode {episode} | "
                f"Score: {score} | "
                f"Best: {best_score} | "
                f"Time: {elapsed_time:.1f}s | "
                f"Epsilon: {agent.epsilon:.2f}"
            )

    return scores_history