"""
rl_common.py
============

Shared utilities for the Reinforcement Learning demos in this folder.
Handles environment wrappers, discretizers, replay buffers, training evaluation,
GIF rendering, and plotting learning curves.
"""

from __future__ import annotations

import os
import random
import argparse
from collections import deque
import numpy as np
import torch
import gymnasium as gym
import warnings

# Suppress pygame setup tools deprecation warnings in Python 3.13
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

# Common directory
_FIGURE_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# State Discretization Helper for Tabular Methods
# --------------------------------------------------------------------------- #
class StateDiscretizer:
    """Discretizes continuous state observations into a fixed grid of buckets."""
    def __init__(self, low: list[float], high: list[float], num_buckets: tuple[int, ...]):
        self.low = np.array(low)
        self.high = np.array(high)
        self.num_buckets = num_buckets

    def discretize(self, state: np.ndarray) -> tuple[int, ...]:
        ratios = (state - self.low) / (self.high - self.low)
        ratios = np.clip(ratios, 0.0, 1.0)
        # Map ratio to bucket indices
        buckets = [int(np.floor(r * (b - 1))) for r, b in zip(ratios, self.num_buckets)]
        return tuple(buckets)


def get_cartpole_discretizer() -> StateDiscretizer:
    # State: [position, velocity, angle, angular_velocity]
    # Restrict bounds to active learning zones
    # Cart position and velocity are ignored (1 bucket) to focus learning purely on pole stability
    low = [-2.4, -3.0, -0.27, -3.0]
    high = [2.4, 3.0, 0.27, 3.0]
    buckets = (1, 1, 6, 12) # 72 total states instead of 10,368
    return StateDiscretizer(low, high, buckets)


def get_mountaincar_discretizer() -> StateDiscretizer:
    # State: [position, velocity]
    low = [-1.2, -0.07]
    high = [0.6, 0.07]
    buckets = (18, 14)
    return StateDiscretizer(low, high, buckets)


# --------------------------------------------------------------------------- #
# Experience Replay Buffer for DQN Models
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    """A circular buffer to store transition tuples for value-based Deep RL."""
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


# --------------------------------------------------------------------------- #
# Device, Argparser, and Evaluation Loops
# --------------------------------------------------------------------------- #
def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_argparser(description: str, episodes: int = 500, lr: float = 1e-3):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--env", type=str, default="CartPole-v1",
                   choices=["CartPole-v1", "MountainCar-v0", "Pendulum-v1"])
    p.add_argument("--episodes", type=int, default=episodes, help="number of training episodes")
    p.add_argument("--lr", type=float, default=lr, help="learning rate")
    p.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--no-figure", action="store_true", help="do not save learning curves and agent GIFs")
    p.add_argument("--variant", type=str, default=None)
    return p


def evaluate_agent(env_id: str, act_fn, num_episodes: int = 20) -> tuple[float, float]:
    """Runs agent policy function across multiple episodes. Returns mean and std reward."""
    test_env = gym.make(env_id)
    episode_rewards = []

    for _ in range(num_episodes):
        state, _ = test_env.reset()
        total_reward = 0.0
        while True:
            action = act_fn(state)
            next_state, reward, terminated, truncated, _ = test_env.step(action)
            total_reward += reward
            state = next_state
            if terminated or truncated:
                break
        episode_rewards.append(total_reward)

    test_env.close()
    return float(np.mean(episode_rewards)), float(np.std(episode_rewards))


# --------------------------------------------------------------------------- #
# Visualizations: Curves & GIFs
# --------------------------------------------------------------------------- #
def plot_learning_curve(rewards: list[float], save_path: str, window: int = 25):
    """Plot cumulative rewards over training episodes with rolling average smoothing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping learning curve plot: {e})")
        return

    episodes = np.arange(1, len(rewards) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Raw rewards
    ax.plot(episodes, rewards, alpha=0.3, color="blue", label="Raw Episode Reward")

    # Rolling average
    if len(rewards) >= window:
        rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
        rolling_idx = np.arange(window, len(rewards) + 1)
        ax.plot(rolling_idx, rolling_mean, color="darkblue", linewidth=2, label=f"Rolling Mean (W={window})")

    ax.set_title("Training Reward Curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved training reward curve -> {save_path}")


def save_agent_gif(env_id: str, act_fn, filename: str, max_steps: int = 500):
    """Render and save trained agent episode rollout to a clean animated GIF file."""
    try:
        from PIL import Image
    except Exception as e:
        print(f"(skipping GIF compilation: PIL missing: {e})")
        return

    # Create visual environment
    env = gym.make(env_id, render_mode="rgb_array")
    state, _ = env.reset(seed=42)
    frames = []

    for _ in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(Image.fromarray(frame))

        action = act_fn(state)
        next_state, _, terminated, truncated, _ = env.step(action)
        state = next_state
        if terminated or truncated:
            break

    env.close()

    if frames:
        frames[0].save(filename, save_all=True, append_images=frames[1:], duration=25, loop=0)
        print(f"Saved agent rendering simulation GIF -> {filename}")
    else:
        print("(Warning: no frames captured, skipping GIF save)")
