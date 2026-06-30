"""
04. Policy Gradient (REINFORCE)
===============================

Monte Carlo Policy Gradient (REINFORCE) algorithm optimizing policy probabilities directly (Williams, 1992).

Loss Function (surrogate objective):
    Loss = -sum_t log(pi(a_t | s_t)) * G_t
    where G_t is the discounted cumulative return from timestep t to the end of the episode.

Key insights / educational takeaways:
    * Policy gradients optimize strategies directly, avoiding intermediate value estimation.
    * Gradients of chosen actions are scaled up or down by the cumulative rewards achieved, reinforcing good strategies.

Run:
    python "04.reinforce.py" --env CartPole-v1 --episodes 600
    python "04.reinforce.py" --env MountainCar-v0 --episodes 1000
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
import rl_common as mc


class PolicyNetwork(nn.Module):
    """Policy Network mapping continuous states to action probability distributions."""
    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


def main():
    p = mc.build_argparser("Policy Gradient (REINFORCE)")
    args = p.parse_args()

    env_id = args.env
    env = gym.make(env_id)

    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    device = mc.get_device(args.device)

    # Initialize Policy network
    policy = PolicyNetwork(state_dim, num_actions).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    gamma = args.gamma
    episode_rewards = []

    print(f"Training REINFORCE on {env_id} for {args.episodes} episodes...")
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()

        saved_log_probs = []
        rewards = []

        while True:
            # Action selection: sample from categorical distribution
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            probs = policy(state_t)
            m = Categorical(probs)
            action = m.sample()

            saved_log_probs.append(m.log_prob(action))

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            rewards.append(reward)
            state = next_state

            if done:
                break

        episode_rewards.append(sum(rewards))

        # Calculate discounted returns G_t backwards
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        # Normalize returns for stable gradients
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Calculate policy loss: -sum_t log(pi) * G_t
        policy_loss = []
        for log_prob, G_t in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * G_t)

        loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % max(1, args.episodes // 10) == 0:
            avg_rew = np.mean(episode_rewards[-30:])
            print(f"Episode {ep:4d}/{args.episodes} | last-30 avg reward: {avg_rew:.2f}")

    print("-" * 64)
    env.close()

    # Define evaluation action selection
    def eval_act_fn(s):
        s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            probs = policy(s_t)
        return torch.argmax(probs, dim=1).item()

    # Evaluate
    print("Evaluating trained REINFORCE...")
    mean_reward, std_reward = mc.evaluate_agent(env_id, eval_act_fn, num_episodes=20)
    print(f"Evaluation over 20 runs: mean_reward {mean_reward:.2f} | std_reward {std_reward:.2f}")

    # Plot metrics and render agent GIF
    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        curve_path = os.path.join(save_dir, f"reinforce_{env_id.lower()}_curve.png")
        mc.plot_learning_curve(episode_rewards, curve_path)

        gif_path = os.path.join(save_dir, f"reinforce_{env_id.lower()}_agent.gif")
        mc.save_agent_gif(env_id, eval_act_fn, gif_path)


if __name__ == "__main__":
    main()
