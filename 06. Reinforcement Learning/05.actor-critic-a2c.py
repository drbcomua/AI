"""
05. Advantage Actor-Critic (A2C)
================================

Advantage Actor-Critic (A2C) utilizing shared representation layers with separate policy and value heads.

Architecture Diagram / Layout:
    Shared Stem: Input Continuous State [Batch, State_Dim] -> Linear [128] -> ReLU
        -> Actor Head: Linear [128, Action_Dim] -> Softmax (Action Probabilities pi)
        -> Critic Head: Linear [128, 1] (State Value V(s))

Key insights / educational takeaways:
    * The Critic reduces REINFORCE policy gradient variance by mapping baseline value estimations V(s).
    * The Advantage metric, A(s, a) = G_t - V(s), measures how much better an action was compared to average expectations.

Run:
    python "05.actor-critic-a2c.py" --env CartPole-v1 --episodes 800
    python "05.actor-critic-a2c.py" --env MountainCar-v0 --episodes 1000
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
import rl_common as mc


class ActorCritic(nn.Module):
    """Actor-Critic network sharing a common representation block."""
    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, num_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        h = self.stem(x)
        probs = torch.softmax(self.actor(h), dim=-1)
        val = self.critic(h)
        return probs, val


def main():
    p = mc.build_argparser("Advantage Actor-Critic (A2C)")
    args = p.parse_args()

    env_id = args.env
    env = gym.make(env_id)

    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    device = mc.get_device(args.device)

    model = ActorCritic(state_dim, num_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion_c = nn.MSELoss()

    gamma = args.gamma
    episode_rewards = []

    print(f"Training A2C on {env_id} for {args.episodes} episodes...")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()

        saved_log_probs = []
        saved_values = []
        rewards = []

        while True:
            # Forward pass: get action probs and state value estimation
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            probs, val = model(state_t)

            m = Categorical(probs)
            action = m.sample()

            saved_log_probs.append(m.log_prob(action))
            saved_values.append(val.squeeze(-1))

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            rewards.append(reward)
            state = next_state

            if done:
                break

        episode_rewards.append(sum(rewards))

        # Calculate discounted returns G_t
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        saved_values = torch.cat(saved_values)
        saved_log_probs = torch.cat(saved_log_probs)

        # Advantage = G_t - V(s_t)
        advantages = returns - saved_values.detach()

        # Actor loss: policy gradient scaled by advantage
        actor_loss = -(saved_log_probs * advantages).sum()

        # Critic loss: MSE between estimated state value and actual returns
        critic_loss = criterion_c(saved_values, returns)

        # Total loss combines policy loss and value network regression
        total_loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        total_loss.backward()
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
            probs, _ = model(s_t)
        return torch.argmax(probs, dim=1).item()

    # Evaluate
    print("Evaluating trained A2C...")
    mean_reward, std_reward = mc.evaluate_agent(env_id, eval_act_fn, num_episodes=20)
    print(f"Evaluation over 20 runs: mean_reward {mean_reward:.2f} | std_reward {std_reward:.2f}")

    # Plot metrics and render agent GIF
    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        curve_path = os.path.join(save_dir, f"a2c_{env_id.lower()}_curve.png")
        mc.plot_learning_curve(episode_rewards, curve_path)

        gif_path = os.path.join(save_dir, f"a2c_{env_id.lower()}_agent.gif")
        mc.save_agent_gif(env_id, eval_act_fn, gif_path)


if __name__ == "__main__":
    main()
