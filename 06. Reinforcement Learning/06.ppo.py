"""
06. Proximal Policy Optimization (PPO)
======================================

Proximal Policy Optimization (PPO) using a clipped surrogate objective to enforce stable policy updates (Schulman et al., 2017).

Loss Function (Surrogate Objective):
    L = -min(r_t * A_t, clip(r_t, 1 - eps, 1 + eps) * A_t)
    where r_t = pi(a_t | s_t) / pi_old(a_t | s_t) and A_t is the advantage.

Key insights / educational takeaways:
    * Clipping prevents a policy update from changing the action probabilities too drastically.
    * Multiple epochs of mini-batch gradient descent can be safely performed on the same trajectory data.

Run:
    python "06.ppo.py" --env CartPole-v1 --episodes 400
    python "06.ppo.py" --env MountainCar-v0 --episodes 800
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
import rl_common as mc


class PPOActorCritic(nn.Module):
    """Separate policy (actor) and value (critic) networks for PPO stability."""
    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Outputs logits (probs computed inside training loop) and state value
        logits = self.actor(x)
        val = self.critic(x)
        return logits, val


def main():
    p = mc.build_argparser("Proximal Policy Optimization (PPO)")
    args = p.parse_args()

    env_id = args.env
    env = gym.make(env_id)

    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    device = mc.get_device(args.device)

    # Initialize model
    model = PPOActorCritic(state_dim, num_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion_c = nn.MSELoss()

    # Hyperparameters
    gamma = args.gamma
    eps_clip = 0.2
    K_epochs = 4
    batch_size = 64

    # Log metrics
    episode_rewards = []

    print(f"Training PPO on {env_id} for {args.episodes} episodes...")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    # Store steps globally across episodes for on-policy batch updates
    states, actions, log_probs, rewards, dones = [], [], [], [], []

    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()
        ep_reward = 0.0

        while True:
            # Action selection
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(state_t)
            probs = torch.softmax(logits, dim=-1)
            m = Categorical(probs)
            action = m.sample()

            action_val = action.item()
            states.append(state)
            actions.append(action_val)
            log_probs.append(m.log_prob(action).item())

            next_state, reward, terminated, truncated, _ = env.step(action_val)
            done = terminated or truncated

            rewards.append(reward)
            dones.append(done)

            state = next_state
            ep_reward += reward

            if done:
                break

        episode_rewards.append(ep_reward)

        # Periodically perform updates (e.g. update policy every 5 episodes)
        if ep % 5 == 0:
            # Convert stored trajectories to PyTorch tensors
            states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
            actions_t = torch.tensor(np.array(actions), dtype=torch.long, device=device)
            old_log_probs_t = torch.tensor(np.array(log_probs), dtype=torch.float32, device=device)

            # Compute discounted returns
            returns = []
            discounted_sum = 0.0
            for r, d in zip(reversed(rewards), reversed(dones)):
                if d:
                    discounted_sum = 0.0
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)

            returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

            # Perform K epochs of policy updates over the collected trajectory
            for _ in range(K_epochs):
                # Forward pass: get current logits, action probs, and values
                logits, state_values = model(states_t)
                state_values = state_values.squeeze(-1)
                probs = torch.softmax(logits, dim=-1)
                m = Categorical(probs)

                # Compute current log_probs and entropy regularizer
                new_log_probs = m.log_prob(actions_t)
                entropy = m.entropy()

                # Advantages = G_t - V(s_t)
                advantages = returns_t - state_values.detach()
                # Normalize advantages
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Probability ratios: r_t = pi(a_t) / pi_old(a_t)
                ratios = torch.exp(new_log_probs - old_log_probs_t)

                # Clipped surrogate losses
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - eps_clip, 1.0 + eps_clip) * advantages

                # Total objective
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * criterion_c(state_values, returns_t)

                total_loss = actor_loss + critic_loss - 0.01 * entropy.mean()

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            # Flush memory buffers
            states, actions, log_probs, rewards, dones = [], [], [], [], []

        if ep % max(1, args.episodes // 10) == 0:
            avg_rew = np.mean(episode_rewards[-30:])
            print(f"Episode {ep:4d}/{args.episodes} | last-30 avg reward: {avg_rew:.2f}")

    print("-" * 64)
    env.close()

    # Define evaluation action selection
    def eval_act_fn(s):
        s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(s_t)
        probs = torch.softmax(logits, dim=-1)
        return torch.argmax(probs, dim=1).item()

    # Evaluate
    print("Evaluating trained PPO...")
    mean_reward, std_reward = mc.evaluate_agent(env_id, eval_act_fn, num_episodes=20)
    print(f"Evaluation over 20 runs: mean_reward {mean_reward:.2f} | std_reward {std_reward:.2f}")

    # Plot metrics and render agent GIF
    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        curve_path = os.path.join(save_dir, f"ppo_{env_id.lower()}_curve.png")
        mc.plot_learning_curve(episode_rewards, curve_path)

        gif_path = os.path.join(save_dir, f"ppo_{env_id.lower()}_agent.gif")
        mc.save_agent_gif(env_id, eval_act_fn, gif_path)


if __name__ == "__main__":
    main()
