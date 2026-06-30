"""
03. Deep Q-Network (DQN)
========================

Function approximation value-based deep reinforcement learning with replay buffers and target networks (Mnih et al., 2013/2015).

Architecture:
    Input Continuous State [Batch, State_Dim]
        -> Linear [State_Dim, 64] -> ReLU
        -> Linear [64, 64] -> ReLU
        -> Linear [64, Action_Dim] (Q-values outputs)

Key insights / educational takeaways:
    * The Experience Replay Buffer breaks the high correlation between sequential state transitions.
    * The Target Network provides a fixed target for learning updates, preventing target values from shifting constantly.

Run:
    python "03.dqn.py" --env CartPole-v1 --episodes 300
    python "03.dqn.py" --env MountainCar-v0 --episodes 500
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import rl_common as mc


class QNetwork(nn.Module):
    """Deep Q-Network function approximator."""
    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return self.net(x)


def main():
    p = mc.build_argparser("Deep Q-Network (DQN)")
    args = p.parse_args()

    env_id = args.env
    env = gym.make(env_id)

    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    device = mc.get_device(args.device)

    # Initialize Main and Target Q-networks
    model = QNetwork(state_dim, num_actions).to(device)
    target_model = QNetwork(state_dim, num_actions).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    replay_buffer = mc.ReplayBuffer(capacity=10000)

    # Hyperparameters
    gamma = args.gamma
    epsilon = 1.0
    min_epsilon = 0.02
    decay_rate = 0.99
    batch_size = 64
    target_update_freq = 10 # episodes

    episode_rewards = []

    print(f"Training DQN on {env_id} for {args.episodes} episodes...")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0

        while True:
            # Action selection: epsilon-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    q_vals = model(state_t)
                action = q_vals.argmax(dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition in Experience Replay
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Update weights if we have enough transitions
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                # Convert to Tensors
                states_t = torch.tensor(states, dtype=torch.float32, device=device)
                actions_t = torch.tensor(actions, dtype=torch.long, device=device)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
                dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

                # Current Q values for chosen actions
                curr_q = model(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)

                # Target Q values using Target Network
                with torch.no_grad():
                    next_q = target_model(next_states_t).max(dim=1)[0]
                target_q = rewards_t + (1.0 - dones_t) * gamma * next_q

                loss = criterion(curr_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # Decay exploration
        epsilon = max(min_epsilon, epsilon * decay_rate)
        episode_rewards.append(total_reward)

        # Periodically update Target Network weights
        if ep % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        if ep % max(1, args.episodes // 10) == 0:
            avg_rew = np.mean(episode_rewards[-30:])
            print(f"Episode {ep:4d}/{args.episodes} | last-30 avg reward: {avg_rew:.2f} | epsilon: {epsilon:.3f}")

    print("-" * 64)
    env.close()

    # Define evaluation action selection
    def eval_act_fn(s):
        s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_vals = model(s_t)
        return int(q_vals.argmax(dim=1).item())

    # Evaluate
    print("Evaluating trained DQN...")
    mean_reward, std_reward = mc.evaluate_agent(env_id, eval_act_fn, num_episodes=20)
    print(f"Evaluation over 20 runs: mean_reward {mean_reward:.2f} | std_reward {std_reward:.2f}")

    # Plot metrics and render agent GIF
    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        curve_path = os.path.join(save_dir, f"dqn_{env_id.lower()}_curve.png")
        mc.plot_learning_curve(episode_rewards, curve_path)

        gif_path = os.path.join(save_dir, f"dqn_{env_id.lower()}_agent.gif")
        mc.save_agent_gif(env_id, eval_act_fn, gif_path)


if __name__ == "__main__":
    main()
