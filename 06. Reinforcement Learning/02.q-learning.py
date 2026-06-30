"""
02. Tabular Q-Learning
======================

Classical model-free value-based reinforcement learning using a discrete lookup table.

Update Rule (Bellman optimality Equation):
    Q(s, a) <- Q(s, a) + lr * [reward + gamma * max_a Q(s', a) - Q(s, a)]

Run:
    python "02.q-learning.py" --env CartPole-v1 --episodes 2000
    python "02.q-learning.py" --env MountainCar-v0 --episodes 3000
"""

import os
import random
import numpy as np
import gymnasium as gym
import rl_common as mc


def main():
    p = mc.build_argparser("Tabular Q-Learning", lr=0.1)
    args = p.parse_args()

    env_id = args.env
    env = gym.make(env_id)

    # Discretizer setup
    if env_id == "CartPole-v1":
        discretizer = mc.get_cartpole_discretizer()
    else:
        discretizer = mc.get_mountaincar_discretizer()

    # Initialize Q-table: dimensions matching discretized state buckets + action count
    num_actions = env.action_space.n
    q_table = np.zeros(discretizer.num_buckets + (num_actions,))

    # Hyperparameters
    alpha = args.lr # learning rate
    gamma = args.gamma # discount factor
    epsilon = 1.0 # start exploration
    min_epsilon = 0.01
    decay_rate = 0.995

    # Log metrics
    episode_rewards = []

    print(f"Training Tabular Q-Learning on {env_id} for {args.episodes} episodes...")
    print("-" * 64)

    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()
        state_idx = discretizer.discretize(state)
        total_reward = 0.0

        while True:
            # Action selection: epsilon-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state_idx])

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_idx = discretizer.discretize(next_state)

            # MountainCar sparse rewards helper: encourage moving up the valley
            if env_id == "MountainCar-v0":
                # State: [position, velocity]
                # Reward is -1.0 per step; encourage moving higher
                reward += 10.0 * abs(next_state[1])

            # Update Bellman optimal Q-value
            q_table[state_idx][action] += alpha * (
                reward + gamma * np.max(q_table[next_idx]) - q_table[state_idx][action]
            )

            state_idx = next_idx
            total_reward += reward

            if terminated or truncated:
                break

        # Decay exploration
        epsilon = max(min_epsilon, epsilon * decay_rate)
        episode_rewards.append(total_reward)

        if ep % max(1, args.episodes // 10) == 0:
            avg_rew = np.mean(episode_rewards[-50:])
            print(f"Episode {ep:4d}/{args.episodes} | last-50 avg reward: {avg_rew:.2f} | epsilon: {epsilon:.3f}")

    print("-" * 64)
    env.close()

    # Define evaluation action selection using discretized state indices
    eval_act_fn = lambda s: int(np.argmax(q_table[discretizer.discretize(s)]))

    # Evaluate agent
    print("Evaluating trained agent...")
    mean_reward, std_reward = mc.evaluate_agent(env_id, eval_act_fn, num_episodes=20)
    print(f"Evaluation over 20 runs: mean_reward {mean_reward:.2f} | std_reward {std_reward:.2f}")

    # Plot metrics and render agent GIF
    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        curve_path = os.path.join(save_dir, f"qlearning_{env_id.lower()}_curve.png")
        mc.plot_learning_curve(episode_rewards, curve_path)

        gif_path = os.path.join(save_dir, f"qlearning_{env_id.lower()}_agent.gif")
        mc.save_agent_gif(env_id, eval_act_fn, gif_path)


if __name__ == "__main__":
    main()
