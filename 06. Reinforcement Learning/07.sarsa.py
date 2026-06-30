"""
07. SARSA (on-policy tabular control)
=====================================

The on-policy twin of tabular Q-learning. Where Q-learning bootstraps from the
*greedy* next action (max_a' Q), SARSA bootstraps from the action the policy
*actually takes* next (the epsilon-greedy a'), so it learns the value of the
policy it is following rather than the optimal one.

Update Rule:
    Q(s, a) <- Q(s, a) + lr * [r + gamma * Q(s', a') - Q(s, a)]
    where a' is the (epsilon-greedy) action actually chosen in s'.

Key insights / educational takeaways:
    * On-policy vs off-policy: SARSA accounts for exploration in its targets, so it
      learns "safer" policies near cliffs/penalties; Q-learning learns the optimal
      (riskier) greedy policy. Same table, one-line difference in the target.

Run:
    python "07.sarsa.py" --env CartPole-v1 --episodes 2000
    python "07.sarsa.py" --env MountainCar-v0 --episodes 3000
"""

import os
import random
import numpy as np
import gymnasium as gym
import rl_common as mc


def main():
    args = mc.build_argparser("SARSA (on-policy tabular)", lr=0.1).parse_args()
    env_id = args.env
    env = gym.make(env_id)
    discretizer = mc.get_cartpole_discretizer() if env_id == "CartPole-v1" else mc.get_mountaincar_discretizer()
    num_actions = env.action_space.n
    q_table = np.zeros(discretizer.num_buckets + (num_actions,))

    alpha, gamma = args.lr, args.gamma
    epsilon, min_epsilon, decay = 1.0, 0.01, 0.995
    episode_rewards = []

    def choose(idx):
        if random.random() < epsilon:
            return env.action_space.sample()
        return int(np.argmax(q_table[idx]))

    print(f"Training SARSA on {env_id} for {args.episodes} episodes...")
    print("-" * 64)
    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()
        s_idx = discretizer.discretize(state)
        action = choose(s_idx)
        total_reward = 0.0
        while True:
            next_state, reward, terminated, truncated, _ = env.step(action)
            n_idx = discretizer.discretize(next_state)
            if env_id == "MountainCar-v0":
                reward += 10.0 * abs(next_state[1])
            next_action = choose(n_idx)                       # on-policy: use the NEXT chosen action
            q_table[s_idx][action] += alpha * (
                reward + gamma * q_table[n_idx][next_action] - q_table[s_idx][action])
            s_idx, action = n_idx, next_action
            total_reward += reward
            if terminated or truncated:
                break
        epsilon = max(min_epsilon, epsilon * decay)
        episode_rewards.append(total_reward)
        if ep % max(1, args.episodes // 10) == 0:
            print(f"Episode {ep:4d}/{args.episodes} | last-50 avg: {np.mean(episode_rewards[-50:]):.2f} | eps {epsilon:.3f}")
    print("-" * 64)
    env.close()

    eval_act_fn = lambda s: int(np.argmax(q_table[discretizer.discretize(s)]))
    mean_r, std_r = mc.evaluate_agent(env_id, eval_act_fn, num_episodes=20)
    print(f"Evaluation over 20 runs: mean_reward {mean_r:.2f} | std_reward {std_r:.2f}")

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        mc.plot_learning_curve(episode_rewards, os.path.join(save_dir, f"sarsa_{env_id.lower()}_curve.png"))
        mc.save_agent_gif(env_id, eval_act_fn, os.path.join(save_dir, f"sarsa_{env_id.lower()}_agent.gif"))


if __name__ == "__main__":
    main()
