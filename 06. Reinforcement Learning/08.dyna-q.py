"""
08. Dyna-Q (tabular model-based RL)
===================================

The first step beyond model-free learning: Dyna-Q learns a *model* of the
environment (which (reward, next-state) follows each (state, action)) and uses it
to "imagine" extra experience. After every real step it performs N planning
updates on randomly recalled (state, action) pairs, applying the same Q-learning
rule to model-predicted transitions — so one real interaction yields many value
updates.

Method:
    real step: Q-learning update + record model[(s,a)] = (r, s')
    planning : repeat N times: sample a seen (s,a); (r,s') = model[(s,a)];
               Q(s,a) <- Q(s,a) + lr * [r + gamma * max_a' Q(s',a') - Q(s,a)]

Key insights / educational takeaways:
    * Planning with a learned model dramatically improves sample efficiency — Dyna-Q
      reaches good policies in far fewer real episodes than tabular Q-learning.
    * Bridges model-free and model-based RL with a tiny amount of extra code.

Run:
    python "08.dyna-q.py" --env CartPole-v1 --episodes 1000
    python "08.dyna-q.py" --env MountainCar-v0 --episodes 1500 --variant 30
"""

import os
import random
import numpy as np
import gymnasium as gym
import rl_common as mc


def main():
    p = mc.build_argparser("Dyna-Q (model-based tabular)", lr=0.1)
    args = p.parse_args()
    n_planning = int(args.variant) if args.variant else 20         # planning steps per real step

    env_id = args.env
    env = gym.make(env_id)
    discretizer = mc.get_cartpole_discretizer() if env_id == "CartPole-v1" else mc.get_mountaincar_discretizer()
    num_actions = env.action_space.n
    q_table = np.zeros(discretizer.num_buckets + (num_actions,))
    model = {}                                                      # (s_idx, a) -> (reward, n_idx)

    alpha, gamma = args.lr, args.gamma
    epsilon, min_epsilon, decay = 1.0, 0.01, 0.995
    episode_rewards = []

    print(f"Training Dyna-Q on {env_id} ({n_planning} planning steps) for {args.episodes} episodes...")
    print("-" * 64)
    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()
        s_idx = discretizer.discretize(state)
        total_reward = 0.0
        while True:
            action = env.action_space.sample() if random.random() < epsilon else int(np.argmax(q_table[s_idx]))
            next_state, reward, terminated, truncated, _ = env.step(action)
            n_idx = discretizer.discretize(next_state)
            if env_id == "MountainCar-v0":
                reward += 10.0 * abs(next_state[1])
            # 1) direct RL update
            q_table[s_idx][action] += alpha * (reward + gamma * np.max(q_table[n_idx]) - q_table[s_idx][action])
            # 2) model learning
            model[(s_idx, action)] = (reward, n_idx)
            # 3) planning: replay imagined transitions
            for _ in range(n_planning):
                (ps, pa), (pr, pn) = random.choice(list(model.items()))
                q_table[ps][pa] += alpha * (pr + gamma * np.max(q_table[pn]) - q_table[ps][pa])
            s_idx = n_idx
            total_reward += reward
            if terminated or truncated:
                break
        epsilon = max(min_epsilon, epsilon * decay)
        episode_rewards.append(total_reward)
        if ep % max(1, args.episodes // 10) == 0:
            print(f"Episode {ep:4d}/{args.episodes} | last-50 avg: {np.mean(episode_rewards[-50:]):.2f} | "
                  f"model size {len(model)} | eps {epsilon:.3f}")
    print("-" * 64)
    env.close()

    eval_act_fn = lambda s: int(np.argmax(q_table[discretizer.discretize(s)]))
    mean_r, std_r = mc.evaluate_agent(env_id, eval_act_fn, num_episodes=20)
    print(f"Evaluation over 20 runs: mean_reward {mean_r:.2f} | std_reward {std_r:.2f}")

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        mc.plot_learning_curve(episode_rewards, os.path.join(save_dir, f"dynaq_{env_id.lower()}_curve.png"))
        mc.save_agent_gif(env_id, eval_act_fn, os.path.join(save_dir, f"dynaq_{env_id.lower()}_agent.gif"))


if __name__ == "__main__":
    main()
