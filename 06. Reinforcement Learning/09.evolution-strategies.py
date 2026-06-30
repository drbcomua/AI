"""
09. Evolution Strategies (Salimans et al., 2017 — OpenAI ES)
============================================================

A gradient-free alternative to backprop-based RL. Treat the episodic return as a
black-box function of the policy weights and optimize it by *perturb-and-average*:
sample many Gaussian noise vectors, add each to the current weights, measure the
return of every perturbed policy, then move the weights toward the noise
directions that scored well. No backpropagation, no value function, no replay —
just rollouts, and it parallelizes trivially across the population.

Method (with antithetic sampling for variance reduction):
    for each generation:
        sample noise eps_i; evaluate return of (theta + sigma*eps_i) and (theta - sigma*eps_i)
        A = normalized returns
        theta <- theta + (lr / (pop * sigma)) * sum_i A_i * eps_i

Key insights / educational takeaways:
    * Optimizes the policy directly from scalar episode returns — robust to sparse,
      non-differentiable, or noisy rewards.
    * "Episodes" here are ES *generations*; each runs a whole population of rollouts.

Run:
    python "09.evolution-strategies.py" --env CartPole-v1 --episodes 100
    python "09.evolution-strategies.py" --env CartPole-v1 --episodes 200 --variant 40
"""

import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import rl_common as mc


class Policy(nn.Module):
    def __init__(self, state_dim, num_actions, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, num_actions))

    def forward(self, x):
        return self.net(x)


def get_flat(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat(model, flat):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i + n].view_as(p))
        i += n


@torch.no_grad()
def rollout(model, env, max_steps=500):
    state, _ = env.reset()
    total = 0.0
    for _ in range(max_steps):
        logits = model(torch.tensor(state, dtype=torch.float32))
        action = int(logits.argmax().item())
        state, reward, terminated, truncated, _ = env.step(action)
        total += reward
        if terminated or truncated:
            break
    return total


def main():
    p = mc.build_argparser("Evolution Strategies", episodes=100, lr=0.05)
    args = p.parse_args()
    pop = int(args.variant) if args.variant else 30                # population (antithetic pairs)
    sigma = 0.1

    env_id = args.env
    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    model = Policy(state_dim, num_actions)
    theta = get_flat(model)
    dim = theta.numel()
    print(f"Training ES on {env_id} | pop {pop} (antithetic) | params {dim} | {args.episodes} generations")
    print("-" * 64)

    gen_best = []
    for gen in range(1, args.episodes + 1):
        noises = torch.randn(pop, dim)
        returns = np.zeros(pop * 2)
        for i in range(pop):
            for sgn, slot in ((1.0, 2 * i), (-1.0, 2 * i + 1)):    # antithetic pair
                set_flat(model, theta + sgn * sigma * noises[i])
                returns[slot] = rollout(model, env)
        # advantage = normalized returns; recombine antithetically
        A = (returns - returns.mean()) / (returns.std() + 1e-8)
        grad = torch.zeros(dim)
        for i in range(pop):
            grad += (A[2 * i] - A[2 * i + 1]) * noises[i]
        theta = theta + (args.lr / (pop * sigma)) * grad
        gen_best.append(returns.max())
        if gen % max(1, args.episodes // 10) == 0:
            set_flat(model, theta)
            print(f"Gen {gen:4d}/{args.episodes} | pop best {returns.max():.1f} | pop mean {returns.mean():.1f}")
    print("-" * 64)
    set_flat(model, theta)
    env.close()

    def eval_act_fn(s):
        with torch.no_grad():
            return int(model(torch.tensor(s, dtype=torch.float32)).argmax().item())
    mean_r, std_r = mc.evaluate_agent(env_id, eval_act_fn, num_episodes=20)
    print(f"Evaluation over 20 runs: mean_reward {mean_r:.2f} | std_reward {std_r:.2f}")

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        mc.plot_learning_curve(gen_best, os.path.join(save_dir, f"es_{env_id.lower()}_curve.png"))
        mc.save_agent_gif(env_id, eval_act_fn, os.path.join(save_dir, f"es_{env_id.lower()}_agent.gif"))


if __name__ == "__main__":
    main()
