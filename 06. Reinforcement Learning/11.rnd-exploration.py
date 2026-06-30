"""
11. RND — Random Network Distillation (Burda et al., 2018)
=========================================================

Exploration via curiosity, for sparse-reward problems where extrinsic reward is
almost never seen (e.g. MountainCar: -1 every step until the rare flag). RND adds
an intrinsic bonus that rewards *novelty*: a fixed, randomly-initialized "target"
network maps states to features, and a "predictor" network is trained to imitate
it. Familiar states are predicted well (low error); novel states are predicted
badly (high error) -> high intrinsic reward, which pulls the agent toward the
unexplored.

Method (DQN backbone):
    intrinsic_t = || predictor(s') - target(s') ||^2   (target frozen)
    train DQN on  reward = extrinsic + beta * normalized(intrinsic)
    train predictor by regressing onto the frozen target on visited states.

Key insights / educational takeaways:
    * Decouples exploration from the (absent) reward signal — vanilla DQN/REINFORCE
      typically never solve MountainCar; with an RND bonus it gets there.
    * The bonus naturally decays as states become familiar (predictor catches up).

Run:
    python "11.rnd-exploration.py" --env MountainCar-v0 --episodes 500
    python "11.rnd-exploration.py" --env CartPole-v1 --episodes 300
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import rl_common as mc


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


def main():
    args = mc.build_argparser("RND Exploration (DQN + curiosity)").parse_args()
    env_id = args.env
    env = gym.make(env_id)
    state_dim, num_actions = env.observation_space.shape[0], env.action_space.n
    device = mc.get_device(args.device)

    model = MLP(state_dim, num_actions).to(device)
    target_model = MLP(state_dim, num_actions).to(device)
    target_model.load_state_dict(model.state_dict()); target_model.eval()
    # RND target (frozen) + predictor
    feat_dim = 32
    rnd_target = MLP(state_dim, feat_dim).to(device)
    for q in rnd_target.parameters():
        q.requires_grad_(False)
    rnd_pred = MLP(state_dim, feat_dim).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rnd_opt = torch.optim.Adam(rnd_pred.parameters(), lr=args.lr)
    buffer = mc.ReplayBuffer(10000)

    gamma, epsilon, min_eps, decay, batch_size, target_freq, beta = args.gamma, 1.0, 0.05, 0.99, 64, 10, 1.0
    int_std = 1.0                                                   # running scale of intrinsic reward
    episode_rewards = []
    print(f"Training DQN+RND on {env_id} for {args.episodes} episodes...")
    print("-" * 64)

    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        while True:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = int(model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).argmax(1))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward                                  # log extrinsic only

            if len(buffer) >= batch_size:
                s, a, r, ns, d = buffer.sample(batch_size)
                s = torch.tensor(s, device=device); a = torch.tensor(a, device=device)
                r = torch.tensor(r, device=device); ns = torch.tensor(ns, device=device)
                d = torch.tensor(d, device=device)
                # intrinsic reward = RND prediction error on next states
                with torch.no_grad():
                    tgt = rnd_target(ns)
                pred = rnd_pred(ns)
                intrinsic = (pred - tgt).pow(2).mean(dim=1)
                int_std = 0.99 * int_std + 0.01 * intrinsic.detach().std().item()
                shaped_r = r + beta * intrinsic.detach() / (int_std + 1e-8)
                # DQN update on shaped reward
                curr_q = model(s).gather(1, a.unsqueeze(-1)).squeeze(-1)
                with torch.no_grad():
                    next_q = target_model(ns).max(1)[0]
                    target_q = shaped_r + (1 - d) * gamma * next_q
                loss = (curr_q - target_q).pow(2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                # train predictor toward frozen target
                rnd_loss = intrinsic.mean()
                rnd_opt.zero_grad(); rnd_loss.backward(); rnd_opt.step()
            if done:
                break

        epsilon = max(min_eps, epsilon * decay)
        episode_rewards.append(total_reward)
        if ep % target_freq == 0:
            target_model.load_state_dict(model.state_dict())
        if ep % max(1, args.episodes // 10) == 0:
            print(f"Episode {ep:4d}/{args.episodes} | last-30 avg (extrinsic): {np.mean(episode_rewards[-30:]):.2f} | eps {epsilon:.3f}")
    print("-" * 64)
    env.close()

    def eval_act_fn(s):
        with torch.no_grad():
            return int(model(torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)).argmax(1))
    mean_r, std_r = mc.evaluate_agent(env_id, eval_act_fn, num_episodes=20)
    print(f"Evaluation over 20 runs: mean_reward {mean_r:.2f} | std_reward {std_r:.2f}")

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        mc.plot_learning_curve(episode_rewards, os.path.join(save_dir, f"rnd_{env_id.lower()}_curve.png"))
        mc.save_agent_gif(env_id, eval_act_fn, os.path.join(save_dir, f"rnd_{env_id.lower()}_agent.gif"))


if __name__ == "__main__":
    main()
