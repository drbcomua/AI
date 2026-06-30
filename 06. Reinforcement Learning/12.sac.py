"""
12. Soft Actor-Critic (SAC) — continuous control (Haarnoja et al., 2018)
=======================================================================

The off-policy actor-critic branch, for *continuous* action spaces (here
Pendulum-v1, where the action is a torque in [-2, 2]). SAC maximizes reward *plus*
policy entropy, so it explores well and trains stably. Three ingredients:

    * Stochastic actor: a tanh-squashed Gaussian policy (reparameterized for
      low-variance gradients).
    * Twin critics: two Q-networks; the min of the two fights overestimation (as
      in TD3). Soft-updated target critics stabilize bootstrapping.
    * Automatic temperature: the entropy coefficient alpha is tuned to hit a target
      entropy, balancing exploration vs exploitation automatically.

Key insights / educational takeaways:
    * Maximum-entropy RL: "succeed while acting as randomly as possible."
    * Off-policy + replay makes it far more sample-efficient than on-policy PPO on
      continuous tasks.

Run:
    python "12.sac.py" --env Pendulum-v1 --episodes 150
    python "12.sac.py" --env Pendulum-v1 --episodes 300
"""

import os
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import rl_common as mc


class ContinuousReplay:
    def __init__(self, capacity=100000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def sample(self, batch):
        s, a, r, ns, d = zip(*random.sample(self.buf, batch))
        return (np.array(s, np.float32), np.array(a, np.float32), np.array(r, np.float32),
                np.array(ns, np.float32), np.array(d, np.float32))

    def __len__(self):
        return len(self.buf)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_scale, hidden=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)
        self.action_scale = action_scale

    def sample(self, state):
        h = self.body(state)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(-20, 2)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()                                       # reparameterized
        y = torch.tanh(x)
        action = y * self.action_scale
        # tanh-correction for the squashed log-prob
        log_prob = normal.log_prob(x) - torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        return action, log_prob.sum(1, keepdim=True), torch.tanh(mean) * self.action_scale


class Critic(nn.Module):
    """Twin Q-networks."""
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        def q():
            return nn.Sequential(nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.q1, self.q2 = q(), q()

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=1)
        return self.q1(sa), self.q2(sa)


def soft_update(target, source, tau=0.005):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


def main():
    p = mc.build_argparser("Soft Actor-Critic (SAC)", episodes=150, lr=3e-4)
    args = p.parse_args()
    env_id = args.env
    if env_id != "Pendulum-v1":
        print("Note: SAC needs a continuous action space; forcing --env Pendulum-v1.")
        env_id = "Pendulum-v1"
    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])
    device = mc.get_device(args.device)

    actor = Actor(state_dim, action_dim, action_scale).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    critic_target = Critic(state_dim, action_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())
    a_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    c_opt = torch.optim.Adam(critic.parameters(), lr=args.lr)
    log_alpha = torch.zeros(1, device=device, requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=args.lr)
    target_entropy = -float(action_dim)

    buffer = ContinuousReplay()
    gamma, batch_size, warmup = args.gamma, 256, 1000
    episode_rewards = []
    print(f"Training SAC on {env_id} for {args.episodes} episodes...")
    print(f"Device: {device} | actor params: {sum(q.numel() for q in actor.parameters()):,}")
    print("-" * 64)

    total_steps = 0
    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        while True:
            if total_steps < warmup:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    a, _, _ = actor.sample(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                action = a.squeeze(0).cpu().numpy()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, float(terminated))
            state = next_state
            total_reward += reward
            total_steps += 1

            if len(buffer) >= max(batch_size, warmup):
                s, a, r, ns, d = buffer.sample(batch_size)
                s = torch.tensor(s, device=device); a = torch.tensor(a, device=device)
                r = torch.tensor(r, device=device).unsqueeze(1); ns = torch.tensor(ns, device=device)
                d = torch.tensor(d, device=device).unsqueeze(1)
                alpha = log_alpha.exp()
                # critic update
                with torch.no_grad():
                    na, nlogp, _ = actor.sample(ns)
                    tq1, tq2 = critic_target(ns, na)
                    target_q = r + gamma * (1 - d) * (torch.min(tq1, tq2) - alpha * nlogp)
                q1, q2 = critic(s, a)
                c_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)
                c_opt.zero_grad(); c_loss.backward(); c_opt.step()
                # actor update
                pa, logp, _ = actor.sample(s)
                qa1, qa2 = critic(s, pa)
                a_loss = (alpha.detach() * logp - torch.min(qa1, qa2)).mean()
                a_opt.zero_grad(); a_loss.backward(); a_opt.step()
                # temperature update
                alpha_loss = -(log_alpha * (logp + target_entropy).detach()).mean()
                alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()
                soft_update(critic_target, critic)
            if done:
                break

        episode_rewards.append(total_reward)
        if ep % max(1, args.episodes // 10) == 0:
            print(f"Episode {ep:4d}/{args.episodes} | last-20 avg: {np.mean(episode_rewards[-20:]):.1f} | alpha {log_alpha.exp().item():.3f}")
    print("-" * 64)
    env.close()

    def eval_act_fn(s):
        with torch.no_grad():
            _, _, mean_a = actor.sample(torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0))
        return mean_a.squeeze(0).cpu().numpy()
    mean_r, std_r = mc.evaluate_agent(env_id, eval_act_fn, num_episodes=20)
    print(f"Evaluation over 20 runs: mean_reward {mean_r:.2f} | std_reward {std_r:.2f}")

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        mc.plot_learning_curve(episode_rewards, os.path.join(save_dir, f"sac_{env_id.lower()}_curve.png"))
        mc.save_agent_gif(env_id, eval_act_fn, os.path.join(save_dir, f"sac_{env_id.lower()}_agent.gif"))


if __name__ == "__main__":
    main()
