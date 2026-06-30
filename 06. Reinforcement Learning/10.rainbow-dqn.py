"""
10. Rainbow-lite DQN — Double + Dueling + Prioritized Replay
============================================================

Three of the headline improvements stacked on the vanilla DQN (03), selectable so
you can isolate each one's effect:

    --variant double    Double DQN (van Hasselt et al., 2016): the online net picks
                        the next action, the target net evaluates it — removes the
                        max-operator overestimation bias.
    --variant dueling   Dueling DQN (Wang et al., 2016): split into a state-value
                        V(s) stream and an advantage A(s,a) stream,
                        Q = V + (A - mean(A)) — learns which states are valuable
                        without having to learn every action's value.
    --variant per       Prioritized Experience Replay (Schaul et al., 2016): sample
                        transitions with high TD-error more often (importance-
                        sampling weights correct the resulting bias).
    --variant rainbow   all three together (default).

Run:
    python "10.rainbow-dqn.py" --variant rainbow --episodes 300
    python "10.rainbow-dqn.py" --variant double --episodes 300
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import rl_common as mc


class PrioritizedBuffer:
    """Proportional PER; with alpha=0 it degrades to a uniform replay buffer."""
    def __init__(self, capacity=10000, alpha=0.6):
        self.capacity, self.alpha = capacity, alpha
        self.data = []
        self.prios = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def push(self, *transition):
        max_prio = self.prios[:len(self.data)].max() if self.data else 1.0
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.pos] = transition
        self.prios[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        prios = self.prios[:len(self.data)]
        probs = prios ** self.alpha
        probs = probs / probs.sum()
        idx = np.random.choice(len(self.data), batch_size, p=probs)
        s, a, r, ns, d = zip(*[self.data[i] for i in idx])
        weights = (len(self.data) * probs[idx]) ** (-beta)
        weights = weights / weights.max()
        return (np.array(s, np.float32), np.array(a, np.int64), np.array(r, np.float32),
                np.array(ns, np.float32), np.array(d, np.float32), idx, weights.astype(np.float32))

    def update_priorities(self, idx, prios):
        self.prios[idx] = np.abs(prios) + 1e-5

    def __len__(self):
        return len(self.data)


class QNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, dueling=False):
        super().__init__()
        self.dueling = dueling
        self.feature = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        if dueling:
            self.value = nn.Linear(64, 1)
            self.adv = nn.Linear(64, num_actions)
        else:
            self.head = nn.Linear(64, num_actions)

    def forward(self, x):
        h = self.feature(x)
        if self.dueling:
            v, a = self.value(h), self.adv(h)
            return v + (a - a.mean(dim=1, keepdim=True))
        return self.head(h)


def main():
    p = mc.build_argparser("Rainbow-lite DQN")
    args = p.parse_args()
    variant = args.variant or "rainbow"
    use_double = variant in ("double", "rainbow")
    use_dueling = variant in ("dueling", "rainbow")
    use_per = variant in ("per", "rainbow")

    env_id = args.env
    env = gym.make(env_id)
    state_dim, num_actions = env.observation_space.shape[0], env.action_space.n
    device = mc.get_device(args.device)

    model = QNetwork(state_dim, num_actions, use_dueling).to(device)
    target = QNetwork(state_dim, num_actions, use_dueling).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    buffer = PrioritizedBuffer(10000, alpha=0.6 if use_per else 0.0)

    gamma, epsilon, min_eps, decay, batch_size, target_freq = args.gamma, 1.0, 0.02, 0.99, 64, 10
    episode_rewards = []
    print(f"Training {variant.upper()} DQN on {env_id} "
          f"(double={use_double}, dueling={use_dueling}, per={use_per})")
    print(f"Device: {device} | trainable params: {sum(q.numel() for q in model.parameters()):,}")
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
            total_reward += reward

            if len(buffer) >= batch_size:
                s, a, r, ns, d, idx, w = buffer.sample(batch_size)
                s = torch.tensor(s, device=device); a = torch.tensor(a, device=device)
                r = torch.tensor(r, device=device); ns = torch.tensor(ns, device=device)
                d = torch.tensor(d, device=device); w = torch.tensor(w, device=device)
                curr_q = model(s).gather(1, a.unsqueeze(-1)).squeeze(-1)
                with torch.no_grad():
                    if use_double:                                  # online selects, target evaluates
                        next_a = model(ns).argmax(1, keepdim=True)
                        next_q = target(ns).gather(1, next_a).squeeze(-1)
                    else:
                        next_q = target(ns).max(1)[0]
                    target_q = r + (1 - d) * gamma * next_q
                td_err = curr_q - target_q
                loss = (w * td_err.pow(2)).mean()                   # IS-weighted Huber/MSE
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                if use_per:
                    buffer.update_priorities(idx, td_err.detach().cpu().numpy())
            if done:
                break

        epsilon = max(min_eps, epsilon * decay)
        episode_rewards.append(total_reward)
        if ep % target_freq == 0:
            target.load_state_dict(model.state_dict())
        if ep % max(1, args.episodes // 10) == 0:
            print(f"Episode {ep:4d}/{args.episodes} | last-30 avg: {np.mean(episode_rewards[-30:]):.2f} | eps {epsilon:.3f}")
    print("-" * 64)
    env.close()

    def eval_act_fn(s):
        with torch.no_grad():
            return int(model(torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)).argmax(1))
    mean_r, std_r = mc.evaluate_agent(env_id, eval_act_fn, num_episodes=20)
    print(f"Evaluation over 20 runs: mean_reward {mean_r:.2f} | std_reward {std_r:.2f}")

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        tag = f"rainbow-{variant}"
        mc.plot_learning_curve(episode_rewards, os.path.join(save_dir, f"{tag}_{env_id.lower()}_curve.png"))
        mc.save_agent_gif(env_id, eval_act_fn, os.path.join(save_dir, f"{tag}_{env_id.lower()}_agent.gif"))


if __name__ == "__main__":
    main()
