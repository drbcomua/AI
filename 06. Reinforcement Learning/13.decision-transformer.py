"""
13. Decision Transformer (Chen et al., 2021)
============================================

RL recast as *sequence modeling*. Instead of learning a value function or policy
gradient, train a causal Transformer (a GPT) on offline trajectories represented
as token sequences:

    (return-to-go_1, state_1, action_1, return-to-go_2, state_2, action_2, ...)

The model predicts each action from the preceding tokens. At test time you
*condition on a desired return* (a large return-to-go), feed the observed states,
and let the Transformer autoregressively emit actions that achieve it — no
bootstrapping, no Bellman equation. It ties RL to the same architecture used in
this repo's Language Modeling folder.

Pipeline:
    1. Collect an OFFLINE dataset with a mixed behavior policy (varied returns).
    2. Train the Decision Transformer to predict actions (cross-entropy).
    3. Evaluate online by conditioning on a high target return.

Run:
    python "13.decision-transformer.py" --env CartPole-v1 --episodes 40
    python "13.decision-transformer.py" --env CartPole-v1 --episodes 60 --variant 300
"""

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import rl_common as mc

K = 20            # context length (timesteps)
RTG_SCALE = 100.0  # normalize returns-to-go for stable embeddings


def behavior_action(env_id, state, env):
    """Mixed behavior policy: a simple heuristic perturbed by randomness, to produce
    a spread of trajectory returns for the offline dataset."""
    if random.random() < 0.4:
        return env.action_space.sample()
    if env_id == "CartPole-v1":
        return 1 if (state[2] + 0.5 * state[3]) > 0 else 0          # push toward the fall
    if env_id == "MountainCar-v0":
        return 2 if state[1] >= 0 else 0                            # push with velocity
    return env.action_space.sample()


def collect_dataset(env_id, n_episodes=800):
    env = gym.make(env_id)
    trajectories = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        states, actions, rewards = [], [], []
        while True:
            a = behavior_action(env_id, state, env)
            ns, r, term, trunc, _ = env.step(a)
            states.append(state); actions.append(a); rewards.append(r)
            state = ns
            if term or trunc:
                break
        rtg = np.cumsum(rewards[::-1])[::-1].copy()                 # return-to-go per step
        trajectories.append((np.array(states, np.float32), np.array(actions, np.int64),
                             np.array(rtg, np.float32)))
    env.close()
    return trajectories


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, num_actions, d_model=64, n_layers=3, n_heads=4, max_ep_len=500):
        super().__init__()
        self.num_actions = num_actions
        self.embed_state = nn.Linear(state_dim, d_model)
        self.embed_action = nn.Embedding(num_actions, d_model)
        self.embed_rtg = nn.Linear(1, d_model)
        self.embed_time = nn.Embedding(max_ep_len, d_model)
        self.ln = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, num_actions)
        self.d_model = d_model

    def forward(self, rtg, states, actions, timesteps):
        # rtg:[B,K,1] states:[B,K,sd] actions:[B,K] timesteps:[B,K]
        B, T = actions.shape
        t_emb = self.embed_time(timesteps)
        r = self.embed_rtg(rtg) + t_emb
        s = self.embed_state(states) + t_emb
        a = self.embed_action(actions) + t_emb
        # interleave to (B, 3T, d): [r_0, s_0, a_0, r_1, s_1, a_1, ...]
        tokens = torch.stack([r, s, a], dim=2).reshape(B, 3 * T, self.d_model)
        tokens = self.ln(tokens)
        mask = torch.triu(torch.ones(3 * T, 3 * T, device=tokens.device), diagonal=1).bool()
        h = self.transformer(tokens, mask=mask)
        # predict action from the state-token positions (indices 1, 4, 7, ... = 3k+1)
        s_out = h[:, 1::3, :]
        return self.head(s_out)                                     # [B, T, num_actions]


def main():
    p = mc.build_argparser("Decision Transformer (offline RL as sequence modeling)", episodes=40, lr=1e-3)
    args = p.parse_args()
    env_id = args.env if args.env != "Pendulum-v1" else "CartPole-v1"   # discrete actions only
    device = mc.get_device(args.device)

    print(f"Collecting offline dataset on {env_id}...")
    trajs = collect_dataset(env_id, n_episodes=800)
    returns = [t[2][0] for t in trajs]
    target_return = float(args.variant) if args.variant else float(min(max(returns), 200.0))
    print(f"Dataset: {len(trajs)} trajectories | return min/mean/max: "
          f"{min(returns):.0f}/{np.mean(returns):.0f}/{max(returns):.0f} | eval target: {target_return:.0f}")

    env = gym.make(env_id)
    state_dim, num_actions = env.observation_space.shape[0], env.action_space.n
    env.close()
    model = DecisionTransformer(state_dim, num_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()
    print(f"Device: {device} | trainable params: {sum(q.numel() for q in model.parameters()):,}")
    print("-" * 64)

    def sample_batch(batch=64):
        rtg_b, s_b, a_b, t_b = [], [], [], []
        for _ in range(batch):
            states, actions, rtg = random.choice(trajs)
            L = len(actions)
            start = random.randint(0, max(0, L - 1))
            sl = slice(start, min(start + K, L))
            n = sl.stop - sl.start
            pad = K - n
            s_b.append(np.pad(states[sl], ((0, pad), (0, 0))))
            a_b.append(np.pad(actions[sl], (0, pad)))
            rtg_b.append(np.pad(rtg[sl] / RTG_SCALE, (0, pad)))
            t_b.append(np.pad(np.arange(sl.start, sl.stop), (0, pad)))
        return (torch.tensor(np.array(rtg_b), dtype=torch.float32, device=device).unsqueeze(-1),
                torch.tensor(np.array(s_b), dtype=torch.float32, device=device),
                torch.tensor(np.array(a_b), dtype=torch.long, device=device),
                torch.tensor(np.array(t_b), dtype=torch.long, device=device))

    steps_per_epoch = 100
    for epoch in range(1, args.episodes + 1):
        model.train()
        running = 0.0
        for _ in range(steps_per_epoch):
            rtg, s, a, t = sample_batch()
            logits = model(rtg, s, a, t)
            loss = ce(logits.reshape(-1, num_actions), a.reshape(-1))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running += loss.item()
        if epoch % max(1, args.episodes // 10) == 0:
            print(f"Epoch {epoch:3d}/{args.episodes} | action CE loss: {running / steps_per_epoch:.4f}")
    print("-" * 64)

    # Online evaluation conditioned on the target return (rolling context window)
    @torch.no_grad()
    def run_episode(env, render_frames=None):
        model.eval()
        state, _ = env.reset(seed=42) if render_frames is not None else env.reset()
        states, actions, rtgs, times = [state], [0], [target_return / RTG_SCALE], [0]
        total = 0.0
        for tstep in range(500):
            if render_frames is not None:
                f = env.render()
                if f is not None:
                    from PIL import Image
                    render_frames.append(Image.fromarray(f))
            # last K timesteps
            s = torch.tensor(np.array(states[-K:]), dtype=torch.float32, device=device).unsqueeze(0)
            a = torch.tensor(np.array(actions[-K:]), dtype=torch.long, device=device).unsqueeze(0)
            r = torch.tensor(np.array(rtgs[-K:]), dtype=torch.float32, device=device).view(1, -1, 1)
            tt = torch.tensor(np.array(times[-K:]), dtype=torch.long, device=device).unsqueeze(0)
            logits = model(r, s, a, tt)
            action = int(logits[0, -1].argmax().item())
            ns, reward, term, trunc, _ = env.step(action)
            total += reward
            actions[-1] = action                                   # fill in the action we just took
            states.append(ns); actions.append(0)
            rtgs.append(max(rtgs[-1] - reward / RTG_SCALE, 0.0)); times.append(min(tstep + 1, 499))
            state = ns
            if term or trunc:
                break
        return total

    test_env = gym.make(env_id)
    evals = [run_episode(test_env) for _ in range(20)]
    test_env.close()
    print(f"Evaluation over 20 runs (target={target_return:.0f}): mean_reward {np.mean(evals):.2f} | std {np.std(evals):.2f}")

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        mc.plot_learning_curve(evals, os.path.join(save_dir, f"dt_{env_id.lower()}_curve.png"), window=5)
        # GIF: one rendered rollout
        try:
            render_env = gym.make(env_id, render_mode="rgb_array")
            frames = []
            run_episode(render_env, render_frames=frames)
            render_env.close()
            if frames:
                frames[0].save(os.path.join(save_dir, f"dt_{env_id.lower()}_agent.gif"),
                               save_all=True, append_images=frames[1:], duration=25, loop=0)
                print(f"Saved agent rendering simulation GIF -> dt_{env_id.lower()}_agent.gif")
        except Exception as e:
            print(f"(skipping GIF: {e})")


if __name__ == "__main__":
    main()
