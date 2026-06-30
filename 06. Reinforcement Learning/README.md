# 06. Reinforcement Learning — Sequential Decision Making

This directory demonstrates reinforcement learning (RL) paradigms, tracing from random controls and classical tabular methods through value-based deep RL, policy gradients, actor-critic algorithms, model-based planning, off-policy continuous control, exploration bonuses, gradient-free search, and RL-as-sequence-modeling.

All algorithms are tested on Classic Control environments from the Farama Foundation `gymnasium`:
1.  **CartPole-v1:** Dense reward (+1 per upright timestep). Easy for standard algorithms.
2.  **MountainCar-v0:** Sparse reward (−1 per step until the flag). Exploration is hard — random steps rarely reach the goal (a good test for RND).
3.  **Pendulum-v1:** *Continuous* action (a torque in [−2, 2]). Used by SAC to demonstrate continuous control.

---

## Utility Module: `rl_common.py`

Every script imports `rl_common.py` as `mc`. It handles:
*   Wrapping `gymnasium` environments.
*   Implementing shared Experience Replay buffers for DQN models.
*   Standardizing evaluation loops (averaging episode reward over 10-100 test runs).
*   Recording and saving MP4/GIF videos of trained agents performing the tasks.
*   Plotting and smoothing learning curves (cumulative reward vs. training episodes).

---

## The Catalog of Scripts

The scripts trace the chronological and theoretical progression of RL:

### 01. Baselines (`01.random-heuristic.py`)
*   **Description:** Benchmarks that require no learning loops.
*   **Models:**
    *   *Random:* Pick actions uniformly.
    *   *Heuristic:* A simple, hard-coded rule (e.g., in CartPole, push left if the pole tilts left; in MountainCar, push in the direction of velocity).
*   **Educational Takeaway:** Setting the bare minimum benchmarks and showing how simple physics heuristics compare to learned strategies.

### 02. Tabular Q-Learning (`02.q-learning.py`)
*   **Description:** Classical value-based method using a discrete lookup table.
*   **Method:** States are continuous, so they are discretized into small buckets. The agent updates a table of Q-values using the Bellman Equation: 
    $$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$
*   **Educational Takeaway:** The foundation of value iteration, and experiencing the "curse of dimensionality" when discretization scales.

### 03. Deep Q-Network (`03.dqn.py`)
*   **Description:** Deep value-based learning (Mnih et al., 2013/2015).
*   **Architecture:** A neural network acts as a function approximator to predict $Q(s, a)$. Key additions:
    *   *Experience Replay Buffer:* To break correlation between sequential states.
    *   *Target Q-Network:* To stabilize updates (fixed $Q$-targets updated periodically).
*   **Educational Takeaway:** Learning how deep networks resolve tabular scaling limits, and seeing how target networks resolve training oscillation.

### 04. Policy Gradients (`04.reinforce.py`)
*   **Description:** Monte Carlo Policy Gradient (Williams, 1992).
*   **Method:** REINFORCE. Directly updates the policy parameters $\theta$ (mapping states to probability distributions of actions) by scaling gradients of chosen actions by the discounted cumulative rewards of the episode.
*   **Educational Takeaway:** Shifting from predicting state action values (Q-learning) to directly optimizing the action strategy (Policy).

### 05. Advantage Actor-Critic (`05.actor-critic-a2c.py`)
*   **Description:** Hybrid value and policy network (A2C).
*   **Architecture:** Two heads (or networks):
    *   *Actor:* Learns the policy distribution.
    *   *Critic:* Estimates the state value function $V(s)$ to calculate the "advantage" $A(s, a) = Q(s, a) - V(s)$, reducing the high variance of policy gradients.
*   **Educational Takeaway:** Seeing how combining value-based and policy-based methods dramatically improves learning stability.

### 06. Proximal Policy Optimization (`06.ppo.py`)
*   **Description:** Clipped surrogate objective policy optimization (Schulman et al., 2017).
*   **Method:** Limits policy updates using a clipped ratio $\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)$ to ensure updates do not ruin policy stability.
*   **Educational Takeaway:** Understanding the state-of-the-art on-policy RL algorithm used to align modern LLMs (RLHF).

### 07. SARSA (`07.sarsa.py`)
*   **Description:** On-policy tabular control — the twin of Q-learning.
*   **Method:** Bootstraps from the action actually taken (`Q(s',a')`) instead of the greedy max, so it learns the value of the exploring policy.
*   **Educational Takeaway:** On-policy vs off-policy in one line of difference; SARSA learns "safer" policies under exploration.

### 08. Dyna-Q (`08.dyna-q.py`)
*   **Description:** Tabular *model-based* RL (Sutton).
*   **Method:** Learns a model `(s,a) → (r,s')` and runs N planning updates on recalled transitions after each real step (`--variant N` sets planning steps).
*   **Educational Takeaway:** Planning with a learned model is dramatically more sample-efficient than model-free Q-learning (visible in far fewer episodes).

### 09. Evolution Strategies (`09.evolution-strategies.py`)
*   **Description:** Gradient-free policy optimization (Salimans et al., 2017).
*   **Method:** Perturb policy weights with Gaussian noise (antithetic pairs), evaluate episodic returns, move weights toward high-scoring directions. No backprop. ("Episodes" = ES generations; `--variant` sets population size.)
*   **Educational Takeaway:** Black-box optimization of the return — robust to sparse/non-differentiable rewards, trivially parallel.

### 10. Rainbow-lite DQN (`10.rainbow-dqn.py`)
*   **Description:** The headline DQN improvements, selectable to isolate each.
*   **Variants:** `double` (overestimation fix), `dueling` (value/advantage streams), `per` (prioritized replay), `rainbow` (all three, default).
*   **Educational Takeaway:** How each Rainbow component stabilizes and accelerates value-based deep RL.

### 11. RND Exploration (`11.rnd-exploration.py`)
*   **Description:** Curiosity-driven exploration via Random Network Distillation (Burda et al., 2018), on a DQN backbone.
*   **Method:** Intrinsic reward = prediction error of a predictor network trying to match a frozen random target; novel states give a bonus. Best on **MountainCar-v0**, which vanilla DQN/REINFORCE typically can't solve.
*   **Educational Takeaway:** Decoupling exploration from the (absent) extrinsic reward.

### 12. Soft Actor-Critic (`12.sac.py`)
*   **Description:** Off-policy, maximum-entropy actor-critic for **continuous** control on Pendulum-v1 (Haarnoja et al., 2018).
*   **Architecture:** Tanh-squashed Gaussian actor, twin critics (min for overestimation), soft target updates, and an automatically tuned entropy temperature.
*   **Educational Takeaway:** The continuous-control branch of RL; "succeed while acting as randomly as possible," with high sample efficiency.

### 13. Decision Transformer (`13.decision-transformer.py`)
*   **Description:** Offline RL as sequence modeling (Chen et al., 2021).
*   **Method:** Collect an offline dataset, then train a causal Transformer on `(return-to-go, state, action)` token sequences; at test time condition on a high target return and let it emit actions. (`--variant` sets the eval target return.)
*   **Educational Takeaway:** RL with no value function or Bellman backup — just a GPT, tying RL to the Language Modeling folder.

---

## Expected Performance & Comparisons

Running these scripts produces performance curves and rollout animation GIFs. Comparing them highlights the core challenges and milestones in RL history:

### 1. Tabular Q-Learning vs. Deep Q-Networks (DQN)
*   **The Curse of Dimensionality:** If you discretize the 4-dimensional CartPole state space too finely (e.g., 10,000 buckets), Tabular Q-Learning (`02.q-learning.py`) fails to learn, hovering around a mean reward of $\sim 23$. This is because Q-learning updates cells independently; with too many buckets, the agent never visits the same cell twice. We coarsen the grid to 72 buckets to force cell overlap.
*   **Deep Function Approximation:** DQN (`03.dqn.py`) solves this by replacing the lookup table with a neural network. The network naturally generalizes across adjacent continuous states without discretization, allowing it to predict optimal actions for states it has never explicitly visited.

### 2. Value-Based (DQN) vs. Policy-Based (REINFORCE)
*   **Optimization Target:** DQN estimates the value of every action ($Q$-values) and acts greedily. REINFORCE (`04.reinforce.py`) directly models the probability distribution of actions.
*   **Parameter Efficiency:** REINFORCE learns much faster and requires a fraction of the parameter count (898 parameters vs. DQN's 4,610) to reach a near-perfect reward ($\sim 500$). This is because modeling the boundary of what actions to take is mathematically simpler than fitting exact cumulative value metrics over all states.

### 3. Variance and Stability: REINFORCE vs. Actor-Critic (A2C & PPO)
*   **Gradient Variance:** REINFORCE uses Monte Carlo estimates (actual cumulative episode returns) to update weights. If one step in an episode has an unlucky random transition, the gradient update is skewed. This creates high training variance.
*   **Critic Stabilization:** Actor-Critic (`05.actor-critic-a2c.py` and `06.ppo.py`) solves this by using a Critic network to estimate the baseline state value $V(s)$. By scaling updates with the *Advantage* ($G_t - V(s)$), we measure how much better the action performed compared to average expectations, reducing variance and smoothing the learning curve.
*   **Trust Region Updates:** PPO enforces a clipped ratio constraint to ensure policy updates do not diverge too far in a single step, preventing the agent from "forgetting" how to balance the pole during training updates.

### 4. Visualizing Rollouts (GIFs)
*   Compare `cartpole-v1_heuristic_baseline.gif`, `qlearning_cartpole-v1_agent.gif`, and `reinforce_cartpole-v1_agent.gif`.
*   The baseline heuristic tries to balance the pole but often wobbles out of control.
*   The trained agents (DQN, REINFORCE, and Actor-Critic) display a rock-solid, centered balance, demonstrating active stabilization learned purely through environmental interaction.

