"""
01. Random & Heuristic Baselines
================================

Non-learning baseline controllers to establish benchmark limits.

Heuristics:
    * CartPole-v1: Push in the direction the pole is tilting (tilt left -> push left; tilt right -> push right).
    * MountainCar-v0: Push in the direction of velocity to build momentum (velocity left -> push left; velocity right -> push right).

Run:
    python "01.random-heuristic.py" --env CartPole-v1 --variant heuristic
    python "01.random-heuristic.py" --env MountainCar-v0 --variant random
"""

import os
import random
import numpy as np
import rl_common as mc


def main():
    p = mc.build_argparser("Random & Heuristic Baselines")
    p.add_argument("--baseline-variant", type=str, default="heuristic", choices=["random", "heuristic"])
    args = p.parse_args()

    variant = args.variant or args.baseline_variant
    if variant not in ["random", "heuristic"]:
        variant = "heuristic"

    env_id = args.env

    # Define actions mapping
    if variant == "random":
        import gymnasium as gym
        temp_env = gym.make(env_id)
        act_fn = lambda state: temp_env.action_space.sample()
        model_name = "Random-Agent"
    else:
        # Heuristic rules
        if env_id == "CartPole-v1":
            # state: [position, velocity, angle, angular_velocity]
            act_fn = lambda state: 0 if state[2] < 0 else 1
            model_name = "CartPole-Heuristic"
        else: # MountainCar-v0
            # state: [position, velocity]
            # actions: 0=push left, 1=no push, 2=push right
            act_fn = lambda state: 0 if state[1] < 0 else (2 if state[1] > 0 else 1)
            model_name = "MountainCar-Heuristic"

    print(f"Evaluating {model_name} on {env_id}...")
    mean_reward, std_reward = mc.evaluate_agent(env_id, act_fn, num_episodes=args.episodes if args.episodes <= 50 else 50)
    print(f"Performance over episodes: mean_reward {mean_reward:.2f} | std_reward {std_reward:.2f}")

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        gif_path = os.path.join(save_dir, f"{env_id.lower()}_{variant}_baseline.gif")
        mc.save_agent_gif(env_id, act_fn, gif_path)


if __name__ == "__main__":
    main()
