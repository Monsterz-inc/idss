"""
idss/models/drl/ppo_agent.py
-----------------------------
PPO agent trainer and inference wrapper.
Uses Stable-Baselines3 PPO on the IDSSSchedulingEnv.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from models.drl.drl_env import IDSSSchedulingEnv

MODEL_DIR  = os.path.join(os.path.dirname(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_idss.zip")


# ── Training progress callback ────────────────────────────────────
class ProgressCallback(BaseCallback):

    def __init__(self, total_steps, print_every=10000):
        super().__init__()
        self.total_steps  = total_steps
        self.print_every  = print_every
        self.episode_rewards = []
        self.current_rewards = 0.0

    def _on_step(self) -> bool:
        self.current_rewards += self.locals["rewards"][0]
        if self.num_timesteps % self.print_every == 0:
            pct = 100 * self.num_timesteps / self.total_steps
            print(f"    [{pct:5.1f}%] Steps: {self.num_timesteps:>7,} "
                  f"| Reward so far: {self.current_rewards:>10.2f}")
        return True


def build_env(df, adapter, n_tasks=100, n_resources=10, forecast=None):
    """Build a fresh IDSSSchedulingEnv from a DataFrame."""
    sample      = df.sample(n=min(n_tasks, len(df)), random_state=42)
    tasks       = [adapter.map_task(row) for row in sample.to_dict("records")]
    machine_ids = sample["machine_id"].unique()[:n_resources]
    resources   = [
        adapter.map_resource({"machine_id": mid})
        for mid in machine_ids
    ]
    return IDSSSchedulingEnv(tasks, resources, forecast=forecast)


def train(df, adapter, total_timesteps=200_000, n_tasks=100,
          n_resources=10, forecast=None):
    """
    Train the PPO agent on the scheduling environment.
    Saves the trained model to disk.
    """
    print("  Building environment ...")
    env = build_env(df, adapter, n_tasks, n_resources, forecast)

    print("  Checking environment ...")
    check_env(env, warn=True)

    print(f"  Training PPO for {total_timesteps:,} timesteps ...")
    print(f"  Observation space : {env.observation_space.shape}")
    print(f"  Action space      : {env.action_space.n} actions")
    print()

    model = PPO(
        policy             = "MlpPolicy",
        env                = env,
        learning_rate      = 3e-4,
        n_steps            = 1024,
        batch_size         = 64,
        n_epochs           = 10,
        gamma              = 0.99,
        gae_lambda         = 0.95,
        clip_range         = 0.2,
        ent_coef           = 0.01,
        verbose            = 0,
        policy_kwargs      = dict(net_arch=[64, 64]),
    )

    callback = ProgressCallback(total_timesteps, print_every=20000)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save(MODEL_PATH)
    print(f"\n  Model saved to {MODEL_PATH}")
    return model


def load():
    """Load a previously trained PPO model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}.\n"
            "Run train() first."
        )
    return PPO.load(MODEL_PATH)


def run_episode(model, df, adapter, n_tasks=50,
                n_resources=5, forecast=None):
    """
    Run one full episode using the trained PPO policy.
    Returns the schedule and performance metrics.
    """
    env  = build_env(df, adapter, n_tasks, n_resources, forecast)
    obs, _ = env.reset()

    done = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += reward
        done = terminated or truncated

    results = env.get_results()
    results["total_reward"] = round(total_reward, 3)
    return results