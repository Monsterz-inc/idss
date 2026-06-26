import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from adapters.cloud_adapter import CloudAdapter
from models.drl.ppo_agent import train

df      = pd.read_parquet(os.path.join("data", "processed", "synthetic_all.parquet"))
adapter = CloudAdapter()
forecast = {"arrival_rate": 250, "mean_duration": 45}

print("Retraining PPO agent with updated reward function ...")
train(df, adapter, total_timesteps=200000, n_tasks=100, n_resources=10, forecast=forecast)
print("Done.")