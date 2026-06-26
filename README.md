# IDSS — Intelligent Decision Support System

A multi-domain scheduling application for cloud, manufacturing, and logistics workloads.

## Structure
- `api/app.py` — FastAPI application and websocket dashboard endpoint
- `simulation_engine.py` — live simulation engine for scheduling and anomaly detection
- `models/optimisation/ilp_solver.py` — ILP-based scheduler
- `models/drl/ppo_agent.py` — PPO reinforcement learning training and inference
- `models/predictive/gbr_model.py` — workload forecasting with Gradient Boosting Regressor
- `kg/kg_validator.py` — knowledge graph validation rules
- `adapters/` — domain adapters for task/resource mapping
- `data/` — dataset generation and preprocessing utilities

## Running locally
1. Create a Python virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Start the app:
   ```bash
   python -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
   ```
3. Open the dashboard:
   - `http://127.0.0.1:8000/dashboard`

## Deploying to Render
- Use the repository root as the service source.
- Set the build command to:
  ```bash
  pip install -r requirements.txt
  ```
- Set the start command to:
  ```bash
  uvicorn api.app:app --host 0.0.0.0 --port $PORT
  ```

## Notes
- `data/processed/` and trained model artifacts are ignored in Git.
- If you want to train the PPO or GBR models, run the sample scripts in the repo root.
