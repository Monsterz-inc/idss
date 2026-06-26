"""
idss/models/xai/xai_explainer.py
----------------------------------
LIME and SHAP explainability for IDSS scheduling decisions.
Works on both the GBR model and PPO agent outputs.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pickle
import lime
import lime.lime_tabular
import shap


GBR_ARRIVAL_PATH  = os.path.join(os.path.dirname(__file__),
                    '..', 'predictive', 'gbr_arrival.pkl')
GBR_DURATION_PATH = os.path.join(os.path.dirname(__file__),
                    '..', 'predictive', 'gbr_duration.pkl')
SCALER_PATH       = os.path.join(os.path.dirname(__file__),
                    '..', 'predictive', 'gbr_scaler.pkl')

FEATURE_NAMES = [
    "task_count_lag1", "task_count_lag2", "task_count_lag3",
    "mean_duration_lag1", "mean_duration_lag2", "mean_duration_lag3",
    "mean_cpu_lag1", "mean_cpu_lag2", "mean_cpu_lag3",
    "task_count_roll3", "mean_duration_roll3",
    "mean_cpu", "mean_mem", "fail_rate", "time_of_day",
]


def _load_gbr():
    with open(GBR_ARRIVAL_PATH,  "rb") as f: arrival  = pickle.load(f)
    with open(GBR_DURATION_PATH, "rb") as f: duration = pickle.load(f)
    with open(SCALER_PATH,       "rb") as f: scaler   = pickle.load(f)
    return arrival, duration, scaler


def explain_gbr_lime(X_train: np.ndarray, x_instance: np.ndarray,
                     model_type: str = "arrival") -> dict:
    """
    Generate LIME explanation for one GBR prediction.
    model_type: 'arrival' or 'duration'
    """
    arrival, duration, scaler = _load_gbr()
    model = arrival if model_type == "arrival" else duration

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data  = X_train,
        feature_names  = FEATURE_NAMES,
        mode           = "regression",
        random_state   = 42,
    )

    explanation = explainer.explain_instance(
        data_row       = x_instance,
        predict_fn     = model.predict,
        num_features   = 5,
    )

    prediction = model.predict(x_instance.reshape(1, -1))[0]
    top_features = explanation.as_list()

    fidelity = explanation.score   # R² of local linear model

    return {
        "model":       model_type,
        "prediction":  round(float(prediction), 3),
        "fidelity_r2": round(float(fidelity), 3),
        "top_features": [
            {"feature": f, "weight": round(w, 4)}
            for f, w in top_features
        ],
    }


def explain_gbr_shap(X_train: np.ndarray,
                     X_explain: np.ndarray,
                     model_type: str = "arrival") -> dict:
    """
    Generate SHAP explanation for GBR predictions.
    model_type: 'arrival' or 'duration'
    """
    arrival, duration, scaler = _load_gbr()
    model = arrival if model_type == "arrival" else duration

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)

    # Mean absolute SHAP value per feature
    mean_shap = np.abs(shap_values).mean(axis=0)
    ranked    = sorted(
        zip(FEATURE_NAMES, mean_shap),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    return {
        "model":        model_type,
        "top_features": [
            {"feature": f, "shap_value": round(float(v), 4)}
            for f, v in ranked[:5]
        ],
        "base_value": round(float(np.atleast_1d(explainer.expected_value)[0]), 3),
    }


def explain_schedule_decision(task, resource, schedule_entry) -> dict:
    """
    Generate a human-readable rule-based explanation for a
    specific scheduling decision (task → resource assignment).
    Used when LIME/SHAP cannot directly access the PPO policy.
    """
    reasons = []

    # Priority reasoning
    if task.priority >= 4:
        reasons.append("Task has high priority — scheduled immediately.")
    elif task.priority <= 2:
        reasons.append("Task has low priority — assigned to least loaded resource.")

    # Deadline reasoning
    slack = task.deadline - task.arrival_time
    if slack < task.duration * 1.5:
        reasons.append(f"Deadline is tight ({slack:.1f}s slack) — urgent assignment.")
    else:
        reasons.append(f"Deadline comfortable ({slack:.1f}s slack) — balanced assignment.")

    # Resource reasoning
    reasons.append(
        f"Resource {resource.resource_id} selected — "
        f"cost: {resource.cost_per_second:.4f}/s, "
        f"power: {resource.power_watts}W."
    )

    # Energy reasoning
    energy = task.duration * resource.power_watts / 3600.0
    reasons.append(f"Estimated energy for this task: {energy:.4f} kWh.")

    # SLA check
    completion = schedule_entry.end_time
    if completion > task.deadline:
        reasons.append(
            f"WARNING: Task completes at {completion:.1f}s, "
            f"after deadline {task.deadline:.1f}s — SLA violation."
        )
    else:
        reasons.append(
            f"Task completes at {completion:.1f}s, "
            f"within deadline {task.deadline:.1f}s — SLA met."
        )

    return {
        "task_id":     task.task_id,
        "resource_id": resource.resource_id,
        "reasons":     reasons,
        "sla_met":     schedule_entry.end_time <= task.deadline,
    }