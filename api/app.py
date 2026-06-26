"""
idss/api/app.py
----------------
FastAPI REST API + WebSocket for live IDSS dashboard.
Run with:
    C:\Python311\python.exe -m uvicorn api.app:app --reload
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import time

from adapters import get_adapter
from models.optimisation.ilp_solver import solve
from models.predictive.gbr_model import predict as gbr_predict
from models.drl.ppo_agent import load as load_ppo
from models.xai.xai_explainer import explain_schedule_decision
from kg.kg_validator import KnowledgeGraph
from schema import AbstractTask, AbstractResource
from simulation_engine import SimulationEngine

app = FastAPI(
    title       = "IDSS — Intelligent Decision Support System",
    description = "AI-powered resource allocation and task scheduling API",
    version     = "1.0.0",
)

# ── Global state ──────────────────────────────────────────────────
kg         = KnowledgeGraph()
engine     = SimulationEngine()
clients    = []   # connected WebSocket clients

try:
    ppo_model = load_ppo()
    PPO_READY = True
except FileNotFoundError:
    PPO_READY = False

# ── Request schemas ───────────────────────────────────────────────
class TaskRequest(BaseModel):
    task_id:       str
    duration:      float
    cpu_demand:    float
    memory_demand: float
    priority:      int   = 2
    deadline:      float = 99999.0
    arrival_time:  float = 0.0
    domain:        str   = "cloud"
    job_type:      str   = "batch"

class ResourceRequest(BaseModel):
    resource_id:     str
    cpu_capacity:    float = 1.0
    memory_capacity: float = 1.0
    cost_per_second: float = 0.002
    power_watts:     float = 300.0
    available:       bool  = True

class ScheduleRequest(BaseModel):
    tasks:     List[TaskRequest]
    resources: List[ResourceRequest]
    domain:    str            = "cloud"
    mode:      str            = "ilp"
    weights:   Optional[dict] = None

class OverrideRequest(BaseModel):
    task_id:     str
    resource_id: str

# ── Broadcast to all WebSocket clients ───────────────────────────
async def broadcast(data: dict):
    dead = []
    for ws in clients:
        try:
            await ws.send_text(json.dumps(data))
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.remove(ws)

# ── Background simulation loop ────────────────────────────────────
async def simulation_loop():
    while True:
        try:
            state = engine.tick_once()
            await broadcast(state)
        except Exception as e:
            print(f"Simulation error: {e}")
        await asyncio.sleep(2)

@app.on_event("startup")
async def startup():
    asyncio.create_task(simulation_loop())

# ── Dashboard HTML ────────────────────────────────────────────────
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    with open(os.path.join(os.path.dirname(__file__),
                           "dashboard.html"), "r") as f:
        return f.read()

# ── WebSocket endpoint ────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    try:
        # Send current state immediately on connect
        await websocket.send_text(json.dumps(engine.get_state()))
        while True:
            data = await websocket.receive_text()
            msg  = json.loads(data)
            if msg.get("action") == "override":
                result = engine.override_task(
                    msg["task_id"], msg["resource_id"]
                )
                await websocket.send_text(json.dumps({
                    "override_result": result
                }))
    except WebSocketDisconnect:
        clients.remove(websocket)

# ── REST endpoints ────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "system":    "IDSS",
        "version":   "1.0.0",
        "author":    "Nwanze Christian Uche — 22/11017",
        "status":    "online",
        "ppo_ready": PPO_READY,
        "dashboard": "http://127.0.0.1:8000/dashboard",
    }

@app.get("/health")
def health():
    return {"status": "healthy", "kg_rules": len(kg.get_rules())}

@app.post("/switch-domain")
def switch_domain(data: dict):
    domain = data.get("domain", "cloud")
    result = engine.switch_domain(domain)
    return result

@app.get("/rules")
def get_rules():
    return {"rules": kg.get_rules()}

@app.get("/state")
def get_state():
    return engine.get_state()

@app.post("/override")
def override(request: OverrideRequest):
    return engine.override_task(request.task_id, request.resource_id)

@app.post("/schedule")
def schedule(request: ScheduleRequest):
    t0 = time.time()
    tasks = [
        AbstractTask(
            task_id       = t.task_id,
            duration      = t.duration,
            cpu_demand    = t.cpu_demand,
            memory_demand = t.memory_demand,
            priority      = t.priority,
            deadline      = t.deadline,
            arrival_time  = t.arrival_time,
            domain        = t.domain,
            job_type      = t.job_type,
        )
        for t in request.tasks
    ]
    resources = [
        AbstractResource(
            resource_id     = r.resource_id,
            cpu_capacity    = r.cpu_capacity,
            memory_capacity = r.memory_capacity,
            cost_per_second = r.cost_per_second,
            power_watts     = r.power_watts,
            available       = r.available,
        )
        for r in request.resources
    ]
    weights    = request.weights or {"makespan": 0.5, "cost": 0.25, "energy": 0.25}
    result     = solve(tasks, resources, weights)
    validation = kg.validate_schedule(result["schedule"], tasks, resources)
    explanations = []
    task_map     = {t.task_id: t for t in tasks}
    resource_map = {r.resource_id: r for r in resources}
    for entry in result["schedule"][:5]:
        t = task_map.get(entry.task_id)
        r = resource_map.get(entry.resource_id)
        if t and r:
            explanations.append(explain_schedule_decision(t, r, entry))

    return {
        "status":          result["status"],
        "makespan":        result["makespan"],
        "total_cost":      result["total_cost"],
        "energy":          result["energy"],
        "solve_time":      result["solve_time"],
        "total_time":      round(time.time() - t0, 3),
        "tasks_scheduled": len(result["schedule"]),
        "validation":      {"valid": validation["valid"],
                            "invalid": validation["invalid"]},
        "schedule":        [
            {"task_id":     e.task_id,
             "resource_id": e.resource_id,
             "start_time":  e.start_time,
             "end_time":    e.end_time}
            for e in result["schedule"]
        ],
        "explanations": explanations,
    }

@app.post("/predict")
def predict(data: dict):
    scenario  = data.get("scenario", "mixed")
    data_path = os.path.join("data", "processed",
                             f"synthetic_{scenario}.parquet")
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404,
                            detail=f"Scenario '{scenario}' not found.")
    df       = pd.read_parquet(data_path)
    forecast = gbr_predict(df.tail(3000))
    return {"scenario": scenario, "forecast": forecast}