"""
FastAPI server exposing the OpenEnv interface:
  POST /reset
  POST /step
  GET  /state
  GET  /tasks
  POST /grade
  GET  /health
  GET  /explain
  GET  /dashboard
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

from models.models import ImmigrationAction, StepResult, ResetResult, EpisodeState
from server.environment import ImmigrationEnvironment, TASK_CONFIGS, VALID_TASKS
from graders.graders import run_grader

# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Airport Immigration Processing Environment",
    description=(
        "OpenEnv environment simulating airport immigration officer decisions. "
        "An AI agent processes passengers, verifies documents, detects anomalies, "
        "queries INTERPOL/biometric systems and immigration policy knowledge base, "
        "and makes clear/hold/deny/escalate decisions under time pressure. "
        "Features demographic fairness tracking, system disruption simulation, "
        "and decision explainability."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (single-session)
env = ImmigrationEnvironment()

# Mount dashboard static files if directory exists
DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard")
if os.path.isdir(DASHBOARD_DIR):
    app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="dashboard_static")


# ─── Request/Response schemas ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task1_document_check"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: ImmigrationAction


class GradeRequest(BaseModel):
    task_id: Optional[str] = None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "environment": "airport-immigration-env", "version": "2.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": task_id,
                **config,
            }
            for task_id, config in TASK_CONFIGS.items()
        ]
    }


@app.post("/reset", response_model=ResetResult)
def reset(request: ResetRequest = ResetRequest()):
    try:
        result = env.reset(task_id=request.task_id, seed=request.seed)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


@app.post("/step", response_model=StepResult)
def step(request: StepRequest):
    try:
        result = env.step(request.action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {e}")


@app.get("/state", response_model=EpisodeState)
def state():
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/grade")
def grade():
    try:
        s = env.state()
        result = run_grader(s.model_dump())
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grading failed: {e}")


@app.get("/explain")
def explain():
    """Returns feature attribution / explainability info for the last decision made."""
    try:
        info = env.get_last_decision_info()
        if info is None:
            return {
                "message": "No decision has been made yet. Process a passenger first.",
                "passenger_id": None,
            }
        return info
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Serve the live monitoring dashboard."""
    dashboard_path = os.path.join(DASHBOARD_DIR, "index.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Dashboard not found. Place index.html in /dashboard/</h1>", status_code=404)


@app.get("/")
def root():
    return {
        "name": "Airport Immigration Processing Environment",
        "openenv": True,
        "version": "2.0.0",
        "tasks": VALID_TASKS,
        "features": [
            "Hidden information APIs (INTERPOL, biometrics)",
            "Policy knowledge base (RAG search)",
            "Demographic fairness tracking (nationality + gender + intersectional)",
            "System disruption simulation (API outages, passenger surges)",
            "Decision explainability",
            "Live monitoring dashboard",
        ],
        "endpoints": ["/reset", "/step", "/state", "/grade", "/tasks", "/health", "/explain", "/dashboard"],
        "docs": "/docs",
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
