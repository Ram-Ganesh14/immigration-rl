"""
FastAPI server exposing the OpenEnv interface:
  POST /reset
  POST /step
  GET  /state
  GET  /tasks
  POST /grade
  GET  /health
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
        "and makes clear/hold/deny/escalate decisions under time pressure."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (single-session)
env = ImmigrationEnvironment()


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
    return {"status": "ok", "environment": "airport-immigration-env", "version": "1.0.0"}


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


@app.get("/")
def root():
    return {
        "name": "Airport Immigration Processing Environment",
        "openenv": True,
        "tasks": VALID_TASKS,
        "endpoints": ["/reset", "/step", "/state", "/grade", "/tasks", "/health"],
        "docs": "/docs",
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)
