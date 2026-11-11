"""HTTP API that scores one employer group with the champion model.

    MODEL_DIR=artifacts/model uvicorn eb_rampup.api:app --reload
"""

import os
from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from eb_rampup.serving import LoadedModel, apply_business_rules, load_model_dir


def load_from_env() -> LoadedModel:
    """MODEL_DIR for a local model; otherwise download the champion."""
    if model_dir := os.environ.get("MODEL_DIR"):
        return load_model_dir(model_dir)
    from eb_rampup.registry import load_champion

    return load_champion(
        os.environ["GCP_PROJECT"],
        os.environ.get("GCP_REGION", "us-central1"),
        os.environ.get("MODEL_NAME", "loss-ratio-ltd"),
    )

@asynccontextmanager #(holds a next() call for when uvicorn disconnects)
async def lifespan(app: FastAPI): # async allow other fns to run while this is paused in the background
    print("start app")
    app.state.model = load_from_env()  # once per container start, not per request
    print("model loaded")
    yield

# Tool	                Does
# Uvicorn (the server)	Opens network port 8080, accepts connections, reads raw HTTP bytes and turns them into requests, sends responses back, and manages startup and shutdown
# FastAPI (the framework)	Routes requests to your functions, validates input with Pydantic, injects dependencies, and turns your return values into JSON

# Testing note
# Situation                                 Code after yield, when an error occurs
# pytest fixture	                        Runs. pytest resumes with next().
# @contextmanager / @asynccontextmanager	Skipped, unless it’s in finally. The wrapper resumes with throw().


# Parameter looks like...	                        FastAPI fills it from... 
# A name in the URL path, like /groups/{group_id}	That part of the URL if @app.get("/groups/{group_id}")
# A simple type (int, str, float) not in the path	The query string: /score?threshold=1.5 @app.get("/score") [infers params]
# A Pydantic model (a BaseModel subclass)	        The JSON request body if @app.post("/score") default is to take entire baseClass. can take subdirectories if key is {"group": {"industry": "Manufacturing", ...}, "options": {"explain": true}}
# Depends(...)	                                    Calling that dependency - means “run this reusable function before the endpoint, and inject its result

# Start server with: MODEL_DIR=artifacts/model uvicorn eb_rampup.api:app --reload

app = FastAPI(title="Loss ratio API", lifespan=lifespan) # init app, give app reference to fn to load model

class Group(BaseModel): # BaseModels come from request body
    group_id: str = "request"
    industry: str
    region: str
    group_size: int = Field(ge=2, le=20_000)
    avg_age: float = Field(ge=18, le=70)
    pct_female: float = Field(ge=0, le=1)
    avg_salary: float = Field(gt=0)
    elimination_days: int = Field(gt=0)
    benefit_pct: float = Field(gt=0, le=1)
    premium_y1: float = Field(gt=0)
    premium_y2: float = Field(gt=0)
    premium_y3: float = Field(gt=0)
    claims_y1: float = Field(ge=0)
    claims_y2: float = Field(ge=0)
    claims_y3: float = Field(ge=0)

class Score(BaseModel):
    predicted_loss_ratio: float
    refer_to_underwriter: bool
    refer_reason: str
    model_version: str


def get_model(request: Request) -> LoadedModel:
    return request.app.state.model

ModelDep = Annotated[LoadedModel, Depends(get_model)] # ModelDep is a type, with a value of Depends(get_model)


@app.get("/health")
def health(model: ModelDep) -> dict:
    return {"status": "ok", "model_version": model.version}

@app.post("/score")
def score(group: Group, model: ModelDep) -> Score:
    df = pd.DataFrame([group.model_dump()])
    try:
        predictions = model.predict(df)
    except ValueError as e:  # bad input the schema can't catch, e.g. an unknown industry
        raise HTTPException(status_code=422, detail=str(e)) from e
    row = apply_business_rules(df, predictions).iloc[0]
    return Score(
        predicted_loss_ratio=float(row["predicted_loss_ratio"]),
        refer_to_underwriter=bool(row["refer_to_underwriter"]),
        refer_reason=row["refer_reason"],
        model_version=model.version,
    )