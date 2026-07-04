"""FastAPI service: process documents, explore traces, flag failures."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .analyzer import diagnose
from .feedback import EvalDataset
from .pipeline import run_pipeline
from .tracing import Trace, TraceStore

app = FastAPI(title="Failure Forensics Tool", version="1.0.0")
store = TraceStore()
dataset = EvalDataset()


class ProcessRequest(BaseModel):
    text: str


class FlagRequest(BaseModel):
    corrected_output: str | None = None


@app.post("/v1/process")
def process(request: ProcessRequest):
    trace = run_pipeline(request.text)
    store.save(trace)
    return {"trace_id": trace.trace_id, "status": trace.status,
            "final_output": trace.final_output}


@app.get("/v1/traces")
def list_traces():
    return store.list_traces()


@app.get("/v1/traces/{trace_id}")
def get_trace(trace_id: str):
    trace = store.load(trace_id)
    if trace is None:
        raise HTTPException(404, "trace not found")
    return trace


@app.post("/v1/traces/{trace_id}/flag")
def flag_trace(trace_id: str, request: FlagRequest):
    raw = store.load(trace_id)
    if raw is None:
        raise HTTPException(404, "trace not found")
    diagnosis = diagnose(raw)
    raw["flagged"] = True
    raw["diagnosis"] = diagnosis
    trace = Trace(**{k: raw[k] for k in ("trace_id", "timestamp",
                                         "final_output", "status")})
    trace.flagged = True
    trace.diagnosis = diagnosis
    trace.spans = []  # spans already serialized in the JSON file
    store.save(trace)
    case = dataset.add_case(raw, diagnosis, request.corrected_output)
    return {"diagnosis": diagnosis, "eval_case_created": case["trace_id"]}


@app.get("/v1/analytics")
def analytics():
    base = store.analytics()
    base["eval_dataset_size"] = len(dataset.cases())
    return base
