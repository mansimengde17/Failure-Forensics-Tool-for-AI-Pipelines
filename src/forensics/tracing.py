"""Tracing layer: spans, traces, instrumentation decorator, and storage."""

from __future__ import annotations

import functools
import json
import os
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field


@dataclass
class Span:
    step_name: str
    input_data: dict
    output_data: dict
    prompt: str
    raw_response: str
    tokens: int
    latency_s: float
    confidence: int          # 1-5, self-reported by the model at each step
    error: str | None = None


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: f"tr-{uuid.uuid4().hex[:8]}")
    timestamp: float = field(default_factory=time.time)
    spans: list[Span] = field(default_factory=list)
    final_output: dict = field(default_factory=dict)
    status: str = "success"   # success | degraded | failure
    flagged: bool = False
    diagnosis: dict | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        return data


def instrumented(step_name: str):
    """Decorator that wraps a pipeline step and records a full span.

    The wrapped step returns (output_dict, prompt, raw_response, confidence).
    Instrumenting a new step is one line of code.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(trace: Trace, payload: dict, *args, **kwargs):
            start = time.perf_counter()
            error = None
            try:
                output, prompt, raw, confidence = func(payload, *args, **kwargs)
            except Exception as exc:  # capture, record, re-raise
                output, prompt, raw, confidence = {}, "", "", 1
                error = f"{type(exc).__name__}: {exc}"
            latency = time.perf_counter() - start
            trace.spans.append(Span(
                step_name=step_name,
                input_data=payload,
                output_data=output,
                prompt=prompt,
                raw_response=raw,
                tokens=len(prompt.split()) + len(raw.split()),
                latency_s=round(latency, 4),
                confidence=confidence,
                error=error,
            ))
            if error:
                raise RuntimeError(error)
            return output
        return wrapper
    return decorator


class TraceStore:
    """JSON trace files plus a queryable SQLite index."""

    def __init__(self, db_path: str = "traces.db", trace_dir: str = "traces"):
        self.trace_dir = trace_dir
        os.makedirs(trace_dir, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS traces (
                   trace_id TEXT PRIMARY KEY, timestamp REAL,
                   status TEXT, flagged INTEGER, root_cause_step TEXT,
                   failure_category TEXT)""")
        self.conn.commit()

    def save(self, trace: Trace) -> None:
        path = os.path.join(self.trace_dir, f"{trace.trace_id}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(trace.to_dict(), fh, indent=2)
        diagnosis = trace.diagnosis or {}
        self.conn.execute(
            "INSERT OR REPLACE INTO traces VALUES (?, ?, ?, ?, ?, ?)",
            (trace.trace_id, trace.timestamp, trace.status,
             int(trace.flagged), diagnosis.get("root_cause_step"),
             diagnosis.get("category")))
        self.conn.commit()

    def load(self, trace_id: str) -> dict | None:
        path = os.path.join(self.trace_dir, f"{trace_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def list_traces(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT trace_id, timestamp, status, flagged, root_cause_step,"
            " failure_category FROM traces ORDER BY timestamp").fetchall()
        keys = ["trace_id", "timestamp", "status", "flagged",
                "root_cause_step", "failure_category"]
        return [dict(zip(keys, row)) for row in rows]

    def analytics(self) -> dict:
        by_category = dict(self.conn.execute(
            "SELECT failure_category, COUNT(*) FROM traces"
            " WHERE failure_category IS NOT NULL GROUP BY failure_category"))
        by_step = dict(self.conn.execute(
            "SELECT root_cause_step, COUNT(*) FROM traces"
            " WHERE root_cause_step IS NOT NULL GROUP BY root_cause_step"))
        total, failed = self.conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN status != 'success' THEN 1"
            " ELSE 0 END) FROM traces").fetchone()
        return {"total_traces": total or 0,
                "failed_traces": failed or 0,
                "failure_rate": round((failed or 0) / total, 3) if total else 0,
                "failures_by_category": by_category,
                "failures_by_step": by_step}
