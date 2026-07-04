#!/usr/bin/env python3
"""Offline end-to-end demonstration: trace documents, diagnose failures,
grow the eval dataset."""

from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, "src")

from forensics.analyzer import diagnose
from forensics.feedback import EvalDataset
from forensics.pipeline import run_pipeline
from forensics.tracing import TraceStore


def main() -> None:
    for path in ("traces.db", "data/eval_dataset.jsonl"):
        if os.path.exists(path):
            os.remove(path)
    shutil.rmtree("traces", ignore_errors=True)

    store = TraceStore()
    dataset = EvalDataset()

    with open("data/sample_documents.json", encoding="utf-8") as fh:
        documents = json.load(fh)["documents"]

    traces = []
    for doc in documents:
        trace = run_pipeline(doc["raw_text"])
        store.save(trace)
        traces.append((doc["id"], trace))

    healthy = sum(t.status == "success" for _, t in traces)
    print(f"Processed {len(traces)} documents:"
          f" {healthy} success, {len(traces) - healthy} degraded/failed\n")

    # A human reviewing outputs flags the bad ones; here we flag any trace
    # that is degraded or whose diagnosis finds a step with a quality drop.
    for doc_id, trace in traces:
        raw = trace.to_dict()
        diagnosis = diagnose(raw)
        genuinely_bad = any(s["quality"] < 0.7 for s in diagnosis["step_scores"])
        if trace.status == "success" and not genuinely_bad:
            continue
        trace.flagged = True
        trace.diagnosis = diagnosis
        store.save(trace)
        dataset.add_case(raw, diagnosis)
        print(f"[{doc_id}] trace {trace.trace_id}: {trace.status}")
        print(f"  root cause: {diagnosis['root_cause_step']}"
              f" ({diagnosis['category']})")
        print(f"  {diagnosis['explanation']}\n")

    analytics = store.analytics()
    print("Analytics:")
    print(f"  failure rate: {analytics['failure_rate']:.1%}")
    print(f"  failures by category: {analytics['failures_by_category']}")
    print(f"  failures by step: {analytics['failures_by_step']}")
    print(f"  eval dataset: {len(dataset.cases())} regression cases")

    regression = dataset.regression_check(run_pipeline, diagnose)
    print(f"  regression check: {regression['resolved']} of"
          f" {regression['total_known_issues']} known issues resolved")


if __name__ == "__main__":
    main()
