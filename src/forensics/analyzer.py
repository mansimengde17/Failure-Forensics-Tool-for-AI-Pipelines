"""Backward trace analyzer: root cause identification and evidence chains."""

from __future__ import annotations

CATEGORIES = ("extraction_hallucination", "misclassification",
              "propagation_error", "prompt_failure", "context_loss")


def _step_quality(span: dict, source_text: str) -> tuple[float, str]:
    """Score whether a span's output is a reasonable transformation of its
    input (0-1) and explain any problem found.

    Offline this uses deterministic content checks; with live providers the
    same interface is backed by an LLM-as-judge scoring input/output pairs.
    """
    name = span["step_name"]
    output = span["output_data"]

    if name == "extraction":
        missing = [n for n in output.get("names", [])
                   if n.lower() not in source_text.lower()]
        if missing:
            return 0.2, (f"extracted entity '{missing[0]}' does not appear"
                         " in the source document")
        if not output.get("dates") and "20" in source_text:
            return 0.6, "possible dates in source were not extracted"
        return 0.95, ""

    if name == "classification":
        scores = output.get("scores", {})
        chosen = output.get("doc_type")
        if scores:
            top = max(scores, key=scores.get)
            if chosen != top:
                return 0.3, (f"classified as '{chosen}' although the evidence"
                             f" favored '{top}' (scores: {scores})")
            ranked = sorted(scores.values(), reverse=True)
            if len(ranked) > 1 and ranked[0] - ranked[1] <= 1:
                return 0.65, "ambiguous classification decided on thin margin"
        return 0.95, ""

    if name == "summarization":
        entities = span["input_data"].get("entities", {})
        summary = output.get("summary", "")
        dropped = [n for n in entities.get("names", []) if n not in summary]
        if dropped:
            return 0.5, f"summary dropped entity '{dropped[0]}' (context loss)"
        return 0.9, ""

    return 0.95, ""


def diagnose(trace: dict) -> dict:
    """Walk spans backward; the earliest significant quality drop is the
    root cause. Produces a categorized diagnosis with an evidence chain."""
    spans = trace["spans"]
    source_text = spans[0]["output_data"].get("text", "") if spans else ""

    scored = []
    for index, span in enumerate(spans):
        quality, problem = _step_quality(span, source_text)
        scored.append({"index": index, "step": span["step_name"],
                       "quality": quality, "problem": problem,
                       "confidence": span["confidence"]})

    suspects = [s for s in scored if s["quality"] < 0.7]
    if not suspects:
        low = min(scored, key=lambda s: (s["confidence"], s["quality"]))
        suspects = [low]
    root = min(suspects, key=lambda s: s["index"])
    root_span = spans[root["index"]]

    category = _categorize(root, root_span, scored)
    downstream = [s["step"] for s in scored
                  if s["index"] > root["index"] and s["quality"] < 0.9]
    evidence = {
        "root_step_input": _clip(root_span["input_data"]),
        "root_step_output": _clip(root_span["output_data"]),
        "problem": root["problem"] or "low self-reported confidence",
        "propagated_to": downstream,
    }
    explanation = (
        f"Step {root['index'] + 1} ({root['step']}) is the root cause:"
        f" {evidence['problem']}."
        + (f" The error propagated to: {', '.join(downstream)}."
           if downstream else ""))
    return {"root_cause_step": root["step"],
            "root_cause_index": root["index"],
            "category": category,
            "step_scores": scored,
            "evidence": evidence,
            "explanation": explanation}


def _categorize(root: dict, span: dict, scored: list[dict]) -> str:
    problem = root["problem"]
    if root["step"] == "extraction" and "does not appear" in problem:
        return "extraction_hallucination"
    if root["step"] == "classification":
        return "misclassification"
    if root["step"] == "summarization" and "dropped" in problem:
        return "context_loss"
    upstream_clean = all(s["quality"] >= 0.9 for s in scored
                         if s["index"] < root["index"])
    if root["index"] > 0 and upstream_clean:
        return "propagation_error"
    return "prompt_failure"


def _clip(data: dict, limit: int = 300) -> dict:
    clipped = {}
    for key, value in data.items():
        text = str(value)
        clipped[key] = text[:limit] + ("..." if len(text) > limit else "")
    return clipped
