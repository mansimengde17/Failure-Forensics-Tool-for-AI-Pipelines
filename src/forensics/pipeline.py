"""The 4-step document pipeline under observation.

Steps: intake -> extraction -> classification -> summarization. Each step is
an isolated, typed function wrapped with the tracing decorator. The simulated
LLM behind extraction and classification carries realistic failure modes so
there are genuine failures to trace:

- documents containing the marker phrase "per our conversation" trigger an
  extraction hallucination (an entity that is not in the source),
- documents that mix invoice and report vocabulary get misclassified,
- documents with no dates produce a low-confidence extraction.
"""

from __future__ import annotations

import re

from .tracing import Trace, instrumented

CATEGORY_HINTS = {
    "contract": ["agreement", "party", "hereinafter", "term", "clause",
                 "governing law"],
    "invoice": ["invoice", "amount due", "payment terms", "bill to", "total"],
    "report": ["findings", "quarter", "analysis", "metrics", "summary of",
               "results"],
    "correspondence": ["dear", "regards", "sincerely", "writing to"],
}


@instrumented("intake")
def intake(payload: dict):
    text = payload["raw_text"].strip()
    output = {"text": text, "char_count": len(text),
              "line_count": text.count("\n") + 1}
    return output, "normalize document", text[:80], 5


@instrumented("extraction")
def extraction(payload: dict):
    text = payload["text"]
    prompt = ("Extract structured entities (names, dates, amounts, key terms)"
              " from the document. Return JSON.")
    names = re.findall(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", text)
    dates = re.findall(
        r"\b(?:\d{4}-\d{2}-\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct"
        r"|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b", text)
    amounts = re.findall(r"(?:USD|EUR|GBP|\$|€)\s?[\d,]+(?:\.\d{2})?", text)

    confidence = 4
    if "per our conversation" in text.lower():
        # Injected failure mode: hallucinated entity not present in source.
        names = names + ["John Smith"]
        confidence = 4  # the model is confidently wrong, as they often are
    if not dates:
        confidence = 2

    output = {"names": sorted(set(names)), "dates": dates,
              "amounts": amounts,
              "currencies": sorted(set(re.findall(r"USD|EUR|GBP|\$|€", text)))}
    raw = str(output)
    return output, prompt, raw, confidence


@instrumented("classification")
def classification(payload: dict):
    text = payload.get("_source_text", "")
    prompt = ("Classify the document as contract, invoice, report, or"
              " correspondence.")
    scores = {category: sum(text.lower().count(hint) for hint in hints)
              for category, hints in CATEGORY_HINTS.items()}
    best = max(scores, key=scores.get)
    ranked = sorted(scores.values(), reverse=True)
    ambiguous = len(ranked) > 1 and ranked[0] - ranked[1] <= 1

    confidence = 2 if ambiguous else 5
    if scores["invoice"] >= 2 and scores["report"] >= 2:
        # Injected failure mode: documents mixing invoice and report
        # vocabulary get the label the evidence does not favor, with
        # misplaced confidence.
        best = "invoice" if scores["report"] >= scores["invoice"] else "report"
        confidence = 3

    output = {"doc_type": best, "scores": scores}
    return output, prompt, str(output), confidence


@instrumented("summarization")
def summarization(payload: dict):
    entities = payload["entities"]
    doc_type = payload["doc_type"]
    prompt = (f"Write a structured summary tailored to a {doc_type}."
              " Mention the key entities.")
    names = ", ".join(entities["names"]) or "no named parties"
    dates = ", ".join(entities["dates"]) or "no dates found"
    amounts = ", ".join(entities["amounts"]) or "no amounts"
    summary = (f"This {doc_type} involves {names}. Key dates: {dates}."
               f" Amounts referenced: {amounts}.")
    confidence = 4 if entities["names"] else 3
    output = {"summary": summary}
    return output, prompt, summary, confidence


def run_pipeline(raw_text: str) -> Trace:
    trace = Trace()
    try:
        normalized = intake(trace, {"raw_text": raw_text})
        entities = extraction(trace, {"text": normalized["text"]})
        doc_type = classification(
            trace, {"entities": entities, "_source_text": normalized["text"]})
        final = summarization(
            trace, {"entities": entities, "doc_type": doc_type["doc_type"]})
        trace.final_output = {"doc_type": doc_type["doc_type"],
                              "entities": entities,
                              "summary": final["summary"]}
        low_confidence = any(s.confidence <= 2 for s in trace.spans)
        trace.status = "degraded" if low_confidence else "success"
    except RuntimeError:
        trace.status = "failure"
    return trace
