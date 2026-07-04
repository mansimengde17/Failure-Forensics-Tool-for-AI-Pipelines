# Failure Forensics Tool for AI Pipelines

An observability layer for multi-step AI pipelines that traces every intermediate step, identifies exactly where failures originate when the final output is bad, and feeds flagged failures back into a growing evaluation dataset.

Live demo: https://mansimengde17.github.io/Failure-Forensics-Tool-for-AI-Pipelines/

## The problem

When a multi-step AI pipeline produces garbage, most teams have no idea which step broke. The final output is wrong, but was it the extraction step hallucinating an entity, the classifier mislabeling the document, or a downstream step misreading a correct upstream output? This tool answers "where did this go wrong?" in seconds instead of hours of manual prompt archaeology.

## Architecture

```
  document
     |
     v
 +--------+   +------------+   +----------------+   +---------------+
 | Intake |-->| Extraction |-->| Classification |-->| Summarization |
 +--------+   +------------+   +----------------+   +---------------+
     |              |                  |                    |
     +------- span: input, output, prompt, response, ------+
              tokens, latency, confidence (1-5)
                            |
                            v
              +---------------------------+
              |        Trace Store        |  JSON traces + SQLite index
              +---------------------------+
                            |
              flag as bad   v
              +---------------------------+
              |  Backward Trace Analyzer  |  walks spans in reverse,
              |  scores each step's       |  first significant quality
              |  input -> output quality  |  drop = root cause
              +---------------------------+
                            |
                            v
              +---------------------------+
              |  Failure taxonomy +       |  extraction_hallucination,
              |  evidence chain           |  misclassification,
              +---------------------------+  propagation_error,
                            |                prompt_failure, context_loss
                            v
              +---------------------------+
              |  Feedback-to-Eval Loop    |  every confirmed diagnosis
              +---------------------------+  becomes a regression case
```

## Quick start (offline)

The pipeline ships with a deterministic simulated LLM and a corpus of sample documents that deliberately includes failure-inducing inputs (a contract with no dates, an invoice with mixed currencies, an ambiguous document type):

```bash
pip install -r requirements.txt
python demo.py
```

Output shows every document traced, the failed traces diagnosed, and the eval dataset growing:

```
Processed 12 documents: 8 success, 4 failed
Diagnosing trace tr-004... root cause: step 2 (extraction), extraction_hallucination
Evidence: extracted entity 'John Smith' does not appear in the source document.
Eval dataset: 4 regression cases accumulated
```

To serve the API and trace explorer backend:

```bash
uvicorn src.forensics.api:app --reload
# POST /v1/process        run a document through the pipeline, returns trace_id
# GET  /v1/traces          list traces with status
# GET  /v1/traces/{id}     full span-level trace detail
# POST /v1/traces/{id}/flag  mark output bad, run root cause analysis
# GET  /v1/analytics       failure types, step failure rates, resolution trend
```

## How root cause analysis works

1. When a trace is flagged, the analyzer walks the spans in reverse order.
2. At each step it scores whether the step's output is a reasonable transformation of its input (offline: deterministic content checks; live: LLM-as-judge with the step's prompt as context).
3. The earliest step with a significant quality drop is the root cause. Low self-reported confidence recorded in the span is used as a tiebreaker.
4. The diagnosis is categorized against a failure taxonomy and packaged with an evidence chain quoting the exact input/output pairs.

## Failure taxonomy

- extraction_hallucination: extracted entities that do not exist in the source
- misclassification: wrong document type
- propagation_error: step N was correct but step N+1 misinterpreted its output
- prompt_failure: the model ignored explicit instructions
- context_loss: information from earlier steps was dropped downstream

## The feedback loop

Every confirmed diagnosis automatically becomes an eval case (original input, failing step, bad output, corrected output, category) appended to `data/eval_dataset.jsonl`. Re-running the accumulated dataset against the current pipeline tracks whether known failures stay fixed; `GET /v1/analytics` exposes the resolved-over-time trend.

## Repository layout

```
src/forensics/pipeline.py   the 4-step document pipeline under observation
src/forensics/tracing.py    Trace/Span model, instrumentation decorator, store
src/forensics/analyzer.py   backward analysis, taxonomy, evidence chains
src/forensics/feedback.py   flag handling and eval dataset accumulation
src/forensics/api.py        FastAPI service
data/sample_documents.json  demo corpus with injected failure modes
demo.py                     offline end-to-end run
tests/                      unit tests for the analyzer
```
