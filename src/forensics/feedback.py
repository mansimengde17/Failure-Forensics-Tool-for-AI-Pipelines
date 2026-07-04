"""Feedback-to-eval loop: flagged failures become regression cases."""

from __future__ import annotations

import json
import os
import time


class EvalDataset:
    def __init__(self, path: str = "data/eval_dataset.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def add_case(self, trace: dict, diagnosis: dict,
                 corrected_output: str | None = None) -> dict:
        case = {
            "created": time.time(),
            "trace_id": trace["trace_id"],
            "original_input": trace["spans"][0]["input_data"].get("raw_text", "")[:2000],
            "failing_step": diagnosis["root_cause_step"],
            "failure_category": diagnosis["category"],
            "bad_output": trace["final_output"],
            "corrected_output": corrected_output,
            "resolved": False,
        }
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(case) + "\n")
        return case

    def cases(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def regression_check(self, run_pipeline, diagnose) -> dict:
        """Re-run accumulated failure cases against the current pipeline and
        report which known issues are fixed."""
        cases = self.cases()
        results = []
        for case in cases:
            trace = run_pipeline(case["original_input"]).to_dict()
            diagnosis = diagnose(trace)
            still_failing = (diagnosis["category"] == case["failure_category"]
                             and diagnosis["root_cause_step"] == case["failing_step"]
                             and any(s["quality"] < 0.7
                                     for s in diagnosis["step_scores"]))
            results.append({"trace_id": case["trace_id"],
                            "category": case["failure_category"],
                            "still_failing": still_failing})
        fixed = sum(not r["still_failing"] for r in results)
        return {"total_known_issues": len(results), "resolved": fixed,
                "still_failing": len(results) - fixed, "cases": results}
