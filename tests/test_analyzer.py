import sys
import unittest

sys.path.insert(0, "src")

from forensics.analyzer import diagnose
from forensics.pipeline import run_pipeline


class AnalyzerTests(unittest.TestCase):
    def test_hallucination_diagnosed_at_extraction(self):
        trace = run_pipeline(
            "Dear team, per our conversation, the launch moves to Apr 10,"
            " 2026. Regards, Maya Patel.").to_dict()
        diagnosis = diagnose(trace)
        self.assertEqual(diagnosis["root_cause_step"], "extraction")
        self.assertEqual(diagnosis["category"], "extraction_hallucination")
        self.assertIn("John Smith", diagnosis["evidence"]["problem"])

    def test_ambiguous_invoice_report_misclassified(self):
        trace = run_pipeline(
            "INVOICE #5120 and quarterly analysis. Bill to: Zeta Corp."
            " Amount due: $2,300.00. Summary of results and metrics for the"
            " quarter, findings attached. Date: May 11, 2026.").to_dict()
        diagnosis = diagnose(trace)
        self.assertEqual(diagnosis["root_cause_step"], "classification")
        self.assertEqual(diagnosis["category"], "misclassification")

    def test_clean_document_traces_healthy(self):
        trace = run_pipeline(
            "SERVICE AGREEMENT. This agreement is made 2026-01-15 between"
            " Acme Corp and Beta LLC, hereinafter the parties, under the"
            " governing law of Delaware. Total: USD 120,000.00. Signed by"
            " Priya Sharma.")
        self.assertEqual(trace.status, "success")
        self.assertEqual(len(trace.spans), 4)

    def test_every_span_captured(self):
        trace = run_pipeline("Dear Bob, I am writing to say hello. Regards, Al Smith.")
        names = [s.step_name for s in trace.spans]
        self.assertEqual(names, ["intake", "extraction", "classification",
                                 "summarization"])
        for span in trace.spans:
            self.assertIn(span.confidence, range(1, 6))


if __name__ == "__main__":
    unittest.main()
