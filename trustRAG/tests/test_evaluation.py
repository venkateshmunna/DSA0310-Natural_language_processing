import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import evaluation


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.old_directory = evaluation.EVALUATION_DIR
        self.old_dataset = evaluation.DATASET_PATH
        self.old_results = evaluation.RESULTS_PATH
        evaluation.EVALUATION_DIR = self.root
        evaluation.DATASET_PATH = self.root / "dataset.csv"
        evaluation.RESULTS_PATH = self.root / "results.json"
        self.addCleanup(
            setattr,
            evaluation,
            "EVALUATION_DIR",
            self.old_directory,
        )
        self.addCleanup(
            setattr,
            evaluation,
            "DATASET_PATH",
            self.old_dataset,
        )
        self.addCleanup(
            setattr,
            evaluation,
            "RESULTS_PATH",
            self.old_results,
        )
        self.addCleanup(self.temporary_directory.cleanup)

    def test_saved_dataset_round_trips_and_loads_optional_answer(self):
        evaluation.save_dataset(
            [{"query": "What is shown?", "relevant_asset_ids": "asset-a"}]
        )

        rows = evaluation.load_dataset()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["query"], "What is shown?")
        self.assertEqual(rows[0]["relevant_asset_ids"], "asset-a")
        self.assertEqual(rows[0]["reference_answer"], "")
        self.assertTrue(evaluation.DATASET_PATH.is_file())

    def test_precision_recall_and_reciprocal_rank(self):
        ranked = ["asset-a", "asset-b", "asset-c"]
        relevant = {"asset-b", "asset-c"}

        self.assertAlmostEqual(
            evaluation.precision_at_k(ranked, relevant, k=3),
            2 / 3,
        )
        self.assertAlmostEqual(
            evaluation.recall_at_k(ranked, relevant, k=2),
            0.5,
        )
        self.assertAlmostEqual(
            evaluation.reciprocal_rank(ranked, relevant),
            0.5,
        )
        self.assertEqual(evaluation.reciprocal_rank(ranked, {"missing"}), 0.0)

    def test_faithfulness_overlap_is_measured_against_evidence(self):
        evidence = [
            {
                "caption": "A slide showing twenty five percent revenue growth",
                "ocr_text": "Revenue 25%",
            }
        ]

        faithful = evaluation.faithfulness_from_text(
            "Revenue growth was twenty five percent",
            evidence,
        )
        unfaithful = evaluation.faithfulness_from_text(
            "Mars contains a submarine",
            evidence,
        )

        self.assertGreater(faithful, 0.5)
        self.assertLess(unfaithful, 0.5)

    def test_evaluate_runs_three_methods_and_persists_real_scores(self):
        evaluation.save_dataset(
            [
                {
                    "query": "revenue",
                    "relevant_asset_ids": "asset-a",
                    "reference_answer": "revenue increased",
                }
            ]
        )
        assets = [
            {
                "asset_id": "asset-a",
                "source": "report.png",
                "filename": "report.png",
                "timestamp_seconds": None,
                "caption": "Revenue increased",
                "ocr_text": "Revenue increased",
                "score": 1.0,
            }
        ]

        with mock.patch.object(
            evaluation.retrieval,
            "search",
            return_value=assets,
        ):
            measured = evaluation.evaluate(
                k=1,
                dense_weight=0.5,
                sparse_weight=0.5,
                faithfulness_method="Text overlap",
            )

        self.assertEqual(measured["methods"]["BM25"][ "Precision@1"], 1.0)
        self.assertEqual(measured["methods"]["Dense"]["Recall@1"], 1.0)
        self.assertEqual(measured["methods"]["Hybrid"]["MRR"], 1.0)
        self.assertGreaterEqual(measured["methods"]["Hybrid"]["Faithfulness"], 0.0)
        self.assertEqual(len(measured["details"]), 3)
        self.assertEqual(evaluation.load_results()["k"], 1)

    def test_empty_or_missing_dataset_returns_clear_error(self):
        with self.assertRaisesRegex(ValueError, "dataset"):
            evaluation.evaluate()
        evaluation.save_dataset(
            [{"query": "test", "relevant_asset_ids": "asset-a"}]
        )
        with mock.patch.object(
            evaluation.retrieval,
            "search",
            return_value=[],
        ):
            measured = evaluation.evaluate(k=5)
        self.assertEqual(measured["methods"]["BM25"]["MRR"], 0.0)

    def test_result_loading_handles_missing_file(self):
        self.assertIsNone(evaluation.load_results())


if __name__ == "__main__":
    unittest.main()
