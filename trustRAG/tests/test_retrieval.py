import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np

import media_processor
import retrieval


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.paths = mock.patch.multiple(
            media_processor,
            DATA_DIR=self.root,
            DATABASE_PATH=self.root / "secure_rag.sqlite3",
            UPLOADS_DIR=self.root / "uploads",
            FRAMES_DIR=self.root / "frames",
        )
        self.paths.start()
        self.addCleanup(self.paths.stop)
        self.addCleanup(self.temporary_directory.cleanup)
        media_processor.initialize_database()
        self.record_number = 0

    def add_asset(
        self, filename, caption, ocr_text="", media_type="image", timestamp=None
    ):
        self.record_number += 1
        media_id = uuid.uuid4().hex
        asset_id = uuid.uuid4().hex
        image_path = self.root / f"{asset_id}.jpg"
        image_path.write_bytes(b"test image")

        with media_processor._database_connection() as connection:
            connection.execute(
                """
                INSERT INTO media_files
                    (id, filename, media_type, path, size_bytes, status, created_at)
                VALUES (?, ?, ?, ?, ?, 'processed', ?)
                """,
                (
                    media_id,
                    filename,
                    media_type,
                    str(image_path),
                    image_path.stat().st_size,
                    f"2026-01-01 00:00:{self.record_number:02d}",
                ),
            )
            connection.execute(
                """
                INSERT INTO media_assets
                    (id, media_id, path, timestamp_seconds, ocr_text, caption)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (asset_id, media_id, str(image_path), timestamp, ocr_text, caption),
            )
        return asset_id

    def build_test_index(self, vectors):
        with mock.patch.object(
            retrieval,
            "_encode_image",
            side_effect=lambda path: vectors[Path(path).stem],
        ):
            return retrieval.build_index()

    def test_bm25_finds_lexical_evidence_and_returns_no_unmatched_assets(self):
        frequent_match = self.add_asset(
            "revenue-report.jpg",
            "Revenue report showing annual revenue growth",
            "revenue revenue 2025",
        )
        lesser_match = self.add_asset(
            "summary.jpg", "Financial overview", "Revenue figures"
        )
        self.add_asset("campus.jpg", "Students walking across the campus")

        results = retrieval.search("revenue", method="BM25", top_k=5)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["asset_id"], frequent_match)
        self.assertEqual(results[-1]["asset_id"], lesser_match)
        self.assertEqual(retrieval.search("spaceship", method="BM25"), [])

    def test_search_can_be_limited_to_one_or_multiple_media_files(self):
        first_asset = self.add_asset("traffic.jpg", "Traffic and license plates")
        second_asset = self.add_asset("mountain.jpg", "Mountain scenery")
        assets_by_filename = {
            asset["filename"]: asset for asset in media_processor.list_media_assets()
        }
        traffic_media_id = str(assets_by_filename["traffic.jpg"]["media_id"])
        mountain_media_id = str(assets_by_filename["mountain.jpg"]["media_id"])
        vectors = {
            first_asset: np.array([1.0, 0.0], dtype=np.float32),
            second_asset: np.array([0.0, 1.0], dtype=np.float32),
        }
        self.build_test_index(vectors)

        with mock.patch.object(
            retrieval, "_encode_text", return_value=np.array([0.0, 1.0])
        ):
            for method in ("BM25", "Dense", "Hybrid"):
                with self.subTest(method=method):
                    results = retrieval.search(
                        "traffic",
                        method=method,
                        top_k=5,
                        media_ids=[traffic_media_id],
                    )
                    self.assertTrue(results)
                    self.assertEqual(
                        {result["filename"] for result in results}, {"traffic.jpg"}
                    )

            both_results = retrieval.search(
                "traffic scenery",
                method="BM25",
                top_k=5,
                media_ids=[traffic_media_id, mountain_media_id],
            )
        self.assertEqual(
            {result["filename"] for result in both_results},
            {"traffic.jpg", "mountain.jpg"},
        )

    def test_clip_index_persists_and_dense_results_use_faiss_similarity(self):
        first_id = self.add_asset("mountain.jpg", "Snowy mountain")
        second_id = self.add_asset("ocean.jpg", "Blue ocean")
        vectors = {
            first_id: np.array([1.0, 0.0], dtype=np.float32),
            second_id: np.array([0.0, 1.0], dtype=np.float32),
        }

        result = self.build_test_index(vectors)

        self.assertEqual(result, {"indexed_assets": 2, "dimension": 2})
        self.assertTrue(retrieval.get_index_status()["is_current"])
        with mock.patch.object(
            retrieval, "_encode_text", return_value=np.array([1.0, 0.0])
        ):
            results = retrieval.search("mountain", method="Dense", top_k=2)

        self.assertEqual(results[0]["asset_id"], first_id)
        self.assertAlmostEqual(results[0]["similarity"], 1.0, places=5)
        self.assertTrue((self.root / "faiss" / "clip.index").is_file())
        self.assertTrue((self.root / "faiss" / "clip.json").is_file())

    def test_image_and_text_queries_use_the_same_normalized_clip_space(self):
        import torch
        from PIL import Image

        image_path = self.root / "sample.png"
        Image.new("RGB", (2, 2), color="white").save(image_path)
        model = mock.Mock()
        model.get_image_features.return_value = torch.tensor([[3.0, 4.0]])
        model.get_text_features.return_value = torch.tensor([[3.0, 4.0]])

        def fake_processor(**kwargs):
            if "images" in kwargs:
                return {"pixel_values": torch.ones((1, 3, 2, 2))}
            return {
                "input_ids": torch.ones((1, 2), dtype=torch.long),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
            }

        with mock.patch.object(
            retrieval, "_get_clip_model", return_value=(fake_processor, model)
        ):
            image_vector = retrieval._encode_image(image_path)
            text_vector = retrieval._encode_text("a white image")

        np.testing.assert_allclose(image_vector, [0.6, 0.8])
        np.testing.assert_allclose(text_vector, image_vector)
        model.get_image_features.assert_called_once()
        model.get_text_features.assert_called_once()

    def test_hybrid_weights_select_dense_or_keyword_rankings(self):
        visual_id = self.add_asset("abstract.jpg", "Abstract art")
        keyword_id = self.add_asset("animal.jpg", "A giraffe in a field")
        noise_id = self.add_asset("vehicle.jpg", "A red car")
        vectors = {
            visual_id: np.array([1.0, 0.0], dtype=np.float32),
            keyword_id: np.array([0.0, 1.0], dtype=np.float32),
            noise_id: np.array([-1.0, 0.0], dtype=np.float32),
        }
        self.build_test_index(vectors)

        with mock.patch.object(
            retrieval, "_encode_text", return_value=np.array([1.0, 0.0])
        ):
            dense_only = retrieval.search(
                "giraffe", method="Hybrid", dense_weight=1, sparse_weight=0
            )
            keyword_only = retrieval.search(
                "giraffe", method="Hybrid", dense_weight=0, sparse_weight=1
            )

        self.assertEqual(dense_only[0]["asset_id"], visual_id)
        self.assertEqual(keyword_only[0]["asset_id"], keyword_id)

    def test_new_assets_mark_index_stale_and_dense_search_requires_rebuild(self):
        original_id = self.add_asset("original.jpg", "A sample")
        self.build_test_index(
            {original_id: np.array([1.0, 0.0], dtype=np.float32)}
        )
        self.add_asset("new.jpg", "A new sample")

        status = retrieval.get_index_status()

        self.assertFalse(status["is_current"])
        self.assertEqual(status["indexed_assets"], 1)
        with mock.patch.object(
            retrieval, "_encode_text", return_value=np.array([1.0, 0.0])
        ):
            with self.assertRaisesRegex(RuntimeError, "out of date"):
                retrieval.search("sample", method="Dense")

    def test_validation_rejects_empty_queries_and_zero_hybrid_weights(self):
        with self.assertRaisesRegex(ValueError, "Enter a question"):
            retrieval.search(" ")
        with self.assertRaisesRegex(ValueError, "cannot both be zero"):
            retrieval.search("test", method="Hybrid", dense_weight=0, sparse_weight=0)


if __name__ == "__main__":
    unittest.main()
