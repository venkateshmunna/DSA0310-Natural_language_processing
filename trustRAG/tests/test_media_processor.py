import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import media_processor


class MediaProcessorTests(unittest.TestCase):
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

    def test_saved_upload_and_statistics_survive_database_reopen(self):
        media_id = media_processor.save_uploaded_media("lecture.PNG", b"image bytes")

        files = media_processor.list_media_files()

        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]["id"], media_id)
        self.assertEqual(files[0]["filename"], "lecture.PNG")
        self.assertEqual(files[0]["media_type"], "image")
        self.assertEqual(Path(files[0]["path"]).read_bytes(), b"image bytes")
        self.assertEqual(
            media_processor.get_statistics(),
            {"media_files": 1, "assets": 0, "processed_files": 0},
        )

    def test_rejects_empty_or_unsupported_uploads(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            media_processor.save_uploaded_media("empty.jpg", b"")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            media_processor.save_uploaded_media("notes.txt", b"not media")
        self.assertEqual(media_processor.list_media_files(), [])

    def test_image_processing_saves_ocr_caption_and_processed_status(self):
        from PIL import Image

        image_path = self.root / "sample.png"
        Image.new("RGB", (4, 4), color="white").save(image_path)
        media_id = media_processor.save_uploaded_media("sample.png", image_path.read_bytes())

        with mock.patch.object(
            media_processor, "_describe_image", return_value=("Visible title", "A white slide")
        ):
            media_processor.process_media(media_id, frame_interval_seconds=5)

        asset = media_processor.list_media_assets()[0]
        self.assertEqual(asset["ocr_text"], "Visible title")
        self.assertEqual(asset["caption"], "A white slide")
        self.assertIsNone(asset["timestamp_seconds"])
        self.assertEqual(media_processor.list_media_files()[0]["status"], "processed")

    def test_video_processing_samples_frames_at_requested_interval(self):
        media_id = media_processor.save_uploaded_media("clip.mp4", b"video bytes")

        class FakeCapture:
            def __init__(self):
                self.frame_index = 0

            def isOpened(self):
                return True

            def get(self, _property):
                return 2.0

            def read(self):
                if self.frame_index == 6:
                    return False, None
                self.frame_index += 1
                return True, object()

            def release(self):
                pass

        fake_cv2 = SimpleNamespace(
            CAP_PROP_FPS=5,
            VideoCapture=lambda _path: FakeCapture(),
            imwrite=lambda _path, _frame: True,
        )
        with mock.patch.dict("sys.modules", {"cv2": fake_cv2}):
            with mock.patch.object(
                media_processor, "_describe_image", return_value=("", "A video frame")
            ):
                media_processor.process_media(media_id, frame_interval_seconds=1)

        assets = media_processor.list_media_assets()
        self.assertEqual(
            [asset["timestamp_seconds"] for asset in assets],
            [0.0, 1.0, 2.0],
        )
        self.assertEqual(media_processor.list_media_files()[0]["status"], "processed")

    def test_blip_uses_torch_without_importing_tensorflow(self):
        processor = object()
        model = mock.Mock()
        processor_loader = mock.Mock(return_value=processor)
        model_loader = mock.Mock(return_value=model)
        fake_transformers = SimpleNamespace(
            BlipProcessor=SimpleNamespace(from_pretrained=processor_loader),
            BlipForConditionalGeneration=SimpleNamespace(
                from_pretrained=model_loader
            ),
        )

        with mock.patch.object(media_processor, "_caption_processor", None):
            with mock.patch.object(media_processor, "_caption_model", None):
                with mock.patch.dict(os.environ, {"USE_TF": "1"}):
                    with mock.patch.dict(
                        "sys.modules", {"transformers": fake_transformers}
                    ):
                        loaded_processor, loaded_model = media_processor._get_caption_model()

                    self.assertEqual(os.environ["USE_TF"], "0")

        self.assertIs(loaded_processor, processor)
        self.assertIs(loaded_model, model)
        processor_loader.assert_called_once_with(
            "Salesforce/blip-image-captioning-base"
        )
        model_loader.assert_called_once_with(
            "Salesforce/blip-image-captioning-base"
        )
        model.eval.assert_called_once_with()

    def test_processing_failure_is_persisted_and_reported(self):
        media_id = media_processor.save_uploaded_media("missing.jpg", b"image bytes")
        Path(media_processor.list_media_files()[0]["path"]).unlink()

        with self.assertRaises(FileNotFoundError):
            media_processor.process_media(media_id, frame_interval_seconds=5)

        failed_file = media_processor.list_media_files()[0]
        self.assertEqual(failed_file["status"], "failed")
        self.assertIn("missing", failed_file["error"])

    def test_missing_database_record_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "could not be found"):
            media_processor.process_media("unknown", frame_interval_seconds=5)

    def test_license_plate_candidates_are_normalized_deduplicated_and_limited(self):
        from PIL import Image

        first_frame = self.root / "first.jpg"
        second_frame = self.root / "second.jpg"
        Image.new("RGB", (100, 40), color="white").save(first_frame)
        Image.new("RGB", (100, 40), color="white").save(second_frame)
        assets = [
            {
                "filename": "traffic.mp4",
                "media_type": "video",
                "media_id": "video-id",
                "timestamp_seconds": 0.0,
                "created_at": "2026-01-01",
                "id": "frame-a",
                "path": str(first_frame),
            },
            {
                "filename": "traffic.mp4",
                "media_type": "video",
                "media_id": "video-id",
                "timestamp_seconds": 1.0,
                "created_at": "2026-01-01",
                "id": "frame-b",
                "path": str(second_frame),
            },
        ]

        def detection(text, confidence=0.95):
            return (
                [(10, 5), (90, 5), (90, 25), (10, 25)],
                text,
                confidence,
            )

        reader = mock.Mock()
        reader.readtext.side_effect = [
            [detection("NA13NRU")],
            [detection("M51VSU", 0.99)],
            [detection("MV51VSU", 0.89)],
        ]
        with mock.patch.object(media_processor, "list_media_assets", return_value=assets):
            with mock.patch.object(media_processor, "_get_ocr_reader", return_value=reader):
                candidates = media_processor.extract_license_plate_candidates(
                    "video-id", limit=2
                )

        self.assertEqual(
            [candidate["plate"] for candidate in candidates],
            ["NA13NRU", "MV51VSU"],
        )
        self.assertEqual(candidates[0]["timestamp"], "00:00")
        self.assertEqual(candidates[1]["timestamp"], "00:01")
        self.assertEqual(candidates[1]["confidence"], 0.89)
        with mock.patch.object(
            media_processor, "list_media_assets", return_value=assets
        ):
            cached_candidates = media_processor.extract_license_plate_candidates(
                "video-id", limit=2
            )
        self.assertEqual(cached_candidates, candidates)
        self.assertEqual(reader.readtext.call_count, 3)

    def test_license_plate_crop_enlarges_enough_to_recover_missing_character(self):
        from PIL import Image

        frame_path = self.root / "large-frame.jpg"
        Image.new("RGB", (3840, 2160), color="white").save(frame_path)
        asset = {
            "filename": "traffic.mp4",
            "media_type": "video",
            "media_id": "video-id",
            "timestamp_seconds": 1.0,
            "id": "frame-a",
            "path": str(frame_path),
        }
        detection = (
            [(1500, 762), (1796, 762), (1796, 836), (1500, 836)],
            "M51VSU",
            0.999,
        )
        reader = mock.Mock()
        reader.readtext.side_effect = [
            [detection],
            [(
                [(66, 73), (929, 73), (929, 268), (66, 268)],
                "MV51VSU",
                0.916,
            )],
        ]
        with mock.patch.object(
            media_processor, "list_media_assets", return_value=[asset]
        ):
            with mock.patch.object(
                media_processor, "_get_ocr_reader", return_value=reader
            ):
                candidates = media_processor.extract_license_plate_candidates(
                    "video-id", limit=5, force_refresh=True
                )

        self.assertEqual([candidate["plate"] for candidate in candidates], ["MV51VSU"])
        self.assertEqual(candidates[0]["timestamp"], "00:01")
        self.assertEqual(candidates[0]["confidence"], 0.916)
        self.assertEqual(reader.readtext.call_args_list[1].args[0].shape[:2], (495, 1494))

    def test_uk_plate_normalization_corrects_common_ocr_character_confusions(self):
        self.assertEqual(
            media_processor._normalize_uk_plate("GXIS0GJ"),
            "GX15OGJ",
        )
        self.assertEqual(
            media_processor._normalize_uk_plate("KHOSZZK"),
            "KH05ZZK",
        )
        self.assertIsNone(media_processor._normalize_uk_plate("M51VSU"))

    def test_adjacent_frame_ocr_variants_merge_only_nearby_tracks(self):
        first = {
            "plate": "GX15OGJ",
            "timestamp_seconds": 1.0,
            "confidence": 0.72,
            "_center": (1000, 900),
        }
        near_variant = {
            "plate": "GX15OGU",
            "timestamp_seconds": 2.0,
            "confidence": 0.95,
            "_center": (1100, 1000),
        }
        other_vehicle = {
            "plate": "GX15OGU",
            "timestamp_seconds": 2.0,
            "confidence": 0.95,
            "_center": (2200, 1700),
        }
        candidates = []

        media_processor._add_plate_candidate(candidates, first)
        media_processor._add_plate_candidate(candidates, near_variant)
        media_processor._add_plate_candidate(candidates, other_vehicle)

        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["plate"], "GX15OGJ")
        self.assertEqual(candidates[0]["support_count"], 2)
        self.assertEqual(candidates[0]["alternate_reads"], ["GX15OGU"])


if __name__ == "__main__":
    unittest.main()
