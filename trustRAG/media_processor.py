import os
from contextlib import contextmanager
import re
import sqlite3
import uuid
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DATABASE_PATH = DATA_DIR / "secure_rag.sqlite3"
UPLOADS_DIR = DATA_DIR / "uploads"
FRAMES_DIR = DATA_DIR / "frames"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

_ocr_reader = None
_caption_processor = None
_caption_model = None


def _connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def _database_connection():
    connection = _connect()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database():
    with _database_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS media_files (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                media_type TEXT NOT NULL,
                path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                status TEXT NOT NULL,
                error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS media_assets (
                id TEXT PRIMARY KEY,
                media_id TEXT NOT NULL REFERENCES media_files(id),
                path TEXT NOT NULL,
                timestamp_seconds REAL,
                ocr_text TEXT NOT NULL DEFAULT '',
                caption TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )


def save_uploaded_media(filename, content):
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported media format: {extension or 'no file extension'}")
    if not content:
        raise ValueError(f"{filename} is empty.")

    media_id = uuid.uuid4().hex
    safe_filename = Path(filename).name
    media_path = UPLOADS_DIR / f"{media_id}{extension}"
    media_path.parent.mkdir(parents=True, exist_ok=True)
    media_path.write_bytes(content)
    media_type = "image" if extension in IMAGE_EXTENSIONS else "video"

    try:
        initialize_database()
        with _database_connection() as connection:
            connection.execute(
                """
                INSERT INTO media_files
                    (id, filename, media_type, path, size_bytes, status)
                VALUES (?, ?, ?, ?, ?, 'saved')
                """,
                (media_id, safe_filename, media_type, str(media_path), len(content)),
            )
    except Exception:
        media_path.unlink(missing_ok=True)
        raise

    return media_id


def list_media_files():
    initialize_database()
    with _database_connection() as connection:
        return connection.execute(
            "SELECT * FROM media_files ORDER BY created_at DESC, rowid DESC"
        ).fetchall()


def list_media_assets():
    initialize_database()
    with _database_connection() as connection:
        return connection.execute(
            """
            SELECT assets.*, media.filename, media.media_type
            FROM media_assets AS assets
            JOIN media_files AS media ON media.id = assets.media_id
            ORDER BY media.created_at DESC, media.rowid DESC,
                     assets.timestamp_seconds, assets.rowid
            """
        ).fetchall()


def get_statistics():
    initialize_database()
    with _database_connection() as connection:
        media_count = connection.execute(
            "SELECT COUNT(*) FROM media_files"
        ).fetchone()[0]
        asset_count = connection.execute(
            "SELECT COUNT(*) FROM media_assets"
        ).fetchone()[0]
        processed_count = connection.execute(
            "SELECT COUNT(*) FROM media_files WHERE status = 'processed'"
        ).fetchone()[0]
    return {
        "media_files": media_count,
        "assets": asset_count,
        "processed_files": processed_count,
    }


def _normalize_uk_plate(text):
    normalized = re.sub(r"[^A-Z0-9]", "", text.upper())
    if len(normalized) != 7:
        return None

    letter_confusions = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
    digit_confusions = {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
        "G": "6",
    }
    normalized = "".join(
        digit_confusions.get(character, character)
        if position in (2, 3)
        else letter_confusions.get(character, character)
        for position, character in enumerate(normalized)
    )
    return (
        normalized
        if re.fullmatch(r"[A-Z]{2}\d{2}[A-Z]{3}", normalized)
        else None
    )


def _plate_timestamp(seconds):
    if seconds is None:
        return "No timestamp"
    minutes, remaining_seconds = divmod(int(seconds), 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"


def _add_plate_candidate(candidates, candidate):
    for existing in candidates:
        if existing["plate"] == candidate["plate"]:
            existing["support_count"] += 1
            existing["confidence"] = max(
                existing["confidence"], candidate["confidence"]
            )
            return

        same_length = len(existing["plate"]) == len(candidate["plate"])
        hamming_distance = (
            sum(a != b for a, b in zip(existing["plate"], candidate["plate"]))
            if same_length
            else 99
        )
        old_time = existing["timestamp_seconds"]
        new_time = candidate["timestamp_seconds"]
        centers = (existing.get("_center"), candidate.get("_center"))
        nearby_in_time = (
            old_time is not None
            and new_time is not None
            and abs(old_time - new_time) <= 2.0
        )
        nearby_in_frame = False
        if all(centers):
            old_x, old_y = centers[0]
            new_x, new_y = centers[1]
            distance = ((old_x - new_x) ** 2 + (old_y - new_y) ** 2) ** 0.5
            nearby_in_frame = distance <= 512
        if (
            hamming_distance == 1
            and nearby_in_time
            and nearby_in_frame
            and existing["plate"][:2] == candidate["plate"][:2]
        ):
            existing["support_count"] += 1
            existing.setdefault("alternate_reads", []).append(candidate["plate"])
            return

    candidate["support_count"] = 1
    candidate["alternate_reads"] = []
    candidates.append(candidate)


def extract_license_plate_candidates(
    media_id,
    limit=5,
    progress_callback=None,
    force_refresh=False,
):
    import json
    import numpy as np
    from PIL import Image

    assets = [
        asset
        for asset in list_media_assets()
        if asset["media_type"] == "video" and str(asset["media_id"]) == str(media_id)
    ]
    assets.sort(
        key=lambda asset: (
            asset["timestamp_seconds"] is None,
            asset["timestamp_seconds"] if asset["timestamp_seconds"] is not None else 0,
            asset["id"],
        )
    )
    if not assets:
        raise ValueError("The selected video has no processed frames.")

    cache_directory = DATA_DIR / "plate_cache"
    cache_path = cache_directory / f"{media_id}.json"
    asset_ids = [str(asset["id"]) for asset in assets]
    if not force_refresh and cache_path.is_file():
        try:
            with cache_path.open("r", encoding="utf-8") as file:
                cached = json.load(file)
            if (
                cached.get("version") == 6
                and cached.get("asset_ids") == asset_ids
                and cached.get("limit") == limit
            ):
                return cached.get("candidates", [])
        except (OSError, ValueError, AttributeError):
            pass

    reader = _get_ocr_reader()
    candidates = []
    scan_until_frame = None
    for frame_number, asset in enumerate(assets, start=1):
        image_path = Path(asset["path"])
        if not image_path.is_file():
            continue
        with Image.open(image_path) as source:
            original_frame = source.convert("RGB")
            frame = original_frame
            if frame.width > 2560:
                resized_height = round(frame.height * 2560 / frame.width)
                frame = frame.resize((2560, resized_height), Image.Resampling.LANCZOS)
        frame_scale = original_frame.width / frame.width

        detections = reader.readtext(
            np.asarray(frame),
            detail=1,
            paragraph=False,
            batch_size=1,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )
        text_regions = []
        for box, text, confidence in detections:
            xs = [point[0] for point in box]
            ys = [point[1] for point in box]
            left, right = int(min(xs)), int(max(xs))
            top, bottom = int(min(ys)), int(max(ys))
            width, height = right - left, bottom - top
            normalized = re.sub(r"[^A-Z0-9]", "", str(text).upper())
            if height < 16 or width / max(height, 1) < 2:
                continue
            if not 5 <= len(normalized) <= 8 or float(confidence) < 0.2:
                continue
            text_regions.append(
                {
                    "box": (left, top, right, bottom),
                    "text": str(text),
                    "confidence": float(confidence),
                    "height": height,
                }
            )

        large_regions = [region for region in text_regions if region["height"] >= 45]
        selected_regions = []
        for region in text_regions:
            left, _top, right, _bottom = region["box"]
            center_x = (left + right) / 2
            is_small_duplicate = region["height"] < 45 and any(
                large_left - 40 <= center_x <= large_right + 40
                for large_left, _large_top, large_right, _large_bottom in (
                    other["box"] for other in large_regions
                )
            )
            if not is_small_duplicate:
                selected_regions.append(region)

        selected_regions.sort(
            key=lambda region: (region["box"][1], region["box"][0])
        )
        for region in selected_regions:
            raw_text = region["text"]
            direct_plate = _normalize_uk_plate(raw_text)
            if direct_plate and raw_text.upper().replace(" ", "") == direct_plate:
                plate, confidence = direct_plate, region["confidence"]
            else:
                left, top, right, bottom = region["box"]
                left = round(left * frame_scale)
                top = round(top * frame_scale)
                right = round(right * frame_scale)
                bottom = round(bottom * frame_scale)
                height = bottom - top
                padding = max(6, int(height * 0.25))
                crop = original_frame.crop(
                    (
                        max(0, left - padding),
                        max(0, top - padding),
                        min(original_frame.width, right + padding),
                        min(original_frame.height, bottom + padding),
                    )
                )
                scale = max(3, min(5, round(120 / max(crop.height, 1))))
                crop = crop.resize(
                    (crop.width * scale, crop.height * scale),
                    Image.Resampling.LANCZOS,
                )
                crop_detections = reader.readtext(
                    np.asarray(crop),
                    detail=1,
                    paragraph=False,
                    batch_size=1,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                )
                valid_crop_reads = [
                    (_normalize_uk_plate(crop_text), float(crop_confidence))
                    for _crop_box, crop_text, crop_confidence in crop_detections
                    if _normalize_uk_plate(crop_text) is not None
                ]
                if not valid_crop_reads:
                    continue
                plate, crop_confidence = max(
                    valid_crop_reads, key=lambda read: read[1]
                )
                confidence = crop_confidence

            if confidence < 0.5:
                continue
            left, top, right, bottom = region["box"]
            candidate = {
                "plate": plate,
                "filename": asset["filename"],
                "timestamp_seconds": asset["timestamp_seconds"],
                "timestamp": _plate_timestamp(asset["timestamp_seconds"]),
                "raw_ocr": raw_text,
                "confidence": round(confidence, 3),
                "path": str(image_path),
                "_center": ((left + right) / 2, (top + bottom) / 2),
            }
            _add_plate_candidate(candidates, candidate)

        if len(candidates) >= limit and scan_until_frame is None:
            scan_until_frame = min(len(assets), frame_number + 2)

        if progress_callback:
            progress_callback(frame_number / len(assets))
        if scan_until_frame is not None and frame_number >= scan_until_frame:
            break

    candidates = candidates[:limit]
    for candidate in candidates:
        candidate.pop("_center", None)
    cache_directory.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "version": 6,
                "asset_ids": asset_ids,
                "limit": limit,
                "candidates": candidates,
            },
            file,
            indent=2,
        )
    return candidates


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
        except ImportError as error:
            raise RuntimeError(
                "EasyOCR is not installed. Install project dependencies with "
                "'python -m pip install -r requirements.txt'."
            ) from error
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader


def _get_caption_model():
    global _caption_processor, _caption_model
    if _caption_processor is None or _caption_model is None:
        os.environ["USE_TF"] = "0"
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as error:
            raise RuntimeError(
                "BLIP dependencies are not installed. Install project dependencies "
                "with 'python -m pip install -r requirements.txt'."
            ) from error
        model_name = "Salesforce/blip-image-captioning-base"
        _caption_processor = BlipProcessor.from_pretrained(model_name)
        _caption_model = BlipForConditionalGeneration.from_pretrained(model_name)
        _caption_model.eval()
    return _caption_processor, _caption_model


def _describe_image(image_path, extract_ocr=True, generate_captions=True):
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as source:
        image = source.convert("RGB")
        analysis_image = image.copy()

    max_size = 1024
    if max(analysis_image.size) > max_size:
        analysis_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    ocr_text = ""
    if extract_ocr:
        try:
            text_lines = _get_ocr_reader().readtext(
                np.asarray(analysis_image),
                detail=0,
                paragraph=False,
                batch_size=1,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            )
        except TypeError:
            text_lines = _get_ocr_reader().readtext(
                np.asarray(analysis_image),
                detail=0,
                paragraph=False,
                batch_size=1,
            )
        ocr_text = "\n".join(
            str(line).strip() for line in text_lines if str(line).strip()
        )

    caption = ""
    if generate_captions:
        import torch

        processor, model = _get_caption_model()
        inputs = processor(images=analysis_image, return_tensors="pt")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(
            generated[0], skip_special_tokens=True
        ).strip()
    return ocr_text, caption


def _save_asset(
    connection,
    media_id,
    image_path,
    timestamp_seconds,
    extract_ocr=True,
    generate_captions=True,
):
    ocr_text, caption = _describe_image(
        image_path,
        extract_ocr=extract_ocr,
        generate_captions=generate_captions,
    )
    connection.execute(
        """
        INSERT INTO media_assets
            (id, media_id, path, timestamp_seconds, ocr_text, caption)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            uuid.uuid4().hex,
            media_id,
            str(image_path),
            timestamp_seconds,
            ocr_text,
            caption,
        ),
    )


def _process_image(
    connection,
    media_id,
    path,
    extract_ocr=True,
    generate_captions=True,
):
    try:
        _save_asset(
            connection,
            media_id,
            path,
            None,
            extract_ocr=extract_ocr,
            generate_captions=generate_captions,
        )
    except Exception as error:
        raise RuntimeError(f"Could not process image {path.name}: {error}") from error


def _process_video(
    connection,
    media_id,
    path,
    frame_interval_seconds,
    extract_ocr=True,
    generate_captions=True,
):
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is not installed. Install project dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from error

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video file: {path.name}")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            raise RuntimeError(f"Could not determine the frame rate for {path.name}.")

        output_dir = FRAMES_DIR / media_id
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_index = 0
        next_sample_seconds = 0.0
        saved_frames = 0

        while True:
            success, frame = capture.read()
            if not success:
                break

            timestamp_seconds = frame_index / fps
            if timestamp_seconds + 1e-9 >= next_sample_seconds:
                frame_path = output_dir / f"frame_{frame_index:08d}.jpg"
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"Could not save an extracted frame: {frame_path.name}")
                _save_asset(
                    connection,
                    media_id,
                    frame_path,
                    timestamp_seconds,
                    extract_ocr=extract_ocr,
                    generate_captions=generate_captions,
                )
                saved_frames += 1
                next_sample_seconds += frame_interval_seconds
            frame_index += 1
    finally:
        capture.release()

    if saved_frames == 0:
        raise RuntimeError(f"No readable frames were found in {path.name}.")


def process_media(
    media_id,
    frame_interval_seconds,
    extract_ocr=True,
    generate_captions=True,
):
    if frame_interval_seconds < 1:
        raise ValueError("Frame interval must be at least one second.")

    initialize_database()
    with _database_connection() as connection:
        media = connection.execute(
            "SELECT * FROM media_files WHERE id = ?", (media_id,)
        ).fetchone()
        if media is None:
            raise ValueError("The uploaded media record could not be found.")
        connection.execute(
            "UPDATE media_files SET status = 'processing', error = '' WHERE id = ?",
            (media_id,),
        )
        connection.commit()
        media_path = Path(media["path"])
        try:
            if not media_path.is_file():
                raise FileNotFoundError(f"Saved media file is missing: {media_path}")
            if media["media_type"] == "image":
                _process_image(
                    connection,
                    media_id,
                    media_path,
                    extract_ocr=extract_ocr,
                    generate_captions=generate_captions,
                )
            else:
                _process_video(
                    connection,
                    media_id,
                    media_path,
                    frame_interval_seconds,
                    extract_ocr=extract_ocr,
                    generate_captions=generate_captions,
                )
            connection.execute(
                "UPDATE media_files SET status = 'processed', error = '' WHERE id = ?",
                (media_id,),
            )
            connection.commit()
        except Exception as error:
            connection.rollback()
            connection.execute(
                "UPDATE media_files SET status = 'failed', error = ? WHERE id = ?",
                (str(error), media_id),
            )
            connection.commit()
            raise
