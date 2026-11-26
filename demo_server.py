"""Local browser UI for demonstrating the image batch processor."""

from __future__ import annotations

import argparse
import io
import json
import logging
import mimetypes
import multiprocessing
import threading
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageOps

from image_batch import (
    BatchConfig,
    IMAGE_EXTENSIONS,
    ImagePipeline,
    OUTPUT_FORMATS,
    _validate_operation,
    run_batch,
)


ROOT = Path(__file__).resolve().parent
DEMO_PAGE = ROOT / "demo" / "index.html"
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
PROCESS_LOCK = threading.Lock()
LOGGER = logging.getLogger("image_batch.demo")


def list_images(folder: Path, route: str) -> list[dict[str, object]]:
    if not folder.exists():
        return []
    items = []
    for path in sorted(folder.iterdir(), key=lambda item: item.name.casefold()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            items.append(
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "url": f"/thumbs/{route}/{quote(path.name)}?v={path.stat().st_mtime_ns}",
                    "file_url": f"/files/{route}/{quote(path.name)}",
                }
            )
    return items


@lru_cache(maxsize=512)
def thumbnail_bytes(path_text: str, modified_ns: int) -> bytes:
    """Create a cached WebP preview instead of loading a full image in the gallery."""
    del modified_ns  # Included in the cache key to invalidate changed images.
    with Image.open(path_text) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGBA")
        image.thumbnail((420, 315), Image.Resampling.LANCZOS)
    payload = io.BytesIO()
    image.save(payload, format="WEBP", quality=82, method=4)
    return payload.getvalue()


def load_demo_config() -> dict[str, object]:
    try:
        raw = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        return {
            "output_format": raw.get("output_format"),
            "operations": raw.get("operations", []),
        }
    except (OSError, ValueError):
        return {"output_format": None, "operations": []}


def clear_image_files(folder: Path) -> tuple[int, list[str]]:
    """Delete only recognized image files directly inside a folder."""
    deleted = 0
    failures: list[str] = []
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            path.unlink()
            deleted += 1
        except OSError as error:
            failures.append(f"{path.name}: {error}")
    return deleted, failures


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "ImageBatchDemo/1.0"

    def log_message(self, pattern: str, *args: object) -> None:
        LOGGER.info("%s - %s", self.address_string(), pattern % args)

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        request_path = unquote(urlsplit(self.path).path)
        if request_path == "/":
            self.send_file(DEMO_PAGE)
            return
        if request_path == "/api/state":
            self.send_json(
                {
                    "input": list_images(INPUT_DIR, "input"),
                    "output": list_images(OUTPUT_DIR, "output"),
                    "config": load_demo_config(),
                }
            )
            return
        if request_path.startswith("/files/input/"):
            self._serve_folder_file(INPUT_DIR, request_path.removeprefix("/files/input/"))
            return
        if request_path.startswith("/files/output/"):
            self._serve_folder_file(OUTPUT_DIR, request_path.removeprefix("/files/output/"))
            return
        if request_path.startswith("/thumbs/input/"):
            self._serve_thumbnail(INPUT_DIR, request_path.removeprefix("/thumbs/input/"))
            return
        if request_path.startswith("/thumbs/output/"):
            self._serve_thumbnail(OUTPUT_DIR, request_path.removeprefix("/thumbs/output/"))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _serve_folder_file(self, folder: Path, filename: str) -> None:
        candidate = (folder / filename).resolve()
        if candidate.parent != folder.resolve():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        self.send_file(candidate)

    def _serve_thumbnail(self, folder: Path, filename: str) -> None:
        candidate = (folder / filename).resolve()
        if candidate.parent != folder.resolve() or not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            body = thumbnail_bytes(str(candidate), candidate.stat().st_mtime_ns)
        except (OSError, ValueError):
            self.send_error(HTTPStatus.UNPROCESSABLE_ENTITY)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/webp")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "private, max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        request_path = urlsplit(self.path).path
        if request_path in {"/api/assets", "/api/masks"}:
            self._upload_asset(".masks" if request_path == "/api/masks" else ".assets")
            return
        if request_path == "/api/mask-preview":
            self._mask_preview()
            return
        if request_path == "/api/clear-output":
            self._clear_output()
            return
        if request_path != "/api/process":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not PROCESS_LOCK.acquire(blocking=False):
            self.send_json({"error": "Another batch is already running."}, HTTPStatus.CONFLICT)
            return
        try:
            payload = self._read_recipe()

            requested_format = payload.get("output_format")
            if requested_format in {None, ""}:
                output_format = None
            elif isinstance(requested_format, str):
                normalized = requested_format.lower().lstrip(".")
                if normalized not in OUTPUT_FORMATS:
                    raise ValueError("Unsupported output format")
                output_format = OUTPUT_FORMATS[normalized]
            else:
                raise ValueError("Invalid output format")

            operations = [
                _validate_operation(operation, index)
                for index, operation in enumerate(payload["operations"])
            ]
            config = BatchConfig(
                input_dir=INPUT_DIR,
                output_dir=OUTPUT_DIR,
                output_format=output_format,
                operations=operations,
                config_dir=ROOT,
                workers=4,
            )
            succeeded, failed = run_batch(config)
            self.send_json(
                {
                    "succeeded": succeeded,
                    "failed": failed,
                    "output": list_images(OUTPUT_DIR, "output"),
                }
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception:
            LOGGER.exception("Unexpected demo request failure")
            self.send_json(
                {"error": "The batch could not be completed."},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        finally:
            PROCESS_LOCK.release()

    def _read_recipe(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        if not 0 < content_length <= 1_000_000:
            raise ValueError("Invalid request size")
        payload = json.loads(self.rfile.read(content_length))
        if not isinstance(payload, dict) or not isinstance(payload.get("operations"), list):
            raise ValueError("operations must be a list")
        return payload

    def _mask_preview(self) -> None:
        if self.headers.get_content_type() != "application/json":
            self.send_json({"error": "Send the example and recipe as JSON"}, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
            return
        try:
            payload = self._read_recipe()
            filename = payload.get("filename")
            if not isinstance(filename, str) or not filename or any(char in filename for char in ("/", "\\")):
                raise ValueError("Select an example from the input folder")
            source = (INPUT_DIR / filename).resolve()
            if source.parent != INPUT_DIR.resolve() or not source.is_file() or source.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError("The selected example is unavailable")
            operations = [_validate_operation(op, index) for index, op in enumerate(payload["operations"])]
            config = BatchConfig(INPUT_DIR, OUTPUT_DIR, None, operations, ROOT)
            image = ImagePipeline(config).render_file(source)
            height, width = image.shape[:2]
            if max(height, width) > 960:
                scale = 960 / max(height, width)
                size = (max(1, round(width * scale)), max(1, round(height * scale)))
                image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            if np.issubdtype(image.dtype, np.floating):
                image = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
            success, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if not success:
                raise ValueError("Could not render the example")
        except (OSError, ValueError, TypeError, cv2.error) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return

        body = encoded.tobytes()
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except ConnectionError:
            # The editor cancels the old request when switching examples.
            return

    def _upload_asset(self, folder: str) -> None:
        if self.headers.get_content_type() != "application/octet-stream":
            self.send_json({"error": "Send the raw image file"}, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
            return
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 20_000_000:
                raise ValueError("Choose an image up to 20 MB")
            payload = self.rfile.read(size)
            with Image.open(io.BytesIO(payload)) as opened:
                extension = {
                    "PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp",
                    "TIFF": ".tiff", "BMP": ".bmp", "GIF": ".gif",
                }.get(opened.format)
                if extension is None:
                    raise ValueError("Choose a PNG, JPEG, WebP, TIFF, BMP, or GIF image")
                opened.verify()
        except ValueError as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        except (OSError, SyntaxError, Image.DecompressionBombError):
            self.send_json({"error": "Could not read the image file"}, HTTPStatus.BAD_REQUEST)
            return
        try:
            path = INPUT_DIR / folder / f"{uuid4().hex}{extension}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        except OSError:
            LOGGER.exception("Could not save the uploaded image")
            self.send_json({"error": "Could not save the image file"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self.send_json({"path": path.relative_to(ROOT).as_posix()}, HTTPStatus.CREATED)

    def _clear_output(self) -> None:
        if not PROCESS_LOCK.acquire(blocking=False):
            self.send_json({"error": "Another batch is already running."}, HTTPStatus.CONFLICT)
            return
        try:
            expected_output = (ROOT / "output").resolve()
            if OUTPUT_DIR.resolve() != expected_output:
                raise ValueError("Refusing to clear an unexpected output location")
            deleted, failures = clear_image_files(OUTPUT_DIR)
            thumbnail_bytes.cache_clear()
            self.send_json(
                {
                    "deleted": deleted,
                    "failed": len(failures),
                    "errors": failures,
                    "output": list_images(OUTPUT_DIR, "output"),
                }
            )
        except (OSError, ValueError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        finally:
            PROCESS_LOCK.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DemoHandler)
    print(f"Image Batch Demo: http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
