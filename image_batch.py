"""Config-driven, folder-in/folder-out image batch processor."""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import logging
import math
import multiprocessing
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


LOGGER = logging.getLogger("image_batch")
IMAGE_EXTENSIONS = {
    *Image.registered_extensions().keys(),
    ".jpe",
    ".jp2",
} - {".eps", ".mpeg", ".mpg", ".pdf", ".ps"}
OUTPUT_FORMATS = {
    "bmp": ".bmp",
    "jpeg": ".jpg",
    "jpg": ".jpg",
    "jp2": ".jp2",
    "png": ".png",
    "tif": ".tif",
    "tiff": ".tiff",
    "webp": ".webp",
}


@dataclass(frozen=True)
class BatchConfig:
    input_dir: Path
    output_dir: Path
    output_format: str | None
    operations: list[dict[str, Any]]
    config_dir: Path
    workers: int = 1


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be true or false")
    return value


def _require_color(value: Any, label: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must be an [r, g, b] list")
    color = tuple(_require_int(channel, f"{label} channel", minimum=0) for channel in value)
    if any(channel > 255 for channel in color):
        raise ValueError(f"{label} channels must be between 0 and 255")
    return color


def _require_fill_color(value: Any, label: str) -> tuple[int, int, int] | None:
    return None if value is None else _require_color(value, label)


def _validate_operation(raw: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"operations[{index}] must be an object")
    op = dict(raw)
    kind = op.get("type")
    label = f"operations[{index}] ({kind!r})"

    if kind == "crop":
        for key in ("left", "top", "right", "bottom"):
            op[key] = _require_int(op.get(key, 0), f"{label}.{key}", minimum=0)
    elif kind == "flip":
        for key in ("horizontal", "vertical"):
            op[key] = _require_bool(op.get(key, False), f"{label}.{key}")
    elif kind == "rotate":
        op["angle"] = _require_number(op.get("angle"), f"{label}.angle")
        if op["angle"] < 0:
            raise ValueError(f"{label}.angle must be zero or greater")
        op["direction"] = op.get("direction")
        if op["direction"] not in {"cw", "ccw"}:
            raise ValueError(f"{label}.direction must be 'cw' or 'ccw'")
        op["fill_color"] = _require_fill_color(
            op.get("fill_color", [255, 255, 255]), f"{label}.fill_color"
        )
    elif kind == "border":
        op["width"] = _require_int(op.get("width", 0), f"{label}.width", minimum=0)
        op["height"] = _require_int(op.get("height", 0), f"{label}.height", minimum=0)
        op["color"] = _require_color(op.get("color"), f"{label}.color")
        op["h_align"] = op.get("h_align", "center")
        op["v_align"] = op.get("v_align", "center")
        if op["h_align"] not in {"left", "center", "right"}:
            raise ValueError(f"{label}.h_align must be left, center, or right")
        if op["v_align"] not in {"top", "center", "bottom"}:
            raise ValueError(f"{label}.v_align must be top, center, or bottom")
    elif kind == "border_radius":
        op["radius"] = _require_int(op.get("radius"), f"{label}.radius", minimum=0)
        op["fill_color"] = _require_fill_color(op.get("fill_color"), f"{label}.fill_color")
    elif kind == "resize":
        op["width"] = _require_int(op.get("width"), f"{label}.width", minimum=1)
        op["height"] = _require_int(op.get("height"), f"{label}.height", minimum=1)
    elif kind == "overlay":
        if not isinstance(op.get("image_url"), str) or not op["image_url"].strip():
            raise ValueError(f"{label}.image_url must be a non-empty URL or path")
        op["position"] = op.get("position", "coordinates")
        if op["position"] not in {
            "coordinates",
            "top_left",
            "top_right",
            "bottom_left",
            "bottom_right",
            "center",
        }:
            raise ValueError(f"{label}.position is not supported")
        minimum = None if op["position"] == "coordinates" else 0
        op["x"] = _require_int(op.get("x", 0), f"{label}.x", minimum=minimum)
        op["y"] = _require_int(op.get("y", 0), f"{label}.y", minimum=minimum)
        has_width, has_height = "width" in op, "height" in op
        if has_width != has_height:
            raise ValueError(f"{label}.width and .height must be supplied together")
        if has_width:
            op["width"] = _require_int(op["width"], f"{label}.width", minimum=1)
            op["height"] = _require_int(op["height"], f"{label}.height", minimum=1)
        op["opacity"] = _require_number(op.get("opacity", 1), f"{label}.opacity")
        if not 0 <= op["opacity"] <= 1:
            raise ValueError(f"{label}.opacity must be between 0 and 1")
    else:
        raise ValueError(f"{label} has an unsupported type")
    return op


def load_config(config_path: str | Path) -> BatchConfig:
    """Load and validate the single JSON configuration file."""
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("config root must be a JSON object")
    if not isinstance(raw.get("input_dir"), str) or not raw["input_dir"].strip():
        raise ValueError("input_dir must be a non-empty path string")
    if not isinstance(raw.get("output_dir"), str) or not raw["output_dir"].strip():
        raise ValueError("output_dir must be a non-empty path string")
    if not isinstance(raw.get("operations"), list):
        raise ValueError("operations must be a list")
    workers = _require_int(raw.get("workers", 1), "workers", minimum=1)
    if workers > 32:
        raise ValueError("workers must be 32 or less")
    requested_format = raw.get("output_format")
    if requested_format is not None:
        if not isinstance(requested_format, str):
            raise ValueError("output_format must be a format name such as 'png', or null")
        requested_format = requested_format.lower().lstrip(".")
        if requested_format not in OUTPUT_FORMATS:
            supported = ", ".join(sorted(OUTPUT_FORMATS))
            raise ValueError(f"unsupported output_format {requested_format!r}; choose from: {supported}")

    def resolve_dir(value: str) -> Path:
        candidate = Path(value).expanduser()
        return (path.parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()

    return BatchConfig(
        input_dir=resolve_dir(raw["input_dir"]),
        output_dir=resolve_dir(raw["output_dir"]),
        output_format=OUTPUT_FORMATS[requested_format] if requested_format else None,
        operations=[_validate_operation(op, i) for i, op in enumerate(raw["operations"])],
        config_dir=path.parent,
        workers=workers,
    )


def crop(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    height, width = image.shape[:2]
    left, top = params["left"], params["top"]
    right, bottom = params["right"], params["bottom"]
    if left + right >= width or top + bottom >= height:
        raise ValueError(
            f"crop ({left}, {top}, {right}, {bottom}) removes the entire {width}x{height} image"
        )
    return np.ascontiguousarray(image[top : height - bottom, left : width - right])


def flip(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    horizontal, vertical = params["horizontal"], params["vertical"]
    if horizontal and vertical:
        return cv2.flip(image, -1)
    if horizontal:
        return cv2.flip(image, 1)
    if vertical:
        return cv2.flip(image, 0)
    return image


def _opaque_alpha(dtype: np.dtype) -> int | float:
    return np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0


def _with_alpha(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image.copy()


def rotate(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    height, width = image.shape[:2]
    angle = params["angle"] if params["direction"] == "ccw" else -params["angle"]
    normalized = angle % 360
    if math.isclose(normalized, 0, abs_tol=1e-10):
        return image
    if math.isclose(normalized, 90, abs_tol=1e-10):
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if math.isclose(normalized, 180, abs_tol=1e-10):
        return cv2.rotate(image, cv2.ROTATE_180)
    if math.isclose(normalized, 270, abs_tol=1e-10):
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = max(1, int(math.ceil(round(height * sine + width * cosine, 10))))
    new_height = max(1, int(math.ceil(round(height * cosine + width * sine, 10))))
    matrix[0, 2] += (new_width - width) / 2.0
    matrix[1, 2] += (new_height - height) / 2.0
    color = params.get("fill_color", (255, 255, 255))
    if color is None:
        dtype = image.dtype
        max_alpha = _opaque_alpha(dtype)
        image = _with_alpha(image).astype(np.float32)
        # Interpolate premultiplied colors to avoid a dark fringe at transparent edges.
        image[:, :, :3] *= image[:, :, 3:] / max_alpha
        fill = (0, 0, 0, 0)
    else:
        red, green, blue = color
        fill = (blue, green, red)
        if image.ndim == 3 and image.shape[2] == 4:
            fill += (255,)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )
    if color is None:
        alpha = rotated[:, :, 3:]
        np.divide(rotated[:, :, :3], alpha / max_alpha, out=rotated[:, :, :3], where=alpha > 0)
        np.clip(rotated, 0, max_alpha, out=rotated)
        if np.issubdtype(dtype, np.integer):
            np.rint(rotated, out=rotated)
        return rotated.astype(dtype)
    return rotated


def _to_pillow(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image, mode="L")
    if image.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA), mode="RGBA")
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mode="RGB")


def _from_pillow(image: Image.Image) -> np.ndarray:
    if image.mode == "RGBA":
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGBA2BGRA)
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def border(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    source = _to_pillow(image)
    added_width, added_height = params["width"], params["height"]
    canvas_size = (source.width + added_width, source.height + added_height)
    color = params["color"] + ((255,) if source.mode == "RGBA" else ())
    canvas = Image.new(source.mode, canvas_size, color)
    x = {"left": 0, "center": added_width // 2, "right": added_width}[params["h_align"]]
    y = {"top": 0, "center": added_height // 2, "bottom": added_height}[params["v_align"]]
    canvas.paste(source, (x, y))
    return _from_pillow(canvas)


def border_radius(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Round the image corners with a transparent or solid fill."""
    requested_radius = params["radius"]
    if requested_radius == 0:
        return image
    height, width = image.shape[:2]
    radius = min(requested_radius, width // 2, height // 2)
    rounded = _with_alpha(image)
    if radius == 0:
        return rounded

    # Draw large and downsample for smooth corner edges.
    scale = 4
    mask = Image.new("L", (width * scale, height * scale), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, width * scale - 1, height * scale - 1),
        radius=radius * scale,
        fill=255,
    )
    coverage = np.asarray(mask.resize((width, height), Image.Resampling.LANCZOS))
    color = params.get("fill_color")
    if color is None:
        alpha = rounded[:, :, 3]
        np.multiply(alpha, coverage.astype(np.float32) / 255, out=alpha, casting="unsafe")
        return rounded

    # Blend only the mask edge and exterior, preserving the source alpha inside the shape.
    edge = coverage < 255
    pixels = rounded[edge].astype(np.float32)
    max_alpha = _opaque_alpha(image.dtype)
    foreground_alpha = (pixels[:, 3:] / max_alpha) * (coverage[edge, None] / 255)
    background_alpha = 1 - coverage[edge, None] / 255
    total_alpha = foreground_alpha + background_alpha
    background = np.asarray(color[::-1]) * (max_alpha / 255)
    pixels[:, :3] = (pixels[:, :3] * foreground_alpha + background * background_alpha) / total_alpha
    pixels[:, 3:] = total_alpha * max_alpha
    if np.issubdtype(image.dtype, np.integer):
        np.rint(pixels, out=pixels)
    rounded[edge] = pixels
    return rounded


def resize(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(params["width"] / width, params["height"] / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if size == (width, height):
        return image
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, size, interpolation=interpolation)


class OverlayStore:
    """Loads each fixed overlay variant once for the entire batch."""

    def __init__(self, config_dir: Path) -> None:
        self.config_dir = config_dir
        self._cache: dict[tuple[str, int | None, int | None, float], Image.Image] = {}

    def get(self, params: dict[str, Any]) -> Image.Image:
        key = (params["image_url"], params.get("width"), params.get("height"), params["opacity"])
        if key not in self._cache:
            source = params["image_url"]
            parsed = urlparse(source)
            if parsed.scheme in {"http", "https"}:
                request = Request(source, headers={"User-Agent": "image-batch/1.0"})
                with urlopen(request, timeout=30) as response:
                    payload = response.read()
                with Image.open(io.BytesIO(payload)) as opened:
                    asset = ImageOps.exif_transpose(opened).convert("RGBA")
            else:
                path = Path(source).expanduser()
                if not path.is_absolute():
                    path = self.config_dir / path
                with Image.open(path) as opened:
                    asset = ImageOps.exif_transpose(opened).convert("RGBA")
            if "width" in params:
                asset = asset.resize((params["width"], params["height"]), Image.Resampling.LANCZOS)
            if params["opacity"] < 1:
                alpha = asset.getchannel("A").point(lambda value: round(value * params["opacity"]))
                asset.putalpha(alpha)
            self._cache[key] = asset
        return self._cache[key]


def overlay(image: np.ndarray, params: dict[str, Any], store: OverlayStore) -> np.ndarray:
    source_had_alpha = image.ndim == 3 and image.shape[2] == 4
    base = _to_pillow(image).convert("RGBA")
    asset = store.get(params)
    x, y = params["x"], params["y"]
    position = params.get("position", "coordinates")
    if position in {"top_right", "bottom_right"}:
        x = base.width - asset.width - x
    elif position == "center":
        x = (base.width - asset.width) // 2 + x
    if position in {"bottom_left", "bottom_right"}:
        y = base.height - asset.height - y
    elif position == "center":
        y = (base.height - asset.height) // 2 + y
    base.alpha_composite(asset, dest=(x, y))
    if not source_had_alpha:
        base = base.convert("RGB")
    return _from_pillow(base)


OPERATION_FUNCTIONS: dict[str, Callable[..., np.ndarray]] = {
    "crop": crop,
    "flip": flip,
    "rotate": rotate,
    "border": border,
    "border_radius": border_radius,
    "resize": resize,
}


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        try:
            with Image.open(io.BytesIO(data.tobytes())) as opened:
                loaded = ImageOps.exif_transpose(opened)
                has_alpha = loaded.mode in {"RGBA", "LA"} or "transparency" in loaded.info
                image = _from_pillow(loaded.convert("RGBA" if has_alpha else "RGB"))
        except Exception as error:
            raise ValueError("could not decode image") from error
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] not in {3, 4}:
        raise ValueError(f"unsupported channel count: {image.shape[2]}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    extension = path.suffix.lower()
    write_image = image
    if extension in {".jpg", ".jpeg", ".jpe"} and image.ndim == 3 and image.shape[2] == 4:
        if image.dtype != np.uint8:
            image = np.clip(np.rint(image * (255 / _opaque_alpha(image.dtype))), 0, 255).astype(np.uint8)
        foreground = _to_pillow(image)
        background = Image.new("RGB", foreground.size, (255, 255, 255))
        background.paste(foreground, mask=foreground.getchannel("A"))
        write_image = _from_pillow(background)
    encode_params: list[int] = []
    if extension in {".jpg", ".jpeg", ".jpe"}:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 100]
        if hasattr(cv2, "IMWRITE_JPEG_CHROMA_QUALITY"):
            encode_params += [cv2.IMWRITE_JPEG_CHROMA_QUALITY, 100]
        if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444"):
            encode_params += [
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
            ]
    elif extension == ".png":
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    elif extension == ".webp":
        encode_params = [cv2.IMWRITE_WEBP_QUALITY, 101]
    elif extension in {".jp2", ".j2k", ".j2c"} and hasattr(
        cv2, "IMWRITE_JPEG2000_COMPRESSION_X1000"
    ):
        encode_params = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 1000]
    try:
        success, encoded = cv2.imencode(extension, write_image, encode_params)
    except cv2.error:
        success = False
    if success:
        path.write_bytes(encoded.tobytes())
        return

    pillow_format = Image.registered_extensions().get(extension)
    if pillow_format is None:
        raise ValueError(f"no encoder is available for {extension} output")
    converted = _to_pillow(write_image)
    if pillow_format == "JPEG" and converted.mode != "RGB":
        converted = converted.convert("RGB")
    payload = io.BytesIO()
    try:
        save_options: dict[str, Any] = {}
        if pillow_format == "JPEG":
            save_options = {"quality": 100, "subsampling": 0}
        elif pillow_format == "PNG":
            save_options = {"compress_level": 1}
        elif pillow_format == "WEBP":
            save_options = {"lossless": True, "quality": 100, "method": 6}
        converted.save(payload, format=pillow_format, **save_options)
    except Exception as error:
        raise ValueError(f"no encoder is available for {extension} output") from error
    path.write_bytes(payload.getvalue())


class ImagePipeline:
    """Owns reusable per-run state and processes one source file at a time."""

    def __init__(self, config: BatchConfig) -> None:
        self.config = config
        self.overlays = OverlayStore(config.config_dir)

    def output_path_for(self, source: Path) -> Path:
        if self.config.output_format:
            return self.config.output_dir / f"{source.stem}{self.config.output_format}"
        return self.config.output_dir / source.name

    def process_file(self, source: Path) -> None:
        image = _read_image(source)
        for params in self.config.operations:
            kind = params["type"]
            image = overlay(image, params, self.overlays) if kind == "overlay" else OPERATION_FUNCTIONS[kind](image, params)
        _write_image(self.output_path_for(source), image)


_WORKER_PIPELINE: ImagePipeline | None = None


def _initialize_worker(config: BatchConfig) -> None:
    """Create reusable state once inside each worker process."""
    global _WORKER_PIPELINE
    cv2.setNumThreads(1)
    _WORKER_PIPELINE = ImagePipeline(config)


def _process_file_worker(source: Path) -> None:
    if _WORKER_PIPELINE is None:
        raise RuntimeError("image worker was not initialized")
    _WORKER_PIPELINE.process_file(source)


def run_batch(config: BatchConfig) -> tuple[int, int]:
    """Run the batch and return (successful_files, failed_files)."""
    if not config.input_dir.is_dir():
        raise ValueError(f"input_dir does not exist or is not a directory: {config.input_dir}")
    if config.output_dir == config.input_dir:
        raise ValueError("output_dir must be different from input_dir")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = ImagePipeline(config)
    successful = failed = 0
    sources = [
        source
        for source in sorted(config.input_dir.iterdir(), key=lambda item: item.name.lower())
        if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS
    ]
    destinations: dict[str, list[Path]] = {}
    for source in sources:
        key = pipeline.output_path_for(source).name.casefold()
        destinations.setdefault(key, []).append(source)
    collisions = {
        source
        for matching_sources in destinations.values()
        if len(matching_sources) > 1
        for source in matching_sources
    }

    processable: list[Path] = []
    for source in sources:
        if source in collisions:
            failed += 1
            LOGGER.error(
                "Failed %s: output filename conflicts with another input (%s)",
                source.name,
                pipeline.output_path_for(source).name,
            )
            continue
        processable.append(source)

    worker_count = min(config.workers, len(processable))
    if worker_count <= 1:
        for source in processable:
            try:
                pipeline.process_file(source)
                successful += 1
            except Exception as error:
                failed += 1
                LOGGER.error("Failed %s: %s", source.name, error)
    else:
        LOGGER.info("Processing %d images with %d workers", len(processable), worker_count)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_initialize_worker,
            initargs=(config,),
        ) as executor:
            futures = {
                executor.submit(_process_file_worker, source): source for source in processable
            }
            for future in concurrent.futures.as_completed(futures):
                source = futures[future]
                try:
                    future.result()
                    successful += 1
                except Exception as error:
                    failed += 1
                    LOGGER.error("Failed %s: %s", source.name, error)

    LOGGER.info("Finished: %d succeeded, %d failed", successful, failed)
    return successful, failed


def run_config(config_path: str | Path) -> tuple[int, int]:
    return run_batch(load_config(config_path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="path to the JSON config file")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        _, failed = run_config(args.config)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        LOGGER.error("Cannot run batch: %s", error)
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
