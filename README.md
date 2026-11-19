# Image Batch Processor

A local folder-in/folder-out processor that applies one ordered JSON operation list to every image in a folder. Non-image files are ignored. An error in one image is logged and the remaining images continue.

## Setup and run

Create an `input` folder beside `config.json` and add the images you want to process. The batch creates the `output` folder.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python image_batch.py config.json
```

Alternatively, edit `CONFIG_FILE` in `run_batch.py`, then run:

```powershell
.\venv\Scripts\python.exe run_batch.py
```

## Local demo UI

The demo reads from the project's `input` folder, writes to `output`, and exposes every operation through a modal form. It includes three clickable configurations—**Soft PNG**, **Web Ready**, and **Navy Frame**—that replace the current format and operation sequence before running. Their dark blue, dark red, and navy borders make the results easy to distinguish. Every preset demonstrates all seven operations and adds `demo/assets/login_logo.png` at the bottom right. Its **Clear** button deletes image files directly inside `output` after confirmation, while leaving non-image files untouched. Gallery previews use cached thumbnails so full-resolution source and output files are not downloaded just to render the UI. Start it with:

```powershell
.\venv\Scripts\python.exe demo_server.py
```

Then open `http://127.0.0.1:8000`. The UI uses the real OpenCV/Pillow processing pipeline; it does not upload images anywhere.

Relative `input_dir`, `output_dir`, and local overlay paths are resolved relative to the JSON file. Only files directly inside `input_dir` are processed; subfolders are not traversed. The CLI exits with code `0` on full success, `1` when individual images fail, and `2` for a config or batch setup error.

`output_format` is optional. Omit it or set it to `null` to retain each source image's filename and format. Set it to `"png"` to convert every result to PNG (for example, `photo.jpg` becomes `photo.png`). Supported values are `png`, `jpg`/`jpeg`, `webp`, `bmp`, `tif`/`tiff`, and `jp2`. If inputs such as `photo.jpg` and `photo.webp` would both become `photo.png`, both are logged as filename conflicts instead of overwriting one another.

## Configuration

Operations execute strictly from top to bottom. Remove operations you do not need.

```json
{
  "input_dir": "input",
  "output_dir": "output",
  "output_format": "png",
  "workers": 4,
  "operations": [
    {"type": "crop", "left": 10, "top": 5, "right": 10, "bottom": 5},
    {"type": "flip", "horizontal": true, "vertical": false},
    {"type": "rotate", "angle": 12.5, "direction": "cw", "fill_color": null},
    {"type": "border", "width": 40, "height": 20, "color": null, "h_align": "center", "v_align": "bottom"},
    {"type": "border_radius", "radius": 24, "fill_color": null},
    {"type": "resize", "width": 1200, "height": 800},
    {"type": "overlay", "image_url": "assets/logo.png", "x": 20, "y": 20, "width": 160, "height": 80, "opacity": 0.8}
  ]
}
```

`workers` controls how many images are processed concurrently. It defaults to `1`; use `4` for the local demo and adjust cautiously based on available CPU and memory.

- `crop`: omitted side values default to `0`; the resulting image must have positive dimensions.
- `flip`: `horizontal` and `vertical` default to `false` and can both be true.
- `rotate` takes a non-negative `angle` and a direction of `cw` or `ccw`. For the space around the rotated image, set `fill_color` to `null` for transparency or an `[r, g, b]` list for a solid color. The default is white.
- `border` adds `width` and `height` total pixels. Set `color` to `null` for transparent padding or an `[r, g, b]` list for a solid color. Center alignment puts the smaller half on the left/top when an odd number is split.
- `border_radius` rounds all four corners by `radius` pixels. Set `fill_color` to `null` (the default) for transparent corners or an `[r, g, b]` list for a solid color. Place it after `border` to round the completed bordered canvas.
- `resize`: fits within the box while preserving aspect ratio. It may enlarge smaller images and does not pad.
- `overlay`: accepts an HTTP(S) URL or local path. `width` and `height` are optional but must be supplied together. `position` can be `coordinates` (the default), `top_left`, `top_right`, `bottom_left`, `bottom_right`, or `center`. For anchored positions, `x` and `y` are inward margins; for exact coordinates they retain their original top-left coordinate meaning. `opacity` defaults to `1`.

For a future process pool, `ImagePipeline.process_file()` is already the isolated per-file unit; `run_batch()` is the simple sequential scheduler.

The demo shows transparent areas against a checkerboard. Choose PNG, WebP, or TIFF to retain transparency in exported images. JPEG flattens transparent pixels onto white.

Outputs use high-quality settings: JPEG quality 100 with full chroma resolution, lossless WebP, lossless PNG, and lossless JPEG 2000 where supported. Right-angle rotations avoid interpolation entirely; other rotations use bicubic interpolation, downsizing uses area resampling, and enlargement uses bicubic resampling. Resizing beyond the source dimensions can still soften an image because no resize algorithm can recreate detail absent from the original.
