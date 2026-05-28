# YOLOE-counting

Open-vocabulary object detection, tracking and **line counting** on video, with a web UI for marking examples and iterating fast.

Built around **YOLOE** (Tsinghua, 2025) — the successor to YOLO-World. Describe what you want to find with plain text *or* mark a few visual examples on a frame; the model finds matching objects in the rest of the video and a counter line tallies them as they cross.

![](docs/screenshot.png)

## Features

- **Two prompting modes**:
  - **Text**: comma-separated open-vocabulary prompts (`pipe, valve, gauge`).
  - **Visual**: draw bounding boxes on one or several frames; the model extracts a per-class prototype and looks for similar instances. *Multi-frame visual prompts are pooled by averaging the VPE per class.*
- **Tracking** with ByteTrack — stable IDs across frames, configurable thresholds.
- **Line counting** with `supervision.LineZone` (CENTER anchor by default — robust for objects passing through). Per-class in/out counters update live during processing.
- **ROI polygon** — restrict detection to a free-shape zone. Detections outside the polygon are filtered before tracking, so tracker IDs aren't wasted.
- **Live MJPEG preview** while the job runs (`<img src="/api/jobs/{id}/stream">`) — see annotations as they happen instead of waiting for the full video.
- **Sessions** — every upload's configuration (prompts, bboxes, classes, line, ROI, tracker params, label/color/position) is auto-saved to `data/uploads/{id}.json` and restored on reload. Reopen a video and pick up where you left off.
- **Per-bbox controls** — number each example, change its class via dropdown, delete it, hover to highlight. Crop thumbnails fetched from the server. Multi-frame: each bbox remembers what frame it was drawn on; jump back to it from the list.
- **Customizable overlay**:
  - Show/hide bbox, mask (for segmentation models), class name, ID, confidence.
  - Position labels: top-left/center/right, bottom-*, center.
  - Colors per class / line / ROI (color pickers).
  - Optional counts overlay box and color legend in any corner.
  - Custom text labels for line and ROI, with positioning.
- **GPU support** — automatic device detection (CPU / CUDA / MPS), selectable from the UI.

## Stack

- **Backend**: FastAPI + uvicorn (auto-reload) — `app/main.py`.
- **Detector**: Ultralytics YOLOE, `app/detector.py`.
- **Tracker / annotators / zones**: `supervision`.
- **Frontend**: vanilla HTML/JS single-page, SVG overlay for drawing — `app/static/index.html`.
- **Container**: based on `ultralytics/ultralytics:latest-cpu` (or `:latest` for CUDA).

## Quick start

### CPU

```bash
git clone git@github.com:lucasikruger/YOLOE-counting.git
cd YOLOE-counting
docker compose up --build
```

Open <http://localhost:8001>.

First run pulls the ultralytics image (~2 GB) and downloads the YOLOE weights on first job (~30 MB for `yoloe-11s-seg.pt`).

### GPU (NVIDIA)

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.

```bash
docker compose -f docker-compose.gpu.yml up --build
```

The Dockerfile takes a `BASE_IMAGE` arg; the GPU compose file overrides it to `ultralytics/ultralytics:latest` (with CUDA).

Inside the UI, open **Opciones avanzadas → Device** and pick `cuda:0`.

## Usage

1. Drop a video into the upload box (mp4 / mov / mkv / avi).
2. **Text mode**: type comma-separated prompts. Bias toward short English phrases (`pressure gauge`, not `the pressure measuring instrument`).
3. **Visual mode**: choose the **Bbox** tool, click a class, drag a bbox on the frame. Repeat across as many frames as you want — the model pools the prototype.
4. *(optional)* **Line tool**: drag once across the conveyor / direction of flow.
5. *(optional)* **ROI tool**: click vertices, double-click or *Cerrar polígono* to close.
6. Submit. Watch live preview, see per-class counts tick up. When done, the player loads with controls and an *Exportar mp4* button.

`Ctrl+Z` undoes any drawing action. `Alt+click` on a bbox in the frame deletes it.

## How YOLOE works (one paragraph)

YOLOE has a backbone that extracts image features and a prompt encoder. Text prompts go through a CLIP-style text encoder (`get_text_pe`) producing class prototypes. Visual prompts (a reference image + bboxes) go through a visual prompt encoder (`get_vpe`) producing visual class prototypes from the regions of the bboxes. Either way, the prototypes are loaded as classes via `set_classes(names, embeddings)`. At inference time, image features are compared against the prototypes (cosine similarity) and the best matches above the confidence threshold are returned. We tap into the low-level VPE to compute prototypes from multiple reference frames separately, then **average per class** so each class gets a more robust prototype that captures variation across pose / scale / lighting.

## Architecture

```
upload (video) ──▶ POST /api/uploads ──▶ data/uploads/{id}.<ext>
                                          + data/uploads/{id}.json (metadata + session)

config (text/visual + tracker + line + ROI + visual options) ──▶ saved to {id}.json via PUT /api/uploads/{id}/session

submit ──▶ POST /api/jobs (with mode, bboxes, bbox_frames, line, roi, …)
                │
                ▼
         _compute_pooled_vpe (visual mode only)
                │     extract reference frame
                │     run YOLOEVPSegPredictor manually
                │     get VPE tensor per frame
                │     average per class
                ▼
         model.set_classes(names, pooled_pe)
                │
                ▼
         run_stream loop
                │
                ▼ frame by frame:
            model.predict → sv.Detections
            ROI.trigger → filter
            ByteTrack.update → IDs
            LineZone.trigger → in/out counts
            BoxAnnotator + LabelAnnotator + MaskAnnotator
            line annotator, ROI annotator
            counts overlay, color legend
            cv2.VideoWriter.write
            on_frame(annotated) → MJPEG stream
            on_counts → JSON job status
                ▼
         finalize: ffmpeg → H.264 mp4 for browser playback
                ▼
         GET /api/jobs/{id}/video to download
```

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/uploads` | Multipart upload a video → `{upload_id, width, height, total_frames, fps, …}` |
| `GET` | `/api/uploads` | List all uploads |
| `DELETE` | `/api/uploads/{id}` | Remove an upload and its metadata |
| `GET` | `/api/uploads/{id}/frame?index=N` | Return frame N as JPEG |
| `GET` | `/api/uploads/{id}/crop?frame=N&x1=…&y1=…&x2=…&y2=…` | Cropped thumbnail JPEG |
| `GET` | `/api/uploads/{id}/session` | Persisted UI session |
| `PUT` | `/api/uploads/{id}/session` | Auto-save UI session |
| `POST` | `/api/jobs` | Start a detection job. All knobs in `Form()`. Returns `{job_id}` |
| `GET` | `/api/jobs/{id}` | Job status, frames processed, per-class counts, output path |
| `GET` | `/api/jobs/{id}/stream` | MJPEG live preview |
| `GET` | `/api/jobs/{id}/video` | Final H.264 mp4 |
| `GET` | `/api/devices` | Available compute devices |
| `GET` | `/api/health` | `{ok, jobs, uploads}` |

## Tracker parameters

ByteTrack knobs (in *Opciones avanzadas → Parámetros del tracker*):

| Param | Default | What it does |
|---|---|---|
| `track_activation_threshold` | 0.25 | Minimum confidence to start a new track. Raise to reduce false positives. |
| `lost_track_buffer` | 30 frames | How long a lost track survives before being removed. Raise if fast objects intermittently disappear. |
| `minimum_matching_threshold` | 0.80 (IoU) | Strictness of detection↔track matching. Lower (0.5–0.6) if objects move a lot between frames. |
| `minimum_consecutive_frames` | 1 | Frames needed to confirm a new track. Raise to 2–3 to combat ID flicker. |

`frame_rate` is auto-derived from the video's fps / stride.

## Multi-frame visual prompts

Mark examples of the same class on several different frames. The detector:

1. Groups bboxes by their `frame_index`.
2. For each frame, runs `YOLOEVPSegPredictor` manually (not via `model.predict`) — manual setup avoids ultralytics' predictor caching that would otherwise reset visual prompt state. `predictor.preprocess()` triggers `pre_transform`, which converts the bboxes into a visual-prompt mask tensor. The model is then called with `vpe=prompts, return_vpe=True` to extract embeddings.
3. The VPE tensor is shape `(1, N_unique_classes, D)` — `LoadVisualPrompt.get_visuals` OR-merges bboxes that share a class into one mask, so one slot per unique class, sorted ascending.
4. Embeddings are averaged per class across frames.
5. Pooled embeddings are loaded via `set_classes(names, pooled_pe)`. The subsequent `model.predict()` calls work identically to text mode.

This is more robust than a single reference frame when objects vary in pose / scale / lighting.

## Folder layout

```
YOLOE-counting/
├── app/
│   ├── main.py          # FastAPI: uploads, sessions, jobs, MJPEG stream, devices
│   ├── detector.py      # YOLOE wrapper + multi-frame VPE pooling + supervision pipeline
│   └── static/
│       └── index.html   # single-page UI (vanilla JS + SVG overlay)
├── data/                # uploads/, outputs/, models/ — mounted volume
├── Dockerfile           # BASE_IMAGE arg, defaults to CPU
├── docker-compose.yml   # CPU service (port 8001)
├── docker-compose.gpu.yml  # GPU override (NVIDIA toolkit required)
└── README.md
```

## Notes & limitations

- Jobs are kept in an in-memory dict on the backend; **the job list is wiped on container restart**, but the persisted videos in `data/outputs/{job_id}/` survive. Sessions and uploads do persist (JSON-on-disk).
- The container default port is `8001` on the host. Change in `docker-compose.yml` if it clashes.
- The first job each session loads the YOLOE checkpoint fresh — the global model cache was removed because `set_classes()` mutates the projection layers, causing shape mismatches between runs with different class counts.
- Output is written with `cv2.VideoWriter` using `mp4v` and then re-encoded to H.264 with ffmpeg (bundled in the base image) so the browser can play it without downloading.

## Roadmap

- [ ] Subclass `YOLOEVPSegPredictor` to extract VPE in a single forward pass instead of two.
- [ ] SAM 2.1 alt mode for pixel-perfect masks + dedicated mark-and-track flow.
- [ ] Multi-line counting (multiple zones).
- [ ] Export of counts as CSV / JSON.
- [ ] Job cancellation endpoint (currently background threads run to completion).

## License

AGPL-3.0 (inherited from ultralytics).
