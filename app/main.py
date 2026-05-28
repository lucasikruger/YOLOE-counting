"""FastAPI app: text+visual prompts, live preview, tracker, line counting."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import cv2
from fastapi import FastAPI, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.detector import StreamConfig, extract_frame, probe_video, run_stream

import datetime
from fastapi import Body

DATA_DIR = Path("/app/data")
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _meta_path(upload_id: str) -> Path:
    return UPLOAD_DIR / f"{upload_id}.json"


def _find_video(upload_id: str) -> Path | None:
    for p in UPLOAD_DIR.glob(f"{upload_id}.*"):
        if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}:
            return p
    return None


def _save_meta(upload: "Upload", session: dict | None = None) -> None:
    data = {
        "id": upload.id, "filename": upload.filename,
        "width": upload.width, "height": upload.height,
        "total_frames": upload.total_frames, "fps": upload.fps,
        "video_filename": upload.path.name,
        "created_at": getattr(upload, "created_at", datetime.datetime.utcnow().isoformat()),
    }
    if session is not None:
        data["session"] = session
    else:
        # Preserve existing session if any
        existing = _meta_path(upload.id)
        if existing.exists():
            try:
                prev = json.loads(existing.read_text())
                if "session" in prev:
                    data["session"] = prev["session"]
            except (json.JSONDecodeError, OSError):
                pass
    _meta_path(upload.id).write_text(json.dumps(data, indent=2))

JobStatus = Literal["queued", "processing", "done", "error"]
Mode = Literal["text", "visual"]


@dataclass
class Upload:
    id: str
    path: Path
    filename: str
    width: int
    height: int
    total_frames: int
    fps: float
    created_at: str = ""


@dataclass
class Job:
    id: str
    mode: Mode = "text"
    class_names: list[str] = field(default_factory=list)
    model: str = "yoloe-11s-seg.pt"
    use_tracker: bool = True
    has_line: bool = False
    has_roi: bool = False
    status: JobStatus = "queued"
    frames_processed: int = 0
    counts_in: dict[str, int] = field(default_factory=dict)
    counts_out: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    output_path: str | None = None
    original_filename: str = ""
    latest_frame: bytes | None = field(default=None, repr=False)
    frame_counter: int = 0


UPLOADS: dict[str, Upload] = {}
JOBS: dict[str, Job] = {}
LOCK = threading.Lock()

app = FastAPI(title="pipeline-vision")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def _restore_uploads() -> None:
    """Rebuild UPLOADS dict from persisted JSON metadata, plus rescue any
    orphan video files (uploaded before the persistence feature) by
    auto-generating metadata for them."""
    # 1) Load videos with metadata
    seen_ids: set[str] = set()
    for meta_file in UPLOAD_DIR.glob("*.json"):
        try:
            data = json.loads(meta_file.read_text())
            upload_id = data["id"]
            video_path = UPLOAD_DIR / data["video_filename"]
            if not video_path.exists():
                continue
            UPLOADS[upload_id] = Upload(
                id=upload_id, path=video_path, filename=data["filename"],
                width=data["width"], height=data["height"],
                total_frames=data["total_frames"], fps=data["fps"],
                created_at=data.get("created_at", ""),
            )
            seen_ids.add(upload_id)
        except (json.JSONDecodeError, KeyError, OSError):
            continue

    # 2) Rescue orphan videos (no .json sibling)
    video_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    for video_path in UPLOAD_DIR.iterdir():
        if video_path.suffix.lower() not in video_exts:
            continue
        upload_id = video_path.stem
        if upload_id in seen_ids:
            continue
        try:
            info = probe_video(video_path)
            mtime = video_path.stat().st_mtime
            created = datetime.datetime.fromtimestamp(mtime).isoformat()
            upload = Upload(
                id=upload_id, path=video_path,
                filename=f"{upload_id}{video_path.suffix} (rescatado)",
                created_at=created, **info,
            )
            UPLOADS[upload_id] = upload
            _save_meta(upload)
        except Exception:  # noqa: BLE001
            continue


def _job_public(job: Job) -> dict:
    d = asdict(job)
    d.pop("latest_frame", None)
    return d


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.post("/api/uploads")
async def create_upload(video: UploadFile) -> dict:
    upload_id = uuid.uuid4().hex[:12]
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    path = UPLOAD_DIR / f"{upload_id}{suffix}"
    with path.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    info = probe_video(path)
    upload = Upload(
        id=upload_id, path=path, filename=video.filename or "video.mp4",
        created_at=datetime.datetime.utcnow().isoformat(), **info,
    )
    with LOCK:
        UPLOADS[upload_id] = upload
    _save_meta(upload)
    return {"upload_id": upload_id, **info, "filename": upload.filename,
            "created_at": upload.created_at}


@app.get("/api/uploads")
def list_uploads() -> list[dict]:
    with LOCK:
        items = list(UPLOADS.values())
    items.sort(key=lambda u: u.created_at or "", reverse=True)
    return [
        {"upload_id": u.id, "filename": u.filename, "width": u.width, "height": u.height,
         "total_frames": u.total_frames, "fps": u.fps, "created_at": u.created_at}
        for u in items
    ]


@app.delete("/api/uploads/{upload_id}")
def delete_upload(upload_id: str) -> dict:
    with LOCK:
        upload = UPLOADS.pop(upload_id, None)
    if not upload:
        raise HTTPException(404, "Upload not found")
    # Remove video + meta json
    upload.path.unlink(missing_ok=True)
    _meta_path(upload_id).unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/uploads/{upload_id}/session")
def get_session(upload_id: str) -> dict:
    if upload_id not in UPLOADS:
        raise HTTPException(404, "Upload not found")
    meta = _meta_path(upload_id)
    if not meta.exists():
        return {}
    try:
        data = json.loads(meta.read_text())
        return data.get("session") or {}
    except (json.JSONDecodeError, OSError):
        return {}


@app.put("/api/uploads/{upload_id}/session")
def put_session(upload_id: str, session: dict = Body(...)) -> dict:
    with LOCK:
        upload = UPLOADS.get(upload_id)
    if not upload:
        raise HTTPException(404, "Upload not found")
    session["updated_at"] = datetime.datetime.utcnow().isoformat()
    _save_meta(upload, session=session)
    return {"ok": True}


@app.get("/api/uploads/{upload_id}/crop")
def get_crop(upload_id: str, frame: int, x1: int, y1: int, x2: int, y2: int,
             max_w: int = 80) -> Response:
    with LOCK:
        upload = UPLOADS.get(upload_id)
    if not upload:
        raise HTTPException(404, "Upload not found")
    img = extract_frame(upload.path, max(0, min(frame, upload.total_frames - 1)))
    H, W = img.shape[:2]
    x1 = max(0, min(W, x1)); y1 = max(0, min(H, y1))
    x2 = max(0, min(W, x2)); y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        raise HTTPException(400, "Empty crop")
    crop = img[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    if w > max_w:
        scale = max_w / w
        crop = cv2.resize(crop, (max_w, max(1, int(h * scale))))
    ok, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise HTTPException(500, "Encode failed")
    return Response(content=jpg.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/uploads/{upload_id}/frame")
def get_frame(upload_id: str, index: int = 0) -> Response:
    with LOCK:
        upload = UPLOADS.get(upload_id)
    if not upload:
        raise HTTPException(404, "Upload not found")
    index = max(0, min(index, max(0, upload.total_frames - 1)))
    frame = extract_frame(upload.path, index)
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(500, "Could not encode frame")
    return Response(content=jpg.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/jobs")
async def create_job(
    upload_id: str = Form(...),
    mode: str = Form("text"),
    prompts: str = Form(""),
    bboxes: str = Form("[]"),
    cls: str = Form("[]"),
    bbox_frames: str = Form("[]"),    # frame index per bbox (multi-frame visual prompts)
    class_names: str = Form(""),
    frame_index: int = Form(0),
    line: str = Form(""),
    roi: str = Form(""),
    model: str = Form("yoloe-11s-seg.pt"),
    conf: float = Form(0.25),
    stride: int = Form(1),
    use_tracker: bool = Form(True),
    device: str = Form("cpu"),
    track_activation_threshold: float = Form(0.25),
    lost_track_buffer: int = Form(30),
    minimum_matching_threshold: float = Form(0.8),
    minimum_consecutive_frames: int = Form(1),
    show_id: bool = Form(True),
    show_conf: bool = Form(True),
    line_label: str = Form(""),
    roi_label: str = Form(""),
    in_label: str = Form("in"),
    out_label: str = Form("out"),
    show_in: bool = Form(True),
    show_out: bool = Form(True),
    line_color: str = Form("#f472b6"),
    roi_color: str = Form("#60a5fa"),
    show_counts_overlay: bool = Form(False),
    counts_corner: str = Form("TL"),
    bbox_label_position: str = Form("TOP_LEFT"),
    line_label_position: str = Form("above"),
    roi_label_position: str = Form("center"),
    show_class_name: bool = Form(True),
    show_box: bool = Form(True),
    show_mask: bool = Form(False),
    class_colors: str = Form(""),   # comma-separated hex colors
    show_legend: bool = Form(False),
    legend_corner: str = Form("TR"),
) -> dict:
    with LOCK:
        upload = UPLOADS.get(upload_id)
    if not upload:
        raise HTTPException(404, "Upload not found")

    line_coords: list[float] | None = None
    if line:
        try:
            line_coords = json.loads(line)
            if (not isinstance(line_coords, list) or len(line_coords) != 4):
                raise ValueError("line must be [x1,y1,x2,y2]")
            line_coords = [float(v) for v in line_coords]
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(400, f"Invalid line: {e}")

    roi_polygon: list[list[float]] | None = None
    if roi:
        try:
            roi_polygon = json.loads(roi)
            if not isinstance(roi_polygon, list) or len(roi_polygon) < 3:
                raise ValueError("roi must be a polygon with ≥3 vertices")
            roi_polygon = [[float(p[0]), float(p[1])] for p in roi_polygon]
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            raise HTTPException(400, f"Invalid roi: {e}")

    job_id = uuid.uuid4().hex[:12]
    cfg = StreamConfig(
        video_path=upload.path, output_dir=OUTPUT_DIR / job_id, mode=mode,
        model_name=model, conf=conf, vid_stride=stride, device=device,
        use_tracker=use_tracker or (line_coords is not None),
        line=line_coords, roi_polygon=roi_polygon, frame_index=frame_index,
        track_activation_threshold=track_activation_threshold,
        lost_track_buffer=lost_track_buffer,
        minimum_matching_threshold=minimum_matching_threshold,
        minimum_consecutive_frames=minimum_consecutive_frames,
        show_id=show_id, show_conf=show_conf,
        line_label=line_label, roi_label=roi_label,
        line_color=line_color, roi_color=roi_color,
        show_counts_overlay=show_counts_overlay,
        counts_corner=counts_corner if counts_corner in ("TL","TR","BL","BR") else "TL",
        bbox_label_position=bbox_label_position,
        line_label_position=line_label_position,
        roi_label_position=roi_label_position,
        show_class_name=show_class_name,
        show_box=show_box,
        show_mask=show_mask,
        class_colors=[c.strip() for c in class_colors.split(",") if c.strip()],
        show_legend=show_legend,
        legend_corner=legend_corner if legend_corner in ("TL","TR","BL","BR") else "TR",
        in_label=in_label, out_label=out_label,
        show_in=show_in, show_out=show_out,
    )

    if mode == "text":
        names = [p.strip() for p in prompts.split(",") if p.strip()]
        if not names:
            raise HTTPException(400, "At least one prompt required.")
        cfg.prompts = names
        full_names = names
    elif mode == "visual":
        try:
            all_bboxes = json.loads(bboxes)
            all_cls = json.loads(cls)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"Invalid bboxes/cls JSON: {e}")
        if not all_bboxes:
            raise HTTPException(400, "At least one bbox required.")
        if len(all_bboxes) != len(all_cls):
            raise HTTPException(400, "bboxes and cls length mismatch.")
        # bbox_frames is optional (legacy single-frame clients omit it)
        try:
            parsed_frames = json.loads(bbox_frames) if bbox_frames else []
        except json.JSONDecodeError:
            parsed_frames = []
        if isinstance(parsed_frames, list) and len(parsed_frames) == len(all_bboxes):
            bbox_frame_list = [int(f) for f in parsed_frames]
        else:
            bbox_frame_list = [frame_index] * len(all_bboxes)

        # Multi-frame visual prompts: pass all bboxes with their frame indices through;
        # detector pools VPE per class across frames.
        cfg.bboxes = all_bboxes
        cfg.cls = all_cls
        cfg.bbox_frames = bbox_frame_list

        provided = [n.strip() for n in class_names.split(",") if n.strip()] if class_names else []
        unique_ids = sorted(set(all_cls))
        full_names = []
        for i, cid in enumerate(unique_ids):
            full_names.append(provided[i] if i < len(provided) else f"class_{cid}")
        cfg.class_names = full_names
    else:
        raise HTTPException(400, "mode must be 'text' or 'visual'")

    job = Job(id=job_id, mode=mode, class_names=full_names, model=model,
              use_tracker=cfg.use_tracker, has_line=cfg.line is not None,
              has_roi=cfg.roi_polygon is not None,
              original_filename=upload.filename,
              counts_in={n: 0 for n in full_names},
              counts_out={n: 0 for n in full_names})
    with LOCK:
        JOBS[job_id] = job

    threading.Thread(target=_run_job, args=(job_id, cfg), daemon=True).start()
    return {"job_id": job_id}


def _run_job(job_id: str, cfg: StreamConfig) -> None:
    with LOCK:
        JOBS[job_id].status = "processing"

    def on_frame(n: int, annotated):
        ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if not ok:
            return
        data = jpg.tobytes()
        with LOCK:
            j = JOBS.get(job_id)
            if j is None:
                return
            j.latest_frame = data
            j.frame_counter = n
            j.frames_processed = n

    def on_counts(cin: dict, cout: dict):
        with LOCK:
            j = JOBS.get(job_id)
            if j is None:
                return
            j.counts_in = cin
            j.counts_out = cout

    try:
        result = run_stream(cfg, on_frame=on_frame, on_counts=on_counts)
        final = _ensure_browser_playable(result.output_video)
        with LOCK:
            JOBS[job_id].status = "done"
            JOBS[job_id].output_path = str(final.relative_to(DATA_DIR))
            JOBS[job_id].frames_processed = result.frames_processed
            JOBS[job_id].counts_in = result.counts_in
            JOBS[job_id].counts_out = result.counts_out
    except Exception as exc:  # noqa: BLE001
        with LOCK:
            JOBS[job_id].status = "error"
            JOBS[job_id].error = f"{type(exc).__name__}: {exc}"


def _ensure_browser_playable(path: Path) -> Path:
    target = path.with_name(path.stem.replace("_raw", "") + ".mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
            check=True, capture_output=True,
        )
        if target != path:
            path.unlink(missing_ok=True)
        return target
    except (FileNotFoundError, subprocess.CalledProcessError):
        return path


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    with LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return _job_public(job)


@app.get("/api/jobs/{job_id}/video")
def get_job_video(job_id: str) -> FileResponse:
    with LOCK:
        job = JOBS.get(job_id)
    if not job or job.status != "done" or not job.output_path:
        raise HTTPException(404, "Result not ready.")
    full_path = DATA_DIR / job.output_path
    if not full_path.exists():
        raise HTTPException(404, "Output file missing.")
    return FileResponse(full_path, media_type="video/mp4", filename=full_path.name)


@app.get("/api/jobs/{job_id}/stream")
def stream_job(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")

    BOUNDARY = b"--frame"

    def gen():
        last = -1
        idle_since = time.monotonic()
        while True:
            with LOCK:
                j = JOBS.get(job_id)
                if j is None:
                    break
                status = j.status
                frame = j.latest_frame
                counter = j.frame_counter
            if frame is not None and counter != last:
                last = counter
                idle_since = time.monotonic()
                yield (BOUNDARY + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
            elif status in ("done", "error"):
                if frame is not None and counter == last:
                    yield (BOUNDARY + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                break
            else:
                if time.monotonic() - idle_since > 600:
                    break
                time.sleep(0.05)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "jobs": len(JOBS), "uploads": len(UPLOADS)}


@app.get("/api/devices")
def list_devices() -> dict:
    """Enumerate compute devices visible to torch (cpu / cuda / mps)."""
    import torch
    options: list[dict] = [{"id": "cpu", "label": "CPU"}]
    default = "cpu"
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                options.append({"id": f"cuda:{i}", "label": f"cuda:{i} · {name}"})
            default = "cuda:0"
    except Exception:  # noqa: BLE001
        pass
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            options.append({"id": "mps", "label": "mps (Apple Silicon)"})
            if default == "cpu":
                default = "mps"
    except Exception:  # noqa: BLE001
        pass
    return {"options": options, "default": default}
