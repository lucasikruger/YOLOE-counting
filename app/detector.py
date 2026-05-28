"""YOLO detection: YOLOE (open-vocab) + standard YOLO + YOLO-OBB, with streaming, tracking and line counting."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO, YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor


@dataclass
class StreamConfig:
    video_path: Path
    output_dir: Path
    mode: str  # "text" or "visual"   (only used when model_type == "yoloe")
    # Detector backbone
    model_type: str = "yoloe"          # "yoloe" | "yolo" | "yolo-obb"
    yolo_class_ids: list[int] = field(default_factory=list)
    # text-mode
    prompts: list[str] = field(default_factory=list)
    # visual-mode
    bboxes: list[list[float]] = field(default_factory=list)
    cls: list[int] = field(default_factory=list)
    bbox_frames: list[int] = field(default_factory=list)   # frame index per bbox
    class_names: list[str] = field(default_factory=list)
    frame_index: int = 0
    # common
    model_name: str = "yoloe-11s-seg.pt"
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    device: str = "cpu"
    vid_stride: int = 1
    use_tracker: bool = True
    line: list[float] | None = None
    # Legacy single ROI (still used for the line/counting filter)
    roi_polygon: list[list[float]] | None = None
    # Multi-ROI occupancy: list of {name, points, color, occupancy_threshold}
    rois: list[dict] = field(default_factory=list)
    # ROI occupancy panel
    show_roi_panel: bool = True
    roi_panel_corner: str = "BL"           # TL | TR | BL | BR
    roi_panel_stack: str = "h"             # h | v
    roi_panel_free_color: str = "#34d399"
    roi_panel_occupied_color: str = "#f87171"
    roi_panel_free_text: str = "libre"
    roi_panel_occupied_text: str = "ocupado"
    roi_panel_cell_w: int = 130
    roi_panel_cell_h: int = 78
    # Occupancy stays "occupied" for at least N seconds after the last detection
    # — kills the flicker when a car/object briefly disappears.
    occupied_persistence_sec: float = 0.0
    # Multiplies every text scale (panels, labels, line counts, ROI names…).
    text_scale_mult: float = 1.0
    # Tracker params (ByteTrack)
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8
    minimum_consecutive_frames: int = 2
    # Ghost boxes: render last predicted position for tracks that lost detection
    show_ghost_tracks: bool = True
    ghost_max_age: int = 10
    # Bbox smoothing — exponential moving average per tracker_id over xyxy.
    smooth_bbox: bool = True
    smooth_factor: float = 0.7   # 0 = no smoothing (raw det) · 1 = frozen at first sighting
    # Throughput rate
    show_rate: bool = True
    rate_unit: str = "min"        # "sec" | "min"
    rate_window_min_sec: float = 1.0
    rate_window_max_sec: float = 5.0
    rate_source: str = "both"     # "in" | "out" | "both"
    show_rate_window: bool = True  # show "(vent. Xs)" next to the rate header
    # Stroke / center-dot styling
    outline_line: bool = True
    outline_roi: bool = True
    outline_bbox: bool = True
    line_thickness: int = 2
    bbox_thickness: int = 2
    show_bbox_center: bool = False
    bbox_center_color: str = "#ffffff"
    bbox_center_size: int = 4
    # Per-class baseline (offset added to counts; shown as rate until real events arrive)
    initial_counts_in: dict[str, int] = field(default_factory=dict)
    initial_counts_out: dict[str, int] = field(default_factory=dict)
    initial_rates: dict[str, float] = field(default_factory=dict)
    # Label display options
    show_id: bool = True
    show_conf: bool = True
    # Optional labels for line and ROI overlays
    line_label: str = ""
    roi_label: str = ""
    # Line counter display
    in_label: str = "in"
    out_label: str = "out"
    show_in: bool = True
    show_out: bool = True
    # Colors (hex strings "#RRGGBB"). Empty = use defaults.
    line_color: str = "#f472b6"
    roi_color: str = "#60a5fa"
    # Counts overlay (corner box with per-class in/out totals)
    show_counts_overlay: bool = False
    counts_corner: str = "TL"  # TL, TR, BL, BR
    # Label positions
    bbox_label_position: str = "TOP_LEFT"   # sv.Position name
    line_label_position: str = "above"      # above | below | start | end
    roi_label_position: str = "center"      # center | top | bottom | left | right
    # Detection display toggles
    show_class_name: bool = True
    show_box: bool = True
    show_mask: bool = False
    # Fill / alpha
    bbox_fill: bool = False
    bbox_fill_alpha: float = 0.15
    mask_alpha: float = 0.45
    # Class colors (hex), aligned with class index
    class_colors: list[str] = field(default_factory=list)
    # Legend overlay
    show_legend: bool = False
    legend_corner: str = "TR"  # TL, TR, BL, BR


@dataclass
class DetectionResult:
    output_video: Path
    frames_processed: int
    counts_in: dict[str, int]
    counts_out: dict[str, int]
    rates: dict[str, float] = field(default_factory=dict)
    rate_window_seconds: float = 0.0
    roi_occupancy: dict[str, dict] = field(default_factory=dict)


def _get_model(name: str, model_type: str = "yoloe") -> Any:
    # Always fresh: set_classes() mutates internal projection layers on YOLOE,
    # so a model configured for N classes can crash on the next run with M != N.
    if model_type == "yoloe":
        return YOLOE(name)
    return YOLO(name)


def get_model_class_names(model_name: str, model_type: str) -> dict[int, str]:
    """Briefly load a model and return its class names dict (for the UI picker)."""
    m = _get_model(model_name, model_type)
    return {int(k): str(v) for k, v in m.names.items()}


def _draw_label(img, text: str, center: tuple[int, int], color_rgb: tuple[int, int, int]) -> None:
    """Centered label with dark outline so it reads on any background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    x = center[0] - tw // 2
    y = center[1] + th // 2
    bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, bgr, 1, cv2.LINE_AA)


def _ghost_detections(tracker, class_map: dict[int, int], max_age: int, is_obb: bool = False):
    """Kalman-predicted positions for tracks lost within `max_age` frames.
    When the host model is OBB, also synthesises a degenerate axis-aligned
    xyxyxyxy so the OrientedBoxAnnotator can render the ghost as a regular
    rectangle (we don't have a rotation estimate for lost tracks)."""
    if not getattr(tracker, "lost_tracks", None):
        return None
    cur = getattr(tracker, "frame_id", None)
    rows_xyxy, rows_conf, rows_cls, rows_id = [], [], [], []
    for lt in tracker.lost_tracks:
        tid = getattr(lt, "external_track_id", None)
        if tid is None or int(tid) not in class_map:
            continue
        if cur is not None and getattr(lt, "frame_id", None) is not None:
            if cur - lt.frame_id > max_age:
                continue
        rows_xyxy.append(lt.tlbr)
        rows_conf.append(float(getattr(lt, "score", 0.0)))
        rows_cls.append(class_map[int(tid)])
        rows_id.append(int(tid))
    if not rows_xyxy:
        return None
    xyxy = np.asarray(rows_xyxy, dtype=np.float32)
    kwargs = dict(
        xyxy=xyxy,
        confidence=np.asarray(rows_conf, dtype=np.float32),
        class_id=np.asarray(rows_cls, dtype=np.int64),
        tracker_id=np.asarray(rows_id, dtype=np.int64),
    )
    if is_obb:
        corners = np.zeros((len(xyxy), 4, 2), dtype=np.float32)
        corners[:, 0, :] = xyxy[:, [0, 1]]
        corners[:, 1, :] = xyxy[:, [2, 1]]
        corners[:, 2, :] = xyxy[:, [2, 3]]
        corners[:, 3, :] = xyxy[:, [0, 3]]
        kwargs["data"] = {"xyxyxyxy": corners}
    return sv.Detections(**kwargs)


def _line_label_pos(line, position: str) -> tuple[int, int]:
    x1, y1, x2, y2 = line
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    if position == "below":
        return int(mx), int(my + 18)
    if position == "start":
        return int(x1), int(y1 - 12)
    if position == "end":
        return int(x2), int(y2 - 12)
    # default: above
    return int(mx), int(my - 12)


def _roi_label_pos(roi, position: str) -> tuple[int, int]:
    pts = np.array(roi, dtype=np.int32)
    xs, ys = pts[:, 0], pts[:, 1]
    if position == "top":
        return int(xs.mean()), int(ys.min() + 14)
    if position == "bottom":
        return int(xs.mean()), int(ys.max() - 8)
    if position == "left":
        return int(xs.min() + 30), int(ys.mean())
    if position == "right":
        return int(xs.max() - 30), int(ys.mean())
    # default: center
    return int(xs.mean()), int(ys.mean())


def _hex_to_rgb(s: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
    s = (s or "").strip().lstrip("#")
    if len(s) != 6:
        return default
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError:
        return default


def _draw_roi_status_panel(img, rois: list[dict], occupancy: dict[str, dict], cfg) -> None:
    if not rois:
        return
    free_rgb = _hex_to_rgb(cfg.roi_panel_free_color, (52, 211, 153))
    occ_rgb = _hex_to_rgb(cfg.roi_panel_occupied_color, (248, 113, 113))
    font = cv2.FONT_HERSHEY_SIMPLEX
    name_scale = 0.45
    status_scale = 0.55
    cell_w = max(40, int(getattr(cfg, "roi_panel_cell_w", 130)))
    cell_h = max(30, int(getattr(cfg, "roi_panel_cell_h", 78)))
    tsm = max(0.2, min(5.0, getattr(cfg, "text_scale_mult", 1.0)))
    name_scale = 0.45 * tsm
    status_scale = 0.55 * tsm
    pad = 8
    gap = 6
    n = len(rois)
    if cfg.roi_panel_stack == "v":
        w = cell_w + 2 * pad
        h = n * cell_h + (n - 1) * gap + 2 * pad
    else:
        w = n * cell_w + (n - 1) * gap + 2 * pad
        h = cell_h + 2 * pad
    H, W = img.shape[:2]
    corner = cfg.roi_panel_corner
    if corner == "TR": x0, y0 = W - w - 10, 10
    elif corner == "BL": x0, y0 = 10, H - h - 10
    elif corner == "BR": x0, y0 = W - w - 10, H - h - 10
    else: x0, y0 = 10, 10
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (12, 16, 22), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (60, 80, 100), 1)
    for i, r in enumerate(rois):
        if cfg.roi_panel_stack == "v":
            cx0 = x0 + pad
            cy0 = y0 + pad + i * (cell_h + gap)
        else:
            cx0 = x0 + pad + i * (cell_w + gap)
            cy0 = y0 + pad
        occ = occupancy.get(r["name"], {}).get("occupied", False)
        rgb = occ_rgb if occ else free_rgb
        bgr = (rgb[2], rgb[1], rgb[0])
        cv2.rectangle(img, (cx0, cy0), (cx0 + cell_w, cy0 + cell_h), bgr, -1)
        cv2.rectangle(img, (cx0, cy0), (cx0 + cell_w, cy0 + cell_h), (0, 0, 0), 1)
        # ROI name at top-left
        name = r["name"][:20]
        cv2.putText(img, name, (cx0 + 6, cy0 + 18), font, name_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
        # Status text centered
        text = cfg.roi_panel_occupied_text if occ else cfg.roi_panel_free_text
        (tw, th), _ = cv2.getTextSize(text, font, status_scale, 2)
        tx = cx0 + (cell_w - tw) // 2
        ty = cy0 + cell_h - 14
        cv2.putText(img, text, (tx, ty), font, status_scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (tx, ty), font, status_scale, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_legend(img, items: list[tuple[str, tuple[int, int, int]]], corner: str) -> None:
    """Color → label legend in chosen corner."""
    if not items:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    line_h = 22
    pad = 8
    swatch_w = 16
    widths = [cv2.getTextSize(label, font, scale, thick)[0][0] for label, _ in items]
    w = swatch_w + 8 + max(widths) + pad * 2
    h = line_h * len(items) + pad
    H, W = img.shape[:2]
    if corner == "TR": x0, y0 = W - w - 10, 10
    elif corner == "BL": x0, y0 = 10, H - h - 10
    elif corner == "BR": x0, y0 = W - w - 10, H - h - 10
    else: x0, y0 = 10, 10
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (12, 16, 22), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, dst=img)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (60, 80, 100), 1)
    for i, (label, rgb) in enumerate(items):
        y = y0 + pad + line_h * (i + 1) - 6
        sx, sy = x0 + pad, y - swatch_w + 4
        bgr = (rgb[2], rgb[1], rgb[0])
        cv2.rectangle(img, (sx, sy), (sx + swatch_w, sy + swatch_w), bgr, -1)
        cv2.rectangle(img, (sx, sy), (sx + swatch_w, sy + swatch_w), (230, 232, 235), 1)
        cv2.putText(img, label, (sx + swatch_w + 8, y), font, scale, (230, 232, 235), thick, cv2.LINE_AA)


def _draw_counts_overlay(img, names: list[str], cin: dict[str, int],
                         cout: dict[str, int], corner: str,
                         in_label: str = "in", out_label: str = "out",
                         show_in: bool = True, show_out: bool = True,
                         line_rgb: tuple[int, int, int] = (244, 114, 182),
                         rates: dict[str, float] | None = None,
                         rate_unit: str = "min", rate_window: float = 0.0,
                         show_rate_window: bool = True,
                         class_colors: list[tuple[int, int, int]] | None = None) -> None:
    """Draw a semi-transparent box with per-class in/out counts in a corner."""
    if not names:
        return
    if not (show_in or show_out):
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    line_h = 22
    pad = 8
    NAME_W = 11
    VAL_W = 6
    in_txt = (in_label or "in")[:VAL_W]
    out_txt = (out_label or "out")[:VAL_W]
    cols = []
    if show_in: cols.append(("in", in_txt))
    if show_out: cols.append(("out", out_txt))
    header_parts = ["class".ljust(NAME_W)] + [c[1].rjust(VAL_W) for c in cols]
    lines = [" ".join(header_parts)]
    for n in names:
        row_parts = [n[:NAME_W].ljust(NAME_W)]
        for kind, _ in cols:
            v = cin.get(n, 0) if kind == "in" else cout.get(n, 0)
            row_parts.append(str(v).rjust(VAL_W))
        lines.append(" ".join(row_parts))
    swatch_w = 14
    swatch_gap = 6
    # Compute box size
    widths = [cv2.getTextSize(l, font, scale, thick)[0][0] for l in lines]
    w = max(widths) + pad * 2
    h = line_h * len(lines) + pad
    H, W = img.shape[:2]
    if corner == "TR":
        x0, y0 = W - w - 10, 10
    elif corner == "BL":
        x0, y0 = 10, H - h - 10
    elif corner == "BR":
        x0, y0 = W - w - 10, H - h - 10
    else:  # TL
        x0, y0 = 10, 10
    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (12, 16, 22), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, dst=img)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (60, 80, 100), 1)
    # Text lines. Header row uses the line color (no swatch). Data rows tint
    # by class color.
    bgr_line = (line_rgb[2], line_rgb[1], line_rgb[0])
    text_x = x0 + pad
    for i, line in enumerate(lines):
        y = y0 + pad + line_h * (i + 1) - 6
        if i == 0:
            color = bgr_line
        elif class_colors is None or (i - 1) >= len(class_colors):
            color = (230, 232, 235)
        else:
            r, g, b = class_colors[i - 1]
            color = (b, g, r)
        cv2.putText(img, line, (text_x, y), font, scale, color, thick, cv2.LINE_AA)

    # Rate sub-box BELOW the counts box (or above if corner is BL/BR)
    if rates is not None and names:
        unit_str = "/min" if rate_unit == "min" else "/seg"
        RNAME_W = 11
        RVAL_W = 14
        # Header text centered over the value column
        col_header = f"clase {unit_str}"
        header_val = col_header.center(RVAL_W)
        header_main = f"{'clase':<{RNAME_W}} {header_val}"
        if show_rate_window:
            header = f"{header_main} (vent. {rate_window:.1f}s)"
        else:
            header = header_main
        # Values: format then center in same column width
        rlines = [header] + [
            f"{n[:RNAME_W]:<{RNAME_W}} {f'{rates.get(n, 0.0):.1f}'.center(RVAL_W)}" for n in names
        ]
        rwidths = [cv2.getTextSize(l, font, scale, thick)[0][0] for l in rlines]
        rw = max(rwidths) + pad * 2
        rh = line_h * len(rlines) + pad
        if corner in ("TL", "BL"):
            rx0 = x0
        else:
            rx0 = x0 + w - rw
        if corner in ("TL", "TR"):
            ry0 = y0 + h + 6
        else:
            ry0 = y0 - rh - 6
        roverlay = img.copy()
        cv2.rectangle(roverlay, (rx0, ry0), (rx0 + rw, ry0 + rh), (12, 16, 22), -1)
        cv2.addWeighted(roverlay, 0.65, img, 0.35, 0, dst=img)
        cv2.rectangle(img, (rx0, ry0), (rx0 + rw, ry0 + rh), (60, 80, 100), 1)
        for i, line in enumerate(rlines):
            y = ry0 + pad + line_h * (i + 1) - 6
            if i == 0:
                color = bgr_line
            elif class_colors is None or (i - 1) >= len(class_colors):
                color = (230, 232, 235)
            else:
                r, g, b = class_colors[i - 1]
                color = (b, g, r)
            cv2.putText(img, line, (rx0 + pad, y), font, scale, color, thick, cv2.LINE_AA)


def extract_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def probe_video(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    try:
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
        }
    finally:
        cap.release()


def _compute_pooled_vpe(model: YOLOE, cfg: StreamConfig, tmp_dir: Path) -> tuple[list[str], torch.Tensor]:
    """Multi-frame visual prompt pooling:
    for each unique frame, run the YOLOEVPSegPredictor to capture its VPE tensor,
    then average per class.

    Returns (names, pooled_vpe of shape (1, N_unique_classes, D)).
    """
    # Group bboxes by frame
    by_frame: dict[int, tuple[list, list]] = defaultdict(lambda: ([], []))
    frames = cfg.bbox_frames if len(cfg.bbox_frames) == len(cfg.bboxes) else [cfg.frame_index] * len(cfg.bboxes)
    for b, c, f in zip(cfg.bboxes, cfg.cls, frames):
        by_frame[f][0].append(b)
        by_frame[f][1].append(c)

    tmp_dir.mkdir(parents=True, exist_ok=True)
    per_class_vecs: dict[int, list[torch.Tensor]] = defaultdict(list)

    # Build YOLOEVPSegPredictor once (manual setup — bypassing model.predict() so
    # ultralytics doesn't swap in a plain SegmentationPredictor on cache).
    predictor = YOLOEVPSegPredictor(overrides={
        "imgsz": cfg.imgsz, "device": cfg.device, "conf": 0.5,
        "task": "segment", "mode": "predict", "verbose": False,
    })
    predictor.setup_model(model.model, verbose=False)

    for frame_idx, (bs, cs) in by_frame.items():
        img = extract_frame(cfg.video_path, frame_idx)
        tmp_path = tmp_dir / f"_ref_{frame_idx}.jpg"
        cv2.imwrite(str(tmp_path), img)

        predictor.set_prompts({
            "bboxes": np.array(bs, dtype=np.float32),
            "cls": np.array(cs, dtype=np.int64),
        })
        predictor.setup_source(str(tmp_path))

        # preprocess() calls pre_transform internally, which converts the dict
        # prompts → visual prompt tensor and stores it on predictor.prompts.
        vpe = None
        for _, im0s, _ in predictor.dataset:
            im_tensor = predictor.preprocess(im0s)
            with torch.no_grad():
                vpe = predictor.model(im_tensor, vpe=predictor.prompts, return_vpe=True)
            break
        tmp_path.unlink(missing_ok=True)

        if vpe is None:
            continue
        # VPE shape: (1, num_unique_classes_in_frame, D), classes sorted ascending.
        # Multiple bboxes of the same class are OR'd into a single mask by
        # LoadVisualPrompt.get_visuals → one slot per unique class.
        unique_in_frame = sorted({int(c) for c in cs})
        for i, cid in enumerate(unique_in_frame):
            per_class_vecs[cid].append(vpe[0, i].detach().cpu())

    # Build names + averaged VPE aligned with sorted unique class ids
    unique_ids = sorted(set(cfg.cls))
    names: list[str] = []
    avg_embeds: list[torch.Tensor] = []
    for i, cid in enumerate(unique_ids):
        names.append(cfg.class_names[i] if i < len(cfg.class_names) else f"class_{cid}")
        vecs = per_class_vecs[cid]
        if not vecs:
            raise ValueError(f"No VPE computed for class {cid}")
        avg_embeds.append(torch.stack(vecs).mean(dim=0))

    pooled = torch.stack(avg_embeds).unsqueeze(0)  # (1, N_classes, D)
    return names, pooled


def _prepare(cfg: StreamConfig) -> tuple[Any, list[str]]:
    model = _get_model(cfg.model_name, cfg.model_type)
    # Standard YOLO / YOLO-OBB: use the model's built-in class list. The user
    # picked a subset of class IDs in the UI; we pass those to predict to
    # filter detections, and align our display names to that subset.
    if cfg.model_type in ("yolo", "yolo-obb"):
        all_names: dict[int, str] = {int(k): str(v) for k, v in model.names.items()}
        ids = cfg.yolo_class_ids or list(all_names.keys())
        ids = [i for i in ids if i in all_names]
        if not ids:
            raise ValueError("Seleccioná al menos una clase del modelo.")
        cfg.yolo_class_ids = ids
        names = [all_names[i] for i in ids]
        return model, names
    if cfg.mode == "text":
        if not cfg.prompts:
            raise ValueError("At least one prompt required.")
        names = cfg.prompts
        model.set_classes(names, model.get_text_pe(names))
        return model, names
    if cfg.mode == "visual":
        if not cfg.bboxes:
            raise ValueError("At least one bbox required.")
        if len(cfg.bboxes) != len(cfg.cls):
            raise ValueError("bboxes and cls length mismatch.")
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        names, pooled_pe = _compute_pooled_vpe(model, cfg, cfg.output_dir / "_refs")
        model.set_classes(names, pooled_pe)
        return model, names
    raise ValueError(f"Unknown mode: {cfg.mode}")


def run_stream(
    cfg: StreamConfig,
    on_frame: Callable[[int, np.ndarray], None] | None = None,
    on_counts: Callable[[dict, dict], None] | None = None,
) -> DetectionResult:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model, names = _prepare(cfg)

    src_info = probe_video(cfg.video_path)
    fr = max(1.0, src_info["fps"] / max(1, cfg.vid_stride))

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        tracker = sv.ByteTrack(
            track_activation_threshold=cfg.track_activation_threshold,
            lost_track_buffer=cfg.lost_track_buffer,
            minimum_matching_threshold=cfg.minimum_matching_threshold,
            minimum_consecutive_frames=cfg.minimum_consecutive_frames,
            frame_rate=fr,
        ) if (cfg.use_tracker or cfg.line) else None

    # Build a per-class palette from the user-chosen colors (fall back to default).
    if cfg.class_colors:
        palette = sv.ColorPalette(colors=[
            sv.Color(*_hex_to_rgb(c, (110, 231, 183))) for c in cfg.class_colors
        ])
    else:
        palette = sv.ColorPalette.DEFAULT

    is_obb = cfg.model_type == "yolo-obb"
    if is_obb:
        box_outline_ann = sv.OrientedBoxAnnotator(
            thickness=cfg.bbox_thickness + 3, color=sv.Color.BLACK,
        ) if cfg.outline_bbox else None
        box_ann = sv.OrientedBoxAnnotator(thickness=cfg.bbox_thickness, color=palette)
    else:
        box_outline_ann = sv.BoxAnnotator(thickness=cfg.bbox_thickness + 3, color=sv.Color.BLACK) if cfg.outline_bbox else None
        box_ann = sv.BoxAnnotator(thickness=cfg.bbox_thickness, color=palette)
    try:
        bbox_pos = sv.Position[cfg.bbox_label_position]
    except KeyError:
        bbox_pos = sv.Position.TOP_LEFT
    tsm = max(0.2, min(5.0, cfg.text_scale_mult))
    # Label outline: stack two LabelAnnotators with different paddings — the
    # bigger black one peeks out around the smaller colored one as a 1-2 px ring.
    label_outline_ann = sv.LabelAnnotator(
        text_scale=0.5 * tsm, text_padding=5, text_thickness=1,
        text_position=bbox_pos,
        color=sv.Color.BLACK, text_color=sv.Color.BLACK,
    ) if cfg.outline_bbox else None
    label_ann = sv.LabelAnnotator(text_scale=0.5 * tsm, text_padding=3, text_thickness=1,
                                  text_position=bbox_pos, color=palette)
    mask_ann = sv.MaskAnnotator(color=palette, opacity=max(0.0, min(1.0, cfg.mask_alpha))) if cfg.show_mask else None
    color_ann = sv.ColorAnnotator(color=palette, opacity=max(0.0, min(1.0, cfg.bbox_fill_alpha))) if cfg.bbox_fill else None
    bbox_center_rgb = _hex_to_rgb(cfg.bbox_center_color, (255, 255, 255))

    line_rgb = _hex_to_rgb(cfg.line_color, (244, 114, 182))
    roi_rgb = _hex_to_rgb(cfg.roi_color, (96, 165, 250))

    line_zone = None
    line_annotator = None
    if cfg.line is not None:
        x1, y1, x2, y2 = cfg.line
        line_zone = sv.LineZone(
            start=sv.Point(x1, y1), end=sv.Point(x2, y2),
            triggering_anchors=[sv.Position.CENTER],
        )
        # If counts overlay is on, suppress on-line text — counts go in the corner box.
        on_line = not cfg.show_counts_overlay
        line_annotator = sv.LineZoneAnnotator(
            thickness=cfg.line_thickness, text_thickness=1, text_scale=0.6 * tsm, text_padding=4,
            color=sv.Color(*line_rgb),
            custom_in_text=cfg.in_label or "in",
            custom_out_text=cfg.out_label or "out",
            display_in_count=cfg.show_in and on_line,
            display_out_count=cfg.show_out and on_line,
        )

    # Legacy single ROI (used as a detection filter for line counting).
    roi_zone = None
    roi_annotator = None
    if cfg.roi_polygon and len(cfg.roi_polygon) >= 3:
        polygon = np.array(cfg.roi_polygon, dtype=np.int32)
        roi_zone = sv.PolygonZone(polygon=polygon, triggering_anchors=[sv.Position.CENTER])
        roi_annotator = sv.PolygonZoneAnnotator(
            zone=roi_zone, color=sv.Color(*roi_rgb),
            thickness=2, text_scale=0.5 * tsm, text_thickness=1, text_padding=4,
            display_in_zone_count=False, opacity=0.10,
        )

    # Multi-ROI occupancy (independent of the filter ROI above).
    multi_rois: list[dict] = []
    for r in cfg.rois:
        pts = r.get("points") or []
        if len(pts) < 3:
            continue
        try:
            poly = np.array(pts, dtype=np.int32)
        except Exception:  # noqa: BLE001
            continue
        rgb = _hex_to_rgb(r.get("color", ""), (96, 165, 250))
        zone = sv.PolygonZone(polygon=poly, triggering_anchors=[sv.Position.CENTER])
        annotator = sv.PolygonZoneAnnotator(
            zone=zone, color=sv.Color(*rgb),
            thickness=2, text_scale=0.5 * tsm, text_thickness=1, text_padding=4,
            display_in_zone_count=False, opacity=0.10,
        )
        multi_rois.append({
            "name": str(r.get("name") or f"ROI {len(multi_rois) + 1}"),
            "points": pts,
            "color_rgb": rgb,
            "threshold": int(r.get("occupancy_threshold", 1) or 1),
            "label_position": str(r.get("label_position") or "center"),
            "zone": zone,
            "annotator": annotator,
        })
    roi_occupancy: dict[str, dict] = {r["name"]: {"occupied": False, "count": 0} for r in multi_rois}

    counts_in: dict[str, int] = {n: int(cfg.initial_counts_in.get(n, 0)) for n in names}
    counts_out: dict[str, int] = {n: int(cfg.initial_counts_out.get(n, 0)) for n in names}
    # Throughput: per-class deque of video timestamps when crossings happened
    from collections import deque as _deque
    event_times: dict[str, _deque] = {n: _deque() for n in names}
    rates: dict[str, float] = {n: float(cfg.initial_rates.get(n, 0.0)) for n in names}
    current_window = 0.0
    out_fps = fr

    kwargs = dict(
        source=str(cfg.video_path),
        conf=cfg.conf, iou=cfg.iou, imgsz=cfg.imgsz, device=cfg.device,
        stream=True, vid_stride=cfg.vid_stride, verbose=False,
    )
    # Standard YOLO / YOLO-OBB: filter detections to the user-selected class IDs
    if cfg.model_type in ("yolo", "yolo-obb") and cfg.yolo_class_ids:
        kwargs["classes"] = list(cfg.yolo_class_ids)

    output_video = cfg.output_dir / f"{cfg.video_path.stem}_raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer: cv2.VideoWriter | None = None

    n = 0
    tracker_class_map: dict[int, int] = {}  # tracker_id → class_id (for ghost reconstruction)
    smoothed_xyxy: dict[int, np.ndarray] = {}  # tracker_id → EMA-smoothed xyxy
    roi_persistence_frames = max(0, int(round(cfg.occupied_persistence_sec * fr)))
    roi_last_occupied_frame: dict[str, int] = {}
    try:
        for result in model.predict(**kwargs):
            scene = result.orig_img.copy()  # BGR ndarray
            detections = sv.Detections.from_ultralytics(result)
            # For standard YOLO modes, remap class_id from the model's full
            # vocabulary down to our selected subset so palette + label arrays
            # stay aligned with `names`.
            if cfg.model_type in ("yolo", "yolo-obb") and cfg.yolo_class_ids \
               and detections.class_id is not None and len(detections) > 0:
                id_remap = {orig_id: i for i, orig_id in enumerate(cfg.yolo_class_ids)}
                new_ids = np.array(
                    [id_remap.get(int(c), 0) for c in detections.class_id],
                    dtype=detections.class_id.dtype,
                )
                detections.class_id = new_ids
            # ROI: filter detections to those whose center is inside the polygon
            if roi_zone is not None and len(detections) > 0:
                inside = roi_zone.trigger(detections)
                detections = detections[inside]
            # Always keep a reference to the pre-merge real detections so we can
            # render masks later (the merge below strips the mask field).
            real_dets_with_masks = detections
            if tracker is not None:
                detections = tracker.update_with_detections(detections)
                if detections.tracker_id is not None and detections.class_id is not None:
                    for i in range(len(detections)):
                        tid = detections.tracker_id[i]
                        if tid is not None:
                            tracker_class_map[int(tid)] = int(detections.class_id[i])
                if cfg.smooth_bbox and detections.tracker_id is not None and len(detections) > 0:
                    a = max(0.0, min(0.95, cfg.smooth_factor))
                    new_xyxy = detections.xyxy.copy().astype(np.float32)
                    for i in range(len(detections)):
                        tid = detections.tracker_id[i]
                        if tid is None:
                            continue
                        tid_i = int(tid)
                        if tid_i in smoothed_xyxy:
                            smoothed_xyxy[tid_i] = a * smoothed_xyxy[tid_i] + (1.0 - a) * new_xyxy[i]
                        else:
                            smoothed_xyxy[tid_i] = new_xyxy[i].copy()
                        new_xyxy[i] = smoothed_xyxy[tid_i]
                    detections.xyxy = new_xyxy
                real_dets_with_masks = detections  # refresh after tracker filtering
                if cfg.show_ghost_tracks:
                    ghosts = _ghost_detections(tracker, tracker_class_map,
                                               cfg.ghost_max_age, is_obb=is_obb)
                    if ghosts is not None and len(ghosts) > 0:
                        keep_data: dict = {}
                        if is_obb and "xyxyxyxy" in (detections.data or {}):
                            keep_data["xyxyxyxy"] = detections.data["xyxyxyxy"]
                        detections = sv.Detections(
                            xyxy=detections.xyxy,
                            confidence=detections.confidence,
                            class_id=detections.class_id,
                            tracker_id=detections.tracker_id,
                            data=keep_data,
                        )
                        detections = sv.Detections.merge([detections, ghosts])

            # ----- Phase 2: triggers (counts, rates, occupancy) -----
            in_mask = out_mask = None
            if line_zone is not None:
                in_mask, out_mask = line_zone.trigger(detections)
                t_video = n / max(1e-6, fr)
                if len(detections) > 0 and detections.class_id is not None:
                    for i in range(len(detections)):
                        cid = int(detections.class_id[i])
                        cname = names[cid] if cid < len(names) else f"class_{cid}"
                        if in_mask[i]:
                            counts_in[cname] = counts_in.get(cname, 0) + 1
                            if cfg.rate_source in ("in", "both"):
                                event_times.setdefault(cname, _deque()).append(t_video)
                        if out_mask[i]:
                            counts_out[cname] = counts_out.get(cname, 0) + 1
                            if cfg.rate_source in ("out", "both"):
                                event_times.setdefault(cname, _deque()).append(t_video)
                current_window = max(cfg.rate_window_min_sec,
                                     min(cfg.rate_window_max_sec, t_video))
                cutoff = t_video - current_window
                factor = 60.0 if cfg.rate_unit == "min" else 1.0
                for cname, dq in event_times.items():
                    while dq and dq[0] < cutoff:
                        dq.popleft()
                    if len(dq) > 0 and current_window > 0:
                        rates[cname] = (len(dq) / current_window) * factor
                    else:
                        rates[cname] = float(cfg.initial_rates.get(cname, 0.0))
            for r in multi_rois:
                if len(detections) > 0:
                    inside_roi = r["zone"].trigger(detections)
                    cnt = int(np.asarray(inside_roi).sum())
                else:
                    cnt = 0
                raw_occ = cnt >= r["threshold"]
                if raw_occ:
                    roi_last_occupied_frame[r["name"]] = n
                last_n = roi_last_occupied_frame.get(r["name"], -10 ** 9)
                sticky_occ = raw_occ or (n - last_n) <= roi_persistence_frames
                roi_occupancy[r["name"]] = {
                    "occupied": bool(sticky_occ), "count": cnt,
                }

            # ----- Phase 3: labels -----
            labels: list[str] = []
            if len(detections) > 0:
                for i in range(len(detections)):
                    cid = int(detections.class_id[i]) if detections.class_id is not None else 0
                    cname = names[cid] if cid < len(names) else f"class_{cid}"
                    parts: list[str] = []
                    if cfg.show_class_name:
                        parts.append(cname)
                    if cfg.show_id and detections.tracker_id is not None and detections.tracker_id[i] is not None:
                        parts.append(f"#{int(detections.tracker_id[i])}")
                    if cfg.show_conf and detections.confidence is not None:
                        parts.append(f"{float(detections.confidence[i]):.2f}")
                    labels.append(" ".join(parts))

            # ----- Phase 4: draw the UNDER-bbox layer (ROIs + line) -----
            if roi_annotator is not None:
                if cfg.outline_roi and cfg.roi_polygon:
                    pts = np.array(cfg.roi_polygon, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(scene, [pts], True, (0, 0, 0), 4, lineType=cv2.LINE_AA)
                scene = roi_annotator.annotate(scene=scene)
                if cfg.roi_label and cfg.roi_polygon:
                    pos = _roi_label_pos(cfg.roi_polygon, cfg.roi_label_position)
                    _draw_label(scene, cfg.roi_label, pos, roi_rgb)
            for r in multi_rois:
                if cfg.outline_roi:
                    pts = np.array(r["points"], dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(scene, [pts], True, (0, 0, 0), 4, lineType=cv2.LINE_AA)
                scene = r["annotator"].annotate(scene=scene)
                if r["name"]:
                    pos = _roi_label_pos(r["points"], r.get("label_position", "center"))
                    _draw_label(scene, r["name"], pos, r["color_rgb"])
            if line_zone is not None:
                if cfg.outline_line:
                    cv2.line(scene,
                             (int(cfg.line[0]), int(cfg.line[1])),
                             (int(cfg.line[2]), int(cfg.line[3])),
                             (0, 0, 0), cfg.line_thickness + 3, lineType=cv2.LINE_AA)
                scene = line_annotator.annotate(frame=scene, line_counter=line_zone)
                if cfg.line_label:
                    pos = _line_label_pos(cfg.line, cfg.line_label_position)
                    _draw_label(scene, cfg.line_label, pos, line_rgb)

            # ----- Phase 5: draw the BBOX layer (mask → box → center → label) -----
            if mask_ann is not None and real_dets_with_masks is not None \
               and real_dets_with_masks.mask is not None and len(real_dets_with_masks) > 0:
                scene = mask_ann.annotate(scene=scene, detections=real_dets_with_masks)
            if len(detections) > 0:
                # Translucent fill inside each bbox (axis-aligned). Draw before the
                # outline so the box border stays sharp.
                if color_ann is not None:
                    scene = color_ann.annotate(scene=scene, detections=detections)
                if cfg.show_box:
                    if box_outline_ann is not None:
                        scene = box_outline_ann.annotate(scene=scene, detections=detections)
                    scene = box_ann.annotate(scene=scene, detections=detections)
                if cfg.show_bbox_center:
                    for i in range(len(detections)):
                        x1b, y1b, x2b, y2b = detections.xyxy[i].astype(int)
                        cx, cy = (x1b + x2b) // 2, (y1b + y2b) // 2
                        if cfg.outline_bbox:
                            cv2.circle(scene, (cx, cy), cfg.bbox_center_size + 2, (0, 0, 0), -1)
                        cv2.circle(scene, (cx, cy),
                                   max(1, cfg.bbox_center_size),
                                   (bbox_center_rgb[2], bbox_center_rgb[1], bbox_center_rgb[0]), -1)
                if any(l for l in labels):
                    if label_outline_ann is not None:
                        scene = label_outline_ann.annotate(scene=scene, detections=detections, labels=labels)
                    scene = label_ann.annotate(scene=scene, detections=detections, labels=labels)

            if on_counts is not None and (line_zone is not None or multi_rois):
                on_counts(dict(counts_in), dict(counts_out),
                          rates=dict(rates) if cfg.show_rate else None,
                          window=current_window if cfg.show_rate else 0.0,
                          roi_occupancy=dict(roi_occupancy))

            # ROI status panel (libre / ocupado per ROI)
            if cfg.show_roi_panel and multi_rois:
                _draw_roi_status_panel(scene, multi_rois, roi_occupancy, cfg)

            # Counts overlay in corner (after line so it sits on top)
            if cfg.show_counts_overlay and (line_zone is not None or roi_zone is not None):
                # Per-class colors for tinted rows in the overlay
                cls_rgb_list = [
                    _hex_to_rgb(cfg.class_colors[i] if i < len(cfg.class_colors) else "", (230, 232, 235))
                    for i in range(len(names))
                ] if cfg.class_colors else None
                _draw_counts_overlay(
                    scene, names, counts_in, counts_out, cfg.counts_corner,
                    in_label=cfg.in_label, out_label=cfg.out_label,
                    show_in=cfg.show_in, show_out=cfg.show_out,
                    line_rgb=line_rgb,
                    rates=rates if cfg.show_rate else None,
                    rate_unit=cfg.rate_unit, rate_window=current_window,
                    show_rate_window=cfg.show_rate_window,
                    class_colors=cls_rgb_list,
                )

            # Legend overlay
            if cfg.show_legend:
                legend_items: list[tuple[str, tuple[int, int, int]]] = []
                for i, n in enumerate(names):
                    color = palette.by_idx(i).as_rgb() if hasattr(palette, "by_idx") else (
                        _hex_to_rgb(cfg.class_colors[i], (110, 231, 183))
                        if i < len(cfg.class_colors) else (110, 231, 183))
                    legend_items.append((n, color))
                if cfg.roi_polygon:
                    legend_items.append((cfg.roi_label or "ROI", roi_rgb))
                if cfg.line is not None:
                    legend_items.append((cfg.line_label or "Línea", line_rgb))
                _draw_legend(scene, legend_items, cfg.legend_corner)

            if writer is None:
                h, w = scene.shape[:2]
                writer = cv2.VideoWriter(str(output_video), fourcc, out_fps, (w, h))
            writer.write(scene)
            n += 1
            if on_frame is not None:
                on_frame(n, scene)
    finally:
        if writer is not None:
            writer.release()

    if n == 0 or writer is None:
        raise RuntimeError("No frames produced — check video format or model.")
    return DetectionResult(
        output_video=output_video, frames_processed=n,
        counts_in=counts_in, counts_out=counts_out,
        rates=dict(rates), rate_window_seconds=current_window,
        roi_occupancy=dict(roi_occupancy),
    )
