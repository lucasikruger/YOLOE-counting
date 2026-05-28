"""YOLOE detection: text + visual prompts (multi-frame pooling), streaming, tracking, line counting."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor


@dataclass
class StreamConfig:
    video_path: Path
    output_dir: Path
    mode: str  # "text" or "visual"
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
    # ROI: list of [x, y] vertices (polygon)
    roi_polygon: list[list[float]] | None = None
    # Tracker params (ByteTrack)
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8
    minimum_consecutive_frames: int = 1
    # Ghost boxes: render last predicted position for tracks that lost detection
    show_ghost_tracks: bool = True
    ghost_max_age: int = 10
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


def _get_model(name: str) -> YOLOE:
    # Always fresh: set_classes() mutates internal projection layers, so a model
    # configured for N classes can crash on the next run with M != N.
    return YOLOE(name)


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


def _ghost_detections(tracker, class_map: dict[int, int], max_age: int):
    """Build a Detections of Kalman-predicted positions for tracks that were
    lost within the last `max_age` frames. The tracker keeps advancing each
    lost track's position via its Kalman filter every frame, so tlbr reflects
    where the object is *expected* to be even though no detection landed."""
    if not getattr(tracker, "lost_tracks", None):
        return None
    cur = getattr(tracker, "frame_id", None)
    rows_xyxy, rows_conf, rows_cls, rows_id = [], [], [], []
    for lt in tracker.lost_tracks:
        tid = getattr(lt, "external_track_id", None)
        if tid is None or int(tid) not in class_map:
            continue
        # Skip tracks lost too long (predictions become unreliable)
        if cur is not None and getattr(lt, "frame_id", None) is not None:
            if cur - lt.frame_id > max_age:
                continue
        rows_xyxy.append(lt.tlbr)
        rows_conf.append(float(getattr(lt, "score", 0.0)))
        rows_cls.append(class_map[int(tid)])
        rows_id.append(int(tid))
    if not rows_xyxy:
        return None
    return sv.Detections(
        xyxy=np.asarray(rows_xyxy, dtype=np.float32),
        confidence=np.asarray(rows_conf, dtype=np.float32),
        class_id=np.asarray(rows_cls, dtype=np.int64),
        tracker_id=np.asarray(rows_id, dtype=np.int64),
    )


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
                         line_rgb: tuple[int, int, int] = (244, 114, 182)) -> None:
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
    in_txt = (in_label or "in")[:6]
    out_txt = (out_label or "out")[:6]
    cols = []
    if show_in: cols.append(("in", in_txt))
    if show_out: cols.append(("out", out_txt))
    header_parts = ["class".ljust(11)] + [c[1].rjust(6) for c in cols]
    lines = [" ".join(header_parts)]
    for n in names:
        row_parts = [n[:11].ljust(11)]
        for kind, _ in cols:
            v = cin.get(n, 0) if kind == "in" else cout.get(n, 0)
            row_parts.append(str(v).rjust(6))
        lines.append(" ".join(row_parts))
    swatch_w = 14
    swatch_gap = 6
    # Compute box size
    widths = [cv2.getTextSize(l, font, scale, thick)[0][0] for l in lines]
    w = swatch_w + swatch_gap + max(widths) + pad * 2
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
    # Line swatch in header row (identifies which line these counts belong to)
    bgr_line = (line_rgb[2], line_rgb[1], line_rgb[0])
    sw_x = x0 + pad
    sw_y = y0 + pad + 4
    cv2.rectangle(img, (sw_x, sw_y), (sw_x + swatch_w, sw_y + swatch_w), bgr_line, -1)
    cv2.rectangle(img, (sw_x, sw_y), (sw_x + swatch_w, sw_y + swatch_w), (230, 232, 235), 1)
    # Text lines, shifted right to leave room for swatch column
    text_x = x0 + pad + swatch_w + swatch_gap
    for i, line in enumerate(lines):
        y = y0 + pad + line_h * (i + 1) - 6
        cv2.putText(img, line, (text_x, y), font, scale, (230, 232, 235), thick, cv2.LINE_AA)


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


def _prepare(cfg: StreamConfig) -> tuple[YOLOE, list[str]]:
    model = _get_model(cfg.model_name)
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

    box_ann = sv.BoxAnnotator(thickness=2, color=palette)
    try:
        bbox_pos = sv.Position[cfg.bbox_label_position]
    except KeyError:
        bbox_pos = sv.Position.TOP_LEFT
    label_ann = sv.LabelAnnotator(text_scale=0.5, text_padding=3, text_thickness=1,
                                  text_position=bbox_pos, color=palette)
    mask_ann = sv.MaskAnnotator(color=palette, opacity=0.45) if cfg.show_mask else None

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
            thickness=2, text_thickness=1, text_scale=0.6, text_padding=4,
            color=sv.Color(*line_rgb),
            custom_in_text=cfg.in_label or "in",
            custom_out_text=cfg.out_label or "out",
            display_in_count=cfg.show_in and on_line,
            display_out_count=cfg.show_out and on_line,
        )

    roi_zone = None
    roi_annotator = None
    if cfg.roi_polygon and len(cfg.roi_polygon) >= 3:
        polygon = np.array(cfg.roi_polygon, dtype=np.int32)
        roi_zone = sv.PolygonZone(polygon=polygon, triggering_anchors=[sv.Position.CENTER])
        roi_annotator = sv.PolygonZoneAnnotator(
            zone=roi_zone, color=sv.Color(*roi_rgb),
            thickness=2, text_scale=0.5, text_thickness=1, text_padding=4,
            display_in_zone_count=False, opacity=0.10,
        )

    counts_in: dict[str, int] = {n: 0 for n in names}
    counts_out: dict[str, int] = {n: 0 for n in names}
    out_fps = fr

    kwargs = dict(
        source=str(cfg.video_path),
        conf=cfg.conf, iou=cfg.iou, imgsz=cfg.imgsz, device=cfg.device,
        stream=True, vid_stride=cfg.vid_stride, verbose=False,
    )

    output_video = cfg.output_dir / f"{cfg.video_path.stem}_raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer: cv2.VideoWriter | None = None

    n = 0
    tracker_class_map: dict[int, int] = {}  # tracker_id → class_id (for ghost reconstruction)
    try:
        for result in model.predict(**kwargs):
            scene = result.orig_img.copy()  # BGR ndarray
            detections = sv.Detections.from_ultralytics(result)
            # ROI: filter detections to those whose center is inside the polygon
            if roi_zone is not None and len(detections) > 0:
                inside = roi_zone.trigger(detections)
                detections = detections[inside]
            if tracker is not None:
                detections = tracker.update_with_detections(detections)
                # Remember class for each tracker_id (lost tracks don't expose class_id)
                if detections.tracker_id is not None and detections.class_id is not None:
                    for i in range(len(detections)):
                        tid = detections.tracker_id[i]
                        if tid is not None:
                            tracker_class_map[int(tid)] = int(detections.class_id[i])
                # Ghost detections from Kalman predictions of recently-lost tracks
                if cfg.show_ghost_tracks:
                    ghosts = _ghost_detections(tracker, tracker_class_map, cfg.ghost_max_age)
                    if ghosts is not None and len(ghosts) > 0:
                        detections = sv.Detections.merge([detections, ghosts])

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
                if mask_ann is not None and detections.mask is not None:
                    scene = mask_ann.annotate(scene=scene, detections=detections)
                if cfg.show_box:
                    scene = box_ann.annotate(scene=scene, detections=detections)
                if any(l for l in labels):
                    scene = label_ann.annotate(scene=scene, detections=detections, labels=labels)

            # ROI outline always visible (after detections so it overlays cleanly)
            if roi_annotator is not None:
                scene = roi_annotator.annotate(scene=scene)
                if cfg.roi_label and cfg.roi_polygon:
                    pos = _roi_label_pos(cfg.roi_polygon, cfg.roi_label_position)
                    _draw_label(scene, cfg.roi_label, pos, roi_rgb)

            if line_zone is not None:
                in_mask, out_mask = line_zone.trigger(detections)
                if len(detections) > 0 and detections.class_id is not None:
                    for i in range(len(detections)):
                        cid = int(detections.class_id[i])
                        cname = names[cid] if cid < len(names) else f"class_{cid}"
                        if in_mask[i]:
                            counts_in[cname] = counts_in.get(cname, 0) + 1
                        if out_mask[i]:
                            counts_out[cname] = counts_out.get(cname, 0) + 1
                scene = line_annotator.annotate(frame=scene, line_counter=line_zone)
                if cfg.line_label:
                    pos = _line_label_pos(cfg.line, cfg.line_label_position)
                    _draw_label(scene, cfg.line_label, pos, line_rgb)
                if on_counts is not None:
                    on_counts(dict(counts_in), dict(counts_out))

            # Counts overlay in corner (after line so it sits on top)
            if cfg.show_counts_overlay and (line_zone is not None or roi_zone is not None):
                _draw_counts_overlay(
                    scene, names, counts_in, counts_out, cfg.counts_corner,
                    in_label=cfg.in_label, out_label=cfg.out_label,
                    show_in=cfg.show_in, show_out=cfg.show_out,
                    line_rgb=line_rgb,
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
    )
