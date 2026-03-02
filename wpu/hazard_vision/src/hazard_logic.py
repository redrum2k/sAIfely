"""Hazard classification: approaching vehicles, person collision, pole ahead, red light."""

from collections import deque

import numpy as np

from .utils import (
    HazardEvent,
    TrackedObject,
    bbox_height,
    is_in_corridor,
    is_tall_thin_bbox,
    red_pixel_ratio_hsv,
)

# YOLO COCO class names (yolov8n.pt)
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
PERSON_CLASS = "person"
POLE_LIKE_CLASSES = {"traffic light", "stop sign", "fire hydrant", "bench"}  # pole-like


def compute_growth_rate(history: deque[TrackedObject], use_area: bool = True) -> float:
    """Growth rate of bbox area or height over history. Positive = approaching."""
    if len(history) < 2:
        return 0.0
    vals = [
        h.area if use_area else bbox_height(h.bbox_xyxy)
        for h in list(history)
    ]
    if vals[-1] <= 0:
        return 0.0
    growth = (vals[-1] - vals[0]) / vals[0] if vals[0] > 0 else 0
    return growth / max(1, len(vals) - 1)


def compute_ttc_proxy(history: deque[TrackedObject], fps: float = 20.0) -> float | None:
    """TTC-like proxy (seconds) from bbox growth. Larger growth -> smaller TTC."""
    if len(history) < 3 or fps <= 0:
        return None
    areas = [h.area for h in list(history) if h.area > 0]
    if len(areas) < 2 or areas[0] <= 0:
        return None
    growth_per_frame = (areas[-1] - areas[0]) / (len(areas) - 1)
    if growth_per_frame <= 0:
        return None
    # Approximate: ttc ~ 1 / (growth_rate * fps)
    growth_rate = growth_per_frame / areas[0]
    ttc = 1.0 / (growth_rate * fps) if growth_rate > 0 else None
    return round(ttc, 2) if ttc is not None and ttc > 0 else None


def check_vehicle_approaching(
    obj: TrackedObject,
    history: deque[TrackedObject],
    frame_width: float,
    corridor_width_ratio: float,
    min_history: int = 3,
    min_growth: float = 0.1,
    fps: float = 20.0,
) -> HazardEvent | None:
    """Hazard 1: Car/truck/bus approaching (in corridor, bbox growing)."""
    if obj.cls_name not in VEHICLE_CLASSES:
        return None
    if len(history) < min_history:
        return None
    if not is_in_corridor(obj.center[0], frame_width, corridor_width_ratio):
        return None
    growth = compute_growth_rate(history, use_area=True)
    if growth < min_growth:
        return None
    ttc = compute_ttc_proxy(history, fps)
    return HazardEvent(
        type="VEHICLE_APPROACHING",
        severity="medium",
        object_id=obj.id,
        confidence=obj.conf,
        details={
            "ttc_proxy": ttc,
            "growth": round(growth, 3),
            "bbox": obj.bbox_xyxy,
        },
        timestamp=obj.timestamp,
    )


def check_person_collision_course(
    obj: TrackedObject,
    history: deque[TrackedObject],
    frame_width: float,
    corridor_width_ratio: float,
    min_history: int = 3,
    min_growth: float = 0.08,
) -> HazardEvent | None:
    """Hazard 3: Person in central corridor + bbox growing (collision path)."""
    if obj.cls_name != PERSON_CLASS:
        return None
    if len(history) < min_history:
        return None
    if not is_in_corridor(obj.center[0], frame_width, corridor_width_ratio):
        return None
    growth = compute_growth_rate(history, use_area=True)
    if growth < min_growth:
        return None
    return HazardEvent(
        type="PERSON_ON_COLLISION_COURSE",
        severity="high",
        object_id=obj.id,
        confidence=obj.conf,
        details={
            "corridor": corridor_width_ratio,
            "growth": round(growth, 3),
            "bbox": obj.bbox_xyxy,
        },
        timestamp=obj.timestamp,
    )


def check_pole_ahead(
    obj: TrackedObject,
    frame_width: float,
    frame_height: float,
    corridor_width_ratio: float,
    min_height_ratio: float = 0.15,
) -> HazardEvent | None:
    """Hazard 4: Pole/static obstacle in central corridor."""
    in_corridor = is_in_corridor(obj.center[0], frame_width, corridor_width_ratio)
    if not in_corridor:
        return None
    # Use YOLO class if pole-like
    if obj.cls_name in POLE_LIKE_CLASSES:
        return HazardEvent(
            type="POLE_AHEAD",
            severity="medium",
            object_id=obj.id,
            confidence=obj.conf,
            details={
                "distance_proxy": "yolo_class",
                "bbox": obj.bbox_xyxy,
                "class": obj.cls_name,
            },
            timestamp=obj.timestamp,
        )
    # Heuristic: tall/thin bbox in corridor with sufficient height
    h = bbox_height(obj.bbox_xyxy)
    if h >= frame_height * min_height_ratio and is_tall_thin_bbox(obj.bbox_xyxy):
        return HazardEvent(
            type="POLE_AHEAD",
            severity="medium",
            object_id=obj.id,
            confidence=obj.conf,
            details={
                "distance_proxy": "heuristic",
                "bbox": obj.bbox_xyxy,
            },
            timestamp=obj.timestamp,
        )
    return None


def check_red_light(
    frame_bgr: np.ndarray,
    obj: TrackedObject,
    red_ratio_threshold: float = 0.15,
) -> HazardEvent | None:
    """Hazard 2 (optional): Traffic light in red state (HSV red ROI)."""
    if obj.cls_name != "traffic light":
        return None
    ratio = red_pixel_ratio_hsv(frame_bgr, obj.bbox_xyxy)
    if ratio < red_ratio_threshold:
        return None
    return HazardEvent(
        type="RED_LIGHT",
        severity="high",
        object_id=obj.id,
        confidence=obj.conf,
        details={
            "roi_red_ratio": round(ratio, 3),
            "bbox": obj.bbox_xyxy,
        },
        timestamp=obj.timestamp,
    )


def evaluate_hazards(
    frame_bgr: np.ndarray,
    objects: list[TrackedObject],
    history: dict[int, deque[TrackedObject]],
    frame_width: int,
    frame_height: int,
    corridor_width_ratio: float,
    enable_red_light: bool,
    fps: float = 20.0,
) -> list[HazardEvent]:
    """Evaluate all hazards for current frame."""
    events: list[HazardEvent] = []
    for obj in objects:
        h = history.get(obj.id, deque())
        if e := check_vehicle_approaching(
            obj, h, frame_width, corridor_width_ratio, fps=fps
        ):
            events.append(e)
        if e := check_person_collision_course(
            obj, h, frame_width, corridor_width_ratio
        ):
            events.append(e)
        if e := check_pole_ahead(
            obj, frame_width, frame_height, corridor_width_ratio
        ):
            events.append(e)
        if enable_red_light and (e := check_red_light(frame_bgr, obj)):
            events.append(e)
    return events
