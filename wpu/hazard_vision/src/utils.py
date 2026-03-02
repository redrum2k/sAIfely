"""Utility functions and data structures for Hazard Vision."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TrackedObject:
    """A tracked object from YOLO + tracker."""

    id: int
    cls_name: str
    conf: float
    bbox_xyxy: tuple[float, float, float, float]
    center: tuple[float, float]
    area: float
    timestamp: float


@dataclass
class HazardEvent:
    """A hazard event to be reported."""

    type: str
    severity: str
    object_id: int | None
    confidence: float
    details: dict[str, Any]
    timestamp: float


def bbox_center(bbox_xyxy: np.ndarray | tuple) -> tuple[float, float]:
    """Compute center of bbox [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bbox_area(bbox_xyxy: np.ndarray | tuple) -> float:
    """Compute area of bbox."""
    x1, y1, x2, y2 = bbox_xyxy
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return w * h


def bbox_height(bbox_xyxy: np.ndarray | tuple) -> float:
    """Compute height of bbox."""
    _, y1, _, y2 = bbox_xyxy
    return max(0, y2 - y1)


def is_in_corridor(
    center_x: float,
    frame_width: float,
    corridor_width_ratio: float,
) -> bool:
    """Check if x-coordinate is within central corridor."""
    cx = frame_width / 2
    half = frame_width * corridor_width_ratio / 2
    return abs(center_x - cx) <= half


def corridor_bounds(frame_width: float, corridor_width_ratio: float) -> tuple[int, int]:
    """Return (left, right) x bounds of central corridor in pixels."""
    cx = frame_width / 2
    half = frame_width * corridor_width_ratio / 2
    return (int(cx - half), int(cx + half))


def is_tall_thin_bbox(bbox_xyxy: np.ndarray | tuple, aspect_threshold: float = 0.4) -> bool:
    """Heuristic: tall/thin bbox (e.g. pole-like). aspect = width/height < threshold."""
    x1, y1, x2, y2 = bbox_xyxy
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    aspect = w / h
    return aspect < aspect_threshold


def crop_traffic_light_bulb(
    bbox_xyxy: tuple[float, float, float, float],
    bulb: str = "red",
) -> tuple[float, float, float, float]:
    """Return (x1,y1,x2,y2) for bulb region. US layout: red=top, yellow=mid, green=bot."""
    x1, y1, x2, y2 = bbox_xyxy
    h = y2 - y1
    if bulb == "red":
        return (x1, y1, x2, y1 + h / 3)
    if bulb == "yellow":
        return (x1, y1 + h / 3, x2, y1 + 2 * h / 3)
    if bulb == "green":
        return (x1, y1 + 2 * h / 3, x2, y2)
    return (x1, y1, x2, y2)


def red_pixel_ratio_hsv(
    frame_bgr: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    bulb_region: str = "top",
) -> float:
    """Crop ROI (bulb region of traffic light) and compute ratio of red pixels. Returns 0..1."""
    import cv2

    if bulb_region in ("red", "top"):
        roi_bbox = crop_traffic_light_bulb(bbox_xyxy, bulb="red")
    elif bulb_region == "full":
        roi_bbox = bbox_xyxy
    else:
        roi_bbox = crop_traffic_light_bulb(bbox_xyxy, bulb=bulb_region)

    x1, y1, x2, y2 = map(int, roi_bbox)
    h_frame, w_frame = frame_bgr.shape[:2]
    x1 = max(0, min(x1, w_frame - 1))
    x2 = max(x1 + 1, min(x2, w_frame))
    y1 = max(0, min(y1, h_frame - 1))
    y2 = max(y1 + 1, min(y2, h_frame))
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Relaxed HSV: lower sat/value for overexposed/dim red; wider hue for LED
    lower1 = np.array([0, 50, 50])
    upper1 = np.array([15, 255, 255])
    lower2 = np.array([165, 50, 50])
    upper2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    total = roi.shape[0] * roi.shape[1]
    return float(np.sum(red_mask > 0)) / total if total > 0 else 0.0


def green_pixel_ratio_hsv(
    frame_bgr: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
) -> float:
    """Ratio of green pixels in top bulb region. High green = green light reflected; avoid red false positive."""
    import cv2

    roi_bbox = crop_traffic_light_bulb(bbox_xyxy, bulb="red")
    x1, y1, x2, y2 = map(int, roi_bbox)
    h_frame, w_frame = frame_bgr.shape[:2]
    x1 = max(0, min(x1, w_frame - 1))
    x2 = max(x1 + 1, min(x2, w_frame))
    y1 = max(0, min(y1, h_frame - 1))
    y2 = max(y1 + 1, min(y2, h_frame))
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 50, 50])
    upper = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower, upper)
    total = roi.shape[0] * roi.shape[1]
    return float(np.sum(green_mask > 0)) / total if total > 0 else 0.0
