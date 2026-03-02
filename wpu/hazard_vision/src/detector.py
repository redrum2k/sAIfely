"""YOLOv8 detection + tracking wrapper using ByteTrack."""

from typing import Any

import numpy as np

from .utils import TrackedObject, bbox_area, bbox_center


class YOLODetector:
    """YOLOv8 model with track() for detection + ByteTrack tracking."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        imgsz: int = 640,
        conf: float = 0.5,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self._class_names = self.model.names or {}

    def track(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
    ) -> list[TrackedObject]:
        """Run YOLO track() and return list of TrackedObject."""
        results = self.model.track(
            frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            persist=True,
            verbose=False,
        )
        objects: list[TrackedObject] = []
        if not results or len(results) == 0:
            return objects
        r = results[0]
        boxes = r.boxes
        if boxes is None:
            return objects
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            cls_id = int(boxes.cls[i].item())
            conf_val = float(boxes.conf[i].item())
            try:
                tid = int(boxes.id[i].item()) if boxes.id is not None else -1 - i
            except (AttributeError, TypeError):
                tid = -1 - i
            cls_name = self._class_names.get(cls_id, f"class_{cls_id}")
            center = bbox_center(xyxy)
            area = bbox_area(xyxy)
            objects.append(
                TrackedObject(
                    id=tid,
                    cls_name=cls_name,
                    conf=conf_val,
                    bbox_xyxy=tuple(float(x) for x in xyxy),
                    center=center,
                    area=area,
                    timestamp=timestamp,
                )
            )
        return objects
