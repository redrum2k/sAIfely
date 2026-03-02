#!/usr/bin/env python3
"""Hazard Vision MVP: wearable hazard detection using OpenCV + YOLOv8."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from config import DEFAULTS
from src.detector import YOLODetector
from src.hazard_logic import evaluate_hazards
from src.notifier import HazardNotifier
from src.tracker import TrackHistory
from src.utils import corridor_bounds
from src.video_stream import CameraSource, VideoFileSource


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hazard Vision - wearable hazard detection")
    p.add_argument("--source", choices=["video", "webcam"], default=DEFAULTS["source"])
    p.add_argument("--video_path", default=DEFAULTS["video_path"])
    p.add_argument("--cam_index", type=int, default=DEFAULTS["cam_index"])
    p.add_argument("--model", default=DEFAULTS["model"])
    p.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"])
    p.add_argument("--conf", type=float, default=DEFAULTS["conf"])
    p.add_argument("--display", type=int, choices=[0, 1], default=DEFAULTS["display"])
    p.add_argument("--realtime", type=int, choices=[0, 1], default=DEFAULTS["realtime"])
    p.add_argument("--max_fps", type=float, default=DEFAULTS["max_fps"])
    p.add_argument("--loop", type=int, choices=[0, 1], default=DEFAULTS["loop"])
    p.add_argument("--start_sec", type=float, default=DEFAULTS["start_sec"])
    p.add_argument("--end_sec", type=float, default=DEFAULTS["end_sec"])
    p.add_argument("--enable_red_light", type=int, choices=[0, 1], default=DEFAULTS["enable_red_light"])
    p.add_argument("--corridor_width_ratio", type=float, default=DEFAULTS["corridor_width_ratio"])
    p.add_argument("--history_len", type=int, default=DEFAULTS["history_len"])
    p.add_argument("--cooldown_s", type=float, default=DEFAULTS["cooldown_s"])
    p.add_argument("--skip", type=int, default=DEFAULTS["skip"])
    return p.parse_args()


def create_frame_source(args: argparse.Namespace) -> "VideoFileSource | CameraSource":
    if args.source == "video":
        return VideoFileSource(
            args.video_path,
            realtime=bool(args.realtime),
            max_fps=args.max_fps,
            start_sec=args.start_sec,
            end_sec=args.end_sec,
            loop=bool(args.loop),
        )
    return CameraSource(args.cam_index)


def draw_overlay(
    frame: np.ndarray,
    objects: list,
    hazards: list,
    corridor_width_ratio: float,
) -> np.ndarray:
    """Draw bboxes, labels, corridor, hazard annotations."""
    out = frame.copy()
    h, w = out.shape[:2]
    left, right = corridor_bounds(w, corridor_width_ratio)
    cv2.rectangle(out, (left, 0), (right, h), (0, 255, 255), 2)
    cv2.putText(
        out,
        "corridor",
        (left, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        1,
    )
    for obj in objects:
        x1, y1, x2, y2 = map(int, obj.bbox_xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{obj.cls_name} {obj.conf:.2f} id={obj.id}"
        cv2.putText(out, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for ev in hazards:
        if ev.object_id is not None:
            obj = next((o for o in objects if o.id == ev.object_id), None)
            if obj:
                x1, y1, x2, y2 = map(int, obj.bbox_xyxy)
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    out,
                    f"HAZARD: {ev.type}",
                    (x1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
    return out


def main() -> int:
    args = parse_args()
    if args.source == "video" and not Path(args.video_path).exists():
        print(f"Video file not found: {args.video_path}")
        print("Add a test video to assets/ or use --video_path <path>.")
        return 1

    try:
        source = create_frame_source(args)
    except (FileNotFoundError, RuntimeError) as e:
        print(e)
        return 1

    detector = YOLODetector(model_path=args.model, imgsz=args.imgsz, conf=args.conf)
    track_history = TrackHistory(history_len=args.history_len)
    notifier = HazardNotifier(cooldown_s=args.cooldown_s)

    # Estimate FPS for TTC (video or default)
    fps = 20.0
    if args.source == "video" and args.realtime:
        cap = cv2.VideoCapture(args.video_path)
        f = cap.get(cv2.CAP_PROP_FPS)
        fps = f if f and f > 0 else 20.0
        cap.release()
    if args.max_fps is not None and args.max_fps > 0:
        fps = args.max_fps

    try:
        for frame_idx, (frame, timestamp_s, _) in enumerate(source):
            if args.skip > 0 and frame_idx % (args.skip + 1) != 0:
                continue

            frame_resized = frame
            if frame.shape[1] != args.imgsz:
                scale = args.imgsz / frame.shape[1]
                new_w = args.imgsz
                new_h = int(frame.shape[0] * scale)
                frame_resized = cv2.resize(frame, (new_w, new_h))

            objects = detector.track(frame_resized, timestamp_s)
            history = track_history.update(objects)

            h_img, w_img = frame_resized.shape[:2]
            hazards = evaluate_hazards(
                frame_resized,
                objects,
                history,
                w_img,
                h_img,
                args.corridor_width_ratio,
                bool(args.enable_red_light),
                fps=fps,
            )

            for ev in hazards:
                if notifier.should_report(ev, timestamp_s):
                    notifier.report(ev)

            if args.display:
                display_frame = draw_overlay(
                    frame_resized, objects, hazards, args.corridor_width_ratio
                )
                cv2.imshow("Hazard Vision", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except RuntimeError as e:
        print(e)
        return 1
    finally:
        source.release()
        if args.display:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
