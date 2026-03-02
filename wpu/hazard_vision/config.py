"""Default configuration for Hazard Vision MVP."""

DEFAULTS = {
    "source": "video",
    "video_path": "./assets/test.mp4",
    "cam_index": 0,
    "model": "yolov8n.pt",
    "imgsz": 640,
    "conf": 0.5,
    "display": 0,
    "realtime": 1,
    "max_fps": None,
    "loop": 0,
    "start_sec": None,
    "end_sec": None,
    "enable_red_light": 0,
    "red_ratio_threshold": 0.12,
    "red_bulb_region": "top",
    "corridor_width_ratio": 0.35,
    "history_len": 10,
    "cooldown_s": 0.5,
    "skip": 0,
    "debug_red_light": 0,
}
