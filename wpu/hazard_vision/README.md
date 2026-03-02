# Hazard Vision MVP

A Python MVP for a wearable hazard-detection system using OpenCV + YOLOv8 (ultralytics). Designed for first-person view (camera on neck). Runs **video-first** — no camera required.

## Setup

```bash
cd hazard_vision
pip install -r requirements.txt
```

To create a minimal placeholder video (if you don't have one):

```bash
python create_placeholder_video.py
```

## Quick Start

Use `assets/test.mp4` (create with script above) or your own video path:

```bash
# Video with debug overlay (throttled to video FPS)
python main.py --source video --video_path ./assets/test.mp4 --display 1

# Fast processing (no throttling)
python main.py --source video --video_path ./assets/test.mp4 --display 0 --realtime 0

# With webcam (fails gracefully if no camera)
python main.py --source webcam --cam_index 0 --display 1

# Red light detection (required for green-to-red transition)
python main.py --source video --video_path ./assets/test_footage.mp4 --display 1 --enable_red_light 1

# Debug red light (log red_ratio per frame for traffic lights)
python main.py --source video --video_path ./assets/test_footage.mp4 --display 1 --enable_red_light 1 --debug_red_light 1
```

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | video | `video` or `webcam` |
| `--video_path` | ./assets/test.mp4 | Input video path |
| `--cam_index` | 0 | Webcam index |
| `--model` | yolov8n.pt | YOLO weights |
| `--imgsz` | 640 | Inference size |
| `--conf` | 0.5 | Detection confidence threshold |
| `--display` | 0 | 1 = show overlay window |
| `--realtime` | 1 | 1 = throttle to video FPS |
| `--max_fps` | - | Optional processing cap |
| `--loop` | 0 | Repeat video |
| `--start_sec`, `--end_sec` | - | Seek range (seconds) |
| `--enable_red_light` | 0 | Enable red-light hazard (required for green-to-red detection) |
| `--red_ratio_threshold` | 0.12 | Red pixel ratio threshold in top bulb ROI |
| `--red_bulb_region` | top | `top` (US layout) or `full` (legacy) |
| `--debug_red_light` | 0 | Log red_ratio and bbox for traffic lights each frame |
| `--corridor_width_ratio` | 0.35 | Central corridor width |
| `--history_len` | 10 | Track history length |
| `--cooldown_s` | 0.5 | Debounce per hazard (seconds) |
| `--skip` | 0 | Process every (N+1)th frame |

## Implemented Hazards

1. **VEHICLE_APPROACHING** — Cars/trucks/buses in central corridor with growing bbox (TTC proxy)
2. **PERSON_ON_COLLISION_COURSE** — Person in corridor with bbox growth
3. **POLE_AHEAD** — Traffic lights, stop signs, fire hydrants; else tall/thin heuristic
4. **RED_LIGHT** (optional, `--enable_red_light 1`) — Top bulb ROI (US layout), HSV red ratio, green rejection

## Output Format

```
HAZARD: VEHICLE_APPROACHING (id=12 conf=0.71 ttc_proxy=1.4 bbox=...)
HAZARD: PERSON_ON_COLLISION_COURSE (id=5 conf=0.83 corridor=0.35 growth=...)
HAZARD: POLE_AHEAD (id=9 conf=0.55 distance_proxy=... bbox=...)
HAZARD: RED_LIGHT (id=9 conf=0.6 roi_red_ratio=0.2 ...)
```

## Known Limitations

- Uses stock YOLOv8n (COCO). No custom weights or datasets.
- Rule-based on bbox growth and corridor position; no custom-trained classifiers.
- Red-light detection uses top-third bulb ROI (US layout: red on top). Use `--red_bulb_region full` for legacy behavior. May misclassify in varied lighting.
- TTC proxy is approximate; not calibrated to real distance.
- Pole heuristic (tall/thin) can have false positives.

## Next Steps (Future Hazards)

- **Hazard 5 (ground-level threats)**: Uneven terrain, trip hazards — would need depth or segmentation.
- **Hazard 6 (head-level hazards)**: Low doorframes, bridges — similar corridor + height logic.

## Architecture

Modular design for future CoreML/TFLite or server inference swap:

- `src/video_stream.py` — FrameSource, VideoFileSource, CameraSource
- `src/detector.py` — YOLOv8 + ByteTrack
- `src/tracker.py` — Track history for growth/TTC
- `src/hazard_logic.py` — Hazard classification
- `src/notifier.py` — Debouncing, console output

## Mobile / Server Port

To port to mobile (CoreML/TFLite) or server:

1. Replace `src/detector.py` with platform-specific inference.
2. Keep `FrameSource` interface; implement mobile camera or server stream.
3. Reuse `hazard_logic`, `tracker`, `notifier` as-is.
