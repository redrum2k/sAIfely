You are an expert computer vision + ML engineer. Build a Python MVP for a wearable hazard-detection system using OpenCV + YOLOv8 (ultralytics) that runs on a laptop/PC at 15–20 FPS. The camera is worn on the user’s neck facing forward (first-person view). Important: we don’t know the final camera yet, so the MVP must be VIDEO-FIRST and run with a video file even with no camera connected.
1) Core goals
Process frames at ~15–20 FPS (or controllable via throttling flags).
Detect hazards and print warnings to the console with debouncing (don’t spam every frame).
Provide a debug overlay window when enabled.
Implement at least 3 hazards in MVP (prefer 3–4):
(1) approaching cars, (3) approaching people on collision course, (4) poles/static obstacles ahead.
Optional: (2) red light if feasible without custom training.
2) Hazards (full list for future scaling)
Cars rapidly approaching the person
Red light
People coming at the person and not turning (collision path)
Objects that pose collision threat (light pole / traffic light pole / sign pole)
Ground-level threats (uneven terrain / trip hazards)
Head-level hazards (low doorframes / bridges)
MVP requirement: implement hazards 1 + 3 + 4 (red-light optional behind a flag).
3) Strict video-first requirements (MUST)
Must run without a camera
Default should run on a video file:
python main.py --source video --video_path ./assets/test.mp4 --display 1
Also support fast processing (no sleeping):
python main.py --source video --video_path ./assets/test.mp4 --display 0 --realtime 0
Video timing controls (MUST)
Add CLI flags:
--realtime 1/0
If 1: throttle to match video FPS (CAP_PROP_FPS, fallback 30).
If 0: process as fast as possible.
--max_fps <float> optional cap (e.g., 20). If set, throttle so processing does not exceed it.
--start_sec, --end_sec (seek and stop early)
--loop 1/0 (repeat the video for testing)
Camera placeholder (MUST, but can be “stubby”)
Implement a webcam source module with the SAME interface as video source:
python main.py --source webcam --cam_index 0 --display 1
If no camera is available, it must fail gracefully with a friendly message and clean exit (no ugly stack trace).
4) Technical approach (MUST follow)
Use YOLOv8 detection + tracking to estimate approaching motion.
Use ultralytics YOLO (default yolov8n.pt) for speed.
Use YOLOv8 built-in track() (ByteTrack) OR a lightweight tracker (SORT) to maintain stable IDs.
For “approaching” compute a TTC-like proxy using bbox growth:
approaching if bbox area/height increases consistently over last N frames
compute growth_rate and ttc_proxy from history
For “person collision course”:
person tracked near central corridor of frame for N frames AND bbox grows → hazard
For “static obstacle ahead / poles”:
If YOLO has a relevant class (traffic light, stop sign, fire hydrant, pole-like objects), use it.
Otherwise implement a heuristic: tall/thin bbox near central corridor + sufficient height in frame.
Red light detection (optional behind --enable_red_light 1):
Detect “traffic light” then crop ROI and use HSV red pixel ratio to classify red state.
Document limitations; do not assume custom weights or datasets.
Performance:
Resize frames (e.g. width ~640) and run YOLO at 640 (--imgsz flag).
Use GPU if available; otherwise CPU.
Optional --skip N to process every (N+1)th frame if needed.
5) Output requirements (MUST)
Console prints hazard events in a structured way, e.g.:
HAZARD: VEHICLE_APPROACHING (id=12 conf=0.71 ttc_proxy=1.4s bbox=...)
HAZARD: PERSON_ON_COLLISION_COURSE (id=5 conf=0.83 corridor=0.35 growth=...)
HAZARD: POLE_AHEAD (id=9 conf=0.55 distance_proxy=... bbox=...)
HAZARD: RED_LIGHT (conf=... roi_red_ratio=...) (only if enabled)
Debounce:
Implement per-(hazard_type, object_id) cooldown, default --cooldown_s 0.5.
Debug overlay (when --display 1):
Draw bboxes, class, confidence, track ID
Draw hazard labels
Draw central corridor region
6) Code structure (MUST implement exactly)
Create a repo-style project:
hazard_vision/
  main.py
  requirements.txt
  config.py
  src/
    video_stream.py
    detector.py
    tracker.py
    hazard_logic.py
    notifier.py
    utils.py
  assets/
    test.mp4  (placeholder only; do NOT download)
  README.md
7) Interfaces + data structures (MUST be explicit)
Unified frame source interface
In src/video_stream.py define:
class FrameSource:
__iter__() yields (frame_bgr: np.ndarray, timestamp_s: float, frame_idx: int)
release()
Implement:
VideoFileSource(FrameSource)
CameraSource(FrameSource) (placeholder but functional if webcam exists; graceful failure if not)
Standard structures
Define dataclasses:
TrackedObject: id, cls_name, conf, bbox_xyxy, center, area, timestamp
Track history: dict[int, deque[TrackedObject]] with --history_len default ~10
HazardEvent: type, severity, object_id, confidence, details, timestamp
8) CLI requirements (MUST)
Support:
--source {video,webcam}
--video_path
--cam_index
--model
--imgsz
--conf
--display 1/0
--realtime 1/0
--max_fps
--loop 1/0
--start_sec
--end_sec
--enable_red_light 1/0
--corridor_width_ratio (default ~0.35)
--history_len (default ~10)
--cooldown_s (default ~0.5)
--skip (default 0)
9) Deliverables (MUST)
Working code end-to-end in video mode (default) and webcam mode (placeholder + graceful error).
Hazard detection for MVP hazards (1, 3, 4) with debouncing.
Debug overlay when enabled.
requirements.txt (ultralytics, opencv-python, numpy, etc).
README.md with setup + commands + which hazards are implemented + known limitations + next steps for hazards #5 and #6 + notes on future mobile/server port.
10) Important constraints
Do NOT assume custom-trained weights exist.
Do NOT download datasets.
MVP must be rule-based on top of YOLO detections + tracking.
Keep architecture modular for later swap to CoreML/TFLite or server inference.
Start by implementing the simplest stable MVP: hazards (1), (3), (4) in video mode. Then add optional red-light behind a flag. Produce the full code and README.