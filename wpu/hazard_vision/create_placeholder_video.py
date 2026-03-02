#!/usr/bin/env python3
"""Create a minimal placeholder video for testing when no video is available."""

import numpy as np

try:
    import cv2
except ImportError:
    print("opencv-python required: pip install opencv-python")
    raise SystemExit(1)

path = "assets/test.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(path, fourcc, 10.0, (640, 480))
for _ in range(30):  # ~3 seconds
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)  # dark gray
    out.write(frame)
out.release()
print(f"Created {path} (30 frames, 640x480)")
