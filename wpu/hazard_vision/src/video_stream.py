"""Unified frame source interface: VideoFileSource and CameraSource."""

from abc import ABC, abstractmethod
from typing import Iterator

import cv2
import numpy as np


class FrameSource(ABC):
    """Abstract base for frame sources. Same interface for video and webcam."""

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[np.ndarray, float, int]]:
        """Yield (frame_bgr, timestamp_s, frame_idx)."""
        ...

    @abstractmethod
    def release(self) -> None:
        """Release resources."""
        ...


class VideoFileSource(FrameSource):
    """Video file source with realtime throttling, loop, seek."""

    def __init__(
        self,
        video_path: str,
        *,
        realtime: bool = True,
        max_fps: float | None = None,
        start_sec: float | None = None,
        end_sec: float | None = None,
        loop: bool = False,
    ):
        self.video_path = video_path
        self.realtime = realtime
        self.max_fps = max_fps
        self.start_sec = start_sec if start_sec is not None else 0.0
        self.end_sec = end_sec
        self.loop = loop
        self._cap: cv2.VideoCapture | None = None
        self._fps: float = 30.0

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and fps > 0 else 30.0
        if self.start_sec > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, self.start_sec * 1000)
        return cap

    def __iter__(self) -> Iterator[tuple[np.ndarray, float, int]]:
        import time

        frame_idx = 0
        while True:
            if self._cap is None:
                self._cap = self._open()
            ret, frame = self._cap.read()
            if not ret or frame is None:
                if self.loop:
                    self._cap.release()
                    self._cap = None
                    continue
                break
            pos_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp_s = pos_ms / 1000.0
            if self.end_sec is not None and timestamp_s >= self.end_sec:
                if self.loop:
                    self._cap.release()
                    self._cap = None
                    continue
                break
            target_fps = self.max_fps if self.max_fps is not None else self._fps
            if self.realtime and target_fps > 0:
                time.sleep(1.0 / target_fps)
            yield (frame, timestamp_s, frame_idx)
            frame_idx += 1

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class CameraSource(FrameSource):
    """Webcam source. Fails gracefully if no camera available."""

    def __init__(self, cam_index: int = 0):
        self.cam_index = cam_index
        self._cap: cv2.VideoCapture | None = None

    def __iter__(self) -> Iterator[tuple[np.ndarray, float, int]]:
        import time

        try:
            self._cap = cv2.VideoCapture(self.cam_index)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"No webcam available at index {self.cam_index}. "
                    "Please connect a camera or use --source video --video_path <file>."
                )
        except Exception as e:
            raise RuntimeError(
                f"Camera unavailable: {e}. "
                "Use --source video --video_path <path> to run with a video file instead."
            ) from e

        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0
        t0 = time.perf_counter()
        while True:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                break
            timestamp_s = time.perf_counter() - t0
            yield (frame, timestamp_s, frame_idx)
            frame_idx += 1
            time.sleep(max(0, 1.0 / fps - 0.001))

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
