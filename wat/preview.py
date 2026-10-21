"""Build the exact request the pipeline would send at a given video time, for inspection (and one optional call)."""

import cv2
import numpy as np

from .detect_track import PersonTracker
from .llm import FrameRequest
from .request_builder import FrameHistory, build_requests


def preview_request(source: str, at_s: float, cfg: dict) -> FrameRequest | None:
    """Track from the start of `source` up to `at_s` and return the first request built there (None: no workers)."""
    det, llm = cfg["detector"], cfg["llm"]
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(fps / det["fps"]))
    tracker, history = PersonTracker(det), FrameHistory(llm["history_s"])
    first_seen: dict[int, float] = {}
    idx, frame, tracks, t = -1, None, [], 0.0
    while cap.grab():
        idx += 1
        if idx % stride:
            continue
        _, frame = cap.retrieve()
        t = idx / fps
        tracks = tracker.update(frame)
        for tr in tracks:
            first_seen.setdefault(tr.track_id, t)
        history.add(t, frame, tracks)
        if t >= at_s:
            break
    cap.release()
    eligible = [tr for tr in tracks if t - first_seen[tr.track_id] >= cfg["tracks"]["min_age_s"]]
    if frame is None or not eligible:
        return None
    return build_requests(t, frame, eligible, history, llm)[0]


def decode(jpeg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
