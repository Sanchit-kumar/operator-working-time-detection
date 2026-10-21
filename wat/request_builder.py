"""Builds one LLM request per frame: full frame with numbered boxes + two crops per worker (~2 s ago, now)."""

from collections import deque

import cv2
import numpy as np

from .detect_track import Track
from .llm import FrameRequest


def encode_jpeg(img: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def resize_long_side(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = size / max(h, w)
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA   # upscale tiny far-away workers
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)


def crop(frame: np.ndarray, box, margin: float) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    x2, y2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
    return frame[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]


def side_by_side(crops: list[np.ndarray], size: int) -> np.ndarray:
    """One image per worker: [~2 s ago | now], each crop scaled to `size` px tall, split by a white bar.
    Vision models charge per image, so this halves the image count versus sending the crops separately."""
    scaled = [cv2.resize(c, (max(1, round(c.shape[1] * size / c.shape[0])), size),
                         interpolation=cv2.INTER_CUBIC if c.shape[0] < size else cv2.INTER_AREA) for c in crops]
    bar = np.full((size, max(4, size // 32), 3), 255, np.uint8)
    return np.hstack([scaled[0], bar, *scaled[1:]])


def draw_numbered_boxes(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    out = frame.copy()
    thick = max(2, frame.shape[1] // 640)
    scale = max(0.6, frame.shape[1] / 1600)
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), thick)
        label = str(t.track_id)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (0, 0, 255), -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick)
    return out


class FrameHistory:
    """Recent processed frames with their track boxes, to crop each worker from ~history_s ago."""

    def __init__(self, history_s: float):
        self.history_s = history_s
        self.items: deque[tuple[float, np.ndarray, dict[int, tuple]]] = deque()

    def add(self, t: float, frame: np.ndarray, tracks: list[Track]):
        self.items.append((t, frame, {tr.track_id: tr.box for tr in tracks}))
        while self.items and t - self.items[0][0] > self.history_s + 1.0:
            self.items.popleft()

    def at(self, t: float):
        """The stored frame closest to time t."""
        return min(self.items, key=lambda it: abs(it[0] - t))


def labelled(img: np.ndarray, label: str) -> np.ndarray:
    """Put a black banner with the worker id above a close-up."""
    h = max(24, img.shape[0] // 8)
    banner = np.zeros((h, img.shape[1], 3), np.uint8)
    cv2.putText(banner, label, (4, int(h * 0.78)), cv2.FONT_HERSHEY_SIMPLEX, h / 32, (255, 255, 255), 2)
    return np.vstack([banner, img])


def contact_sheet(cells: list[np.ndarray], max_width: int) -> np.ndarray:
    """Grid of labelled close-ups, rows packed left to right, white gaps between cells."""
    gap, rows, row, width = 8, [], [], 0
    for c in cells:
        if row and width + c.shape[1] + gap > max_width:
            rows.append(row)
            row, width = [], 0
        row.append(c)
        width += c.shape[1] + gap
    rows.append(row)
    out_w = max(sum(c.shape[1] + gap for c in r) for r in rows)
    out_rows = []
    for r in rows:
        h = max(c.shape[0] for c in r)
        canvas = np.full((h + gap, out_w, 3), 255, np.uint8)
        x = 0
        for c in r:
            canvas[:c.shape[0], x:x + c.shape[1]] = c
            x += c.shape[1] + gap
        out_rows.append(canvas)
    return np.vstack(out_rows)


def build_requests(t: float, frame: np.ndarray, tracks: list[Track], history: FrameHistory,
                   cfg: dict) -> list[FrameRequest]:
    """One request per chunk of at most `max_workers_per_call` workers (normally just one request).

    layout "sheet": full frame + one contact sheet of all workers (2 images per call, cheapest).
    layout "per_worker": full frame + one image per worker.
    """
    q = cfg["jpeg_quality"]
    full = encode_jpeg(resize_long_side(draw_numbered_boxes(frame, tracks), cfg["frame_width"]), q)
    _, past_frame, past_boxes = history.at(t - cfg["history_s"])

    pairs = []
    for tr in tracks:
        past_box = past_boxes.get(tr.track_id, tr.box)     # worker may be new: fall back to the current box
        crops = [crop(past_frame, past_box, cfg["crop_margin"]), crop(frame, tr.box, cfg["crop_margin"])]
        pairs.append((tr.track_id, side_by_side(crops, cfg["crop_size"])))

    requests, n = [], cfg["max_workers_per_call"]
    for i in range(0, len(pairs), n):
        chunk = pairs[i:i + n]
        ids = [tid for tid, _ in chunk]
        images = [("Full frame with numbered worker boxes.", full)]
        if cfg.get("layout", "sheet") == "sheet":
            sheet = contact_sheet([labelled(img, f"id {tid}") for tid, img in chunk], cfg["sheet_width"])
            images.append((f"Close-ups of workers {', '.join(map(str, ids))}, each labelled with its id.",
                           encode_jpeg(sheet, q)))
        else:
            images += [(f"Worker {tid}: left = ~2 s ago, right = now.", encode_jpeg(img, q)) for tid, img in chunk]
        requests.append(FrameRequest(t, ids, images))
    return requests
