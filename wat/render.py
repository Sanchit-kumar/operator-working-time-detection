"""Live overlay: box colour = smoothed state, label = ID, state, last activity and detection class + confidence."""

import cv2
import numpy as np

from .activities import IDLE, WORKING
from .detect_track import Track

COLORS = {WORKING: (60, 200, 60), IDLE: (40, 40, 230)}   # BGR; anything else is grey
GREY = (160, 160, 160)


def id_color(track_id: int) -> tuple[int, int, int]:
    """Stable, distinct colour per track id (golden-ratio hue steps), so ID switches are easy to spot."""
    hue = int((track_id * 0.618034 % 1.0) * 180)
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in bgr)


def draw_overlay(frame: np.ndarray, tracks: list[Track], labels: dict[int, tuple[str, str]],
                 t: float, info: str = "", trails: dict[int, list[tuple[int, int]]] | None = None,
                 show_ids: bool = True) -> np.ndarray:
    """With `trails` (tracking-only mode) boxes are coloured per ID with their recent path; otherwise by state.
    `show_ids=False` (tracking bypassed) labels boxes with class and confidence only."""
    out = frame.copy()
    thick = max(1, frame.shape[1] // 800)
    scale = max(0.45, frame.shape[1] / 2400)
    counts = {WORKING: 0, IDLE: 0}
    for tr in tracks:
        state, activity = labels.get(tr.track_id, ("pending", ""))
        if state in counts:
            counts[state] += 1
        x1, y1, x2, y2 = map(int, tr.box)
        det = f"{tr.name} {tr.conf:.0%}"
        if not show_ids:
            color, text = id_color(tr.track_id), det
        elif trails is not None:
            color = id_color(tr.track_id)
            pts = trails.get(tr.track_id, [])
            for a, b in zip(pts, pts[1:]):
                cv2.line(out, a, b, color, thick + 1)
            text = f"id {tr.track_id} {det}"
        else:
            color = COLORS.get(state, GREY)
            text = f"{tr.track_id} {state.upper()}" + (f" {activity}" if activity else "") + f" | {det}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick + 1)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        lx = max(0, min(x1, out.shape[1] - tw - 4))        # keep the label inside the frame
        ly = max(th + 6, y1)
        cv2.rectangle(out, (lx, ly - th - 6), (lx + tw + 4, ly), color, -1)
        cv2.putText(out, text, (lx + 2, ly - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick)

    header = (f"t={t:6.1f}s  people={len(tracks)}  {info}" if trails is not None else
              f"t={t:6.1f}s  working={counts[WORKING]}  idle={counts[IDLE]}  {info}")
    cv2.rectangle(out, (0, 0), (out.shape[1], int(30 * scale * 2)), (0, 0, 0), -1)
    cv2.putText(out, header, (8, int(22 * scale * 2)), cv2.FONT_HERSHEY_SIMPLEX, scale * 1.2, (255, 255, 255),
                thick)
    return out
