"""Person detection + ByteTrack on CPU (or GPU). Gives stable worker IDs between LLM calls.
With `tracker: null` tracking is bypassed: plain detection, IDs are just 1..N within each frame."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ultralytics
import yaml
from ultralytics import YOLO

from .config import resolve


@dataclass
class Track:
    track_id: int
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in frame pixels
    conf: float
    name: str = "person"                    # detector class name


BUILTIN_TRACKERS = Path(ultralytics.__file__).parent / "cfg" / "trackers"


def tracker_yaml(tracker: str, overrides: dict | None = None) -> str:
    """`tracker` is a yaml path or a built-in name (bytetrack, botsort, ocsort, deepocsort, fasttrack, tracktrack).
    With overrides, the merged settings are written to outputs/ and that file's path is returned."""
    path = resolve(tracker)
    if not path.exists():
        path = BUILTIN_TRACKERS / f"{Path(tracker).stem}.yaml"
        if not path.exists():
            names = sorted(p.stem for p in BUILTIN_TRACKERS.glob("*.yaml"))
            raise FileNotFoundError(f"tracker {tracker!r} is neither a file nor a built-in ({', '.join(names)})")
    if not overrides:
        return str(path)
    settings = {**yaml.safe_load(path.read_text()), **overrides}
    out = resolve("outputs") / f"_tracker_{settings['tracker_type']}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(settings, sort_keys=False))
    return str(out)


class PersonTracker:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = YOLO(str(resolve(cfg["weights"])))
        self.tracking = cfg.get("tracker") is not None
        self.tracker_cfg = tracker_yaml(cfg["tracker"], cfg.get("tracker_overrides")) if self.tracking else None

    def update(self, frame: np.ndarray) -> list[Track]:
        args = dict(classes=[0], conf=self.cfg["conf"], imgsz=self.cfg["imgsz"], device=self.cfg["device"],
                    verbose=False)
        if self.tracking:
            res = self.model.track(frame, persist=True, tracker=self.tracker_cfg, **args)[0]
        else:
            res = self.model.predict(frame, **args)[0]
        boxes = res.boxes
        if boxes is None or (self.tracking and boxes.id is None):
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().tolist() if self.tracking else list(range(1, len(xyxy) + 1))
        confs = boxes.conf.cpu().tolist()
        names = [res.names[int(c)] for c in boxes.cls.cpu().tolist()]
        return [Track(i, tuple(map(float, b)), c, n) for i, b, c, n in zip(ids, xyxy, confs, names)]
