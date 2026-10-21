"""Main loop: track at ~4 fps, ask the LLM every 2 s (in the background), smooth, draw, and write results.

Outputs in outputs/<video stem>/:
  decisions.csv  one row per worker per LLM decision
  timeline.csv   merged segments: track_id,start_s,end_s,state
  summary.json   seconds and % working/idle per worker, plus run stats
  annotated.mp4  overlay video (if output.annotated_video)
"""

import json
import time
from collections.abc import Callable
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .activities import IDLE, UNKNOWN, WORKING
from .config import resolve
from .detect_track import PersonTracker
from .llm import FrameResult, make_backend
from .render import draw_overlay
from .request_builder import FrameHistory, build_requests
from .smoothing import Smoother


def run(source: str | int, cfg: dict, show: bool = False, realtime: bool = False,
        max_seconds: float | None = None, on_frame: Callable[[np.ndarray, float], bool | None] | None = None) -> dict:
    """Process a file or stream. `on_frame(overlay_bgr, t)` is called for every processed frame (e.g. to display it
    in a notebook); returning False stops the run. Results are written even when stopped early."""
    det, llm = cfg["detector"], cfg["llm"]
    is_file = isinstance(source, str) and Path(source).exists()   # else an rtsp:// URL or webcam index
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(src_fps / det["fps"]))
    tracking = det.get("tracker") is not None     # None = detection only: no IDs across frames
    classify = llm["backend"] != "none"           # "none" = tracking only: no LLM calls, boxes coloured per ID
    if classify and not tracking:
        print("tracker is None: working/idle needs stable IDs across frames, so LLM classification is off")
        classify = False
    suffix = "" if classify else "_tracking" if tracking else "_detection"
    name = (Path(source).stem if is_file else "stream") + suffix
    out_dir = resolve(cfg["output"]["dir"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = PersonTracker(det)
    backend = make_backend(llm) if classify else None
    if backend:
        backend.warm_up()                         # load the model now so the first tick isn't 10 s late
    smoother = Smoother(cfg["smoothing"]["idle_after"], cfg["smoothing"]["work_after"])
    history = FrameHistory(llm["history_s"])
    pool = ThreadPoolExecutor(max_workers=llm["concurrency"])
    pending: deque[Future] = deque()              # in submit order, so results are applied in time order

    first_seen: dict[int, float] = {}
    last_seen: dict[int, float] = {}
    track_frames: dict[int, int] = {}
    max_gap: dict[int, float] = {}                # longest time a track was lost and then re-found
    trails: dict[int, deque] = {}
    labels: dict[int, tuple[str, str]] = {}       # track_id -> (smoothed state, activity) shown on screen
    rows: list[dict] = []
    stats = {"frames": 0, "detections": 0, "calls": 0, "errors": 0, "skipped_ticks": 0, "latency_s": [], "lag_s": []}
    next_tick, t = 0.0, 0.0
    writer = None
    wall_start = time.perf_counter()

    def apply(res: FrameResult, now_t: float):
        stats["latency_s"].append(res.latency_s)
        stats["lag_s"].append(now_t - res.t)       # how far behind the video the label appears
        if res.error:
            stats["errors"] += 1
        for tid, d in sorted(res.decisions.items()):
            state = smoother.update(tid, d.state)
            labels[tid] = (state, d.activity)
            rows.append({"t": round(res.t, 2), "track_id": tid, "activity": d.activity, "raw_state": d.state,
                         "state": state, "confidence": d.confidence, "reason": d.reason,
                         "latency_s": round(res.latency_s, 2), "error": res.error or ""})

    def drain(now_t: float, wait: bool = False):
        while pending and (wait or pending[0].done()):
            apply(pending.popleft().result(), now_t)

    interrupted = False
    try:
        frame_idx = -1
        while True:
            ok = cap.grab()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % stride:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = frame_idx / src_fps
            if max_seconds is not None and t > max_seconds:
                break

            tracks = tracker.update(frame)
            stats["detections"] += len(tracks)
            for tr in (tracks if tracking else []):        # per-ID stats are meaningless without tracking
                tid = tr.track_id
                if tid in last_seen:
                    max_gap[tid] = max(max_gap.get(tid, 0.0), t - last_seen[tid])
                first_seen.setdefault(tid, t)
                last_seen[tid] = t
                track_frames[tid] = track_frames.get(tid, 0) + 1
                x1, y1, x2, y2 = tr.box
                trails.setdefault(tid, deque(maxlen=cfg.get("render", {}).get("trail_len", 20))).append(
                    (int((x1 + x2) / 2), int(y2)))  # feet point
            history.add(t, frame, tracks)
            stats["frames"] += 1

            if classify and t >= next_tick:
                eligible = [tr for tr in tracks if t - first_seen[tr.track_id] >= cfg["tracks"]["min_age_s"]]
                if len(pending) >= llm["concurrency"]:   # model can't keep up: skip this tick rather than queue it
                    stats["skipped_ticks"] += 1
                elif eligible:
                    for req in build_requests(t, frame, eligible, history, llm):
                        pending.append(pool.submit(backend.classify, req))
                        stats["calls"] += 1
                next_tick += llm["interval_s"]
                while next_tick <= t:                  # skip ticks if processing fell behind
                    next_tick += llm["interval_s"]

            drain(t)
            lag = f"lag={stats['lag_s'][-1]:.1f}s" if stats["lag_s"] else ""
            if classify:
                vis = draw_overlay(frame, tracks, labels, t, f"in-flight={len(pending)} {lag}")
            elif not tracking:
                vis = draw_overlay(frame, tracks, labels, t, "detection only", trails={}, show_ids=False)
            else:
                vis = draw_overlay(frame, tracks, labels, t, f"ids so far={len(first_seen)}",
                                   trails={tid: list(trails[tid]) for tid in trails})

            if cfg["output"]["annotated_video"]:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                             src_fps / stride, (w, h))
                writer.write(vis)
            if show:
                cv2.imshow("worker activity", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if on_frame is not None and on_frame(vis, t) is False:
                break
            if realtime and is_file:                   # pace a file like a live camera
                ahead = t - (time.perf_counter() - wall_start)
                if ahead > 0:
                    time.sleep(ahead)
    except KeyboardInterrupt:                  # notebook Interrupt button / Ctrl+C: stop but keep results
        interrupted = True

    drain(t, wait=True)                            # answers already requested still count
    pool.shutdown()
    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    wall = time.perf_counter() - wall_start
    summary = write_outputs(out_dir, rows, last_seen, llm["interval_s"], llm["max_hold_s"], {
        "source": source, "video_s": round(t, 2), "interrupted": interrupted, "wall_s": round(wall, 2),
        "processed_fps": round(stats["frames"] / wall, 2) if wall else 0.0,
        "target_det_fps": round(src_fps / stride, 2), "tracker": det.get("tracker"),
        "backend": llm["backend"] if classify else "none (tracking only)" if tracking else "none (detection only)",
        "mean_people_per_frame": round(stats["detections"] / stats["frames"], 2) if stats["frames"] else 0.0,
        "llm_calls": stats["calls"], "llm_errors": stats["errors"], "skipped_ticks": stats["skipped_ticks"],
        "mean_latency_s": round(sum(stats["latency_s"]) / len(stats["latency_s"]), 2) if stats["latency_s"] else None,
        "mean_label_lag_s": round(sum(stats["lag_s"]) / len(stats["lag_s"]), 2) if stats["lag_s"] else None,
    })
    summary["tracks"] = {int(tid): {"first_s": round(first_seen[tid], 2), "last_s": round(last_seen[tid], 2),
                                    "frames": track_frames[tid], "max_gap_s": round(max_gap.get(tid, 0.0), 2)}
                         for tid in sorted(first_seen)}
    summary["run"]["track_ids"] = len(first_seen)
    pd.DataFrame.from_dict(summary["tracks"], orient="index").rename_axis("track_id").to_csv(out_dir / "tracks.csv")
    summary["output_dir"] = str(out_dir)
    return summary


def write_outputs(out_dir: Path, rows: list[dict], last_seen: dict[int, float], interval_s: float, max_hold_s: float,
                  run_info: dict) -> dict:
    cols = ["t", "track_id", "activity", "raw_state", "state", "confidence", "reason", "latency_s", "error"]
    df = pd.DataFrame(rows, columns=cols).sort_values(["track_id", "t"])
    df.to_csv(out_dir / "decisions.csv", index=False)

    # Each decision covers [t, next decision) for that worker, capped at max_hold_s (ticks may be skipped when the
    # model is slow, so gaps can exceed interval_s) and at when the track was last seen.
    segments, workers = [], {}
    for tid, g in df.groupby("track_id"):
        ts, states = g["t"].tolist(), g["state"].tolist()
        secs = {WORKING: 0.0, IDLE: 0.0, UNKNOWN: 0.0}
        seen_until = last_seen.get(tid, ts[-1])
        for i, (start, state) in enumerate(zip(ts, states)):
            nxt = ts[i + 1] if i + 1 < len(ts) else seen_until + interval_s / 2
            end = min(nxt, start + max_hold_s, max(seen_until, start + 0.01))
            secs[state] += end - start
            if segments and segments[-1]["track_id"] == tid and segments[-1]["state"] == state \
                    and abs(segments[-1]["end_s"] - start) < 1e-6:
                segments[-1]["end_s"] = round(end, 2)
            else:
                segments.append({"track_id": int(tid), "start_s": round(start, 2), "end_s": round(end, 2),
                                 "state": state})
        known = secs[WORKING] + secs[IDLE]
        workers[int(tid)] = {"working_s": round(secs[WORKING], 1), "idle_s": round(secs[IDLE], 1),
                             "unknown_s": round(secs[UNKNOWN], 1),
                             "pct_working": round(100 * secs[WORKING] / known, 1) if known >= 0.1 else None}
    pd.DataFrame(segments, columns=["track_id", "start_s", "end_s", "state"]).to_csv(out_dir / "timeline.csv",
                                                                                   index=False)
    summary = {"run": run_info, "workers": workers}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
