"""CLI: python -m wat data/video1.mp4 [--show] [--realtime] [--device cuda] [--backend dummy]"""

import argparse
import json

from .config import load_config
from .pipeline import run


def main():
    p = argparse.ArgumentParser(description="Per-worker Working/Idle from CCTV video or an RTSP stream")
    p.add_argument("sources", nargs="+", help="video files or rtsp:// URLs")
    p.add_argument("--config", default=None)
    p.add_argument("--backend", choices=["none", "dummy", "ollama", "huggingface", "gemini"], help="override llm.backend (none = tracking only)")
    p.add_argument("--no-track", action="store_true", help="bypass the tracker: detection only (implies no LLM)")
    p.add_argument("--device", help="override detector.device (cpu / cuda)")
    p.add_argument("--show", action="store_true", help="show the live overlay window (q to quit)")
    p.add_argument("--realtime", action="store_true", help="pace files at video speed, like a live camera")
    p.add_argument("--max-seconds", type=float, help="stop after this much video time")
    p.add_argument("--no-video", action="store_true", help="don't write annotated.mp4")
    a = p.parse_args()

    cfg = load_config(a.config) if a.config else load_config()
    if a.backend:
        cfg["llm"]["backend"] = a.backend
    if a.no_track:
        cfg["detector"]["tracker"] = None
    if a.device:
        cfg["detector"]["device"] = a.device
    if a.no_video:
        cfg["output"]["annotated_video"] = False

    for src in a.sources:
        summary = run(src, cfg, show=a.show, realtime=a.realtime, max_seconds=a.max_seconds)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
