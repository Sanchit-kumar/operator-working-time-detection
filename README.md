# Worker Activity Tracker 2

Per-worker **Working / Idle** from factory and warehouse CCTV. A CPU-only box tracks people; a multimodal LLM decides
what each worker is doing every 2 s (0.5 fps).

```
video/RTSP ─► YOLO11n + ByteTrack (CPU, ~4 fps) ─► every 2 s: frame with numbered boxes + 2 crops per worker
          ─► LLM (background thread) ─► JSON activity per worker ─► hysteresis ─► overlay, timeline, summary
```

## Run
Uses the existing venv from `../worker-activity-tracker/.venv` (nothing extra to install).

```bash
PY="../worker-activity-tracker/.venv/bin/python"
$PY -m wat data/video1.mp4                 # writes outputs/video1/
$PY -m wat data/video2.mp4 --max-seconds 120 --show --realtime
$PY -m wat rtsp://camera/stream --show
$PY -m pytest -q tests
```

**Notebook:** `live_inference.ipynb` shows the live overlay inline. All settings are in one cell:
`DISABLE_CLASSIFY` (default `False`; `True` = detection + tracking only, boxes coloured per ID with trails),
`TRACKER = None` to bypass tracking (detection only, no LLM), YOLO weights / device /
imgsz / conf / detection fps, tracker (any Ultralytics built-in: bytetrack, botsort, ocsort, deepocsort, fasttrack,
tracktrack) with parameter overrides, and LLM backend / model / interval / layout / smoothing. Interrupt to stop;
results so far are kept. Then a per-ID tracking-quality table, per-worker results and a "what the model sees"
preview. Select the `../worker-activity-tracker/.venv` kernel; VS Code will ask to install `ipykernel` into it once.

Tracking only from the CLI: `$PY -m wat data/video3.mp4 --backend none` (writes `outputs/video3_tracking/`).
Detection only: `--no-track` (writes `outputs/video3_detection/`). Every box shows the detection class and confidence
(`person 87%`); with the LLM it is `3 WORKING packing_sorting | person 87%`.
In config.yaml, `detector.tracker` takes a yaml path or a built-in name, and `detector.tracker_overrides` a dict.

Outputs in `outputs/<video>/`: `decisions.csv` (every LLM answer), `timeline.csv` (segments), `summary.json`
(seconds and % working per worker, latency, lag), `annotated.mp4`.

## LLM backends (`llm.backend` in config.yaml, or `--backend`)
- `ollama` (default): local server at `http://localhost:11434`, no key needed (`ollama serve`; model must be
  pulled). For Ollama cloud set `llm.ollama.base_url: https://ollama.com` and export `OLLAMA_API_KEY`.
  Default model `qwen3.5:4b` with `think: false`; `qwen2.5vl:3b` answered "packing_sorting" for everyone.
- `huggingface`: runs the model in-process with transformers, no server (`llm.huggingface` in config.yaml, or
  `BACKEND = "huggingface"` + `HF_MODEL` in the notebook). Default `Qwen/Qwen3-VL-2B-Instruct`, downloaded to
  `~/.cache/huggingface` on first use (~4.5 GB; `HF_TOKEN` for gated models). On an RTX 4060 laptop: 15 s load,
  4.6 GB VRAM, ~7 s per call for 6 workers (decode ~45 ms/token, so it asks for compact one-line JSON). No
  constrained decoding: the parser takes the first JSON value in the reply. CPU works but is far slower.
- `dummy`: random activity per worker, with simulated latency; same JSON path as a real model. No server needed.
- `gemini`: stub in `wat/llm.py` (`complete()` raises); key will come from `GEMINI_API_KEY`.

Each call sends 2 images: the full frame with numbered boxes, and one contact sheet with a `[~2 s ago | now]` pair
per worker (`llm.layout: sheet`). Ollama's Qwen models cost ~1000 tokens per image whatever its size, so this is
~3x fewer tokens than one image per worker (`layout: per_worker`). If a call is still running when the next 2 s tick
comes, the tick is skipped (`skipped_ticks` in summary.json), so a slow model lowers the decision rate rather
than building a backlog.

The model picks one activity from `wat/activities.py`; that table decides Working / Idle / Unknown.

## Layout
`wat/detect_track.py` tracking · `wat/request_builder.py` images per call · `wat/prompt.py` prompt + schema ·
`wat/llm.py` backends + parser · `wat/smoothing.py` hysteresis · `wat/render.py` overlay · `wat/pipeline.py` loop.
