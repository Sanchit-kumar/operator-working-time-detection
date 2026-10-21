import json

import numpy as np

from wat.activities import IDLE, UNKNOWN, WORKING
from wat.detect_track import Track
from wat.llm import DummyBackend, FrameRequest, parse_response
from wat.request_builder import FrameHistory, build_requests
from wat.smoothing import Smoother

LLM_CFG = {"history_s": 2.0, "max_workers_per_call": 2, "frame_width": 320, "crop_size": 64, "crop_margin": 0.2,
           "layout": "sheet", "sheet_width": 400,
           "jpeg_quality": 80, "dummy": {"seed": 1, "latency_s": [0, 0]}}


def test_parse_maps_activity_and_fills_missing():
    text = json.dumps({"workers": [
        {"id": 1, "activity": "phone", "state": "working", "confidence": 0.9},   # model's state is ignored
        {"id": 2, "activity": "made_up", "state": "idle", "confidence": 0.5},
        {"id": 99, "activity": "carrying", "state": "working", "confidence": 1},  # not asked about
    ]})
    out = parse_response(text, [1, 2, 3])
    assert out[1].state == IDLE
    assert out[2].activity == "occluded" and out[2].state == UNKNOWN
    assert out[3].state == UNKNOWN
    assert 99 not in out


def test_parse_strips_code_fence():
    out = parse_response('```json\n{"workers": [{"id": 4, "activity": "carrying", "state": "working", '
                         '"confidence": 1}]}\n```', [4])
    assert out[4].state == WORKING


def test_smoother_hysteresis():
    s = Smoother(idle_after=2, work_after=1)
    assert s.update(1, UNKNOWN) == UNKNOWN
    assert s.update(1, WORKING) == WORKING      # first answer sets the state
    assert s.update(1, IDLE) == WORKING         # one idle answer is not enough
    assert s.update(1, UNKNOWN) == WORKING      # unknown keeps the state
    assert s.update(1, IDLE) == IDLE            # unknown doesn't break the idle streak
    assert s.update(1, WORKING) == WORKING


def test_smoother_idle_needs_two_in_a_row():
    s = Smoother(idle_after=2, work_after=1)
    s.update(1, WORKING)
    assert s.update(1, IDLE) == WORKING
    assert s.update(1, IDLE) == IDLE
    assert s.update(1, WORKING) == WORKING


def test_build_requests_and_dummy_roundtrip():
    frame = np.zeros((360, 640, 3), np.uint8)
    tracks = [Track(i, (10.0 * i, 20.0, 10.0 * i + 30, 100.0), 0.9) for i in range(1, 4)]
    hist = FrameHistory(2.0)
    hist.add(0.0, frame, tracks[:1])
    hist.add(2.0, frame, tracks)
    reqs = build_requests(2.0, frame, tracks, hist, LLM_CFG)
    assert [r.worker_ids for r in reqs] == [[1, 2], [3]]                    # chunked by max_workers_per_call
    assert [len(r.images) for r in reqs] == [2, 2]                          # frame + one contact sheet
    per_worker = build_requests(2.0, frame, tracks, hist, {**LLM_CFG, "layout": "per_worker"})
    assert [len(r.images) for r in per_worker] == [3, 2]                    # frame + one image per worker

    res = DummyBackend(LLM_CFG).classify(reqs[0])
    assert res.error is None and set(res.decisions) == {1, 2}


def test_backend_failure_becomes_unknown():
    class Broken(DummyBackend):
        def complete(self, req):
            return "not json"
    req = FrameRequest(0.0, [5], [])
    res = Broken(LLM_CFG).classify(req)
    assert res.error and res.decisions[5].state == UNKNOWN


def test_parse_accepts_bare_list():
    out = parse_response('[{"id": 7, "activity": "phone", "confidence": 0.8}]', [7])
    assert out[7].state == IDLE


def test_missing_confidence_is_none():
    out = parse_response('[{"id": "3", "activity": "carrying"}]', [3])    # string id, no confidence (qwen3.5)
    assert out[3].state == WORKING and out[3].confidence is None


def test_trailing_text_after_json_is_ignored():
    text = 'Sure:\n{"workers":[{"id":3,"activity":"carrying","confidence":0.9}]}\n\n{"workers":[]}'   # hf models ramble on
    out = parse_response(text, [3])
    assert out[3].activity == "carrying" and out[3].confidence == 0.9


def test_json_cut_off_before_closing_brace():
    out = parse_response('{"workers":[{"id":3,"activity":"carrying","confidence":0.9}]', [3])   # qwen2.5-vl + stop string
    assert out[3].activity == "carrying"
