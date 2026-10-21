"""Multimodal LLM backends. Every backend takes a FrameRequest and returns the model's raw text;
`parse_response` turns that text into validated WorkerDecisions, so the dummy and real backends share one path.

Switch backend with `llm.backend` in config.yaml: dummy | ollama | huggingface (transformers, in-process) | gemini
(stub, raises until enabled).
"""

import base64
import io
import json
import os
import random
import time
from dataclasses import dataclass, field

import requests

from .activities import ACTIVITIES, UNKNOWN, state_of
from .prompt import RESPONSE_SCHEMA, system_prompt, user_text


@dataclass
class FrameRequest:
    t: float                           # video time of the decision, seconds
    worker_ids: list[int]
    images: list[tuple[str, bytes]]    # (caption, jpeg): full frame first, then the worker close-ups


@dataclass
class WorkerDecision:
    track_id: int
    activity: str
    state: str                         # from our mapping, not the model's own claim
    confidence: float | None           # None when the model didn't give one
    reason: str = ""


@dataclass
class FrameResult:
    t: float
    decisions: dict[int, WorkerDecision] = field(default_factory=dict)
    latency_s: float = 0.0
    error: str | None = None


def first_json(text: str):
    """First JSON value in `text`, ignoring anything the model adds after it. A reply cut off just before its
    closing brackets (a stop string can end generation one token early) is closed and parsed."""
    for suffix in ("", "}", "]}", "}]}"):
        try:
            return json.JSONDecoder().raw_decode(text + suffix)[0]
        except json.JSONDecodeError as e:
            err = err if suffix else e
    raise err


def parse_response(text: str, worker_ids: list[int]) -> dict[int, WorkerDecision]:
    """Parse and validate the model's JSON. Unknown ids are dropped; missing ids become `unknown`."""
    text = text.strip()
    if text.startswith("```"):                       # some models wrap JSON in a code fence
        text = text.strip("`").removeprefix("json").strip()
    start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=0)
    data = first_json(text[start:])
    items = data if isinstance(data, list) else data.get("workers", [])   # some models drop the wrapper object
    out: dict[int, WorkerDecision] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            tid = int(item["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if tid not in worker_ids:
            continue
        activity = item.get("activity", "")
        if activity not in ACTIVITIES:
            activity = "occluded"
        conf = item.get("confidence")
        out[tid] = WorkerDecision(tid, activity, state_of(activity),
                                  float(conf) if isinstance(conf, (int, float)) else None,
                                  str(item.get("reason", "")))
    for tid in worker_ids:
        out.setdefault(tid, WorkerDecision(tid, "occluded", UNKNOWN, 0.0, "missing from response"))
    return out


class Backend:
    def complete(self, req: FrameRequest) -> str:
        raise NotImplementedError

    def warm_up(self) -> float:
        """Load the model before the first real call; returns seconds taken. No-op for remote APIs."""
        return 0.0

    def classify(self, req: FrameRequest) -> FrameResult:
        start = time.perf_counter()
        try:
            text = self.complete(req)
            decisions = parse_response(text, req.worker_ids)
            return FrameResult(req.t, decisions, time.perf_counter() - start)
        except Exception as e:                      # a failed call must never stop the video loop
            decisions = {tid: WorkerDecision(tid, "occluded", UNKNOWN, 0.0, "call failed") for tid in req.worker_ids}
            return FrameResult(req.t, decisions, time.perf_counter() - start, error=f"{type(e).__name__}: {e}")


class DummyBackend(Backend):
    """Returns a random activity per worker, in the same JSON shape a real model must return."""

    def __init__(self, cfg: dict):
        d = cfg.get("dummy", {})
        self.rng = random.Random(d.get("seed"))
        self.latency = d.get("latency_s", [0.0, 0.0])
        names = list(ACTIVITIES)
        # Mostly real activities; occluded / not_a_worker are rare, as they would be from a real model.
        self.weights = [0.3 if state_of(n) == UNKNOWN else 1.0 for n in names]
        self.names = names

    def complete(self, req: FrameRequest) -> str:
        time.sleep(self.rng.uniform(*self.latency))
        workers = []
        for tid in req.worker_ids:
            activity = self.rng.choices(self.names, self.weights)[0]
            workers.append({"id": tid, "activity": activity, "state": state_of(activity),
                            "confidence": round(self.rng.uniform(0.5, 1.0), 2), "reason": "dummy"})
        return json.dumps({"workers": workers})


class OllamaBackend(Backend):
    """Ollama /api/chat: local server (no auth) or Ollama cloud (base_url https://ollama.com + API key).

    Each image goes in its own user message with its caption, so the model can tell which image shows whom.
    The response is constrained to RESPONSE_SCHEMA with Ollama's `format`.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg["ollama"]
        self.timeout = cfg.get("timeout_s", 30)
        self.url = self.cfg["base_url"].rstrip("/") + "/api/chat"
        key = os.environ.get(self.cfg.get("api_key_env", ""), "")
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.session = requests.Session()

    def messages(self, req: FrameRequest) -> list[dict]:
        msgs = [{"role": "system", "content": system_prompt()}]
        for caption, img in req.images:
            msgs.append({"role": "user", "content": caption, "images": [base64.b64encode(img).decode()]})
        msgs.append({"role": "user", "content": user_text(req.worker_ids)})
        return msgs

    def options(self) -> dict:
        # Must be identical for warm-up and real calls: a different num_ctx makes Ollama reload the model.
        return {"temperature": 0, "num_ctx": self.cfg.get("num_ctx", 8192)}

    def warm_up(self) -> float:
        start = time.perf_counter()
        r = self.session.post(self.url, json={"model": self.cfg["model"], "messages": [], "options": self.options(),
                                              "keep_alive": self.cfg.get("keep_alive", "10m")},
                              headers=self.headers, timeout=120)
        if not r.ok:
            raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
        return time.perf_counter() - start

    def complete(self, req: FrameRequest) -> str:
        body = {"model": self.cfg["model"], "messages": self.messages(req), "stream": False,
                "format": RESPONSE_SCHEMA, "keep_alive": self.cfg.get("keep_alive", "10m"),
                "options": self.options()}
        if self.cfg.get("think") is not None:      # only for thinking models; others reject the field
            body["think"] = self.cfg["think"]
        r = self.session.post(self.url, json=body, headers=self.headers, timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
        return r.json()["message"]["content"]


class HuggingFaceBackend(Backend):
    """Runs a Hugging Face vision-language model in-process with transformers (no server). The model is
    downloaded to the HF cache on first use (set HF_TOKEN for gated models) and loaded in warm_up().

    No constrained decoding here, so the JSON shape comes from the prompt; parse_response tolerates code fences
    and a missing wrapper object.
    """

    _loaded: dict[tuple, tuple] = {}                  # one copy per settings, shared by every backend instance

    def __init__(self, cfg: dict):
        self.cfg = cfg["huggingface"]
        self.model = self.processor = None

    def load(self):
        key = tuple(self.cfg.get(k) for k in ("model", "device", "dtype", "quantize", "max_pixels"))
        if key not in self._loaded:
            self._loaded.clear()                      # a different model/settings replaces the old one
            free_gpu_memory()
            self._loaded[key] = self._load()
        self.model, self.processor = self._loaded[key]

    def _load(self):
        import torch                                  # heavy imports only when this backend is used
        from transformers import AutoModelForImageTextToText, AutoProcessor
        device = self.cfg.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = getattr(torch, self.cfg.get("dtype", "bfloat16")) if device != "cpu" else torch.float32
        processor = AutoProcessor.from_pretrained(self.cfg["model"])
        if self.cfg.get("max_pixels"):                # caps image tokens (Qwen2.5-VL: 28x28 px, Qwen3-VL: 32x32 px)
            processor.image_processor.size["longest_edge"] = self.cfg["max_pixels"]
        kwargs = {}
        if self.cfg.get("quantize") in ("4bit", "8bit"):   # needs `pip install bitsandbytes` and a GPU
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = (
                BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
                if self.cfg["quantize"] == "4bit" else BitsAndBytesConfig(load_in_8bit=True))
        model = AutoModelForImageTextToText.from_pretrained(self.cfg["model"], dtype=dtype, device_map=device,
                                                            **kwargs).eval()
        return model, processor

    def warm_up(self) -> float:
        start = time.perf_counter()
        if self.model is None:
            self.load()
        return time.perf_counter() - start

    def messages(self, req: FrameRequest) -> list[dict]:
        from PIL import Image
        content = []
        for caption, img in req.images:
            content += [{"type": "text", "text": caption},
                        {"type": "image", "image": Image.open(io.BytesIO(img)).convert("RGB")}]
        # Compact one-line JSON: decoding is ~45 ms/token on a laptop GPU, and pretty-printed output is 2x longer.
        content.append({"type": "text", "text": user_text(req.worker_ids) + "\nReply with one line of compact JSON, "
                        'no markdown: {"workers":[{"id":<int>,"activity":"<listed activity>","confidence":<0-1>}]}'})
        return [{"role": "system", "content": [{"type": "text", "text": system_prompt()}]},
                {"role": "user", "content": content}]

    def complete(self, req: FrameRequest) -> str:
        import torch
        if self.model is None:
            self.load()
        inputs = self.processor.apply_chat_template(self.messages(req), add_generation_prompt=True, tokenize=True,
                                                    return_dict=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.cfg.get("max_new_tokens", 400), do_sample=False,
                                      stop_strings=["}]}", "}]\n}"], tokenizer=self.processor.tokenizer)
        return self.processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def free_gpu_memory():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def unload_ollama(base_url: str = "http://localhost:11434") -> list[str]:
    """Ask a local Ollama server to drop its loaded models from (GPU) memory; returns their names.
    Ollama keeps models for keep_alive (10 min), which would leave too little VRAM for a transformers model."""
    try:
        running = [m["name"] for m in requests.get(f"{base_url}/api/ps", timeout=3).json().get("models", [])]
        for name in running:
            requests.post(f"{base_url}/api/generate", json={"model": name, "keep_alive": 0}, timeout=30)
        return running
    except requests.RequestException:
        return []                                     # no server: nothing to free


class GeminiBackend(Backend):
    """Google Gemini generateContent with a response JSON schema. Not enabled yet."""

    def __init__(self, cfg: dict):
        self.cfg = cfg["gemini"]
        self.timeout = cfg.get("timeout_s", 30)
        self.api_key = os.environ.get(self.cfg["api_key_env"], "")

    def complete(self, req: FrameRequest) -> str:
        raise NotImplementedError("Gemini backend not enabled yet; set llm.backend: dummy")


BACKENDS = {"dummy": DummyBackend, "ollama": OllamaBackend, "huggingface": HuggingFaceBackend,
            "gemini": GeminiBackend}


def make_backend(llm_cfg: dict) -> Backend:
    return BACKENDS[llm_cfg["backend"]](llm_cfg)


# Re-exported so real backends can build their payloads from one place.
__all__ = ["FrameRequest", "WorkerDecision", "FrameResult", "make_backend", "parse_response",
           "system_prompt", "user_text", "RESPONSE_SCHEMA"]
