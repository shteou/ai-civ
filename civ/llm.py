"""Thin, instrumented wrapper around the Ollama chat API."""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict

from ollama import Client


@dataclass
class CallStats:
    wall_s: float            # total time as seen by the caller
    load_s: float            # model (re)load time inside Ollama - should be ~0 after warmup
    prompt_tokens: int       # prompt tokens Ollama actually evaluated (cached prefix excluded)
    prompt_s: float          # time spent evaluating the prompt
    out_tokens: int
    out_s: float             # time spent generating
    overhead_s: float        # wall time not accounted for by Ollama (HTTP, JSON, queueing)

    @property
    def tok_per_s(self) -> float:
        return self.out_tokens / self.out_s if self.out_s > 0 else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tok_per_s"] = round(self.tok_per_s, 1)
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}


class LLM:
    def __init__(self, model: str, num_ctx: int = 4096, num_predict: int = 160,
                 temperature: float = 0.7, seed: int | None = None,
                 keep_alive: str = "30m", host: str | None = None):
        self.model = model
        self.client = Client(host=host) if host else Client()
        self.keep_alive = keep_alive
        self.options = {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": temperature}
        if seed is not None:
            self.options["seed"] = seed
        # Thinking models (e.g. qwen3.5) spend 10-50x more tokens reasoning unless told not to.
        # Only send think=False if the model supports the flag, otherwise Ollama errors.
        caps = getattr(self.client.show(model), "capabilities", None) or []
        self.think = False if "thinking" in caps else None
        self.supports_tools = "tools" in caps

    def warmup(self) -> float:
        """Load the model into VRAM so the first real turn isn't charged for it."""
        t = time.perf_counter()
        self.client.chat(self.model, messages=[{"role": "user", "content": "hi"}], think=self.think,
                         keep_alive=self.keep_alive, options={**self.options, "num_predict": 1})
        return time.perf_counter() - t

    def chat(self, messages: list[dict], tools: list[dict] | None = None, fmt: dict | None = None):
        t0 = time.perf_counter()
        resp = self.client.chat(self.model, messages=messages, tools=tools, format=fmt, think=self.think,
                                keep_alive=self.keep_alive, options=self.options)
        wall = time.perf_counter() - t0
        ns = 1e-9
        total = (resp.total_duration or 0) * ns
        return resp, CallStats(
            wall_s=wall,
            load_s=(resp.load_duration or 0) * ns,
            prompt_tokens=resp.prompt_eval_count or 0,
            prompt_s=(resp.prompt_eval_duration or 0) * ns,
            out_tokens=resp.eval_count or 0,
            out_s=(resp.eval_duration or 0) * ns,
            overhead_s=max(0.0, wall - total),
        )
