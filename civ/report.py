"""Human-readable console output, a plain-text log, a JSONL trace, and the performance summary."""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

from .agent import Decision
from .world import World

COLORS = [36, 33, 35, 32, 34, 31, 96, 93]
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class Reporter:
    def __init__(self, out_dir: Path, color: bool | None = None, verbose: bool = False):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.color = sys.stdout.isatty() if color is None else color
        self.verbose = verbose
        self.trace = open(out_dir / "trace.jsonl", "w", buffering=1)
        self.log = open(out_dir / "log.txt", "w", buffering=1)

    def close(self) -> None:
        self.trace.close()
        self.log.close()

    # -- low level ------------------------------------------------------------------------
    def c(self, text: str, code: int | str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.color else text

    def out(self, text: str = "") -> None:
        print(text if self.color else _ANSI.sub("", text), flush=True)
        self.log.write(_ANSI.sub("", text) + "\n")

    def event(self, kind: str, **data) -> None:
        self.trace.write(json.dumps({"event": kind, **data}, default=str) + "\n")

    # -- narrative ------------------------------------------------------------------------
    def run_start(self, world: World, config: dict, system_prompt: str) -> None:
        self.event("run_start", config=config, system_prompt=system_prompt, world=world.snapshot())
        self.out(self.c(f"Village of {len(world.villagers)}: ", 1) + ", ".join(
            f"{v.name} ({v.persona})" for v in world.villagers))
        self.out(f"model={config['model']}  seed={config['seed']}  trace -> {self.out_dir}/trace.jsonl")

    def round_start(self, world: World) -> None:
        self.out()
        self.out(self.c(f"━━━ Round {world.round} ━━━", 1) +
                 f"  pool: food {world.pool['food']}, wood {world.pool['wood']}")

    def turn(self, world: World, idx: int, d: Decision) -> None:
        v = world.get(d.villager)
        col = COLORS[idx % len(COLORS)]
        s = d.stats
        who = self.c(f"{v.name:<5}", f"1;{col}") + self.c(f"{v.persona:<10}", 2)
        for sp in d.speech:
            self.out(f"  {who} 💬 to {sp['to']}: {self.c(repr(sp['message']), 3)}")
        if d.action:
            a = d.action
            args = ", ".join(f"{k}={val!r}" for k, val in a["args"].items() if k != "reason")
            mark = self.c("✓", 32) if d.action_ok else self.c("✗", 31)
            self.out(f"  {who} {mark} {self.c(a['name'] + '(' + args + ')', 1)} → {d.action_result}")
            if a["args"].get("reason"):
                self.out(f"  {'':<15} because: {self.c(a['args']['reason'], 3)}")
        else:
            self.out(f"  {who} {self.c('– no action', 33)}")
        for n in d.notes:
            self.out(f"  {'':<15} {self.c('! ' + n, 33)}")
        self.out(self.c(f"  {'':<15} ⏱ {s.wall_s:.2f}s  (prompt {s.prompt_tokens} tok/{s.prompt_s:.2f}s, "
                        f"gen {s.out_tokens} tok/{s.out_s:.2f}s = {s.tok_per_s:.0f} tok/s)", 2))
        if self.verbose:
            self.out(self.c("    ┌─ prompt", 2))
            for line in d.prompt.splitlines():
                self.out(self.c(f"    │ {line}", 2))
            self.out(self.c(f"    └─ raw: content={d.raw_content!r} tool_calls={d.tool_calls}", 2))
        self.event("turn", round=world.round, villager=d.villager, persona=v.persona, prompt=d.prompt,
                   raw_content=d.raw_content, tool_calls=d.tool_calls, action=d.action,
                   action_ok=d.action_ok, action_result=d.action_result, speech=d.speech,
                   notes=d.notes, stats=s.to_dict(), state_after=world.snapshot()["villagers"][idx])

    def round_end(self, world: World, notes: list[str], timing: dict) -> None:
        for n in notes:
            self.out(self.c(f"  ⚠ {n}", 31))
        rows = []
        for v in world.villagers:
            if not v.alive:
                rows.append(self.c(f"{v.name}: dead", 2))
            else:
                rows.append(f"{v.name}: {v.food}f {v.wood}w hp{v.hp}{' 🏠' if v.house else ''}")
        houses = sum(v.house for v in world.living())
        self.out(self.c(f"  ── end of round: {' | '.join(rows)}", 2))
        self.out(self.c(f"     houses {houses}/{len(world.living())}  gini {world.gini():.2f}  "
                        f"round took {timing['round_s']:.2f}s (LLM {timing['llm_s']:.2f}s)", 2))
        self.event("round_end", world=world.snapshot(), notes=notes, timing=timing)

    # -- final summary ----------------------------------------------------------------------
    def summary(self, world: World, calls: list[dict], wall_s: float, warmup_s: float, reason: str) -> dict:
        living = world.living()
        houses = sum(v.house for v in living)
        llm_wall = sum(c["wall_s"] for c in calls)
        gen_tok = sum(c["out_tokens"] for c in calls)
        gen_s = sum(c["out_s"] for c in calls)
        prompt_tok = sum(c["prompt_tokens"] for c in calls)
        prompt_s = sum(c["prompt_s"] for c in calls)
        lat = sorted(c["wall_s"] for c in calls) or [0.0]
        p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
        valid = sum(1 for c in calls if c.get("ok"))
        s = {
            "ended_because": reason, "rounds": world.round, "alive": len(living),
            "villagers": len(world.villagers), "houses": houses, "gini": round(world.gini(), 3),
            "llm_calls": len(calls), "valid_actions": valid,
            "wall_s": round(wall_s, 2), "warmup_s": round(warmup_s, 2), "llm_wall_s": round(llm_wall, 2),
            "llm_share_pct": round(100 * llm_wall / wall_s, 1) if wall_s else 0,
            "latency_mean_s": round(statistics.mean(lat), 3), "latency_median_s": round(statistics.median(lat), 3),
            "latency_p95_s": round(p95, 3),
            "gen_tokens": gen_tok, "gen_tok_per_s": round(gen_tok / gen_s, 1) if gen_s else 0,
            "prompt_tokens_evaluated": prompt_tok,
            "prompt_tok_per_s": round(prompt_tok / prompt_s, 1) if prompt_s else 0,
            "model_load_s_during_run": round(sum(c["load_s"] for c in calls), 2),
            "http_overhead_s": round(sum(c["overhead_s"] for c in calls), 2),
        }
        self.event("run_end", summary=s, final=world.snapshot())
        B = lambda t: self.c(t, 1)
        self.out()
        self.out(B("━━━ Outcome ━━━"))
        self.out(f"  {reason}. {len(living)}/{len(world.villagers)} alive, {houses} houses, "
                 f"gini {s['gini']} (0 = equal) after {world.round} rounds")
        for v in world.villagers:
            self.out(f"    {v.name:<5}{v.persona:<11}" + ("DEAD" if not v.alive else
                     f"food {v.food:<3} wood {v.wood:<3} hp {v.hp:<3} {'house' if v.house else 'no house'}"))
        self.out(B("━━━ Performance ━━━"))
        self.out(f"  total {s['wall_s']}s  (+{s['warmup_s']}s one-off model load)   "
                 f"{s['llm_calls']} LLM calls, {valid} produced a valid action")
        self.out(f"  latency per decision: mean {s['latency_mean_s']}s  median {s['latency_median_s']}s  p95 {s['latency_p95_s']}s")
        self.out(f"  generation: {gen_tok} tokens @ {s['gen_tok_per_s']} tok/s    "
                 f"prompt: {prompt_tok} tokens evaluated @ {s['prompt_tok_per_s']} tok/s")
        self.out(f"  time in LLM {s['llm_share_pct']}% of wall; HTTP/other overhead {s['http_overhead_s']}s; "
                 f"model reloads during run {s['model_load_s_during_run']}s")
        return s
