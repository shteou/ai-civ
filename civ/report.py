"""Human-readable console output, a plain-text log, a JSONL trace, and the performance summary."""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .agent import Decision
from .world import World, LOCATIONS

COLORS = [36, 33, 35, 32, 34, 31, 96, 93, 95, 92]
ICONS = {"gather": "🧺", "move": "🚶", "build_house": "🏠", "give": "🎁", "offer": "📜",
         "accept": "🤝", "steal": "🦹", "contribute": "🏛 ", "wait": "💤"}
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class Reporter:
    def __init__(self, out_dir: Path, color: bool | None = None, verbose: bool = False):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.color = sys.stdout.isatty() if color is None else color
        self.verbose = verbose
        self.trace = open(out_dir / "trace.jsonl", "w", buffering=1)
        self.log = open(out_dir / "log.txt", "w", buffering=1)
        self.mix: dict[str, Counter] = defaultdict(Counter)    # persona -> action counts

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
        self.out(self.c(f"Village of {len(world.villagers)}", 1))
        for i, v in enumerate(world.villagers):
            self.out(f"  {self.c(f'{v.name:<5}', f'1;{COLORS[i % len(COLORS)]}')} {v.persona:<11}{self.c(v.temperament, 3)}")
        hc = world.hall_cost
        self.out(f"Goal: everyone housed + Town Hall ({hc['wood']} wood, {hc['stone']} stone). Winter = last 3 rounds of every 12.")
        self.out(f"model={config['model']}  mode={config['mode']}  seed={config['seed']}  trace -> {self.out_dir}/trace.jsonl")

    def _positions(self, world: World) -> str:
        by = defaultdict(list)
        for v in world.living():
            by[v.loc].append(v.name)
        return "  ".join(f"{loc}[{' '.join(by[loc])}]" for loc in LOCATIONS if by[loc])

    def round_start(self, world: World) -> None:
        winter = world.winter()
        tag = self.c("❄ WINTER", "1;96") if winter else self.c("☀ summer", 33)
        hall = "BUILT" if world.hall_done else " ".join(f"{world.hall[r]}/{c}{r[0]}" for r, c in world.hall_cost.items())
        self.out()
        self.out(self.c(f"━━━ Round {world.round} ━━━", 1) + f" {tag}   hall {hall}   " + self.c(
            "pools " + " ".join(f"{p} {n}" for p, n in world.pools.items() if p != "Square"), 2))
        self.out(self.c(f"  map: {self._positions(world)}", 2))

    def turn(self, world: World, idx: int, d: Decision) -> None:
        v = world.get(d.villager)
        col = COLORS[idx % len(COLORS)]
        s = d.stats
        who = self.c(f"{v.name:<5}", f"1;{col}") + self.c(f"{v.persona:<11}", 2) + self.c(f"@{d.loc_before:<7}", 2)
        pad = " " * 25
        for sp in d.speech:
            self.out(f"  {who} 💬 to {sp['to']}: {self.c(repr(sp['message']), 3)}")
        if d.action:
            a = d.action
            self.mix[v.persona][a["name"]] += 1
            args = ", ".join(f"{k}={val!r}" for k, val in a["args"].items() if k != "reason")
            mark = self.c("✓", 32) if d.action_ok else self.c("✗", 31)
            self.out(f"  {who} {mark} {ICONS.get(a['name'], '?')} {self.c(a['name'] + '(' + args + ')', 1)} → {d.action_result}")
            if a["args"].get("reason"):
                self.out(f"{pad}because: {self.c(a['args']['reason'], 3)}")
        else:
            self.out(f"  {who} {self.c('– no action', 33)}")
        for n in d.notes:
            self.out(f"{pad}{self.c('! ' + n, 33)}")
        self.out(self.c(f"{pad}⏱ {s.wall_s:.2f}s  (prompt {s.prompt_tokens} tok/{s.prompt_s:.2f}s, "
                        f"gen {s.out_tokens} tok/{s.out_s:.2f}s = {s.tok_per_s:.0f} tok/s)", 2))
        if self.verbose:
            self.out(self.c("    ┌─ prompt", 2))
            for line in d.prompt.splitlines():
                self.out(self.c(f"    │ {line}", 2))
            self.out(self.c(f"    └─ raw: content={d.raw_content!r} tool_calls={d.tool_calls}", 2))
        self.event("turn", round=world.round, villager=d.villager, persona=v.persona, loc_before=d.loc_before,
                   prompt=d.prompt, raw_content=d.raw_content, tool_calls=d.tool_calls, action=d.action,
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
                rows.append(f"{v.name}@{v.loc[:3]}: {v.inv['food']}f {v.inv['wood']}w {v.inv['stone']}s hp{v.hp}{' 🏠' if v.house else ''}")
        self.out(self.c("  ── end of round: " + " | ".join(rows), 2))
        houses = sum(v.house for v in world.living())
        self.out(self.c(f"     houses {houses}/{len(world.living())}  gini {world.gini():.2f}  "
                        f"round took {timing['round_s']:.2f}s (LLM {timing['llm_s']:.2f}s)", 2))
        self.event("round_end", world=world.snapshot(), notes=notes, timing=timing)

    # -- emergent behaviour report ---------------------------------------------------------
    def emergence(self, world: World) -> dict:
        ev = world.events
        kinds = Counter(e["kind"] for e in ev)
        gifts = Counter((e["actor"], e["target"]) for e in ev if e["kind"] == "gift")
        trades = [e for e in ev if e["kind"] == "trade"]
        thefts = [e for e in ev if e["kind"] in ("theft", "theft_failed")]
        stats = {
            "gifts": kinds["gift"], "offers": kinds["offer"], "trades": kinds["trade"],
            "thefts_ok": kinds["theft"], "thefts_failed": kinds["theft_failed"],
            "thefts_caught": sum(1 for e in ev if e["kind"] == "theft" and e.get("caught")),
            "speeches": kinds["say"], "deaths": kinds["death"],
            "hall": dict(world.hall), "hall_contributors": dict(world.hall_by),
            "reputation": {v.name: v.rep for v in world.villagers},
            "action_mix": {p: dict(c) for p, c in self.mix.items()},
        }
        B = lambda t: self.c(t, 1)
        self.out(B("━━━ What emerged ━━━"))
        self.out(f"  social: {kinds['say']} things said · {kinds['gift']} gifts · {kinds['offer']} offers → {kinds['trade']} trades · "
                 f"{kinds['theft']} thefts ({stats['thefts_caught']} seen) + {kinds['theft_failed']} failed attempts")
        if gifts:
            top = ", ".join(f"{a}→{b} ×{n}" for (a, b), n in gifts.most_common(4))
            self.out(f"  gift network: {top}")
        if trades:
            self.out("  trades: " + "; ".join(f"R{e['round']} {e['actor']}↔{e['target']} ({e['give'][1]} {e['give'][0]} for {e['want'][1]} {e['want'][0]})" for e in trades[:5]))
        if thefts:
            self.out("  crime: " + "; ".join(f"R{e['round']} {e['actor']}→{e['target']}{' ✗' if e['kind'] == 'theft_failed' else ''}" for e in thefts[:6]))
        done = "BUILT" if world.hall_done else "unfinished"
        who = ", ".join(f"{n} {k}" for n, k in sorted(world.hall_by.items(), key=lambda x: -x[1])) or "nobody"
        self.out(f"  Town Hall {done}: {world.hall['wood']}/{world.hall_cost['wood']} wood, {world.hall['stone']}/{world.hall_cost['stone']} stone  (contributors: {who})")
        rep = sorted(world.villagers, key=lambda v: -v.rep)
        self.out("  reputation: " + ", ".join(f"{v.name} {v.rep:+d}" for v in rep))
        for persona, c in self.mix.items():
            self.out(self.c(f"  {persona:<11}" + " ".join(f"{a} {n}" for a, n in c.most_common()), 2))
        marks = [e for e in ev if e["kind"] in ("house", "death", "hall_done", "winter")]
        if marks:
            self.out("  timeline: " + " · ".join(f"R{e['round']} {e['actor'] + ' ' if e['actor'] != 'world' else ''}{e['kind']}" for e in marks[:14]))
        return stats

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
        B = lambda t: self.c(t, 1)
        self.out()
        self.out(B("━━━ Outcome ━━━"))
        self.out(f"  {reason}. {len(living)}/{len(world.villagers)} alive, {houses} houses, "
                 f"Town Hall {'built' if world.hall_done else 'not built'}, gini {s['gini']} (0 = equal) after {world.round} rounds")
        for v in world.villagers:
            self.out(f"    {v.name:<5}{v.persona:<11}{v.temperament:<12}" + ("DEAD" if not v.alive else
                     f"{v.inv['food']}f {v.inv['wood']}w {v.inv['stone']}s  hp {v.hp:<3} {'house' if v.house else 'no house'}"))
        s["emergence"] = self.emergence(world)
        self.out(B("━━━ Performance ━━━"))
        self.out(f"  total {s['wall_s']}s  (+{s['warmup_s']}s one-off model load)   "
                 f"{s['llm_calls']} LLM calls, {valid} produced a valid action")
        self.out(f"  latency per decision: mean {s['latency_mean_s']}s  median {s['latency_median_s']}s  p95 {s['latency_p95_s']}s")
        self.out(f"  generation: {gen_tok} tokens @ {s['gen_tok_per_s']} tok/s    "
                 f"prompt: {prompt_tok} tokens evaluated @ {s['prompt_tok_per_s']} tok/s")
        self.out(f"  time in LLM {s['llm_share_pct']}% of wall; HTTP/other overhead {s['http_overhead_s']}s; "
                 f"model reloads during run {s['model_load_s_during_run']}s")
        self.event("run_end", summary=s, final=world.snapshot(), events=world.events)
        return s
