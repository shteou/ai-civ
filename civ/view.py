"""UI-independent view state for the TUI.

It consumes the exact event dicts that `Reporter` writes to `trace.jsonl` ("run_start", "round_start",
"turn", "round_end", "run_end"), so a live run and a replay of a saved trace go through one code path.
Nothing here imports Textual, which keeps it unit-testable.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from rich.markup import escape

from .report import ICONS
from .world import WINTER_LEN, YEAR

COLORS = ["cyan", "yellow", "magenta", "green", "blue", "red", "bright_cyan", "bright_yellow",
          "bright_magenta", "bright_green"]
PLACES = ["Hills", "Forest", "Square", "Fields", "Lake"]


def section(prompt: str, start: str, *ends: str) -> list[str]:
    """Bullet lines of a prompt section, e.g. section(p, 'Recent events:', 'Choose your action.')."""
    if start not in prompt:
        return []
    body = prompt.split(start, 1)[1]
    for e in ends:
        if e in body:
            body = body.split(e, 1)[0]
            break
    return [l.strip() for l in body.strip().splitlines() if l.strip()]


@dataclass
class VillagerView:
    name: str
    persona: str
    temperament: str
    color: str
    loc: str = "Square"
    food: int = 0
    wood: int = 0
    stone: int = 0
    hp: int = 10
    house: bool = False
    alive: bool = True
    rep: int = 0
    last: dict | None = None                       # most recent "turn" event for this villager
    history: list[tuple] = field(default_factory=list)   # (round, action name, ok, summary)
    memory: list[str] = field(default_factory=list)      # "Recent events" as the model last saw them
    situation: list[str] = field(default_factory=list)   # derived hints the model last saw

    def update(self, d: dict) -> None:
        for k in ("loc", "food", "wood", "stone", "hp", "house", "alive", "rep"):
            if k in d:
                setattr(self, k, d[k])

    @property
    def last_icon(self) -> str:
        a = (self.last or {}).get("action")
        return ICONS.get(a["name"], "?") if a else " "


class ViewState:
    def __init__(self) -> None:
        self.config: dict = {}
        self.system_prompt = ""
        self.round = 0
        self.winter = False
        self.pools: dict = {}
        self.hall: dict = {}
        self.hall_cost: dict = {}
        self.hall_done = False
        self.gini = 0.0
        self.villagers: dict[str, VillagerView] = {}
        self.latencies: list[float] = []
        self.out_tokens = 0
        self.out_s = 0.0
        self.turns = 0
        self.valid = 0
        self.round_times: list[float] = []
        self.summary: dict | None = None
        self.world_events: list[dict] = []
        self.finished = False
        self.current: str | None = None       # villager who acted most recently

    # -- derived -------------------------------------------------------------------------
    @property
    def alive(self) -> list[VillagerView]:
        return [v for v in self.villagers.values() if v.alive]

    @property
    def houses(self) -> int:
        return sum(v.house for v in self.alive)

    @property
    def mean_latency(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0.0

    @property
    def tok_per_s(self) -> float:
        return self.out_tokens / self.out_s if self.out_s else 0.0

    def at(self, place: str) -> list[VillagerView]:
        return [v for v in self.alive if v.loc == place]

    # -- event ingestion -------------------------------------------------------------------
    def _sync_world(self, w: dict) -> None:
        self.round = w.get("round", self.round)
        self.winter = w.get("winter", self.winter)
        self.pools = w.get("pools", self.pools)
        self.hall = w.get("hall", self.hall)
        self.hall_cost = w.get("hall_cost", self.hall_cost)
        self.hall_done = w.get("hall_done", self.hall_done)
        self.gini = w.get("gini", self.gini)
        for i, d in enumerate(w.get("villagers", [])):
            v = self.villagers.get(d["name"])
            if v is None:
                v = self.villagers[d["name"]] = VillagerView(
                    d["name"], d["persona"], d.get("temperament", ""), COLORS[i % len(COLORS)])
            v.update(d)

    def apply(self, ev: dict) -> list[str]:
        """Update state from one trace event; return rich-markup lines to append to the feed."""
        kind = ev.get("event")
        if kind == "run_start":
            self.config = ev.get("config", {})
            self.system_prompt = ev.get("system_prompt", "")
            self._sync_world(ev["world"])
            return [f"[b]Village of {len(self.villagers)}[/b]  model {self.config.get('model')}  seed {self.config.get('seed')}"]
        if kind == "round_start":
            self._sync_world(ev["world"])
            tag = "[bold bright_cyan]❄ WINTER[/]" if self.winter else "[yellow]☀ summer[/]"
            return ["", f"[b]━━━ Round {self.round} ━━━[/b] {tag}"]
        if kind == "turn":
            return self._turn(ev)
        if kind == "round_end":
            self._sync_world(ev["world"])
            if "timing" in ev:
                self.round_times.append(ev["timing"].get("round_s", 0.0))
            return [f"[red]⚠ {escape(n)}[/red]" for n in ev.get("notes", [])]
        if kind == "run_end":
            self.summary = ev.get("summary")
            self.world_events = ev.get("events", [])
            self._sync_world(ev["final"])
            self.finished = True
            s = self.summary or {}
            return ["", f"[b]━━━ Finished: {escape(str(s.get('ended_because', '')))} ━━━[/b]"]
        return []

    def _turn(self, ev: dict) -> list[str]:
        v = self.villagers[ev["villager"]]
        v.update(ev["state_after"])
        self.round = ev.get("round", self.round)
        if "pools" not in ev:             # trace written before round_start/pools existed
            self.winter = (self.round - 1) % YEAR >= YEAR - WINTER_LEN
        self.pools = ev.get("pools", self.pools)
        self.hall = ev.get("hall", self.hall)
        v.last = ev
        v.memory = section(ev["prompt"], "Recent events:", "Choose your action.")
        v.situation = section(ev["prompt"], "Situation:", "Recent events:")
        st = ev.get("stats", {})
        self.latencies.append(st.get("wall_s", 0.0))
        self.out_tokens += st.get("out_tokens", 0)
        self.out_s += st.get("out_s", 0.0)
        self.turns += 1
        self.current = v.name
        a = ev.get("action")
        if ev.get("action_ok"):
            self.valid += 1
        c = v.color
        who = f"[bold {c}]{v.name:<5}[/] [dim]{v.persona}@{ev.get('loc_before', '')}[/dim]"
        lines = []
        for sp in ev.get("speech", []):
            lines.append(f"{who} 💬 [i]to {escape(str(sp['to']))}:[/i] {escape(str(sp['message']))}")
        if a:
            args = ", ".join(f"{k}={val}" for k, val in a["args"].items() if k != "reason")
            mark = "[green]✓[/green]" if ev["action_ok"] else "[red]✗[/red]"
            lines.append(f"{who} {mark} {ICONS.get(a['name'], '?')} [b]{a['name']}({escape(args)})[/b] → {escape(ev['action_result'])}")
            why = a["args"].get("reason")
            if why:
                lines.append(f"      [dim italic]because: {escape(str(why))}[/dim italic]")
            v.history.append((ev["round"], a["name"], bool(ev["action_ok"]), ev["action_result"]))
        else:
            lines.append(f"{who} [yellow]– no action[/yellow]")
            v.history.append((ev["round"], "-", False, "no action"))
        for n in ev.get("notes", []):
            lines.append(f"      [yellow]! {escape(n)}[/yellow]")
        return lines
