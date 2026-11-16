"""Textual TUI: world map, villager stats, and per-villager response / memory / prompt inspection.

Two sources feed the same view (see civ/view.py):
  * live   - the simulation runs in a worker thread and pushes every trace event to the UI
  * replay - events are read back from a saved trace.jsonl (no GPU / Ollama needed)
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from rich.markup import escape
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.widgets import DataTable, Footer, RichLog, Sparkline, Static, TabbedContent, TabPane

from .view import ViewState, VillagerView
from .world import MAP

RES = {p: m["res"] for p, m in MAP.items()}


def bar(n: int, total: int, width: int = 12) -> str:
    total = max(total, 1)
    filled = min(width, round(width * n / total))
    return "█" * filled + "░" * (width - filled)


def hp_markup(hp: int) -> str:
    colour = "green" if hp >= 7 else "yellow" if hp >= 4 else "red"
    return f"[{colour}]♥{hp:<2}[/]"


def load_trace(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


class Place(Static):
    """One bordered box on the map listing the villagers currently standing there."""

    def __init__(self, place: str, **kw):
        kw.setdefault("id", f"place-{place.lower()}")
        super().__init__("", classes="place", **kw)
        self.place = place
        self.border_title = place

    def show(self, st: ViewState) -> None:
        res = RES.get(self.place)
        if res:
            frozen = " ❄ frozen" if self.place == "Fields" and st.winter else ""
            self.border_subtitle = f"{res} {st.pools.get(self.place, 0)}{frozen}"
        else:
            self.border_subtitle = "town hall"
        lines = []
        for v in st.at(self.place):
            name = f"[bold {v.color}]{v.name:<5}[/]"
            if v.name == st.current:
                name = f"[reverse]{name}[/reverse]"
            warn = "[red]⚠[/]" if v.food == 0 else " "
            lines.append(f"{name} {hp_markup(v.hp)} f{v.food:<2} w{v.wood:<2} s{v.stone:<2}{warn}{'🏠' if v.house else ''}{v.last_icon}")
        self.set_class(bool(lines), "occupied")
        self.update("\n".join(lines))


class CivApp(App):
    TITLE = "ai-civilization"
    CSS = """
    #top { height: 3; }
    #status { width: 1fr; padding: 1 1; }
    #spark { width: 32; height: 3; margin: 0 1; }
    #main { height: 1fr; }
    #left { width: 56%; }
    #right { width: 44%; }
    #map { grid-size: 3 3; grid-columns: 1fr 1fr 1fr; grid-rows: 1fr 1fr 1fr; height: 56%; }
    .place { border: round $primary-darken-2; padding: 0 1; overflow: hidden; }
    .place.occupied { border: round $primary; }
    #hall { column-span: 2; border: round $warning; padding: 0 1; }
    #legend { padding: 0 1; color: $text-muted; }
    #lake { column-span: 2; }
    #villagers { height: 44%; }
    #detail { height: 55%; }
    #feed { height: 45%; border: round $primary-darken-2; padding: 0 1; }
    .pane { padding: 0 1; }
    """
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("space", "toggle_pause", "Pause/Resume"),
        Binding("n", "step", "Step turn (replay)"),
        Binding("f", "follow", "Follow actor"),
        Binding("plus,equals_sign", "faster", "Faster"),
        Binding("minus", "slower", "Slower"),
        Binding("1", "tab('t-resp')", "Response"),
        Binding("2", "tab('t-mem')", "Memory"),
        Binding("3", "tab('t-prompt')", "Prompt"),
        Binding("4", "tab('t-sys')", "System"),
    ]

    def __init__(self, live=None, replay: list[dict] | None = None, speed: float = 1.0):
        super().__init__()
        self.live = live                      # (world, llm, reporter, rounds, config, mode)
        self.replay = replay
        self.speed = speed
        self.state = ViewState()
        self.paused = False
        self.follow = True
        self.selected: str | None = None
        self._steps = 0
        self._programmatic = False
        self._run_flag = threading.Event()
        self._run_flag.set()
        self._error: str | None = None

    # -- layout --------------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        with Horizontal(id="top"):
            yield Static("", id="status")
            yield Sparkline([], id="spark")
        with Horizontal(id="main"):
            with Vertical(id="left"):
                with Grid(id="map"):
                    yield Place("Hills")
                    yield Static("", id="hall")
                    yield Place("Forest")
                    yield Place("Square")
                    yield Place("Fields")
                    yield Static("[b]legend[/b]  🧺 gather  🚶 move  🏠 house  📜 offer  🤝 accept\n🎁 give  🦹 steal  🏛 contribute  💤 wait   ⚠ no food", id="legend")
                    yield Place("Lake", id="lake")
                yield DataTable(id="villagers", cursor_type="row", zebra_stripes=True)
            with Vertical(id="right"):
                with TabbedContent(id="detail"):
                    with TabPane("1 Response", id="t-resp"):
                        with VerticalScroll():
                            yield Static("", id="resp", classes="pane")
                    with TabPane("2 Memory", id="t-mem"):
                        with VerticalScroll():
                            yield Static("", id="mem", classes="pane")
                    with TabPane("3 Prompt", id="t-prompt"):
                        with VerticalScroll():
                            yield Static("", id="prompt", classes="pane")
                    with TabPane("4 System", id="t-sys"):
                        with VerticalScroll():
                            yield Static("", id="sys", classes="pane")
                yield RichLog(id="feed", markup=True, wrap=True, highlight=False, max_lines=4000)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#villagers", DataTable)
        table.add_columns("", "villager", "role", "temperament", "at", "food", "wood", "stone", "hp", "rep", "last")
        self.query_one("#feed", RichLog).border_title = "log"
        self.query_one("#map").border_title = "map"
        if self.replay is not None:
            self.run_worker(self._replay_loop(), exclusive=True)
        elif self.live is not None:
            self._start_live()
        self._refresh()

    # -- sources -------------------------------------------------------------------------
    def _start_live(self) -> None:
        from .sim import run
        world, llm, rep, rounds, config, mode = self.live
        rep.listeners.append(lambda ev: self.call_from_thread(self.ingest, ev))
        rep.gate = self._run_flag.wait          # blocks the sim between turns while paused

        def work():
            try:
                run(world, llm, rep, rounds, config, mode)
            except Exception as e:  # surface it in the UI instead of dying silently in a thread
                self.call_from_thread(self._fail, repr(e))
            finally:
                rep.close()

        threading.Thread(target=work, daemon=True).start()

    def _fail(self, msg: str) -> None:
        self._error = msg
        self.query_one("#feed", RichLog).write(f"[bold red]simulation error: {escape(msg)}[/]")
        self._refresh()

    async def _replay_loop(self) -> None:
        i, events = 0, self.replay
        while i < len(events):
            if self.paused and self._steps <= 0:
                await asyncio.sleep(0.05)
                continue
            ev = events[i]
            i += 1
            self.ingest(ev)
            if ev["event"] == "turn":
                if self.paused:
                    self._steps -= 1
                else:
                    await asyncio.sleep(0.35 / self.speed)

    # -- state -> widgets ----------------------------------------------------------------
    def ingest(self, ev: dict) -> None:
        lines = self.state.apply(ev)
        log = self.query_one("#feed", RichLog)
        for l in lines:
            log.write(l)
        if ev["event"] == "turn" and self.follow:
            self.selected = self.state.current
        if self.selected is None and self.state.villagers:
            self.selected = next(iter(self.state.villagers))
        self._refresh()

    def _status(self) -> str:
        s = self.state
        total = s.config.get("rounds", "?")
        season = "[bold bright_cyan]❄ WINTER[/]" if s.winter else "[yellow]☀ summer[/]"
        if s.finished:
            mode = "[green]FINISHED[/]"
        elif self._error:
            mode = "[red]ERROR[/]"
        elif self.paused or not self._run_flag.is_set():
            mode = "[yellow]PAUSED[/]"
        else:
            mode = "[green]RUNNING[/]" if (self.live or self.replay) else ""
        src = "replay" if self.replay is not None else "live"
        valid = f"{100 * s.valid // s.turns}%" if s.turns else "-"
        speed = f"  x{self.speed:g}" if self.replay is not None else ""
        return (f"[b]Round {s.round}/{total}[/b] {season}  alive {len(s.alive)}/{len(s.villagers)}  "
                f"houses {s.houses}  gini {s.gini:.2f}  │  ⏱ {s.mean_latency:.2f}s/decision  "
                f"{s.tok_per_s:.0f} tok/s  valid {valid}  │  {src} {mode}{speed}")

    def _hall(self) -> str:
        s = self.state
        if s.hall_done:
            body = "[b green]TOWN HALL BUILT[/]"
        else:
            body = "  ".join(f"{r} {bar(s.hall.get(r, 0), c, 10)} {s.hall.get(r, 0)}/{c}" for r, c in s.hall_cost.items())
        dead = [v.name for v in s.villagers.values() if not v.alive]
        extra = f"\n[red]☠ {', '.join(dead)}[/]" if dead else ""
        houses = f"\nhouses {bar(s.houses, max(len(s.alive), 1), 10)} {s.houses}/{len(s.alive)}"
        return f"[b]Town Hall[/b]  {body}{houses}{extra}"

    def _refresh(self) -> None:
        s = self.state
        self.query_one("#status", Static).update(self._status())
        self.query_one("#spark", Sparkline).data = s.latencies[-60:] or [0]
        self.query_one("#hall", Static).update(self._hall())
        self.query_one("#hall").border_title = "village"
        for p in self.query(Place):
            p.show(s)
        self._refresh_table()
        self._refresh_detail()

    def _refresh_table(self) -> None:
        t = self.query_one("#villagers", DataTable)
        keep = t.cursor_row
        t.clear()
        for v in self.state.villagers.values():
            dim = "" if v.alive else "dim strike "
            name = f"[{dim}bold {v.color}]{v.name}[/]" if dim else f"[bold {v.color}]{v.name}[/]"
            t.add_row("☠" if not v.alive else ("●" if v.name == self.state.current else " "),
                      name, v.persona, v.temperament, v.loc if v.alive else "-",
                      str(v.food), str(v.wood), str(v.stone), str(v.hp), f"{v.rep:+d}",
                      v.last_icon, key=v.name)
        if self.state.villagers:
            names = list(self.state.villagers)
            row = names.index(self.selected) if self.selected in names else min(keep, len(names) - 1)
            self._programmatic = True
            t.move_cursor(row=row)
            self.call_after_refresh(setattr, self, "_programmatic", False)

    def _refresh_detail(self) -> None:
        v = self.state.villagers.get(self.selected or "")
        if v is None:
            return
        self.query_one("#resp", Static).update(self._resp(v))
        self.query_one("#mem", Static).update(self._mem(v))
        self.query_one("#prompt", Static).update(escape(v.last["prompt"]) if v.last else "(no turn yet)")
        self.query_one("#sys", Static).update(escape(self.state.system_prompt) or "(waiting for run start)")
        self.query_one("#detail").border_title = f"{v.name} · {v.persona} · {v.temperament}"

    def _resp(self, v: VillagerView) -> str:
        head = (f"[bold {v.color}]{v.name}[/] — {v.persona}, [i]{v.temperament}[/i]   rep {v.rep:+d}   "
                f"at {v.loc}   {hp_markup(v.hp)}  food {v.food}  wood {v.wood}  stone {v.stone}  "
                f"{'🏠 has house' if v.house else 'no house'}{'' if v.alive else '  [red]DEAD[/]'}")
        e = v.last
        if not e:
            return head + "\n\n(has not acted yet)"
        a = e.get("action")
        out = [head, "", f"[b]Round {e['round']}[/b] at {e['loc_before']}"]
        for sp in e.get("speech", []):
            out.append(f"💬 to {escape(str(sp['to']))}: [i]{escape(str(sp['message']))}[/i]")
        if a:
            args = ", ".join(f"{k}={val!r}" for k, val in a["args"].items() if k != "reason")
            ok = "[green]✓ OK[/]" if e["action_ok"] else "[red]✗ FAILED[/]"
            out += [f"{ok}  [b]{a['name']}({escape(args)})[/b]", f"→ {escape(e['action_result'])}",
                    f"[italic]“{escape(str(a['args'].get('reason', '')))}”[/italic]"]
        else:
            out.append("[yellow]no usable action[/]")
        for n in e.get("notes", []):
            out.append(f"[yellow]! {escape(n)}[/]")
        st = e.get("stats", {})
        out += ["", f"[dim]⏱ {st.get('wall_s', 0):.2f}s · prompt {st.get('prompt_tokens', 0)} tok / "
                    f"{st.get('prompt_s', 0):.2f}s · generated {st.get('out_tokens', 0)} tok @ {st.get('tok_per_s', 0):.0f} tok/s[/]",
                "", "[b]raw model reply[/b]", f"[dim]{escape(e.get('raw_content', ''))}[/]", "", "[b]recent actions[/b]"]
        for rnd, name, ok, text in v.history[-8:][::-1]:
            out.append(f"{'[green]✓[/]' if ok else '[red]✗[/]'} R{rnd:<2} {name:<11} [dim]{escape(text[:70])}[/]")
        return "\n".join(out)

    def _mem(self, v: VillagerView) -> str:
        if not v.last:
            return "(nothing yet)"
        mem = "\n".join(escape(m) for m in v.memory) or "(nothing yet)"
        sit = "\n".join(escape(m) for m in v.situation) or "(none)"
        return (f"[b]What {v.name} remembered when deciding (round {v.last['round']})[/b]\n{mem}\n\n"
                f"[b]Derived hints the prompt gave {v.name}[/b]\n{sit}")

    # -- events --------------------------------------------------------------------------
    @on(DataTable.RowHighlighted)
    def _row_highlighted(self, ev: DataTable.RowHighlighted) -> None:
        if self._programmatic or ev.row_key is None or ev.row_key.value is None:
            return
        if ev.row_key.value != self.selected:
            self.follow = False
            self.selected = ev.row_key.value
            self._refresh_detail()

    def action_toggle_pause(self) -> None:
        if self.live is not None:
            if self._run_flag.is_set():
                self._run_flag.clear()
            else:
                self._run_flag.set()
        else:
            self.paused = not self.paused
        self._refresh()

    def action_step(self) -> None:
        if self.replay is None:
            self.notify("Stepping works in replay mode only (use space to pause a live run)")
            return
        self.paused = True
        self._steps += 1
        self._refresh()

    def action_follow(self) -> None:
        self.follow = True
        self.selected = self.state.current or self.selected
        self.notify("following the acting villager")
        self._refresh()

    def action_faster(self) -> None:
        self.speed = min(32.0, self.speed * 1.5)
        self._refresh()

    def action_slower(self) -> None:
        self.speed = max(0.1, self.speed / 1.5)
        self._refresh()

    def action_tab(self, tab: str) -> None:
        self.query_one(TabbedContent).active = tab

    async def action_quit(self) -> None:
        self._run_flag.set()                   # release a paused sim thread so we can exit cleanly
        self.exit()
