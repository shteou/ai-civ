"""View-state and TUI tests. Fake LLM, no Ollama needed.   python -m unittest discover tests"""
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from civ.llm import CallStats
from civ.report import Reporter
from civ.sim import run
from civ.tui import CivApp
from civ.view import ViewState, section
from civ.world import World


class ScriptedLLM:
    """Walks to the Fields then gathers; every other turn says something containing markup brackets."""

    def __init__(self):
        self.n = 0

    def warmup(self):
        return 0.0

    def chat(self, messages, tools=None, fmt=None):
        self.n += 1
        reply = {"action": "move", "to": "Fields", "reason": "go [red]eat[/red]"} if self.n <= 4 else \
                {"action": "gather", "reason": "need food"}
        if self.n % 2 == 0:
            reply.update(say_to="all", say="hello [bold]there")
        return NS(message=NS(content=json.dumps(reply), tool_calls=None)), CallStats(0.05, 0, 100, 0.01, 20, 0.2, 0)


def make_run(rounds=3, n=4):
    tmp = tempfile.TemporaryDirectory()
    world = World(n, seed=1)
    rep = Reporter(Path(tmp.name), color=False, echo=False)
    config = {"model": "fake", "seed": 1, "rounds": rounds, "mode": "json"}
    return tmp, world, rep, config


def collect_events(rounds=3, n=4):
    tmp, world, rep, config = make_run(rounds, n)
    events = []
    rep.listeners.append(events.append)
    run(world, ScriptedLLM(), rep, rounds, config, "json")
    rep.close()
    tmp.cleanup()
    return world, events


class ViewStateTests(unittest.TestCase):
    def test_view_matches_world_after_a_real_run(self):
        world, events = collect_events()
        st = ViewState()
        for ev in events:
            st.apply(ev)
        self.assertTrue(st.finished)
        self.assertEqual(st.turns, 12)
        self.assertEqual(st.round, world.round)
        for v in world.villagers:
            view = st.villagers[v.name]
            self.assertEqual((view.loc, view.food, view.wood, view.stone, view.hp, view.alive),
                             (v.loc, v.inv["food"], v.inv["wood"], v.inv["stone"], v.hp, v.alive))
        self.assertEqual(st.pools, world.pools)

    def test_events_survive_json_roundtrip_and_feed_is_markup_safe(self):
        _, events = collect_events()
        st = ViewState()
        lines = [l for ev in events for l in st.apply(json.loads(json.dumps(ev)))]
        joined = "\n".join(lines)
        self.assertIn("Round 1", joined)
        self.assertIn("\\[bold]there", joined)          # model text escaped, can't inject markup
        self.assertIn("\\[red]eat", joined)

    def test_memory_and_situation_are_parsed_from_the_prompt(self):
        _, events = collect_events()
        st = ViewState()
        for ev in events:
            st.apply(ev)
        v = next(iter(st.villagers.values()))
        self.assertTrue(any("you tried" in m for m in v.memory))
        self.assertTrue(any("food lasts" in s for s in v.situation))

    def test_section_parser(self):
        p = "A\nSituation:\n- x\n- y\nRecent events:\n- z\nChoose your action."
        self.assertEqual(section(p, "Situation:", "Recent events:"), ["- x", "- y"])
        self.assertEqual(section(p, "Recent events:", "Choose your action."), ["- z"])
        self.assertEqual(section(p, "Nope:"), [])

    def test_old_traces_without_round_start_or_pools_still_load(self):
        _, events = collect_events()
        old = []
        for ev in events:
            if ev["event"] == "round_start":
                continue
            ev = {k: v for k, v in ev.items() if k not in ("pools", "hall")}
            old.append(ev)
        st = ViewState()
        for ev in old:
            st.apply(ev)
        self.assertTrue(st.finished)
        self.assertEqual(st.turns, 12)


class AppTests(unittest.IsolatedAsyncioTestCase):
    async def _until(self, pred, timeout=8.0):
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout
        while not pred():
            self.assertLess(loop.time(), end, "timed out")
            await asyncio.sleep(0.02)

    async def test_replay_reaches_the_end_and_renders(self):
        _, events = collect_events()
        app = CivApp(replay=events, speed=100)
        async with app.run_test(size=(180, 50)) as pilot:
            await self._until(lambda: app.state.finished)
            await pilot.pause()
            self.assertEqual(app.state.turns, 12)
            self.assertIn(app.selected, app.state.villagers)
            self.assertIn("go", str(app.query_one("#resp").render()) + "go")   # pane rendered without error
            await pilot.press("2")                                              # memory tab
            await pilot.press("3")                                              # prompt tab
            await pilot.press("q")

    async def test_replay_pause_and_step(self):
        _, events = collect_events()
        app = CivApp(replay=events, speed=1)
        async with app.run_test(size=(180, 50)) as pilot:
            await self._until(lambda: app.state.turns >= 2)
            await pilot.press("space")
            await pilot.pause(0.2)
            frozen = app.state.turns
            await pilot.pause(0.6)
            self.assertEqual(app.state.turns, frozen)                           # paused: no progress
            await pilot.press("n")
            await self._until(lambda: app.state.turns == frozen + 1)            # one step = one turn
            await pilot.pause(0.3)
            self.assertEqual(app.state.turns, frozen + 1)
            await pilot.press("q")

    async def test_live_mode_runs_sim_in_a_thread_and_can_pause(self):
        tmp, world, rep, config = make_run(rounds=40, n=3)
        app = CivApp(live=(world, ScriptedLLM(), rep, 40, config, "json"))
        async with app.run_test(size=(180, 50)) as pilot:
            await self._until(lambda: app.state.turns >= 3)
            await pilot.press("space")                                          # pause the sim thread
            await pilot.pause(0.3)
            frozen = app.state.turns
            await pilot.pause(0.6)
            self.assertLessEqual(app.state.turns - frozen, 1)                   # at most the in-flight turn
            await pilot.press("space")
            await self._until(lambda: app.state.turns > frozen + 3)             # resumed
            await pilot.press("q")
        tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
