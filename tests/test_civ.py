"""Rule and turn-loop tests. No Ollama required: the LLM is faked.   python -m unittest discover tests"""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from civ import world as W
from civ.agent import JSON_SCHEMA, TOOLS, build_prompt, parse_json_reply, take_turn
from civ.llm import CallStats
from civ.world import World, MAX_HP


class FakeLLM:
    def __init__(self, content="", tool_calls=None):
        self.content, self.tool_calls = content, tool_calls

    def chat(self, messages, tools=None, fmt=None):
        msg = NS(content=self.content, tool_calls=self.tool_calls)
        return NS(message=msg), CallStats(0.1, 0, 10, 0.01, 5, 0.05, 0)


class WorldRules(unittest.TestCase):
    def setUp(self):
        self.w = World(5, seed=0)
        self.w.round = 1
        self.farmer, self.lumber, self.miner, self.fisher, self.builder = self.w.villagers

    def at(self, v, loc):
        v.loc = loc
        return v

    def test_gather_needs_the_right_place_and_skill_doubles_yield(self):
        self.assertFalse(self.w.gather(self.farmer).ok)                    # Square has nothing
        self.at(self.farmer, "Fields"); self.at(self.lumber, "Fields")
        self.assertTrue(self.w.gather(self.farmer).ok)
        self.assertTrue(self.w.gather(self.lumber).ok)
        self.assertEqual(self.farmer.inv["food"], W.START_FOOD + W.SKILL_YIELD)
        self.assertEqual(self.lumber.inv["food"], W.START_FOOD + W.BASE_YIELD)
        self.assertEqual(self.w.pools["Fields"], 60 - W.SKILL_YIELD - W.BASE_YIELD)

    def test_move_routes_one_step_at_a_time(self):
        r = self.w.move(self.miner, "Hills")
        self.assertEqual((self.miner.loc, r.ok), ("Forest", True))
        self.assertIn("more step", r.text)
        self.w.move(self.miner, "Hills")
        self.assertEqual(self.miner.loc, "Hills")
        self.assertFalse(self.w.move(self.miner, "Mars").ok)
        self.assertFalse(self.w.move(self.miner, "Hills").ok)

    def test_winter_freezes_fields_and_stops_regrowth(self):
        self.w.round = 10
        self.assertTrue(self.w.winter())
        self.at(self.farmer, "Fields")
        self.assertFalse(self.w.gather(self.farmer).ok)
        self.w.pools["Lake"] = 10
        self.w.end_round()
        self.assertEqual(self.w.pools["Lake"], 10)
        self.w.round = 1
        self.w.end_round()
        self.assertGreater(self.w.pools["Lake"], 10)

    def test_build_needs_wood_and_stone_builders_cheaper(self):
        self.assertFalse(self.w.build_house(self.farmer).ok)
        self.builder.inv.update(wood=3, stone=1)
        self.assertTrue(self.w.build_house(self.builder).ok)
        self.assertTrue(self.builder.house)
        self.farmer.inv.update(wood=5, stone=1)
        self.assertFalse(self.w.build_house(self.farmer).ok)

    def test_give_requires_same_place_and_updates_reputation(self):
        self.at(self.lumber, "Forest")
        self.assertFalse(self.w.give(self.farmer, "Bram", "food", 1).ok)   # not here
        self.at(self.lumber, "Square")
        self.assertFalse(self.w.give(self.farmer, "Bram", "food", 99).ok)
        self.assertFalse(self.w.give(self.farmer, "Ada", "food", 1).ok)
        self.assertTrue(self.w.give(self.farmer, "bram", "food", 1).ok)
        self.assertEqual(self.farmer.rep, W.GIFT_REP)
        self.assertIn("Ada gave you 1 food", self.lumber.memory[-1])

    def test_offer_accept_swaps_and_validates(self):
        self.farmer.inv["food"], self.lumber.inv["wood"] = 5, 4
        self.assertFalse(self.w.offer(self.farmer, "Bram", "food", 50, "wood", 1).ok)
        self.assertFalse(self.w.offer(self.farmer, "Bram", "food", 1, "food", 1).ok)
        self.assertTrue(self.w.offer(self.farmer, "Bram", "food", 2, "wood", 3).ok)
        self.assertFalse(self.w.accept(self.miner, "Ada").ok)               # not their offer
        self.assertTrue(self.w.accept(self.lumber, "Ada").ok)
        self.assertEqual((self.farmer.inv["food"], self.farmer.inv["wood"]), (3, 3))
        self.assertEqual((self.lumber.inv["food"], self.lumber.inv["wood"]), (W.START_FOOD + 2, 1))
        self.assertFalse(self.w.accept(self.lumber, "Ada").ok)              # consumed

    def test_offers_expire_after_one_round(self):
        self.farmer.inv["food"] = 5
        self.w.offer(self.farmer, "Bram", "food", 1, "wood", 1)
        self.w.end_round()
        self.assertTrue(self.w.offers)
        self.w.round += 1
        self.w.end_round()
        self.assertFalse(self.w.offers)

    def test_steal_outcomes_are_witnessed_and_cost_reputation(self):
        self.w.rng.random = lambda: 0.0                                     # force success + noticed
        r = self.w.steal(self.farmer, "Bram", "food")
        self.assertTrue(r.ok)
        self.assertEqual(self.lumber.inv["food"], W.START_FOOD - 2)
        self.assertEqual(self.farmer.rep, W.THEFT_REP)
        self.assertIn("STOLE", self.lumber.memory[-1])
        self.assertIn("you saw Ada steal", self.miner.memory[-1])           # a witness in the Square
        self.w.rng.random = lambda: 0.99                                    # force failure
        self.assertFalse(self.w.steal(self.farmer, "Bram", "food").ok)

    def test_speech_is_local(self):
        self.at(self.miner, "Hills")
        self.w.say(self.farmer, "all", "hello")
        self.assertIn("hello", self.lumber.memory[-1])
        self.assertEqual(self.miner.memory, [])
        self.assertFalse(self.w.say(self.farmer, "Cleo", "hi").ok)

    def test_town_hall_completes_and_heals_everyone(self):
        self.farmer.hp = 3
        self.farmer.inv.update(wood=self.w.hall_cost["wood"], stone=self.w.hall_cost["stone"])
        self.assertTrue(self.w.contribute(self.farmer, "wood", 999).ok is False)   # more than held
        self.assertTrue(self.w.contribute(self.farmer, "wood", self.w.hall_cost["wood"]).ok)
        r = self.w.contribute(self.farmer, "stone", self.w.hall_cost["stone"])
        self.assertIn("COMPLETING", r.text)
        self.assertTrue(self.w.hall_done)
        self.assertEqual(self.farmer.hp, MAX_HP)
        self.at(self.lumber, "Forest")
        self.lumber.inv["wood"] = 1
        self.assertFalse(self.w.contribute(self.lumber, "wood", 1).ok)

    def test_starvation_and_death(self):
        self.farmer.inv["food"] = 0
        for _ in range(MAX_HP // W.STARVE_DAMAGE):
            notes = self.w.end_round()
        self.assertFalse(self.farmer.alive)
        self.assertTrue(any("starved" in n for n in notes))

    def test_victory_needs_houses_and_hall(self):
        for v in self.w.villagers:
            v.house = True
        self.assertFalse(self.w.done())
        self.w.hall_done = True
        self.assertTrue(self.w.done())

    def test_gini(self):
        for v in self.w.villagers:
            v.inv.update(food=4, wood=0, stone=0)
        self.assertEqual(self.w.gini(), 0.0)
        for v in self.w.villagers[1:]:
            v.inv["food"] = 0
        self.assertAlmostEqual(self.w.gini(), 0.8, places=3)

    def test_apply_never_raises_on_garbage(self):
        for name, args in [("give", {"to": None, "amount": "x"}), ("offer", {}), ("move", {"to": None}),
                           ("steal", {"target": 5}), ("bogus", {})]:
            self.assertFalse(self.w.apply(self.farmer, name, args).ok)


class TurnLoop(unittest.TestCase):
    def setUp(self):
        self.w = World(3, seed=0)
        self.w.round = 1
        self.ada = self.w.villagers[0]

    def test_json_reply_executes_speech_and_action(self):
        reply = json.dumps({"action": "move", "to": "Fields", "say_to": "all", "say": "hi", "reason": "x"})
        d = take_turn(self.w, self.ada, FakeLLM(reply), 10, "json")
        self.assertTrue(d.action_ok)
        self.assertEqual(self.ada.loc, "Fields")
        self.assertEqual(d.speech[0]["message"], "hi")
        self.assertIn("Ada says", self.w.villagers[1].memory[-1])   # heard: spoken before leaving, same place

    def test_garbage_reply_is_survivable(self):
        d = take_turn(self.w, self.ada, FakeLLM("not json"), 10, "json")
        self.assertIsNone(d.action)
        self.assertTrue(d.notes)

    def test_failed_action_is_fed_back_to_memory(self):
        d = take_turn(self.w, self.ada, FakeLLM(json.dumps({"action": "gather", "reason": "x"})), 10, "json")
        self.assertFalse(d.action_ok)
        self.assertIn("FAILED", self.ada.memory[-1])

    def test_native_tool_calls_only_one_world_action(self):
        tc = lambda n, **a: NS(function=NS(name=n, arguments=a))
        llm = FakeLLM("", [tc("move", to="Fields", reason="x"), tc("move", to="Forest", reason="y")])
        d = take_turn(self.w, self.ada, llm, 10, "tools")
        self.assertEqual(self.ada.loc, "Fields")
        self.assertTrue(any("ignored extra" in n for n in d.notes))

    def test_parse_json_reply_non_object(self):
        notes = []
        self.assertEqual(parse_json_reply("[1,2]", notes), [])
        self.assertTrue(notes)


class Prompts(unittest.TestCase):
    def test_every_action_has_schema_and_tool(self):
        names = {v["properties"]["action"]["enum"][0] for v in JSON_SCHEMA["anyOf"]}
        tools = {t["function"]["name"] for t in TOOLS} - {"say"}
        self.assertEqual(names, tools)

    def test_prompt_hides_far_away_inventories_and_flags_urgency(self):
        w = World(3, seed=0)
        w.round = 1
        w.villagers[1].loc = "Forest"
        w.villagers[0].inv["food"] = 1
        p = build_prompt(w, w.villagers[0], 10)
        self.assertIn("URGENT", p)
        self.assertIn("Bram (lumberjack, rep +0) at the Forest", p)     # location only, no inventory
        self.assertNotIn("Bram (lumberjack, rep +0): food", p)


if __name__ == "__main__":
    unittest.main()
