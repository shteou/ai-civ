"""Rule and turn-loop tests. No Ollama required: the LLM is faked.   python -m unittest discover tests"""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from civ.agent import take_turn, parse_json_reply
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
        self.w = World(3, seed=0)
        self.w.round = 1
        self.ada, self.bram, self.cleo = self.w.villagers  # farmer, lumberjack, builder

    def test_gather_uses_persona_yield_and_depletes_pool(self):
        self.assertTrue(self.w.gather(self.ada, "food").ok)
        self.assertEqual((self.ada.food, self.w.pool["food"]), (2 + 3, 57))

    def test_gather_fails_on_empty_pool_or_bad_resource(self):
        self.w.pool["wood"] = 0
        self.assertFalse(self.w.gather(self.bram, "wood").ok)
        self.assertFalse(self.w.gather(self.bram, "gold").ok)

    def test_build_requires_wood_and_builders_are_cheaper(self):
        self.assertFalse(self.w.build_house(self.ada).ok)
        self.cleo.wood = 3
        self.assertTrue(self.w.build_house(self.cleo).ok)
        self.assertTrue(self.cleo.house and self.cleo.wood == 0)

    def test_give_validates_and_notifies_receiver(self):
        self.assertFalse(self.w.give(self.ada, "Bram", "food", 99).ok)
        self.assertFalse(self.w.give(self.ada, "Nobody", "food", 1).ok)
        self.assertFalse(self.w.give(self.ada, "Ada", "food", 1).ok)
        self.assertFalse(self.w.give(self.ada, "Bram", "food", "lots").ok)
        self.assertTrue(self.w.give(self.ada, "bram", "food", 1).ok)   # case-insensitive
        self.assertIn("Ada gave you 1 food", self.bram.memory[-1])

    def test_starvation_and_death(self):
        self.ada.food = 0
        for _ in range(MAX_HP // 2):
            notes = self.w.end_round()
        self.assertFalse(self.ada.alive)
        self.assertTrue(any("starved" in n for n in notes))

    def test_eating_heals_and_pool_regrows(self):
        self.ada.hp = 5
        self.w.pool["food"] = 10
        self.w.end_round()
        self.assertEqual((self.ada.hp, self.ada.food, self.w.pool["food"]), (6, 1, 12))

    def test_gini(self):
        for v in self.w.villagers:
            v.food, v.wood = 4, 0
        self.assertEqual(self.w.gini(), 0.0)
        self.ada.food, self.bram.food, self.cleo.food = 12, 0, 0
        self.assertAlmostEqual(self.w.gini(), 2 / 3, places=3)


class TurnLoop(unittest.TestCase):
    def setUp(self):
        self.w = World(2, seed=0)
        self.w.round = 1
        self.ada = self.w.villagers[0]

    def test_json_reply_executes_speech_and_action(self):
        reply = json.dumps({"action": "gather", "resource": "wood", "say_to": "all", "say": "hi", "reason": "x"})
        d = take_turn(self.w, self.ada, FakeLLM(reply), 10, "json")
        self.assertTrue(d.action_ok)
        self.assertEqual(d.action["name"], "gather")
        self.assertEqual(d.speech[0]["message"], "hi")
        self.assertIn("Ada says to everyone", self.w.villagers[1].memory[-1])

    def test_garbage_reply_is_survivable(self):
        d = take_turn(self.w, self.ada, FakeLLM("not json"), 10, "json")
        self.assertIsNone(d.action)
        self.assertTrue(d.notes)

    def test_failed_action_is_fed_back_to_memory(self):
        d = take_turn(self.w, self.ada, FakeLLM(json.dumps({"action": "build_house", "reason": "x"})), 10, "json")
        self.assertFalse(d.action_ok)
        self.assertIn("FAILED", self.ada.memory[-1])

    def test_native_tool_calls_only_one_world_action(self):
        tc = lambda n, **a: NS(function=NS(name=n, arguments=a))
        llm = FakeLLM("", [tc("gather", resource="food", reason="x"), tc("gather", resource="wood", reason="y")])
        d = take_turn(self.w, self.ada, llm, 10, "tools")
        self.assertEqual(d.action["args"]["resource"], "food")
        self.assertTrue(any("ignored extra" in n for n in d.notes))

    def test_parse_json_reply_non_object(self):
        notes = []
        self.assertEqual(parse_json_reply("[1,2]", notes), [])
        self.assertTrue(notes)


if __name__ == "__main__":
    unittest.main()
