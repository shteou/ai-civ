"""Pure game rules. No LLM, no I/O - so it is fast to test and easy to reason about.

The village: every villager gets one world action per round (gather / build / give / wait)
and may also speak. At the end of each round everyone eats one food (or starves), and the
shared pool regrows a little. The goal is for everybody to live in a house, fairly.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

RESOURCES = ("food", "wood")
START_FOOD = 2
MAX_HP = 10
STARVE_DAMAGE = 2
POOL_START = 60
POOL_REGROW = 2          # per resource, per round
MAX_POOL = 80

# persona -> (yield per gather of own speciality, yield of the other resource, wood cost of a house)
PERSONAS = {
    "farmer":     {"food": 3, "wood": 1, "house_cost": 6},
    "lumberjack": {"food": 1, "wood": 3, "house_cost": 6},
    "builder":    {"food": 1, "wood": 1, "house_cost": 3},
}
PERSONA_ORDER = ["farmer", "lumberjack", "builder"]
NAMES = ["Ada", "Bram", "Cleo", "Dax", "Eve", "Finn", "Gus", "Hana"]


@dataclass
class Villager:
    name: str
    persona: str
    food: int = START_FOOD
    wood: int = 0
    hp: int = MAX_HP
    house: bool = False
    alive: bool = True
    memory: list[str] = field(default_factory=list)   # recent events this villager has witnessed

    @property
    def house_cost(self) -> int:
        return PERSONAS[self.persona]["house_cost"]

    def yield_of(self, resource: str) -> int:
        return PERSONAS[self.persona][resource]

    def wealth(self) -> int:
        return self.food + self.wood + (PERSONAS[self.persona]["house_cost"] * 2 if self.house else 0)

    def remember(self, rnd: int, text: str, limit: int = 8) -> None:
        self.memory.append(f"R{rnd}: {text}")
        del self.memory[:-limit]


@dataclass
class Result:
    ok: bool
    text: str


class World:
    def __init__(self, n_villagers: int = 5, seed: int | None = None):
        self.rng = random.Random(seed)
        self.round = 0
        self.pool = {"food": POOL_START, "wood": POOL_START}
        self.villagers = [
            Villager(NAMES[i % len(NAMES)], PERSONA_ORDER[i % len(PERSONA_ORDER)])
            for i in range(n_villagers)
        ]

    # --- lookups -------------------------------------------------------------------------
    def get(self, name: str) -> Villager | None:
        for v in self.villagers:
            if v.name.lower() == str(name).strip().lower():
                return v
        return None

    def living(self) -> list[Villager]:
        return [v for v in self.villagers if v.alive]

    def done(self) -> bool:
        living = self.living()
        return not living or all(v.house for v in living)

    # --- actions -------------------------------------------------------------------------
    def gather(self, who: Villager, resource: str) -> Result:
        if resource not in RESOURCES:
            return Result(False, f"unknown resource {resource!r} (must be food or wood)")
        got = min(who.yield_of(resource), self.pool[resource])
        if got <= 0:
            return Result(False, f"the shared pool has no {resource} left")
        self.pool[resource] -= got
        setattr(who, resource, getattr(who, resource) + got)
        return Result(True, f"gathered {got} {resource} (pool {resource} now {self.pool[resource]})")

    def build_house(self, who: Villager) -> Result:
        if who.house:
            return Result(False, "already has a house")
        if who.wood < who.house_cost:
            return Result(False, f"needs {who.house_cost} wood to build a house but has {who.wood}")
        who.wood -= who.house_cost
        who.house = True
        return Result(True, f"built a house (spent {who.house_cost} wood)")

    def give(self, who: Villager, to: str, resource: str, amount) -> Result:
        target = self.get(to)
        if target is None or not target.alive or target is who:
            return Result(False, f"cannot give to {to!r} (not another living villager)")
        if resource not in RESOURCES:
            return Result(False, f"unknown resource {resource!r} (must be food or wood)")
        try:
            amount = int(amount)
        except (TypeError, ValueError):
            return Result(False, f"amount {amount!r} is not a whole number")
        if amount < 1 or getattr(who, resource) < amount:
            return Result(False, f"cannot give {amount} {resource}; only has {getattr(who, resource)}")
        setattr(who, resource, getattr(who, resource) - amount)
        setattr(target, resource, getattr(target, resource) + amount)
        target.remember(self.round, f"{who.name} gave you {amount} {resource}")
        return Result(True, f"gave {amount} {resource} to {target.name}")

    def say(self, who: Villager, to: str, message: str) -> Result:
        message = str(message).strip()[:200]
        if not message:
            return Result(False, "empty message")
        to = str(to).strip() or "all"
        if to.lower() == "all":
            for v in self.living():
                if v is not who:
                    v.remember(self.round, f'{who.name} says to everyone: "{message}"')
            return Result(True, f'said to everyone: "{message}"')
        target = self.get(to)
        if target is None or not target.alive or target is who:
            return Result(False, f"cannot talk to {to!r}")
        target.remember(self.round, f'{who.name} says to you: "{message}"')
        return Result(True, f'said to {target.name}: "{message}"')

    # --- round bookkeeping ---------------------------------------------------------------
    def end_round(self) -> list[str]:
        """Everyone eats, the pool regrows. Returns human-readable notes."""
        notes = []
        for v in self.living():
            if v.food > 0:
                v.food -= 1
                v.hp = min(MAX_HP, v.hp + 1)
            else:
                v.hp -= STARVE_DAMAGE
                notes.append(f"{v.name} went hungry (hp {v.hp})")
                v.remember(self.round, "you went hungry and lost health")
                if v.hp <= 0:
                    v.alive = False
                    notes.append(f"{v.name} starved to death")
        for r in RESOURCES:
            self.pool[r] = min(MAX_POOL, self.pool[r] + POOL_REGROW)
        return notes

    # --- metrics -------------------------------------------------------------------------
    def gini(self) -> float:
        """0 = perfectly equal wealth, 1 = one villager owns everything."""
        w = sorted(v.wealth() for v in self.living())
        n, total = len(w), sum(w)
        if n == 0 or total == 0:
            return 0.0
        return sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(w)) / (n * total)

    def snapshot(self) -> dict:
        return {
            "round": self.round,
            "pool": dict(self.pool),
            "gini": round(self.gini(), 3),
            "villagers": [
                {"name": v.name, "persona": v.persona, "food": v.food, "wood": v.wood,
                 "hp": v.hp, "house": v.house, "alive": v.alive}
                for v in self.villagers
            ],
        }
