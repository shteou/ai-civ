"""Pure game rules. No LLM, no I/O - so it is fast to test and easy to reason about.

A small map of five places. Each villager has a location and gets ONE world action per round
(gather / move / build / give / offer / accept / steal / contribute / wait) and may also speak.
Speech and inventory are only visible to villagers standing in the same place, so gathering
spots and the Square (where the Town Hall is) become meeting points.

Pressures that can produce emergent behaviour:
  * local scarcity  - each place has its own finite, slowly regrowing pool
  * winter          - the fields give nothing and nothing regrows; you must stockpile
  * specialisation  - skilled villagers gather 2x as much, but only at their home place
  * trade / theft   - binding offers vs. stealing, with witnesses and a public reputation
  * a shared goal   - the Town Hall needs contributions from everyone
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

RESOURCES = ("food", "wood", "stone")
START_FOOD = 5
MAX_HP = 10
STARVE_DAMAGE = 2
SKILL_YIELD, BASE_YIELD = 4, 2

# place -> neighbours, resource found there, starting pool, regrowth per round
MAP = {
    "Square": {"adj": ["Fields", "Forest", "Lake"], "res": None, "pool": 0, "regrow": 0},
    "Fields": {"adj": ["Square", "Lake"], "res": "food", "pool": 60, "regrow": 4},
    "Lake":   {"adj": ["Square", "Fields"], "res": "food", "pool": 60, "regrow": 3},
    "Forest": {"adj": ["Square", "Hills"], "res": "wood", "pool": 60, "regrow": 3},
    "Hills":  {"adj": ["Forest"], "res": "stone", "pool": 40, "regrow": 2},
}
LOCATIONS = list(MAP)

# persona -> place where they are skilled (None = no gathering skill)
PERSONAS = {
    "farmer": "Fields", "fisher": "Lake", "lumberjack": "Forest", "miner": "Hills", "builder": None,
}
PERSONA_ORDER = ["farmer", "lumberjack", "miner", "fisher", "builder", "farmer", "lumberjack", "miner"]
HOUSE_COST = {"wood": 5, "stone": 2}
BUILDER_HOUSE_COST = {"wood": 3, "stone": 1}

TEMPERAMENTS = {
    "generous": "You love helping others and share what you can spare.",
    "greedy": "You like to hoard and drive hard bargains.",
    "sociable": "You like to talk, organise people and make plans together.",
    "cautious": "You worry about winter and always keep a big food reserve.",
    "opportunist": "You will take what isn't nailed down if you think you can get away with it.",
    "honest": "You are fair and straightforward, and you dislike thieves.",
}
NAMES = ["Ada", "Bram", "Cleo", "Dax", "Eve", "Finn", "Gus", "Hana", "Ivo", "Jun"]

YEAR, WINTER_LEN = 12, 3          # rounds 10-12 of each 12-round year are winter
THEFT_SUCCESS = 0.5
THEFT_REP, GIFT_REP, TRADE_REP = -3, 1, 1


def fmt_cost(cost: dict) -> str:
    return " + ".join(f"{n} {r}" for r, n in cost.items() if n)


@dataclass
class Villager:
    name: str
    persona: str
    temperament: str
    loc: str = "Square"
    inv: dict = field(default_factory=lambda: {"food": START_FOOD, "wood": 0, "stone": 0})
    hp: int = MAX_HP
    house: bool = False
    alive: bool = True
    rep: int = 0                                       # public reputation
    memory: list[str] = field(default_factory=list)    # recent events this villager has witnessed

    @property
    def house_cost(self) -> dict:
        return BUILDER_HOUSE_COST if self.persona == "builder" else HOUSE_COST

    def needs(self) -> set:
        """Resources this villager is currently short of."""
        out = set(self.missing_for_house()) if not self.house else set()
        if self.inv["food"] <= 3:
            out.add("food")
        return out

    def missing_for_house(self) -> dict:
        return {r: n - self.inv[r] for r, n in self.house_cost.items() if n > self.inv[r]}

    def spare(self) -> dict:
        """What this villager could trade away without hurting themselves."""
        need = self.missing_for_house() if not self.house else {}
        out = {}
        for r, n in self.inv.items():
            keep = 3 if r == "food" else (self.house_cost[r] if r in need or not self.house else 0)
            if n - keep > 0:
                out[r] = n - keep
        return out

    def wealth(self) -> int:
        return sum(self.inv.values()) + (sum(self.house_cost.values()) * 2 if self.house else 0)

    def remember(self, rnd: int, text: str, limit: int = 10) -> None:
        self.memory.append(f"R{rnd}: {text}")
        del self.memory[:-limit]


@dataclass
class Result:
    ok: bool
    text: str


class World:
    def __init__(self, n_villagers: int = 8, seed: int | None = None):
        self.rng = random.Random(seed)
        self.round = 0
        self.pools = {p: m["pool"] for p, m in MAP.items()}
        temps = list(TEMPERAMENTS)
        self.rng.shuffle(temps)
        self.villagers = [
            Villager(NAMES[i % len(NAMES)], PERSONA_ORDER[i % len(PERSONA_ORDER)], temps[i % len(temps)])
            for i in range(n_villagers)
        ]
        n = n_villagers
        self.hall_cost = {"wood": n, "stone": max(2, n // 2)}
        self.hall = {"wood": 0, "stone": 0}
        self.hall_by: dict[str, int] = {}
        self.hall_done = False
        self.offers: list[dict] = []
        self.events: list[dict] = []    # structured log of social/economic events, for analysis

    # --- lookups -------------------------------------------------------------------------
    def get(self, name: str) -> Villager | None:
        for v in self.villagers:
            if v.name.lower() == str(name).strip().lower():
                return v
        return None

    def living(self) -> list[Villager]:
        return [v for v in self.villagers if v.alive]

    def here(self, who: Villager) -> list[Villager]:
        return [v for v in self.living() if v is not who and v.loc == who.loc]

    def done(self) -> bool:
        living = self.living()
        return not living or (self.hall_done and all(v.house for v in living))

    def path(self, a: str, b: str) -> list[str]:
        """Shortest route from a to b (exclusive of a)."""
        prev, q = {a: None}, deque([a])
        while q:
            cur = q.popleft()
            if cur == b:
                break
            for nxt in MAP[cur]["adj"]:
                if nxt not in prev:
                    prev[nxt] = cur
                    q.append(nxt)
        out, cur = [], b
        while cur != a:
            out.append(cur)
            cur = prev[cur]
        return out[::-1]

    def log(self, kind: str, actor: str, target: str = "", **detail) -> None:
        self.events.append({"round": self.round, "kind": kind, "actor": actor, "target": target, **detail})

    # --- seasons -------------------------------------------------------------------------
    def winter(self, rnd: int | None = None) -> bool:
        return ((rnd or self.round) - 1) % YEAR >= YEAR - WINTER_LEN

    def season_note(self) -> str:
        pos = (self.round - 1) % YEAR
        if self.winter():
            left = YEAR - pos
            return f"WINTER ({left} more round(s)): the Fields give no food and nothing regrows."
        until = YEAR - WINTER_LEN - pos
        return f"Summer. Winter begins in {until} round(s) and lasts {WINTER_LEN}."

    # --- actions -------------------------------------------------------------------------
    def apply(self, who: Villager, name: str, args: dict) -> Result:
        """Execute one action by name. Never raises on bad model input."""
        try:
            if name == "gather":
                return self.gather(who)
            if name == "move":
                return self.move(who, str(args.get("to", "")))
            if name == "build_house":
                return self.build_house(who)
            if name == "give":
                return self.give(who, args.get("to", ""), str(args.get("resource", "")), args.get("amount"))
            if name == "offer":
                return self.offer(who, args.get("to", ""), str(args.get("give_resource", "")), args.get("give_amount"),
                                  str(args.get("want_resource", "")), args.get("want_amount"))
            if name == "accept":
                return self.accept(who, args.get("target", ""))
            if name == "steal":
                return self.steal(who, args.get("target", ""), str(args.get("resource", "")))
            if name == "contribute":
                return self.contribute(who, str(args.get("resource", "")), args.get("amount"))
            if name == "wait":
                return Result(True, "waited")
            return Result(False, f"unknown action {name!r}")
        except Exception as e:   # a malformed call must never crash the simulation
            return Result(False, f"error executing {name}: {e}")

    @staticmethod
    def _int(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return None

    def gather(self, who: Villager) -> Result:
        res = MAP[who.loc]["res"]
        if res is None:
            return Result(False, f"nothing to gather at the {who.loc}; walk to Fields/Lake (food), Forest (wood) or Hills (stone)")
        if who.loc == "Fields" and self.winter():
            return Result(False, "the Fields are frozen in winter")
        skilled = PERSONAS[who.persona] == who.loc
        got = min(SKILL_YIELD if skilled else BASE_YIELD, self.pools[who.loc])
        if got <= 0:
            return Result(False, f"the {who.loc} has no {res} left")
        self.pools[who.loc] -= got
        who.inv[res] += got
        return Result(True, f"gathered {got} {res} at the {who.loc} ({self.pools[who.loc]} left there)")

    def move(self, who: Villager, to: str) -> Result:
        dest = next((l for l in LOCATIONS if l.lower() == to.strip().lower()), None)
        if dest is None:
            return Result(False, f"no such place {to!r}; places are {', '.join(LOCATIONS)}")
        if dest == who.loc:
            return Result(False, f"already at the {dest}")
        step = self.path(who.loc, dest)[0]
        before, who.loc = who.loc, step
        via = "" if step == dest else f" (heading to {dest}, {len(self.path(step, dest))} more step(s))"
        return Result(True, f"walked {before} -> {step}{via}")

    def build_house(self, who: Villager) -> Result:
        if who.house:
            return Result(False, "already has a house")
        miss = who.missing_for_house()
        if miss:
            return Result(False, f"cannot build yet: needs {fmt_cost(who.house_cost)}, still missing {fmt_cost(miss)}")
        for r, n in who.house_cost.items():
            who.inv[r] -= n
        who.house = True
        self.log("house", who.name)
        return Result(True, f"built a house (spent {fmt_cost(who.house_cost)})")

    def _co_located(self, who: Villager, name) -> tuple[Villager | None, str]:
        t = self.get(name)
        if t is None or not t.alive or t is who:
            return None, f"{name!r} is not another living villager"
        if t.loc != who.loc:
            return None, f"{t.name} is not here (they are at the {t.loc})"
        return t, ""

    def say(self, who: Villager, to, message) -> Result:
        """Speech is only heard by villagers standing in the same place."""
        message = str(message).strip()[:200]
        if not message:
            return Result(False, "empty message")
        to = str(to or "all").strip()
        if to.lower() in ("all", "everyone", ""):
            listeners = self.here(who)
            for v in listeners:
                v.remember(self.round, f'{who.name} says: "{message}"')
            self.log("say", who.name, "all", heard_by=[v.name for v in listeners])
            who_ = ", ".join(v.name for v in listeners) or "nobody"
            return Result(True, f'said to everyone here ({who_}): "{message}"')
        t, err = self._co_located(who, to)
        if t is None:
            return Result(False, f"cannot talk: {err}")
        t.remember(self.round, f'{who.name} says to you: "{message}"')
        self.log("say", who.name, t.name, heard_by=[t.name])
        return Result(True, f'said to {t.name}: "{message}"')

    def give(self, who: Villager, to, resource: str, amount) -> Result:
        t, err = self._co_located(who, to)
        if t is None:
            return Result(False, f"cannot give: {err}")
        n = self._int(amount)
        if resource not in RESOURCES:
            return Result(False, f"unknown resource {resource!r} (food, wood or stone)")
        if n is None or n < 1 or who.inv[resource] < n:
            return Result(False, f"cannot give {amount} {resource}; only has {who.inv[resource]}")
        who.inv[resource] -= n
        t.inv[resource] += n
        who.rep += GIFT_REP
        t.remember(self.round, f"{who.name} gave you {n} {resource}")
        self.log("gift", who.name, t.name, resource=resource, amount=n)
        return Result(True, f"gave {n} {resource} to {t.name}")

    def offer(self, who: Villager, to, give_res: str, give_n, want_res: str, want_n) -> Result:
        t, err = self._co_located(who, to)
        if t is None:
            return Result(False, f"cannot offer: {err}")
        gn, wn = self._int(give_n), self._int(want_n)
        if give_res not in RESOURCES or want_res not in RESOURCES or give_res == want_res:
            return Result(False, "an offer must swap two different resources")
        if not gn or not wn or gn < 1 or wn < 1:
            return Result(False, "amounts must be whole numbers >= 1")
        if who.inv[give_res] < gn:
            return Result(False, f"you only have {who.inv[give_res]} {give_res}")
        self.offers = [o for o in self.offers if not (o["from"] == who.name and o["to"] == t.name)]
        self.offers.append({"from": who.name, "to": t.name, "give": (give_res, gn), "want": (want_res, wn), "round": self.round})
        t.remember(self.round, f"{who.name} offered you {gn} {give_res} for {wn} {want_res} (use accept to take it)")
        self.log("offer", who.name, t.name, give=(give_res, gn), want=(want_res, wn))
        return Result(True, f"offered {t.name} {gn} {give_res} for {wn} {want_res}")

    def offers_for(self, who: Villager) -> list[dict]:
        return [o for o in self.offers if o["to"] == who.name]

    def accept(self, who: Villager, target) -> Result:
        o = next((o for o in self.offers if o["to"] == who.name and o["from"].lower() == str(target).strip().lower()), None)
        if o is None:
            return Result(False, f"no pending offer from {target!r}")
        other = self.get(o["from"])
        if other is None or not other.alive or other.loc != who.loc:
            return Result(False, f"{o['from']} is no longer here")
        (gr, gn), (wr, wn) = o["give"], o["want"]
        if other.inv[gr] < gn:
            self.offers.remove(o)
            return Result(False, f"{other.name} no longer has {gn} {gr}")
        if who.inv[wr] < wn:
            return Result(False, f"you only have {who.inv[wr]} {wr} but the deal needs {wn}")
        other.inv[gr] -= gn; who.inv[gr] += gn
        who.inv[wr] -= wn; other.inv[wr] += wn
        self.offers.remove(o)
        who.rep += TRADE_REP; other.rep += TRADE_REP
        other.remember(self.round, f"{who.name} accepted your deal: you gave {gn} {gr} and got {wn} {wr}")
        self.log("trade", other.name, who.name, give=(gr, gn), want=(wr, wn))
        return Result(True, f"traded with {other.name}: got {gn} {gr} for {wn} {wr}")

    def steal(self, who: Villager, target, resource: str) -> Result:
        t, err = self._co_located(who, target)
        if t is None:
            return Result(False, f"cannot steal: {err}")
        if resource not in RESOURCES:
            return Result(False, f"unknown resource {resource!r}")
        witnesses = [v for v in self.here(who) if v is not t]
        if self.rng.random() < THEFT_SUCCESS:
            n = min(2, t.inv[resource])
            if n == 0:
                return Result(False, f"{t.name} has no {resource} to steal")
            t.inv[resource] -= n; who.inv[resource] += n
            victim_notices = self.rng.random() < 0.6
            caught = victim_notices or bool(witnesses)
            if victim_notices:
                t.remember(self.round, f"{who.name} STOLE {n} {resource} from you!")
            for w in witnesses:
                w.remember(self.round, f"you saw {who.name} steal {n} {resource} from {t.name}")
            if caught:
                who.rep += THEFT_REP
            self.log("theft", who.name, t.name, resource=resource, amount=n, caught=caught)
            return Result(True, f"stole {n} {resource} from {t.name}" + (" - and was seen!" if caught else " - unnoticed"))
        t.remember(self.round, f"{who.name} tried to steal {resource} from you but failed")
        for w in witnesses:
            w.remember(self.round, f"you saw {who.name} try to steal from {t.name}")
        who.rep += THEFT_REP
        self.log("theft_failed", who.name, t.name, resource=resource)
        return Result(False, f"caught trying to steal {resource} from {t.name}")

    def contribute(self, who: Villager, resource: str, amount) -> Result:
        if who.loc != "Square":
            return Result(False, "the Town Hall is in the Square; walk there first")
        if self.hall_done:
            return Result(False, "the Town Hall is already built")
        if resource not in self.hall:
            return Result(False, "the Town Hall only needs wood and stone")
        n = self._int(amount)
        need = self.hall_cost[resource] - self.hall[resource]
        if n is None or n < 1 or who.inv[resource] < n:
            return Result(False, f"cannot contribute {amount} {resource}; only has {who.inv[resource]}")
        if need <= 0:
            return Result(False, f"the Town Hall already has all the {resource} it needs")
        n = min(n, need)
        who.inv[resource] -= n
        self.hall[resource] += n
        self.hall_by[who.name] = self.hall_by.get(who.name, 0) + n
        who.rep += GIFT_REP
        self.log("contribute", who.name, resource=resource, amount=n)
        done = all(self.hall[r] >= c for r, c in self.hall_cost.items())
        if done:
            self.hall_done = True
            self.log("hall_done", who.name)
            for v in self.living():
                v.hp = MAX_HP
                v.remember(self.round, "the Town Hall was completed! Everyone is healed.")
        return Result(True, f"added {n} {resource} to the Town Hall" + (" - COMPLETING IT!" if done else ""))

    # --- round bookkeeping ---------------------------------------------------------------
    def end_round(self) -> list[str]:
        """Everyone eats, pools regrow (except in winter), stale offers expire."""
        notes = []
        for v in self.living():
            if v.inv["food"] > 0:
                v.inv["food"] -= 1
                v.hp = min(MAX_HP, v.hp + 1)
            else:
                v.hp -= STARVE_DAMAGE
                notes.append(f"{v.name} went hungry (hp {v.hp})")
                v.remember(self.round, "you went hungry and lost health")
                if v.hp <= 0:
                    v.alive = False
                    notes.append(f"{v.name} starved to death")
                    self.log("death", v.name)
        if not self.winter():
            for p, m in MAP.items():
                self.pools[p] = min(m["pool"], self.pools[p] + m["regrow"])
        self.offers = [o for o in self.offers if o["round"] >= self.round]
        nxt = self.round + 1
        if self.winter(nxt) and not self.winter(self.round):
            notes.append("winter is coming - the Fields will freeze")
            self.log("winter", "world")
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
            "round": self.round, "winter": self.winter(), "pools": dict(self.pools),
            "hall": dict(self.hall), "hall_cost": dict(self.hall_cost), "hall_done": self.hall_done,
            "gini": round(self.gini(), 3),
            "villagers": [
                {"name": v.name, "persona": v.persona, "temperament": v.temperament, "loc": v.loc,
                 **v.inv, "hp": v.hp, "house": v.house, "alive": v.alive, "rep": v.rep}
                for v in self.villagers
            ],
        }
