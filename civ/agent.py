"""Turns world state into a prompt, asks the LLM for a decision, and applies it to the world."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .llm import LLM, CallStats
from .world import LOCATIONS, MAP, PERSONAS, TEMPERAMENTS, RESOURCES, World, Villager, fmt_cost, YEAR, WINTER_LEN

# ---------------------------------------------------------------------------------------------
# Action spec: single source of truth for the JSON schema, the native tool definitions, and the
# prompt text, so they can't drift apart.
# ---------------------------------------------------------------------------------------------
_S = {"type": "string"}
_I = {"type": "integer"}
_RES = {"type": "string", "enum": list(RESOURCES)}
ACTIONS: dict[str, tuple[str, dict]] = {
    "gather":      ("Gather the resource found at your current location.", {}),
    "move":        ("Walk one step toward a place.", {"to": {"type": "string", "enum": LOCATIONS}}),
    "build_house": ("Build your house from the wood and stone you carry.", {}),
    "give":        ("Gift a resource to a villager in your location.", {"to": _S, "resource": _RES, "amount": _I}),
    "offer":       ("Offer a binding swap to a villager in your location: you give X, you want Y.",
                    {"to": _S, "give_resource": _RES, "give_amount": _I, "want_resource": _RES, "want_amount": _I}),
    "accept":      ("Accept a pending offer made to you by villager `target` (same location).", {"target": _S}),
    "steal":       ("Try to steal up to 2 of a resource from a villager in your location. Risky.",
                    {"target": _S, "resource": _RES}),
    "contribute":  ("Add wood or stone to the Town Hall (only in the Square).",
                    {"resource": {"type": "string", "enum": ["wood", "stone"]}, "amount": _I}),
    "wait":        ("Do nothing this round.", {}),
}
_SPEECH = {"say_to": _S, "say": {"type": "string", "maxLength": 120}}
_REASON = {"reason": {"type": "string", "maxLength": 80, "description": "max 10 words"}}

SYSTEM_PROMPT = """You are a villager in a small village simulation. Each round you choose ONE action, and you may also say something (only villagers in your current place can hear you). Always include a short `reason`.

The map: the Square (village centre, Town Hall) connects to the Fields, the Forest and the Lake. The Fields and Lake connect to each other. The Hills are only reachable through the Forest.
Resources: food at the Fields and Lake, wood in the Forest, stone in the Hills. You can only gather where the resource is. Skilled villagers gather 4 per action at their home place; everyone else gathers 2. Each place has its own finite pool that regrows slowly, so over-harvesting empties it.
Survival: everyone eats 1 food at the end of every round. With no food you lose 2 hp; at 0 hp you die.
Winter: every 12 rounds there are 3 winter rounds (the last 3 of each 12). In winter the Fields give nothing and nothing regrows (the Lake still works). Stock up food beforehand.
Houses cost wood and stone (5 wood + 2 stone; builders only 3 + 1). The Town Hall is a village project in the Square that everyone can contribute wood and stone to; when finished everyone is fully healed. The village succeeds when everyone has a house AND the Town Hall is built.
Social: you see other villagers' belongings only if they are in your place. `give` is a gift; `offer` + `accept` is a binding swap; `steal` can fail and gets you a bad reputation if seen. Reputation is public.

Only speak when you have something specific to say (a request, an offer, a warning); never repeat what was already said.
Think about what the village needs, not only yourself. Priorities: 1) never run out of food, 2) get the materials for your house, 3) help others and trade where it helps, 4) contribute to the Town Hall.
"""


def json_addendum() -> str:
    lines = ["", "Reply with ONE JSON object. Choose `action` from:"]
    for name, (desc, props) in ACTIONS.items():
        lines.append(f"- {name}({', '.join(props)}): {desc}")
    lines.append('Optionally add "say_to" (a villager here, or "all") and "say" (what you say). "reason" (max 10 words) is always required.')
    return "\n".join(lines)


def _variant(action: str, props: dict) -> dict:
    properties = {"action": {"type": "string", "enum": [action]}, **props, **_SPEECH, **_REASON}
    return {"type": "object", "properties": properties, "required": ["action", *props, "reason"]}


# Per-action variants so that e.g. `give` can't omit its target. Grammar-constrained decoding.
JSON_SCHEMA = {"anyOf": [_variant(n, p) for n, (_, p) in ACTIONS.items()]}

TOOLS = [{"type": "function", "function": {
    "name": n, "description": d,
    "parameters": {"type": "object", "properties": {**p, **_REASON}, "required": [*p, "reason"]}}}
    for n, (d, p) in ACTIONS.items()] + [{"type": "function", "function": {
        "name": "say", "description": "Say something to a villager in your place, or 'all'. Doesn't use your action.",
        "parameters": {"type": "object", "properties": {"to": _S, "message": _S}, "required": ["to", "message"]}}}]


# ---------------------------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------------------------
def _route(world: World, a: str, b: str) -> str:
    return " -> ".join([a, *world.path(a, b)])


def situation(world: World, me: Villager) -> list[str]:
    """Facts derived from the state, so the model doesn't have to do arithmetic or path-finding."""
    notes = [f"Your food lasts {me.inv['food']} more round(s)."]
    if me.inv["food"] <= 2:
        spots = [p for p, m in MAP.items() if m["res"] == "food" and world.pools[p] > 0
                 and not (p == "Fields" and world.winter())]
        if spots:
            best = min(spots, key=lambda p: len(world.path(me.loc, p)) if p != me.loc else 0)
            how = "gather it right here" if best == me.loc else f"go to the {best}: {_route(world, me.loc, best)}"
            notes.append(f"URGENT: you are almost out of food. Get food first - {how}.")
        else:
            notes.append("URGENT: you are almost out of food and no place has any left - ask others for food.")
    here_res0 = MAP[me.loc]["res"]
    if here_res0 and world.pools[me.loc] <= 3 and not (me.loc == "Fields" and world.winter()):
        notes.append(f"The {me.loc} is nearly out of {here_res0}: gathering here will soon fail; "
                     "go where there is more (see Stocks).")
    pos = (world.round - 1) % YEAR
    to_winter = YEAR - WINTER_LEN - pos
    if not world.winter() and to_winter <= 3 and me.inv["food"] < 6:
        notes.append(f"Winter is {to_winter} round(s) away and you only hold {me.inv['food']} food: build a reserve (the Lake works in winter).")
    if world.winter() and me.inv["food"] < 3:
        notes.append("It is winter and you are low on food: the Lake is the only place to gather it.")
    if not me.house:
        miss = me.missing_for_house()
        if not miss:
            notes.append("You can afford your house: use build_house now.")
        else:
            where = []
            for r, n in miss.items():
                spot = next(p for p, m in MAP.items() if m["res"] == r)
                where.append(f"{n} {r} ({r} is at the {spot}; from here: {_route(world, me.loc, spot)})" if spot != me.loc
                             else f"{n} {r} (you can gather it right here)")
            notes.append("For your house you still need " + "; ".join(where) + ".")
    spare = me.spare()
    if spare:
        notes.append("You could spare for trade or gifts: " + ", ".join(f"{n} {r}" for r, n in spare.items())
                     + " (never offer more than you hold).")
    here_res = MAP[me.loc]["res"]
    if here_res and not (me.loc == "Fields" and world.winter()):
        skilled = PERSONAS[me.persona] == me.loc
        notes.append(f"Gathering here yields {4 if skilled else 2} {here_res} per action.")
    if me.loc == "Square" and not world.hall_done and (me.house or not me.missing_for_house()):
        notes.append("You are in the Square: you can contribute spare wood/stone to the Town Hall.")
    home = PERSONAS[me.persona]
    if home and MAP[home]["res"] != "food":
        r = MAP[home]["res"]
        demand = sum(v.missing_for_house().get(r, 0) for v in world.living() if not v.house)
        if not world.hall_done:
            demand += max(0, world.hall_cost.get(r, 0) - world.hall.get(r, 0))
        if demand > 0 and me.loc != home:
            notes.append(f"The village still needs {demand} {r} in total, and you are the best at gathering it (4 per action at the {home}); "
                         f"{r}-holders can trade it for food or wood.")
    my_need = me.needs()
    for v in world.here(me):
        if v.inv["food"] == 0:
            notes.append(f"{v.name} is here and has NO food.")
        their_need = v.needs()
        i_want = [r for r in my_need if v.inv[r] >= 3 and r not in their_need]
        they_want = [r for r in their_need if me.inv[r] >= 3 and r not in my_need]
        if i_want and they_want:
            notes.append(f"Possible swap with {v.name}: they have spare {i_want[0]}, you have spare {they_want[0]} - try an offer.")
        elif i_want:
            notes.append(f"{v.name} has spare {i_want[0]} that you need: offer them something or ask.")
        elif they_want and me.temperament in ("generous", "sociable"):
            notes.append(f"{v.name} needs {they_want[0]}, which you can spare.")
    weak = [f"{v.name} (hp {v.hp}, at the {v.loc})" for v in world.living() if v is not me and v.hp <= 5]
    if weak and me.inv["food"] >= 6:
        notes.append(f"Villagers looking starved: {', '.join(weak)} - you hold spare food.")
    if me.temperament in ("opportunist", "greedy"):
        for v in world.here(me):
            r = next((r for r in my_need if v.inv[r] >= 4), None)
            if r:
                others = len(world.here(me)) - 1
                seen = "nobody else would see" if others == 0 else f"{others} other villager(s) here could see"
                notes.append(f"Temptation: {v.name} carries {v.inv[r]} {r} that you need. Stealing succeeds about half the time; {seen}.")
                break
    return notes


def build_prompt(world: World, me: Villager, max_rounds: int) -> str:
    skill = PERSONAS[me.persona]
    skill_txt = f"skilled at the {skill}" if skill else "a cheap house-builder with no gathering skill"
    res = MAP[me.loc]["res"]
    here_txt = (f"{res} ({world.pools[me.loc]} left)" if res else "nothing to gather")
    inv = ", ".join(f"{r} {me.inv[r]}" for r in RESOURCES)
    miss = me.missing_for_house()
    house = "you HAVE a house" if me.house else f"no house yet (needs {fmt_cost(me.house_cost)})"

    def vline(v: Villager, full: bool) -> str:
        head = f"- {v.name} ({v.persona}, rep {v.rep:+d})"
        if full:
            return head + f": {', '.join(f'{r} {v.inv[r]}' for r in RESOURCES)}, hp {v.hp}, {'HAS house' if v.house else 'no house'}"
        return head + f" at the {v.loc}, hp {v.hp}, {'has house' if v.house else 'no house'}"

    present = world.here(me)
    elsewhere = [v for v in world.living() if v is not me and v.loc != me.loc]
    offers = world.offers_for(me)
    parts = [
        f"You are {me.name}, a {me.persona} ({skill_txt}). {TEMPERAMENTS[me.temperament]}",
        f"Round {world.round} of {max_rounds}. {world.season_note()}",
        f"You are at the {me.loc}. Here: {here_txt}. Neighbouring places: {', '.join(MAP[me.loc]['adj'])}.",
        "Stocks left: " + ", ".join(f"{p} {world.pools[p]} {m['res']}" for p, m in MAP.items() if m["res"]) + ".",
        f"You carry: {inv}; hp {me.hp}; {house}; your reputation {me.rep:+d}.",
        "Town Hall: " + ("BUILT." if world.hall_done else
                         ", ".join(f"{world.hall[r]}/{c} {r}" for r, c in world.hall_cost.items()) + " contributed."),
        "Here with you:\n" + ("\n".join(vline(v, True) for v in present) or "- (nobody)"),
        "Elsewhere:\n" + ("\n".join(vline(v, False) for v in elsewhere) or "- (nobody)"),
    ]
    if offers:
        parts.append("Offers made to you (reply with accept, target=<name>):\n" + "\n".join(
            f"- {o['from']} offers {o['give'][1]} {o['give'][0]} for your {o['want'][1]} {o['want'][0]}" for o in offers))
    parts.append("Situation:\n" + "\n".join(f"- {n}" for n in situation(world, me)))
    parts.append("Recent events:\n" + ("\n".join(f"- {m}" for m in me.memory) or "- (nothing yet)"))
    parts.append("Choose your action.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------------------------
# One turn
# ---------------------------------------------------------------------------------------------
@dataclass
class Decision:
    """Everything that happened in one villager turn - this is what gets traced."""
    villager: str
    prompt: str
    raw_content: str
    tool_calls: list[dict]
    stats: CallStats
    action: dict | None = None            # the world action chosen (None = nothing valid)
    action_result: str = ""
    action_ok: bool = False
    speech: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)   # parsing problems, ignored extra calls, etc.
    loc_before: str = ""


def parse_json_reply(text: str, notes: list[str]) -> list[dict]:
    """Convert the JSON reply into the same call list that native tool calling produces."""
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("not an object")
    except ValueError as e:
        notes.append(f"unparseable JSON reply ({e})")
        return []
    calls = []
    if obj.get("say"):
        calls.append({"name": "say", "args": {"to": obj.get("say_to") or "all", "message": obj["say"]}})
    args = {k: v for k, v in obj.items() if k not in ("action", "say", "say_to")}
    calls.append({"name": obj.get("action", "wait"), "args": args})
    return calls


def take_turn(world: World, me: Villager, llm: LLM, max_rounds: int, mode: str = "json") -> Decision:
    prompt = build_prompt(world, me, max_rounds)
    notes: list[str] = []
    if mode == "json":
        messages = [{"role": "system", "content": SYSTEM_PROMPT + json_addendum()}, {"role": "user", "content": prompt}]
        resp, stats = llm.chat(messages, fmt=JSON_SCHEMA)
        calls = parse_json_reply(resp.message.content or "", notes)
    else:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        resp, stats = llm.chat(messages, tools=TOOLS)
        calls = [{"name": c.function.name, "args": dict(c.function.arguments or {})}
                 for c in (resp.message.tool_calls or [])]
    d = Decision(me.name, prompt, resp.message.content or "", calls, stats, notes=notes, loc_before=me.loc)

    if not calls:
        notes.append("model made no usable reply - treated as wait")
        d.action_result = "no action"
        return d

    for call in calls:
        name, args = call["name"], call["args"]
        if name == "say":
            r = world.say(me, args.get("to", "all"), args.get("message", ""))
            d.speech.append({"to": args.get("to", "all"), "message": args.get("message", ""), "ok": r.ok, "result": r.text})
            if not r.ok:
                notes.append(f"say failed: {r.text}")
            continue
        if d.action is not None:
            notes.append(f"ignored extra action {name}() - one action per round")
            continue
        d.action = call
        r = world.apply(me, name, args)
        d.action_ok, d.action_result = r.ok, r.text
    if d.action is None:
        notes.append("only spoke; no world action taken")
        d.action_result = "no action"
    else:
        me.remember(world.round, f"you tried {d.action['name']} -> {'OK: ' if d.action_ok else 'FAILED: '}{d.action_result}")
    return d
