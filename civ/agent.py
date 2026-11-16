"""Turns world state into a prompt, asks the LLM for a decision, and parses the tool calls."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .llm import LLM, CallStats
from .world import PERSONAS, World, Villager

# NOTE: this system prompt and the tool list are identical for every villager and every turn.
# Ollama/llama.cpp reuses the KV cache for a shared prompt prefix, so keeping all the static
# text up front (and all the per-turn state in the user message) makes prompt evaluation
# nearly free after the first call.
SYSTEM_PROMPT = """You are a villager in a small village simulation. Each round you choose ONE action by calling a function, and you may also say something. Always include a short `reason`.

Rules:
- Everyone eats 1 food at the end of every round. With no food you lose 2 hp; at 0 hp you die.
- Gathering takes from a shared pool that regrows slowly. Farmers gather 3 food (1 wood); lumberjacks gather 3 wood (1 food); builders gather 1 of each.
- Building a house costs wood (6 for most, 3 for builders). A house is permanent.
- The village succeeds when every living villager has a house. Wealth should be shared fairly: use `give` to trade or help others, and `say` to coordinate.
- You can't build without enough wood. Don't give away what you need to survive.
- The pool is SHARED and finite. Taking more than you need leaves others to starve, so stop gathering a resource once you hold plenty of it.

Decide in this priority order, using the "Situation" notes:
1. If you are about to run out of food, get food (gather it, or `say` to ask someone rich in food).
2. If another villager is starving and you hold 4+ food, `give` them 2 food (never more than you hold).
3. If you can afford your house, `build_house`.
4. Otherwise gather your strongest resource that you still need.
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "gather", "description": "Gather a resource from the shared pool.",
        "parameters": {"type": "object", "properties": {
            "resource": {"type": "string", "enum": ["food", "wood"]},
            "reason": {"type": "string", "description": "max 10 words"}},
            "required": ["resource", "reason"]}}},
    {"type": "function", "function": {
        "name": "build_house", "description": "Build your house using your wood.",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string", "description": "max 10 words"}},
            "required": ["reason"]}}},
    {"type": "function", "function": {
        "name": "give", "description": "Give some of your food or wood to another villager.",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string", "description": "villager name"},
            "resource": {"type": "string", "enum": ["food", "wood"]},
            "amount": {"type": "integer"},
            "reason": {"type": "string", "description": "max 10 words"}},
            "required": ["to", "resource", "amount", "reason"]}}},
    {"type": "function", "function": {
        "name": "say", "description": "Say something to one villager (by name) or to everyone ('all'). Does not use your action.",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string", "description": "villager name or 'all'"},
            "message": {"type": "string"}},
            "required": ["to", "message"]}}},
    {"type": "function", "function": {
        "name": "wait", "description": "Do nothing this round.",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string", "description": "max 10 words"}},
            "required": ["reason"]}}},
]


# Alternative to native tool calling: one compact, grammar-constrained JSON object per turn.
# Fewer prompt tokens (no tool definitions) and fewer generated tokens (no XML wrapper), and the
# reply is guaranteed to be well-formed.
_RES = {"type": "string", "enum": ["food", "wood"]}
_SPEECH = {"say_to": {"type": "string"}, "say": {"type": "string"}}
_REASON = {"reason": {"type": "string"}}


def _variant(action: str, **props) -> dict:
    """One allowed reply shape. Property order = generation order; required fields are enforced."""
    properties = {"action": {"type": "string", "enum": [action]}, **props, **_SPEECH, **_REASON}
    return {"type": "object", "properties": properties,
            "required": ["action", *props, "reason"]}


JSON_SCHEMA = {"anyOf": [
    _variant("gather", resource=_RES),
    _variant("give", to={"type": "string"}, resource=_RES, amount={"type": "integer"}),
    _variant("build_house"),
    _variant("wait"),
]}

JSON_ADDENDUM = """
Reply with one JSON object: {"action": "gather|build_house|give|wait", "resource": "food|wood" (gather/give), "to": "<villager>" and "amount": <int> (give only), "say_to": "<villager or all>" and "say": "<text>" (optional), "reason": "max 10 words"}. Omit fields you don't need."""


def situation(world: World, me: Villager) -> list[str]:
    """Facts derived from the state, so the model doesn't have to do arithmetic."""
    others = [v for v in world.living() if v is not me]
    notes = [f"Your food lasts {me.food} more round(s)."]
    if not me.house:
        if me.wood >= me.house_cost:
            notes.append("You have enough wood to build your house RIGHT NOW.")
        else:
            notes.append(f"You need {me.house_cost - me.wood} more wood to build your house.")
    if world.pool["food"] == 0:
        notes.append("The shared pool has NO food left: do not gather food; trade with others instead.")
    if world.pool["wood"] == 0:
        notes.append("The shared pool has NO wood left: do not gather wood; trade with others instead.")
    if me.food >= 6:
        notes.append(f"You already hold {me.food} food: gathering more just takes it from others.")
    rich_food = [f"{v.name} ({v.food})" for v in others if v.food >= 5]
    hungry = [f"{v.name} ({v.food})" for v in others if v.food <= 1]
    if me.food <= 2 and rich_food:
        notes.append(f"Villagers with plenty of food: {', '.join(rich_food)} - you could ask them or take your strong resource and trade.")
    if me.food >= 4 and hungry:
        notes.append(f"STARVING villagers (you should give them food): {', '.join(hungry)}.")
    if not me.house and me.wood < me.house_cost:
        rich_wood = [f"{v.name} ({v.wood})" for v in others if v.wood >= 4 and not v.house]
        if rich_wood:
            notes.append(f"Villagers holding extra wood: {', '.join(rich_wood)}.")
    return notes


def build_prompt(world: World, me: Villager, max_rounds: int) -> str:
    others = "\n".join(
        f"- {v.name} ({v.persona}): food {v.food}, wood {v.wood}, hp {v.hp}, "
        f"{'HAS house' if v.house else f'no house (needs {v.house_cost} wood)'}"
        for v in world.living() if v is not me
    ) or "- (nobody else is alive)"
    memory = "\n".join(f"- {m}" for m in me.memory) or "- (nothing yet)"
    house = "you HAVE a house" if me.house else f"no house yet (costs {me.house_cost} wood)"
    notes = "\n".join(f"- {n}" for n in situation(world, me))
    return (
        f"You are {me.name}, a {me.persona}. Round {world.round} of {max_rounds}.\n"
        f"You: food {me.food}, wood {me.wood}, hp {me.hp}; {house}.\n"
        f"Shared pool: food {world.pool['food']}, wood {world.pool['wood']}.\n"
        f"Other villagers:\n{others}\n"
        f"Situation:\n{notes}\n"
        f"Recent events:\n{memory}\n"
        f"Choose your action."
    )


@dataclass
class Decision:
    """Everything that happened in one villager turn - this is what gets traced."""
    villager: str
    prompt: str
    raw_content: str
    tool_calls: list[dict]
    stats: CallStats
    action: dict | None = None            # the world action chosen (None = wait / invalid)
    action_result: str = ""
    action_ok: bool = False
    speech: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)   # parsing problems, ignored extra calls, etc.


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
    action = obj.get("action", "wait")
    args = {k: obj[k] for k in ("resource", "to", "amount", "reason") if k in obj}
    calls.append({"name": "build_house" if action == "build_house" else action, "args": args})
    return calls


def take_turn(world: World, me: Villager, llm: LLM, max_rounds: int, mode: str = "json") -> Decision:
    prompt = build_prompt(world, me, max_rounds)
    notes: list[str] = []
    if mode == "json":
        messages = [{"role": "system", "content": SYSTEM_PROMPT + JSON_ADDENDUM}, {"role": "user", "content": prompt}]
        resp, stats = llm.chat(messages, fmt=JSON_SCHEMA)
        calls = parse_json_reply(resp.message.content or "", notes)
    else:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        resp, stats = llm.chat(messages, tools=TOOLS)
        calls = [{"name": c.function.name, "args": dict(c.function.arguments or {})}
                 for c in (resp.message.tool_calls or [])]
    d = Decision(me.name, prompt, resp.message.content or "", calls, stats, notes=notes)

    if not calls:
        d.notes.append("model made no tool call - treated as wait")
        d.action_result = "no action"
        return d

    for call in calls:
        name, args = call["name"], call["args"]
        try:
            if name == "say":
                r = world.say(me, args.get("to", "all"), args.get("message", ""))
                d.speech.append({"to": args.get("to", "all"), "message": args.get("message", ""), "ok": r.ok})
                if not r.ok:
                    d.notes.append(f"say failed: {r.text}")
                continue
            if d.action is not None:
                d.notes.append(f"ignored extra action {name}() - one action per round")
                continue
            d.action = call
            if name == "gather":
                r = world.gather(me, str(args.get("resource", "")))
            elif name == "build_house":
                r = world.build_house(me)
            elif name == "give":
                r = world.give(me, args.get("to", ""), str(args.get("resource", "")), args.get("amount"))
            elif name == "wait":
                r = type("R", (), {"ok": True, "text": "waited"})()
            else:
                r = type("R", (), {"ok": False, "text": f"unknown function {name!r}"})()
            d.action_ok, d.action_result = r.ok, r.text
        except Exception as e:  # a malformed tool call must never crash the simulation
            d.action_ok, d.action_result = False, f"error executing {name}: {e}"
    if d.action is None:
        d.notes.append("only spoke; no world action taken")
        d.action_result = d.action_result or "no action"

    # Feed the outcome back so the villager can learn from failed attempts.
    if d.action is not None:
        verb = d.action["name"]
        me.remember(world.round, f"you tried {verb} -> {'OK: ' if d.action_ok else 'FAILED: '}{d.action_result}")
    return d
