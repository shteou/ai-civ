from __future__ import annotations

import time

from .agent import SYSTEM_PROMPT, take_turn
from .llm import LLM
from .report import Reporter
from .world import World


def run(world: World, llm: LLM, rep: Reporter, max_rounds: int, config: dict, mode: str = "json") -> dict:
    rep.run_start(world, config, SYSTEM_PROMPT)
    warmup_s = llm.warmup()
    rep.out(rep.c(f"model loaded in {warmup_s:.1f}s (one-off)", 2))

    calls: list[dict] = []
    t_start = time.perf_counter()
    reason = f"reached the {max_rounds}-round limit"
    while world.round < max_rounds:
        world.round += 1
        rep.round_start(world)
        t_round, llm_s = time.perf_counter(), 0.0
        order = world.living()
        world.rng.shuffle(order)          # no villager gets a permanent first-mover advantage
        for v in order:
            d = take_turn(world, v, llm, max_rounds, mode)
            llm_s += d.stats.wall_s
            calls.append({**d.stats.to_dict(), "ok": d.action_ok})
            rep.turn(world, world.villagers.index(v), d)
        notes = world.end_round()
        rep.round_end(world, notes, {"round_s": time.perf_counter() - t_round, "llm_s": llm_s})
        if world.done():
            reason = "everyone has a house" if world.living() else "everyone died"
            break
    return rep.summary(world, calls, time.perf_counter() - t_start, warmup_s, reason)
