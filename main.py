"""Little village simulation driven by local Ollama models.

    python main.py                       # run with defaults
    python main.py -r 15 -n 4 --seed 1   # 15 rounds, 4 villagers, reproducible-ish
    python main.py -v                    # also print every prompt and raw model reply
    python main.py inspect runs/<run>/trace.jsonl --round 3 --villager Ada   # drill into a turn
    python main.py tui                   # full-screen TUI: map, stats, responses, memory (live run)
    python main.py tui --replay runs/<run>/trace.jsonl   # replay a saved run in the TUI
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def build_run(a, echo: bool = True):
    """Create world, LLM and reporter from CLI args."""
    from civ.llm import LLM
    from civ.report import Reporter
    from civ.world import World

    seed = a.seed if a.seed is not None else int(datetime.now().timestamp()) % 100000
    out = Path(a.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    llm = LLM(a.model, num_ctx=a.num_ctx, num_predict=a.num_predict, temperature=a.temperature, seed=seed)
    if not llm.supports_tools:
        sys.exit(f"{a.model} does not support tool calling; pick another model (see `ollama list`)")
    rep = Reporter(out, color=False if a.no_color else None, verbose=a.verbose, echo=echo)
    config = {"model": a.model, "seed": seed, "rounds": a.rounds, "villagers": a.villagers,
              "options": llm.options, "think": llm.think, "mode": a.mode}
    return World(a.villagers, seed), llm, rep, config


def cmd_run(a) -> None:
    from civ.sim import run

    world, llm, rep, config = build_run(a)
    try:
        run(world, llm, rep, a.rounds, config, a.mode)
    except KeyboardInterrupt:
        rep.out("\ninterrupted")
    finally:
        rep.close()


def cmd_tui(a) -> None:
    from civ.tui import CivApp, load_trace

    if a.replay:
        CivApp(replay=load_trace(a.replay), speed=a.speed).run()
        return
    world, llm, rep, config = build_run(a, echo=False)
    CivApp(live=(world, llm, rep, a.rounds, config, a.mode)).run()
    print(f"log: {rep.out_dir}/log.txt   trace: {rep.out_dir}/trace.jsonl")


def cmd_inspect(a) -> None:
    """Pretty-print one or more turns from a trace: prompt, raw reply, parsed action, timings."""
    for line in open(a.trace):
        e = json.loads(line)
        if e["event"] == "run_start" and a.system:
            print("SYSTEM PROMPT:\n" + e["system_prompt"])
        if e["event"] != "turn":
            continue
        if a.round is not None and e["round"] != a.round:
            continue
        if a.villager and e["villager"].lower() != a.villager.lower():
            continue
        print(f"\n===== round {e['round']} · {e['villager']} ({e['persona']}) =====")
        print("--- prompt ---\n" + e["prompt"])
        print(f"--- raw reply ---\ncontent: {e['raw_content']!r}\ntool_calls: {json.dumps(e['tool_calls'])}")
        print(f"--- outcome ---\n{'OK' if e['action_ok'] else 'FAILED/none'}: {e['action_result']}")
        for n in e["notes"]:
            print("note:", n)
        print("--- stats ---\n" + json.dumps(e["stats"]))
        print("--- villager after ---\n" + json.dumps(e["state_after"]))


def run_parser(**kw) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, **kw)
    p.add_argument("-m", "--model", default="qwen3.5:4b")
    p.add_argument("-r", "--rounds", type=int, default=30)
    p.add_argument("-n", "--villagers", type=int, default=8)
    p.add_argument("--mode", choices=["json", "tools"], default="json",
                   help="json = schema-constrained reply (faster); tools = native tool calling")
    p.add_argument("--seed", type=int)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--num-ctx", type=int, default=4096)
    p.add_argument("--num-predict", type=int, default=160, help="max tokens per decision")
    p.add_argument("-o", "--out", default="runs")
    p.add_argument("-v", "--verbose", action="store_true", help="print prompts and raw replies")
    p.add_argument("--no-color", action="store_true")
    return p


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "inspect":
        p = argparse.ArgumentParser(prog="main.py inspect")
        p.add_argument("trace")
        p.add_argument("--round", type=int)
        p.add_argument("--villager")
        p.add_argument("--system", action="store_true", help="also print the system prompt")
        return cmd_inspect(p.parse_args(sys.argv[2:]))
    if cmd == "tui":
        p = run_parser(prog="main.py tui", description="Full-screen TUI: live run, or --replay a saved trace.")
        p.add_argument("--replay", metavar="TRACE", help="replay runs/<run>/trace.jsonl (no Ollama needed)")
        p.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier")
        return cmd_tui(p.parse_args(sys.argv[2:]))
    cmd_run(run_parser(description=__doc__).parse_args())


if __name__ == "__main__":
    main()
