"""Little village simulation driven by local Ollama models.

    python main.py                       # run with defaults
    python main.py -r 15 -n 4 --seed 1   # 15 rounds, 4 villagers, reproducible-ish
    python main.py -v                    # also print every prompt and raw model reply
    python main.py inspect runs/<run>/trace.jsonl --round 3 --villager Ada   # drill into a turn
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def cmd_run(a) -> None:
    from civ.llm import LLM
    from civ.report import Reporter
    from civ.sim import run
    from civ.world import World

    seed = a.seed if a.seed is not None else int(datetime.now().timestamp()) % 100000
    out = Path(a.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    llm = LLM(a.model, num_ctx=a.num_ctx, num_predict=a.num_predict, temperature=a.temperature, seed=seed)
    if not llm.supports_tools:
        sys.exit(f"{a.model} does not support tool calling; pick another model (see `ollama list`)")
    rep = Reporter(out, color=False if a.no_color else None, verbose=a.verbose)
    config = {"model": a.model, "seed": seed, "rounds": a.rounds, "villagers": a.villagers,
              "options": llm.options, "think": llm.think, "mode": a.mode}
    try:
        run(World(a.villagers, seed), llm, rep, a.rounds, config, a.mode)
    except KeyboardInterrupt:
        rep.out("\ninterrupted")
    finally:
        rep.close()


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


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        p = argparse.ArgumentParser(prog="main.py inspect")
        p.add_argument("trace")
        p.add_argument("--round", type=int)
        p.add_argument("--villager")
        p.add_argument("--system", action="store_true", help="also print the system prompt")
        return cmd_inspect(p.parse_args(sys.argv[2:]))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model", default="qwen3.5:4b")
    p.add_argument("-r", "--rounds", type=int, default=20)
    p.add_argument("-n", "--villagers", type=int, default=5)
    p.add_argument("--mode", choices=["json", "tools"], default="json",
                   help="json = schema-constrained reply (faster); tools = native tool calling")
    p.add_argument("--seed", type=int)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--num-ctx", type=int, default=4096)
    p.add_argument("--num-predict", type=int, default=160, help="max tokens per decision")
    p.add_argument("-o", "--out", default="runs")
    p.add_argument("-v", "--verbose", action="store_true", help="print prompts and raw replies")
    p.add_argument("--no-color", action="store_true")
    cmd_run(p.parse_args())


if __name__ == "__main__":
    main()
