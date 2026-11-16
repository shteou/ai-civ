# ai-civilization

A village simulation where each villager is a local LLM (via Ollama) choosing one action per round.
Goal: everyone alive, everyone housed, and the Town Hall built - with wealth shared fairly (Gini).

    uv run main.py                  # 8 villagers, 30 rounds, qwen3.5:4b
    uv run main.py -r 36 -n 6 --seed 3
    uv run main.py -v               # also print every prompt + raw model reply
    uv run main.py --mode tools     # native tool-calling instead of JSON mode (slower)
    uv run main.py inspect runs/<run>/trace.jsonl --round 3 --villager Ada
    uv run main.py tui                         # full-screen TUI for a live run (same options as above)
    uv run main.py tui --replay runs/<run>/trace.jsonl   # replay any saved run in the TUI (no Ollama needed)
    uv run python -m unittest discover tests   # rules, turn loop and TUI, no GPU needed

Each run writes `runs/<timestamp>/`: `log.txt` (the readable story) and `trace.jsonl` (per turn: full prompt,
raw reply, parsed action, result, state, timing/token stats; plus every social/economic event at the end).

## The TUI
`main.py tui` shows the world map (Hills above Forest; Forest - Square - Fields in a row; the Lake under Square and
Fields), every villager's stats, the town hall/house progress, a per-decision latency sparkline, and a live log.
Pick a villager in the table (or press `f` to follow whoever just acted) and use the tabs:
`1` Response (action, result, stated reason, raw model reply, recent actions) · `2` Memory (what they remembered and
the derived hints they were given) · `3` Prompt (the exact user prompt) · `4` System prompt.
Keys: `space` pause/resume (live pauses the simulation between turns) · `n` step one turn (replay) · `+`/`-` replay
speed · `q` quit. A live TUI run still writes `log.txt` and `trace.jsonl`, and replays use the same code path as live
runs, so any trace can be scrubbed through later.

## The world
Map: **Square** (Town Hall) - **Fields**, **Lake** (food) - **Forest** (wood) - **Hills** (stone, reached via Forest).
Walking costs an action. Skilled villagers (farmer, fisher, lumberjack, miner) gather 4/action at their home place,
everyone else 2; builders build cheaper houses. Each place has its own finite, slowly regrowing pool.
Everyone eats 1 food/round. The last 3 rounds of every 12 are **winter**: the Fields freeze, nothing regrows.

Villagers only hear speech and see inventories of those in the same place. Actions: gather, move, build_house,
give (gift), offer/accept (binding swap), steal (50%, witnesses, public reputation hit), contribute (Town Hall), wait.
Each villager also has a temperament (generous, greedy, sociable, cautious, opportunist, honest).

The post-run **What emerged** report summarises trades, gift network, thefts, Town Hall contributors, reputation,
what each persona actually did, and a timeline. All rules live in `civ/world.py` (pure, unit-tested).

## Layout
`civ/world.py` rules · `civ/agent.py` action spec, prompt, derived "Situation" hints, parsing · `civ/llm.py`
instrumented Ollama client · `civ/report.py` output/trace/summary · `civ/sim.py` loop.

## Speed notes (RTX 3070 8GB, qwen3.5:4b Q4)
- `think=False` is the big one: thinking models spend 10-50x the tokens.
- One model for all villagers, warmed up front, `keep_alive` 30m: no reloads.
- JSON-schema-constrained replies beat native tool calls (fewer prompt and generated tokens). `maxLength` on
  `reason`/`say` in the schema bounds generation, which is what keeps a decision at ~0.7-1s.
- Generation (~90 tok/s) is the floor; concurrent requests gave no speedup (Ollama serves one at a time here).
