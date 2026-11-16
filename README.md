# ai-civilization

A village simulation where each villager is a local LLM (via Ollama) choosing one action per round.
Goal: everyone alive, everyone housed, and the Town Hall built - with wealth shared fairly (Gini).

    ./venv/bin/python main.py                  # 8 villagers, 30 rounds, qwen3.5:4b
    ./venv/bin/python main.py -r 36 -n 6 --seed 3
    ./venv/bin/python main.py -v               # also print every prompt + raw model reply
    ./venv/bin/python main.py --mode tools     # native tool-calling instead of JSON mode (slower)
    ./venv/bin/python main.py inspect runs/<run>/trace.jsonl --round 3 --villager Ada
    ./venv/bin/python -m unittest discover tests   # rules + turn loop, no GPU needed

Each run writes `runs/<timestamp>/`: `log.txt` (the readable story) and `trace.jsonl` (per turn: full prompt,
raw reply, parsed action, result, state, timing/token stats; plus every social/economic event at the end).

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
