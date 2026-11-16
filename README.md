# ai-civilization

A tiny village simulation where each villager is a local LLM (via Ollama) choosing one action per round.
Goal: everyone alive, everyone housed, wealth shared fairly (reported as a Gini coefficient).

    ./venv/bin/python main.py                  # 5 villagers, 20 rounds, qwen3.5:4b
    ./venv/bin/python main.py -r 30 -n 6 --seed 3
    ./venv/bin/python main.py -v               # also print every prompt + raw model reply
    ./venv/bin/python main.py --mode tools     # native tool-calling instead of JSON mode (slower)
    ./venv/bin/python main.py inspect runs/<run>/trace.jsonl --round 3 --villager Ada
    ./venv/bin/python -m unittest discover tests   # rules + turn loop, no GPU needed

Each run writes `runs/<timestamp>/`: `log.txt` (the readable story) and `trace.jsonl`
(per turn: full prompt, raw reply, parsed action, result, state, and timing/token stats).

Layout: `civ/world.py` rules (pure) · `civ/agent.py` prompt + parsing · `civ/llm.py` instrumented
Ollama client · `civ/report.py` output/trace/summary · `civ/sim.py` loop.

## Speed notes (RTX 3070 8GB, qwen3.5:4b Q4)
- `think=False` is the big one: thinking models spend 10-50x the tokens (~2s+ -> ~0.3s per decision).
- One model for all villagers, warmed up front, `keep_alive` 30m: no reloads (cold load is ~12s).
- JSON-schema-constrained replies beat native tool calls: ~25 vs ~46 generated tokens and ~650 vs ~1300
  prompt tokens per turn -> ~0.42s vs ~0.62s per decision.
- Generation (~100 tok/s) is now the floor. Concurrent requests gave no speedup here (Ollama serves one at a time).
