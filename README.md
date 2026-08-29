# lob-execution-hma

A hierarchical multi-agent RL system for limit-order-book trade execution, tested against real
BTCUSDT order-book data: a local-LLM macro analyst (L1), an RL strategist (L2, SAC), and an RL
executioner (L3, RecurrentPPO), each running on its own decision cadence above a custom
Gymnasium execution environment with a queue-position-aware fill simulator.

**Project complete.** Start with **[`docs/reports/PROJECT_FINAL_REPORT.md`](docs/reports/PROJECT_FINAL_REPORT.md)**
for the full writeup: what was built, headline results, engineering findings, negative results
with evidence, and scope/limitations. `docs/TRACK_STATUS.md` carries the full chronological
working log behind it; `docs/reports/` holds every per-track detail report.
