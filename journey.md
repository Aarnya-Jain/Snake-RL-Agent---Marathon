# Journey Log — Snake: Learning to Live

> Short log of everything done throughout the project, in order.

---

## Day 0 — 2026-04-08 | Project Audit & Planning

- Received the project from a friend — existing codebase with `game.py`, `agent.py`, `model.py`, `helper.py`
- Did a full audit of all files to understand what each does
- Created `about.md` — complete breakdown of the project, algorithms, data flow, and libraries
- Created `optimise.md` — identified components to reimplement from scratch (NumPy neural net, custom replay memory, MSE loss)
- Created `plan.md` — designed the **Algorithm Marathon** idea: 5 snakes running in parallel using Random, BFS, A*, Hamiltonian, and DQN algorithms with a split-screen Pygame UI and live stats

---

<!-- future entries go below this line -->

## Day 1 — 2026-04-09 | Split-Screen Grid Layout

- Created `marathon.py` — standalone Pygame window (1260 × 700 px)
- Renders 5 equal game panels: 3 on top row, 2 centred on bottom row
- Each panel: 400 × 300 px (20 cols × 15 rows × 20 px/block)
- Each panel has: dark background, subtle grid lines, coloured border, agent name label
- Agent colour scheme (Catppuccin Mocha): Random=Pink, BFS=Blue, A*=Green, Hamiltonian=Yellow, DQN=Mauve
- No game logic connected yet — pure layout skeleton

## Day 2 — 2026-04-13 | Game Logic + All 5 Agents

- Created `snake_env.py` — headless `SnakeGame` class (pure logic, no display/clock)
- Created `agents.py` — all 5 agent classes:
  - `RandomAgent` — random choice of 3 actions
  - `BFSAgent` — BFS pathfinding to food, falls back to safe move if trapped
  - `AStarAgent` — A* with Manhattan distance heuristic
  - `HamiltonianAgent` — pre-computed zig-zag cycle covering every cell
  - `DQNAgent` — loads existing `model/model.pth`, continues online training if not pre-trained
- Rewrote `marathon.py` — full game loop driving all 5 agents simultaneously
  - Rendering matches original `game.py` UI: rounded snake, directional eyes, food with leaf, score badge
  - Stats strip at bottom: score, record, mean, game count per agent
  - `+`/`-` keys to change speed at runtime

## Day 2 (continued) — Bug Fixes

- **Root cause found:** `SnakeGame` started head at `Point(200, 150)` — y=150 is not a grid multiple (150/20=7.5). Caused BFS/A* to expand from an off-grid cell, never reaching food → timeout → reset → appeared as random food respawn
- **Fix:** Head now snapped to grid: `Point((w//2//BLOCK_SIZE)*BLOCK_SIZE, (h//2//BLOCK_SIZE)*BLOCK_SIZE)` = Point(200,140)
- **Hamiltonian cycle fixed:** Previous "closing strip" was adding cells already in path (no-ops). New algorithm: zigzag all cols through rows 1..H-1, then step to row 0 at last col, then sweep row 0 leftward — produces proper 300-cell cycle (verified: all unique, last→first adjacent)
- **A* tiebreaker fixed:** Added monotonic counter to heap tuple so Direction enums are never compared
- **DQN cleaned up:** Uses `model.py` directly (Linear_QNet, QTrainer), no dependency on agent.py

## Day 3 — 2026-04-17 | Stats, CSV Export, and Comparison Plot

- Created `stats.py` — CSV logging and multi-agent Matplotlib comparison plot
  - `init_csv()` — creates `stats.csv` with header on each run
  - `log_game(agent_idx, game_num, score, record, mean)` — appends a row per completed game
  - `save_plot(scores, means)` — dual-panel dark-themed plot (scores + running mean per agent, 5 coloured lines)
- Updated `marathon.py`:
  - Imports and initialises stats on startup
  - Logs every game to `stats.csv` as it finishes
  - Saves `stat.png` automatically when window is closed (X button)

## Day 3 (continued) — Model Analysis & Solo Training Setup

- Identified DQN plateau at ~23 mean score: structural ceiling of 11-bit 1-step state on 300-cell grid
- Saved marathon-trained model as `model/model_marathon.pth`
- Added CSV logging to `agent.py` → writes `agent_training.csv` (game_num, score, record, mean_score) with `flush()` on each game so data is safe even if interrupted
- Plan: train solo model on big grid (640×480) overnight → save as `model/model_solo.pth` → compare both models for PBL presentation

## Day 4 — 2026-04-18 | Solo DQN Training Results (agent.py)

- Trained `agent.py` for ~5 hours on the original 640×480 grid → **575 games completed**
- **Record score: 78** (proves model capability on big grid)
- **Final cumulative mean: 27.55** (dragged down by first 50 random-exploration games)
- **Last-100-game mean: 30.82** — the true trained performance metric

**Training curve (per 50-game windows):**

| Games | Mean Score | Notes |
|---|---|---|
| 1–50 | 0.36 | Pure random exploration (epsilon high) |
| 51–100 | 10.22 | Beginning to learn |
| 101–150 | 33.60 | Rapid improvement |
| 201–250 | 39.54 | **Peak performance** |
| 251–575 | 28–34 | Oscillating plateau (~30 mean) |
| last 50 | 31.56 | Ticking back up at time of exit |

**Key insight:** Model peaked at games 200-250 then oscillated between 28-34 for 300+ games — the structural ceiling of the 11-bit 1-step state representation. Record of 78 is an outlier; the model has high variance (can score high but can't maintain it consistently).

**Comparison summary (for final report):**

| Metric | Solo DQN (640×480) | Marathon DQN (400×300) |
|---|---|---|
| Grid cells | 768 | 300 |
| Games played | 575 | ~1000+ (2 hrs) |
| Record score | **78** | ~35 |
| Last-100 mean | **30.82** | ~23 |
| Training ceiling | ~32–34 | ~23–25 |

**Why solo performs better:** Larger grid means snake stays "short" relative to total space for longer — 1-step danger detection survives further into the game before becoming insufficient.

- Saved solo model as `model/model_solo.pth` for PBL showcase

## Day 5 — 2026-04-18 | Course Outcome (CO) Alignment Analysis

Before proceeding to final reporting, a mapping of project components to the syllabus Course Outcomes was performed to ensure all academic requirements are met.

| Course Outcome | Requirement | Project Evidence |
|---|---|---|
| **CO1** | Understand Python framework for AI tasks | Integrated use of `Pygame` (rendering), `PyTorch` (deep learning), `Matplotlib` (visualization), `Pandas` (data analysis), and `CSV` (logging). |
| **CO2** | Demonstrate use of intelligent searching techniques | Implementation of **BFS** (uninformed), **A*** (informed/heuristic), and **Hamiltonian Cycle** (deterministic/graph traversal) agents. |
| **CO3** | Apply learning algorithms for real-world problems | Implementation and training of a **Deep Q-Network (DQN)** reinforcement learning agent across two different environments (Solo vs Marathon). |
| **CO4** | Performance analysis of AI algorithmic approaches | The **Algorithm Marathon** provides a side-by-side comparative analysis of 5 distinct approaches with 20,000+ total games of logged data. |

**Conclusion:** The project is CO-complete. The focus now shifts to deep data analysis and formal reporting.

## Day 5 (continued) — Data Analysis Strategy

To satisfy the specific syllabus requirements for Pandas, SciPy, and various ML models, a comprehensive Jupyter Notebook (`analysis.ipynb`) is being developed.

**Analysis Scope:**
- **Pandas:** Loading and cleaning `data_agent.csv` and `data_marathon.csv`.
- **SciPy:** Image manipulation of `stat.png` (filtering/edge detection) to demonstrate library proficiency.
- **Supervised Learning:**
    - **Linear Regression:** Predicting score trends over time.
    - **Decision Trees/Random Forest:** Classifying agent types based on performance metrics.
    - **ANN:** Evaluation of the existing DQN architecture.
- **Unsupervised Learning:**
    - **K-Means Clustering:** Grouping games into "performance tiers" (e.g., Early Learning, Mid-Tier, High-Performance).
- **Data Prep:** Train/Test splitting for all evaluative models.


## Day 4 (continued) — Marathon 2-Hour Session Analysis (stats.csv)

Session ran for ~2 hours at default speed. All 5 agents ran simultaneously on 400×300 panels.

**Per-agent summary:**

| Agent | Games | Record | Overall Mean | Last-100 Mean |
|---|---|---|---|---|
| Random | 17,255 | ~1 | ~0.11 | ~0.11 |
| BFS | 1,193 | 94 | **50.77** | 50.19 |
| A* | 1,201 | 92 | **50.83** | 51.43 |
| Hamiltonian | 38 | **297** | **297.0** | 297.0 |
| DQN | 2,624 | 63 | 23.17 | 23.25 |

**Key observations:**

**Hamiltonian — mean score 297 on a 300-cell grid.** Played 38 complete games, scoring 297 every single game — essentially a perfect run every time. The 300-cell Hamiltonian cycle guarantees visiting every cell, so it consumes all food before the board fills up. The only reason it's not 299 (max possible starting length 3) is minor — it's functionally perfect. Slowest in games completed because it almost never dies.

**BFS and A* — nearly identical at ~50.8 mean.** Both consistently score 47-55 per window across all 1,200 games with no degradation. They are deterministic path-finders — no learning, no variance, no plateau. A* is marginally better (51.43 vs 50.19 last-100) due to heuristic guidance reducing unnecessary detours. Record of 94 for BFS and 92 for A* show both can handle long snakes.

**DQN — plateau at 23.17 from game ~150 onward.** Learned fast initially (0.34 → 20.48 in first 150 games), then locked into oscillation between 21-27 for the remaining 2,400+ games. Matches the structural ceiling analysis: 11-bit 1-step state insufficient for long-snake avoidance on 300-cell grid.

**Random — 17,255 games, mean ~0.11.** Dies on average within ~10 steps on the 20×15 grid. Plays orders of magnitude more games than any other agent because each game lasts seconds. Useful as a baseline showing pure noise.

**Algorithm ranking by mean score (marathon, 400×300 grid):**

```
1. Hamiltonian  — 297.0  (perfect, guaranteed)
2. A*           —  50.83 (optimal path, slight heuristic edge)
3. BFS          —  50.77 (optimal path, no heuristic)
4. DQN          —  23.17 (learned, plateaued)
5. Random       —   0.11 (baseline noise)
```

**Insight for report:** The gap between classical algorithms (BFS/A*: ~51) and DQN (~23) on the small grid is substantial. On the big grid, DQN closes some gap (last-100 mean: 30.82) but classical algorithms would perform similarly. The Hamiltonian result shows that knowing the environment perfectly (full-grid cycle) trivially dominates learned approaches.


## Day 6 — 2026-04-20 | DQN Demo Bug — Diagnosis & Fix

**Bug:** Marathon DQN agent spinning in place when toggled to demo mode, despite working fine 2 days earlier.

**Root cause:** `agents.py` DQN saved the model **every 20 games blindly** (`if self.n_games % 20 == 0`). When the marathon was re-run in training mode, the save at game 20 or 40 — during the ε-greedy exploration phase — **overwrote the good trained model with random-exploration weights**. Solo `agent.py` never had this issue because it only saves on `if score > record`.

**Fix applied:**
- Changed marathon DQN save logic from "every 20 games" → **"only when a new record is beaten"** (matching solo `agent.py` behaviour)
- Added `_best_score` tracking to `_init_train()` and `_init_demo()`
- `on_game_over()` now accepts `score` parameter; `marathon.py` passes it in
- Added `torch.no_grad()` to DQN inference (good practice, prevents unnecessary gradient tracking)
- Updated all non-DQN agents' `on_game_over` to accept `**kw` for forward compatibility

**Recovery:** Need to retrain the marathon DQN or copy `model_agent_solo.pth` → `model_marathon.pth` as a starting point.


