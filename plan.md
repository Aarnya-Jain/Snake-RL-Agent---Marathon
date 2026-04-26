# Plan: The Algorithm Marathon — 5 Snakes, 5 Algorithms

## The Idea

Run **5 different snake agents simultaneously**, each using a **different algorithm**, displayed side-by-side on one screen. After N games, compare their performance metrics — score, survival time, improvement rate — to see which algorithm "won the marathon."

This turns a standard DQN snake demo into a compelling **comparative AI study**, which is directly aligned with the lab syllabus's theme of evaluating and comparing multiple AI/ML approaches.

---

## Is This Feasible?

**Yes — and it's a strong project.**

Here's why:

| Concern | Reality |
|---|---|
| "5 games at once is complex" | Each game runs its own independent `SnakeAI` instance — they don't interact |
| "Rendering 5 windows is messy" | Use one Pygame window split into a 2x3 grid of sub-surfaces (`pygame.Surface`) |
| "Different algorithms need different code" | Each agent is a self-contained class — easily swappable |
| "Training 5 networks at once is slow" | Most of the 5 algorithms are **not** neural networks — they're lightweight |
| "Hard to compare fairly" | All agents see the same type of game state and same reward structure |

The most complex part is the **multi-panel Pygame rendering**. The algorithm logic itself is simpler than the existing DQN.

---

## The 5 Algorithms — One for Each Snake

These are chosen to span the syllabus and show a natural progression from "dumb" to "smart."

### Snake 1 — Random Agent (Baseline)
**Algorithm:** Pure random action selection
**Syllabus match:** Unit 1 (Python basics), baseline for comparison

At each step, randomly pick `[straight, right, left]`. No learning, no strategy.
- **Purpose:** Sets the floor for comparison. Everything else should beat this.
- **Implementation:** 3 lines. `random.choice([[1,0,0],[0,1,0],[0,0,1]])`

---

### Snake 2 — BFS / Greedy Search Agent
**Algorithm:** Breadth-First Search or Greedy Best-First
**Syllabus match:** Unit 2 — Uninformed Search (BFS), Informed Search (Greedy)

At each step, find the **shortest path to the food** using BFS on the grid. If a path exists, follow it. If blocked, try to survive by following the tail.

- **Purpose:** Shows that classical AI search can solve structured problems well
- **No learning required** — pure graph search at each frame
- **Interesting property:** BFS is guaranteed to find the shortest path if one exists
- **Known limitation:** Ignores future consequences of its path (can cut itself off)

```
Grid = Snake game grid
BFS(start=head, goal=food) → path → first step direction
```

---

### Snake 3 — A* Search Agent
**Algorithm:** A* (A-Star) with Manhattan Distance heuristic
**Syllabus match:** Unit 2 — Informed Search, A*

Improves on BFS by using a heuristic to prioritise moves that head toward food.

- **Heuristic:** `h(node) = |node.x - food.x| + |node.y - food.y|` (Manhattan distance)
- **F-score:** `f(n) = g(n) + h(n)` where `g` = steps so far, `h` = estimated steps to food
- **Purpose:** Demonstrates informed search — smarter path planning than BFS
- **Often scores higher than BFS** because it finds paths faster

---

### Snake 4 — Hamiltonian Cycle Agent
**Algorithm:** Hamiltonian Cycle (deterministic)
**Syllabus match:** Unit 2 — Problem solving agents, graph traversal

Pre-computes a path that visits **every cell** on the grid exactly once, then loops. The snake always follows this fixed cycle, so it will **never die** — but it's slow.

- **Purpose:** Shows a mathematically optimal survival strategy (if implemented correctly)
- **Trade-off:** Extremely safe but very low food collection efficiency
- **Makes a great visual** because the snake traces a perfect pattern

> NOTE: A full Hamiltonian cycle on large grids is complex. A simplified version can use a **checkerboard-pattern shortcut** — snake follows a zig-zag path that covers most of the board.

---

### Snake 5 — Deep Q-Learning Agent (Existing)
**Algorithm:** Deep Q-Network (DQN)
**Syllabus match:** Unit 3 — ANN, Unit 4 — Deep Neural Network

This is the existing agent from the project. Keep it as-is (or optionally swap its PyTorch model for the NumPy version from `optimise.md`).

- **Purpose:** Shows that neural networks can learn behaviour that rivals or beats handcrafted algorithms
- **The star of the show** — the only agent that actually *learns* and improves over time

---

## Architecture Design

### Rendering: Split-Screen Pygame

Use one 1280×720 window divided into 6 panels (2 rows × 3 columns):

```
┌─────────┬─────────┬─────────┐
│ Random  │   BFS   │   A*    │
│ Snake 1 │ Snake 2 │ Snake 3 │
├─────────┼─────────┴─────────┤
│ Hamilt. │     DQN Agent     │
│ Snake 4 │      Snake 5      │
└─────────┴───────────────────┘
            ↓
     [Stats panel below]
  Scores | Record | Games | Trend
```

Each panel is a `pygame.Surface` rendered independently, then `blit`-ed onto the master surface.

### Per-Frame Game Loop

```python
while True:
    for i, (agent, game) in enumerate(zip(agents, games)):
        state = agent.get_state(game)
        action = agent.get_action(state)
        reward, done, score = game.play_step(action)
        agent.update(state, action, reward, done)
        if done:
            game.reset()
            stats[i].record(score)

    render_all_panels(games, stats)
    clock.tick(SPEED)
```

---

## Metrics to Compare (The Marathon Scoreboard)

At the end of each game, record and display:

| Metric | Description |
|---|---|
| `Current Score` | Food eaten this game |
| `Record` | Highest score ever achieved |
| `Mean Score` | Average over all games |
| `Games Played` | Total games run |
| `Survival Rate` | Avg frames survived per game |
| `Improvement` | (DQN only) Score trend over last 10 games |

Display these in a live stats strip below the game panels, updated every game.

---

## What Else Can Be Added

### 1. Live Leaderboard
A ranked list of agents updated every N games — creates a "race" feeling.

### 2. Speed Control
Add keyboard shortcuts to speed up/slow down all snakes simultaneously (change `SPEED`).

### 3. Hill Climbing Agent (Snake 6?)
> **Syllabus match:** Unit 2 — Hill Climbing

A simple agent that always moves in the direction that **minimises Manhattan distance to food**, without lookahead. It gets stuck in local optima (loops) but is interesting to observe.

### 4. Decision Tree Agent
> **Syllabus match:** Unit 3 — Decision Trees

Train a `sklearn` Decision Tree on recorded (state, best_action) pairs from the BFS agent (supervised learning on BFS demonstrations). Compare it to the online-learned DQN.

This is called **imitation learning** and is a great syllabus tie-in.

### 5. Export Performance Data to CSV / Pandas
> **Syllabus match:** Unit 1 — Pandas, Data Analysis

Save `[game_num, agent, score, survival]` rows at the end and analyse with Pandas:
```python
df = pd.DataFrame(all_records)
df.groupby('agent')['score'].describe()
```
Produce a `results.csv` and analysis notebook.

### 6. Final Report Graph
Plot all 5 agents' score histories on one Matplotlib graph at the end of the marathon. Clear visual proof of which algorithm "won" and which ones converged.

---

## Implementation Roadmap

| Phase | Task | Effort |
|---|---|---|
| Phase 1 | Adapt `game.py` to support sub-surface rendering | Medium |
| Phase 2 | Write `BaseAgent` abstract class all agents inherit from | Low |
| Phase 3 | Implement `RandomAgent` | Trivial |
| Phase 4 | Implement `BFSAgent` (grid BFS) | Medium |
| Phase 5 | Implement `AStarAgent` (grid A*) | Medium |
| Phase 6 | Implement `HamiltonianAgent` (zig-zag approximation) | Medium |
| Phase 7 | Wrap existing `Agent` (DQN) as `DQNAgent` | Low |
| Phase 8 | Build the multi-panel `marathon.py` runner | High |
| Phase 9 | Add stats panel + leaderboard rendering | Medium |
| Phase 10 | Add CSV export + Matplotlib final report | Low |

**Total estimated effort:** 3–5 focused sessions (6–10 hours)

---

## Why This Will Impress the Lab Examiner

1. **Covers 3 out of 4 syllabus units in one project** (Python, Search, ML/ANN)
2. **Comparative analysis is explicitly listed in Unit 4** of the syllabus (CNN architectures comparison — same spirit)
3. **Visual and interactive** — live split-screen is far more compelling than a static output
4. **Demonstrates algorithmic thinking** — not just running a library, but implementing search from scratch
5. **Scalable** — you can add or swap agents, which shows architectural thinking
6. **Honest narrative** — DQN may not win immediately, which is itself an interesting finding (classical algos can outperform RL in constrained environments)

---

## Summary

> The marathon idea is not just feasible — it is the right direction for a lab submission that wants to stand out. It naturally demonstrates every concept in the syllabus by letting the algorithms compete, and the visual format of 5 simultaneous snakes makes the comparison concrete and memorable. Start with the BFS and A* agents (pure search, no ML), then add the DQN, and the contrast becomes the project's story.
