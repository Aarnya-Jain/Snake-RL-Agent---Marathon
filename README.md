# 🐍 Snake: Learning to Live — Algorithm Marathon

> Five snake agents. Five algorithms. One screen. Who wins?

A reinforcement learning + classical AI project that runs **5 snake agents simultaneously**, each using a different algorithm, comparing their performance in real time.

![Marathon Screenshot](https://github.com/user-attachments/assets/35f8148d-55f4-4fae-8344-260ebd363e5d)

---

## 🧠 Algorithms Implemented

| Agent | Algorithm | Type | Mean Score |
|---|---|---|---|
| 🎲 Random | Random action | Baseline | ~0.11 |
| 🔵 BFS | Breadth-First Search | Uninformed Search | ~50.8 |
| 🟢 A\* | A\* with Manhattan heuristic | Informed Search | ~50.8 |
| 🟡 Hamiltonian | Hamiltonian Cycle | Graph Traversal | **297 / 300** |
| 🟣 DQN | Deep Q-Network (RL) | Machine Learning | ~23–31 |

---

## 🗂️ Project Structure

```
snake-learning-to-live/
├── marathon.py              # 5-agent marathon — main entry point
├── agent.py                 # Solo DQN training
├── game.py                  # Pygame Snake environment (solo)
├── model.py                 # Neural network (solo agent)
├── helper.py                # Training plot utility
├── snake_env.py             # Headless Snake env (solo)
│
├── core/                    # Shared marathon utilities
│   ├── snake_env.py         # Headless Snake env (marathon)
│   ├── model.py             # Neural net with absolute-path save()
│   └── stats.py             # CSV logging + matplotlib stat plot
│
├── marathon_agents/         # One file per algorithm
│   ├── __init__.py          # Exports all 5 agents
│   ├── _helpers.py          # Shared: direction math, BFS utils
│   ├── random_agent.py
│   ├── bfs_agent.py
│   ├── astar_agent.py
│   ├── hamiltonian_agent.py
│   └── dqn_agent.py         # Self-contained DQN, record-only save
│
├── model/                   # Saved model weights (.pth)
├── analysis.ipynb           # Full data analysis notebook
├── data_agent.csv           # Solo DQN training log
├── data_marathon.csv        # Marathon 20k+ games log
└── stat.png                 # Live performance chart
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- `pip install torch pygame matplotlib pandas scipy scikit-learn`

### Run the Algorithm Marathon (5 agents simultaneously)
```bash
python marathon.py
```

### Run Solo DQN Training
```bash
python agent.py
```

### Keyboard Controls (Marathon)

| Key | Action |
|---|---|
| `D` | Toggle DQN between **Train** ↔ **Demo** mode |
| `+` / `=` | Speed up |
| `-` | Slow down |
| `Q` | Quit and save plot |

---

## 📊 Data Analysis

A full Jupyter notebook (`analysis.ipynb`) covers:

- **Pandas** — EDA, rolling means, per-agent statistics
- **SciPy** — Gaussian blur, Sobel edge detection on stat plots
- **Supervised Learning** — Linear Regression, Logistic Regression, Decision Tree, SVM, Random Forest, ANN
- **Unsupervised Learning** — K-Means clustering (3 phases: Exploration / Learning / Plateau)
- **Model comparison** exported to `model_comparison.csv`

```bash
pip install jupyter
jupyter notebook analysis.ipynb
```

---

## 🤖 DQN Architecture

```
Input  (11 features) → Linear(256, ReLU) → Linear(3 outputs)
```

**State vector (11 bits):**
- Danger straight / right / left
- Current direction (4 bits)
- Food relative position (4 bits)

**Training:** ε-greedy exploration → experience replay → Bellman update
**Saves:** Only when a new record score is beaten (never overwrites a good model)

---

## 📈 Key Results (20,000+ total games)

```
Rank  Agent        Mean Score   Note
 1.   Hamiltonian   297.0       Near-perfect every game
 2.   A*             50.83      Heuristic path planning
 3.   BFS            50.77      Shortest path, no heuristic
 4.   DQN            23.17      Learned, plateaus ~game 150
 5.   Random          0.11      Baseline noise
```

**Insight:** Classical algorithms dominate on small grids. DQN shows faster learning but hits a structural ceiling from the 1-step state representation.

---

## 📋 Course Outcome Mapping (AIML Lab)

| CO | Requirement | Project Evidence |
|---|---|---|
| CO1 | Python for AI tasks | Pygame, PyTorch, Pandas, SciPy |
| CO2 | Intelligent search | BFS, A\*, Hamiltonian Cycle |
| CO3 | Apply learning algorithms | DQN (Reinforcement Learning) |
| CO4 | Performance analysis | Marathon: 5 algorithms, 20k+ games |

---
