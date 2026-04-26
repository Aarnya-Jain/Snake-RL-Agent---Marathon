# Snake: Learning to Live — Project Overview

## What is This Project?

**Snake: Learning to Live** is a classic Snake game where the snake is not controlled by a human — it learns to play on its own using **Reinforcement Learning**, specifically a technique called **Deep Q-Learning (DQN)**. The snake starts knowing nothing, makes random moves, and gradually learns from its mistakes and rewards over hundreds of games, eventually developing a strategy to survive longer and eat more food.

The project sits at the intersection of:
- **Game Development** (Pygame-based custom snake environment)
- **Machine Learning** (Reinforcement Learning with Q-Learning)
- **Deep Learning** (Neural Network function approximation via PyTorch)

---

## Core Concepts

### Reinforcement Learning (RL)
The agent (snake) interacts with its environment (the game) and receives:
- **+10 reward** when it eats food
- **−10 reward** when it dies (hits a wall or itself)
- **0** for all other moves

The goal is to train the agent to maximise cumulative reward over time.

### Deep Q-Learning (DQN)
Classical Q-Learning uses a table to store Q-values for every (state, action) pair. Since the state space here is large, a **Neural Network** is used instead to approximate the Q-function. This is what makes it "Deep" Q-Learning.

The **Bellman Equation** (used in training) is:
```
Q_new = reward + γ * max(Q(next_state))
```
Where `γ` (gamma) is the discount factor (0.9 here), controlling how much the agent values future rewards vs. immediate ones.

### Exploration vs. Exploitation (ε-Greedy)
Early training uses a linearly decaying **epsilon (ε)** value:
- When `ε` is high → the agent takes **random actions** (explores)
- When `ε` is low → the agent uses the **neural network** to decide (exploits)

Formula used: `ε = 80 - n_games` clamped over `[0, 200]`

---

## File-by-File Breakdown

### `game.py` — The Snake Environment
**Role:** Defines the game world the AI interacts with.

| Component | Details |
|---|---|
| `SnakeAI` class | Main game class — initialises the window, handles game logic |
| `reset()` | Resets the snake to starting position (center, length 3), clears score |
| `play_step(action)` | Core game loop tick — moves snake, checks collision, gives reward |
| `is_collision(pt)` | Returns `True` if a point is out-of-bounds or inside the snake body |
| `_move(action)` | Translates 3-value action array `[straight, right, left]` into a direction |
| `_update_ui()` | Renders the grid, snake (with eyes!), food (with leaf), and scoreboard |
| `_place_food()` | Randomly places food, recursively retries if it lands on the snake |

**Action Encoding:**
- `[1, 0, 0]` → Go straight
- `[0, 1, 0]` → Turn right (clockwise)
- `[0, 0, 1]` → Turn left (counter-clockwise)

**Anti-loop mechanism:** If `frame_iteration > 100 * len(snake)`, the game ends (prevents the snake from running in circles forever).

---

### `agent.py` — The Reinforcement Learning Agent
**Role:** The brain that decides what the snake does, trains the model, runs the game loop.

| Component | Details |
|---|---|
| `Agent` class | Holds the neural network, memory, and training logic |
| `get_state(game)` | Encodes the current game situation into an **11-value binary vector** |
| `remember(...)` | Appends a `(state, action, reward, next_state, done)` tuple to a replay buffer |
| `train_short_memory(...)` | Trains on the most recent single step (online learning) |
| `train_long_memory()` | Samples a random mini-batch from replay memory and trains (experience replay) |
| `get_action(state)` | Returns a move — random if exploring, model-predicted if exploiting |
| `train()` | Main training loop: runs games forever, saves best model |

**The 11-value State Vector:**

| Index | Meaning |
|---|---|
| 0 | Danger straight ahead |
| 1 | Danger to the right |
| 2 | Danger to the left |
| 3 | Currently moving LEFT |
| 4 | Currently moving RIGHT |
| 5 | Currently moving UP |
| 6 | Currently moving DOWN |
| 7 | Food is to the LEFT |
| 8 | Food is to the RIGHT |
| 9 | Food is ABOVE |
| 10 | Food is BELOW |

**Hyperparameters:**
- `MAX_MEMORY = 100,000` — Replay buffer size (uses a `deque`)
- `BATCH_SIZE = 1,000` — Number of samples per long-memory training step
- `LR = 0.001` — Learning rate for the Adam optimiser
- `γ = 0.9` — Discount factor

---

### `model.py` — The Neural Network
**Role:** Defines and trains the Q-Network used to approximate Q-values.

| Component | Details |
|---|---|
| `Linear_QNet` | A simple feed-forward neural network with **2 linear layers** |
| Architecture | `Input(11) → ReLU → Hidden(256) → Output(3)` |
| `save()` | Saves model weights to `./model/model.pth` |
| `QTrainer` | Manages training — loss computation, backpropagation, weight updates |
| Optimiser | **Adam** (`lr=0.001`) |
| Loss Function | **Mean Squared Error (MSE)** between predicted and target Q-values |

**Training Logic (Bellman Update):**
```python
if not done:
    Q_new = reward + gamma * max(Q(next_state))
else:
    Q_new = reward  # terminal state, no future reward
```

---

### `helper.py` — Visualisation Utility
**Role:** Plots training progress in real time using Matplotlib.

| Component | Details |
|---|---|
| `plot(scores, mean_scores)` | Draws two lines — per-game score (blue) and running mean score (red) |
| Style | Dark-themed plot matching the game's Catppuccin Mocha colour theme |
| Output | Saves `plot.png` each frame; renders inline via IPython display |

---

## Algorithms Implemented

| Algorithm | Where Used | Purpose |
|---|---|---|
| **Deep Q-Learning (DQN)** | `agent.py` + `model.py` | Core RL algorithm |
| **Experience Replay** | `agent.py` → `train_long_memory()` | Breaks correlation between training samples |
| **ε-Greedy Exploration** | `agent.py` → `get_action()` | Balances exploration vs exploitation |
| **Bellman Equation** | `model.py` → `train_step()` | Computes target Q-values |
| **Backpropagation** | `model.py` (via PyTorch autograd) | Updates neural network weights |
| **Adam Optimisation** | `model.py` | Adaptive gradient descent |
| **Recursive Food Placement** | `game.py` → `_place_food()` | Prevents food spawning inside snake |

---

## Libraries Used

| Library | Version | Purpose |
|---|---|---|
| `pygame` | latest | Game rendering, input handling, game loop |
| `torch` (PyTorch) | latest | Neural network definition, training, saving |
| `numpy` | latest | State vector construction, array ops |
| `matplotlib` | (via torch deps) | Training progress visualisation |
| `IPython` | latest | Inline display of live plots |
| `collections.deque` | stdlib | Fixed-size replay memory buffer |
| `enum.Enum` | stdlib | Direction enumeration |
| `collections.namedtuple` | stdlib | Lightweight `Point(x, y)` data structure |

---

## Data Flow Summary

```
game.py (Environment)
    ↕  state / reward / done
agent.py (Agent)
    ↕  forward pass / Q-values
model.py (Neural Network)
    ↓
helper.py (Visualisation)
```

1. `Agent.get_state()` reads the game and encodes an 11-bit state
2. `Agent.get_action()` queries the neural network (or picks random)
3. `game.play_step(action)` executes the move and returns `(reward, done, score)`
4. `Agent.remember()` stores the transition in replay memory
5. `Agent.train_short_memory()` trains on this single step
6. When game ends → `Agent.train_long_memory()` trains on a random mini-batch
7. Best models are saved; `helper.plot()` updates the training graph

---

## Key Observations

- The neural network is **very small** (11 → 256 → 3) — intentionally lightweight, yet effective
- The agent typically reaches decent scores (~20–40) after **~100 games**
- The game colour theme is **Catppuccin Mocha** (a popular dark colour palette)
- The `model/model.pth` file (pre-trained weights) is saved and can be resumed
- Matplotlib is run non-blockingly via `IPython.display`, saving `plot.png` as a snapshot
