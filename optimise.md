# Optimise: What Can We Build From Scratch?

This document identifies components currently imported from libraries that can be **reimplemented from scratch** — keeping things at an entry-level appropriate for the AIML lab syllabus. The goal is to demonstrate understanding of the underlying algorithms, not just their usage.

---

## Overview of What's Currently Imported

| Library | What We Use From It | Can We Replace? |
|---|---|---|
| `torch.nn.Linear` | Fully connected layers | ✅ Yes — write our own `Linear` layer |
| `torch.nn.functional.relu` | Activation function | ✅ Yes — just a one-liner |
| `torch.optim.Adam` | Gradient-based optimiser | ⚠️ Partially — can use simpler SGD instead |
| `torch.nn.MSELoss` | Loss function | ✅ Yes — trivial to write |
| `numpy` (in agent) | State array, comparisons | ✅ Yes — use plain Python lists |
| `collections.deque` | Replay memory buffer | ✅ Yes — wrap a list with max-size logic |
| `matplotlib` | Training plot | ⚠️ Optional — can log to terminal instead |
| `pygame` | Game rendering | ❌ No — this is infrastructure, not ML |
| `torch` (autograd/backprop) | Backpropagation | ⚠️ Only if we write a full NumPy-based net |

---

## 1. ✅ Reimplement the Neural Network (NumPy only)

> **Syllabus connection:** Unit 3 — ANN (Artificial Neural Networks), Unit 4 — Deep Neural Network basics

This is the highest-impact thing you can replace. The current model (`model.py`) uses two `nn.Linear` layers and PyTorch autograd. We can rebuild this **entirely with NumPy**, manually implementing:

### What to reimplement

#### Forward Pass
Each linear layer computes:
```
output = input @ weights + bias
```
ReLU activation:
```python
def relu(x):
    return np.maximum(0, x)
```

#### Backpropagation (Gradient Descent)
Instead of `loss.backward()`, we compute gradients manually:
```
dL/dW2 = hidden_output.T @ dL/dout
dL/dW1 = input.T @ (dL/dout @ W2.T) * relu_derivative
```
ReLU derivative:
```python
def relu_derivative(x):
    return (x > 0).astype(float)
```

#### MSE Loss (from scratch)
```python
def mse_loss(predicted, target):
    return np.mean((predicted - target) ** 2)
```

#### Weight Update (SGD — simpler than Adam)
```python
weights -= learning_rate * gradient
```

> **Why not Adam from scratch?** Adam requires tracking two moving averages per parameter. SGD is simpler and still valid — and shows that you understand gradient descent, which is all that's needed at this syllabus level.

### Minimal from-scratch MLP class
```python
import numpy as np

class SimpleNet:
    def __init__(self, input_size=11, hidden_size=64, output_size=3, lr=0.01):
        self.lr = lr
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)

    def forward(self, x):
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)   # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2                       # Q-values (no activation on output)

    def backward(self, target):
        pred = self.z2
        loss = np.mean((pred - target) ** 2)
        dout = 2 * (pred - target) / pred.size
        dW2 = self.a1.T @ dout
        db2 = dout.sum(axis=0)
        da1 = dout @ self.W2.T
        dz1 = da1 * (self.z1 > 0)  # ReLU derivative
        dW1 = self.x.T @ dz1
        db1 = dz1.sum(axis=0)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        return loss
```

**Effort:** Medium — can be done in ~1–2 hours. Completely removes PyTorch dependency.

---

## 2. ✅ Reimplement the Replay Memory (Deque → List)

> **Syllabus connection:** Unit 1 — Python fundamentals, Lists

`collections.deque(maxlen=N)` auto-removes old entries when it overflows. We can do this with a plain list:

```python
class ReplayMemory:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size

    def append(self, experience):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)   # remove oldest
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)
```

**Effort:** Very low — 10 minutes. Removes `collections` import.

---

## 3. ✅ Reimplement MSE Loss (from scratch)

> **Syllabus connection:** Unit 3 — Model evaluation, Loss functions

Currently using `nn.MSELoss()`. Replace with:

```python
def mse_loss(predicted, target):
    diff = predicted - target
    return np.mean(diff ** 2)
```

**Effort:** Trivial — one line.

---

## 4. ✅ Remove NumPy Dependency in State Vector

> **Syllabus connection:** Unit 1 — Lists, Booleans

The `get_state()` function in `agent.py` returns `np.array(state, dtype=int)`. Since the state is just a list of `True`/`False` values mapped to 0/1, we can use plain Python:

```python
# Instead of:
return np.array(state, dtype=int)

# Use:
return [int(s) for s in state]
```

And replace `np.array_equal(action, [1, 0, 0])` in `game.py` with:

```python
action == [1, 0, 0]
```

**Effort:** Very low — a few substitutions.

---

## 5. ⚠️ Replace Matplotlib Plot with Terminal Logging (Optional)

> **Syllabus connection:** Unit 1 — String manipulation, print

Matplotlib is not a core ML component. A simple terminal log is equally informative for a lab submission:

```python
def log_progress(game_num, score, record, mean_score):
    bar = '█' * min(score, 40)
    print(f"Game {game_num:4} | Score: {score:3} | Record: {record:3} | Mean: {mean_score:.2f} | {bar}")
```

Or keep `plot.png` saving but remove the IPython import (which is Jupyter-specific and unnecessary in a script).

**Effort:** Low.

---

## Summary Table

| Component | Current | Replacement | Effort | Removes |
|---|---|---|---|---|
| Neural Network | `torch.nn` | NumPy `SimpleNet` class | Medium | `torch` |
| Backpropagation | PyTorch autograd | Manual chain rule in NumPy | Medium | `torch` |
| Loss Function | `nn.MSELoss` | `np.mean((pred-target)**2)` | Trivial | `torch` |
| Optimiser | `Adam` | `SGD` (manual weight update) | Low | `torch.optim` |
| Replay Memory | `collections.deque` | Plain Python list class | Very Low | `collections` |
| State Array | `np.array(...)` | `[int(s) for s in state]` | Very Low | `numpy` (in agent) |
| Visualisation | `matplotlib` + `IPython` | Terminal print log | Low | both |

---

## What NOT to Replace

| Component | Reason |
|---|---|
| `pygame` | This is a game rendering library — not an ML concept. No point reimplementing it. |
| `numpy` (in net) | If you reimplement the neural network in NumPy, numpy itself stays — it's your compute backend. |
| File I/O (`os`, `torch.save`) | Can use `pickle` or `numpy.save` instead, but not worth the effort |

---

## Recommended Priority for Lab Submission

1. **HIGH IMPACT:** Reimplement the neural network in pure NumPy (removes all PyTorch dependency)
2. **EASY WIN:** Replace `deque` with a plain list class
3. **BONUS:** Add terminal score logging instead of Matplotlib
