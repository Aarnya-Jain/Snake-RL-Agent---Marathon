"""
dqn_agent.py — Deep Q-Network agent for the marathon.
Self-contained: owns model/model_marathon.pth exclusively.

Modes:
  'train' — ε-greedy exploration, online learning, saves on new record
  'demo'  — loads saved weights, ε=0, no training (pure exploitation)

Press D in marathon.py to toggle between modes live.
"""
import os
import random
import numpy as np
import torch
from collections import deque
from core.model import Linear_QNet, QTrainer


class DQNAgent:
    name      = "DQN"
    MODEL_PATH = os.path.join('model', 'model_marathon.pth')

    MAX_MEMORY = 100_000
    BATCH_SIZE = 1000
    LR         = 0.001

    def __init__(self):
        self.mode         = 'train'
        self._model       = Linear_QNet(11, 256, 3)
        self._trainer     = QTrainer(self._model, lr=self.LR, gamma=0.9)
        self._memory      = deque(maxlen=self.MAX_MEMORY)
        self.n_games      = 0
        self._best_score  = 0
        self._last_state  = None
        self._last_action = None

        # Load existing model if available — continues from last checkpoint
        if os.path.exists(self.MODEL_PATH):
            self._model.load_state_dict(
                torch.load(self.MODEL_PATH, map_location='cpu', weights_only=False))
            self._model.eval()

    # ── Public API ────────────────────────────────────────────────────────────

    def toggle(self):
        """Live toggle between train ↔ demo modes. Never reinitialises the model."""
        if self.mode == 'train':
            self.mode = 'demo'
            self._model.eval()
        else:
            self.mode = 'train'

    @property
    def display_name(self):
        tag = 'DEMO' if self.mode == 'demo' else 'TRAIN'
        return f'DQN [{tag}]'

    # ── State ─────────────────────────────────────────────────────────────────

    def _get_state(self, game) -> np.ndarray:
        from core.snake_env import Direction, Point, BLOCK_SIZE
        head = game.snake[0]
        pl   = Point(head.x - BLOCK_SIZE, head.y)
        pr   = Point(head.x + BLOCK_SIZE, head.y)
        pu   = Point(head.x, head.y - BLOCK_SIZE)
        pd   = Point(head.x, head.y + BLOCK_SIZE)
        dl = game.direction == Direction.LEFT
        dr = game.direction == Direction.RIGHT
        du = game.direction == Direction.UP
        dd = game.direction == Direction.DOWN
        state = [
            (dr and game.is_collision(pr)) or (dl and game.is_collision(pl)) or
            (du and game.is_collision(pu)) or (dd and game.is_collision(pd)),
            (du and game.is_collision(pr)) or (dd and game.is_collision(pl)) or
            (dl and game.is_collision(pu)) or (dr and game.is_collision(pd)),
            (dd and game.is_collision(pr)) or (du and game.is_collision(pl)) or
            (dr and game.is_collision(pu)) or (dl and game.is_collision(pd)),
            dl, dr, du, dd,
            game.food.x < head.x, game.food.x > head.x,
            game.food.y < head.y, game.food.y > head.y,
        ]
        return np.array(state, dtype=int)

    # ── Action ────────────────────────────────────────────────────────────────

    def get_action(self, game) -> list:
        state   = self._get_state(game)

        if self.mode == 'demo':
            epsilon = 0
        else:
            epsilon = max(0, 80 - self.n_games)

        if random.randint(0, 200) < epsilon:
            move = random.randint(0, 2)
        else:
            with torch.no_grad():
                t    = torch.tensor(state, dtype=torch.float)
                move = torch.argmax(self._model(t)).item()

        action       = [0, 0, 0]
        action[move] = 1
        self._last_state  = state
        self._last_action = action
        return action

    # ── Training hooks ────────────────────────────────────────────────────────

    def on_step(self, reward, game, done):
        if self.mode == 'demo' or self._last_state is None:
            return
        state_new = self._get_state(game)
        self._trainer.train_step(
            self._last_state, self._last_action, reward, state_new, done)
        self._memory.append(
            (self._last_state, self._last_action, reward, state_new, done))

    def on_game_over(self, score: int = 0):
        self.n_games      += 1
        self._last_state   = None
        self._last_action  = None
        if self.mode == 'demo':
            return
        # long memory replay
        if len(self._memory) > self.BATCH_SIZE:
            batch = random.sample(self._memory, self.BATCH_SIZE)
        else:
            batch = list(self._memory)
        if batch:
            s, a, r, ns, d = zip(*batch)
            self._trainer.train_step(s, a, r, ns, d)
        # save ONLY when a new record is beaten — never overwrites a good model
        if score > self._best_score:
            self._best_score = score
            self._model.save('model_marathon.pth')
