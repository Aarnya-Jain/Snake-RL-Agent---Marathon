"""
snake_env.py — headless Snake game environment.
Same logic as game.py but no pygame display, no clock.
Imports Direction and Point from game.py so the original Agent works directly.
"""

import random
import sys
import os

# game.py lives one level up (project root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from game import Direction, Point

BLOCK_SIZE = 20


class SnakeGame:
    def __init__(self, w=400, h=300):
        self.w = w
        self.h = h
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        cx = (self.w // 2 // BLOCK_SIZE) * BLOCK_SIZE
        cy = (self.h // 2 // BLOCK_SIZE) * BLOCK_SIZE
        self.head  = Point(cx, cy)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y),
        ]
        self.score           = 0
        self.food            = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        """Iterative food placement — safe even when snake is very long."""
        snake_set = set(self.snake)
        free = [
            Point(x, y)
            for x in range(0, self.w, BLOCK_SIZE)
            for y in range(0, self.h, BLOCK_SIZE)
            if Point(x, y) not in snake_set
        ]
        if not free:
            self.food = None   # board is full — Hamiltonian win!
            return
        self.food = random.choice(free)

    def play_step(self, action):
        self.frame_iteration += 1
        self._move(action)
        self.snake.insert(0, self.head)

        reward    = 0
        game_over = False

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward    = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self.frame_iteration = 0
            self._place_food()
            if self.food is None:
                return 100, True, self.score
        else:
            self.snake.pop()

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action == [1, 0, 0]:
            new_dir = clock_wise[idx]
        elif action == [0, 1, 0]:
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir
        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT: x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:  x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:  y += BLOCK_SIZE
        elif self.direction == Direction.UP:    y -= BLOCK_SIZE
        self.head = Point(x, y)
