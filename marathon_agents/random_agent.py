"""
random_agent.py — Random action agent.
"""
import random


class RandomAgent:
    name = "Random"

    def get_action(self, game) -> list:
        return random.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def on_game_over(self, *args, **kwargs):
        pass
