"""
bfs_agent.py — Breadth-First Search agent.
Finds the shortest path to food; falls back to safe move if no path exists.
"""
from collections import deque
from core.snake_env import Direction, Point, BLOCK_SIZE
from ._helpers import _dir_to_action, _neighbours, _in_bounds, _safe_fallback


class BFSAgent:
    name = "BFS"

    def get_action(self, game) -> list:
        first_step = self._bfs(game)
        if first_step:
            return _dir_to_action(game.direction, first_step)
        return _safe_fallback(game)

    def _bfs(self, game):
        start   = game.head
        goal    = game.food
        blocked = set(game.snake[1:])
        queue   = deque([(start, None)])
        visited = {start}

        while queue:
            pt, first_dir = queue.popleft()
            for direction, npt in _neighbours(pt, game.w, game.h):
                if not _in_bounds(npt, game.w, game.h): continue
                if npt in blocked or npt in visited:    continue
                fd = first_dir if first_dir else direction
                if npt == goal:
                    return fd
                visited.add(npt)
                queue.append((npt, fd))
        return None

    def on_game_over(self, *args, **kwargs):
        pass
