"""
astar_agent.py — A* (heuristic search) agent.
Uses Manhattan distance as heuristic; falls back to safe move if blocked.
"""
import heapq
from core.snake_env import Direction, Point, BLOCK_SIZE
from ._helpers import _dir_to_action, _neighbours, _in_bounds, _safe_fallback


class AStarAgent:
    name = "A*"

    @staticmethod
    def _h(a: Point, b: Point) -> int:
        return abs(a.x - b.x) + abs(a.y - b.y)

    def get_action(self, game) -> list:
        first_step = self._astar(game)
        if first_step:
            return _dir_to_action(game.direction, first_step)
        return _safe_fallback(game)

    def _astar(self, game):
        start   = game.head
        goal    = game.food
        blocked = set(game.snake[1:])
        counter = 0

        heap    = [(self._h(start, goal), 0, counter, start, None)]
        visited = {}

        while heap:
            f, g, _, pt, first_dir = heapq.heappop(heap)
            if pt in visited and visited[pt] <= g:
                continue
            visited[pt] = g

            for direction, npt in _neighbours(pt, game.w, game.h):
                if not _in_bounds(npt, game.w, game.h): continue
                if npt in blocked:                      continue
                ng = g + 1
                fd = first_dir if first_dir else direction
                if npt == goal:
                    return fd
                if npt not in visited or visited[npt] > ng:
                    counter += 1
                    heapq.heappush(heap, (ng + self._h(npt, goal), ng, counter, npt, fd))
        return None

    def on_game_over(self, *args, **kwargs):
        pass
