"""
hamiltonian_agent.py — Hamiltonian cycle agent.
Pre-computes a full Hamiltonian cycle over the grid and follows it perfectly.
Achieves near-perfect scores every game.
"""
from core.snake_env import Direction, Point, BLOCK_SIZE
from ._helpers import _dir_to_action, _safe_fallback


class HamiltonianAgent:
    """
    Pre-computes a valid Hamiltonian cycle over the grid.

    Construction for W cols (even), H rows:
      - Zigzag all columns through rows 1..(H-1)  [W*(H-1) cells]
      - Step up to row 0 at the last column        [1 cell]
      - Sweep row 0 leftward back to col 0         [W-1 cells]
      Total = W*H cells — a proper closed cycle.
    """
    name = "Hamiltonian"

    def __init__(self, w=400, h=300):
        self.cycle  = self._build_cycle(w, h)
        self.index  = {pt: i for i, pt in enumerate(self.cycle)}
        self.length = len(self.cycle)

    def _build_cycle(self, w, h):
        cols = w // BLOCK_SIZE
        rows = h // BLOCK_SIZE
        path = []

        for col in range(cols):
            if col % 2 == 0:
                row_range = range(1, rows)
            else:
                row_range = range(rows - 1, 0, -1)
            for row in row_range:
                path.append(Point(col * BLOCK_SIZE, row * BLOCK_SIZE))

        path.append(Point((cols - 1) * BLOCK_SIZE, 0))
        for col in range(cols - 2, -1, -1):
            path.append(Point(col * BLOCK_SIZE, 0))

        return path

    def get_action(self, game) -> list:
        head = game.head
        if head not in self.index:
            return _safe_fallback(game)
        next_pt = self.cycle[(self.index[head] + 1) % self.length]

        dx = next_pt.x - head.x
        dy = next_pt.y - head.y
        if   dx > 0: desired = Direction.RIGHT
        elif dx < 0: desired = Direction.LEFT
        elif dy > 0: desired = Direction.DOWN
        else:        desired = Direction.UP

        return _dir_to_action(game.direction, desired)

    def on_game_over(self, *args, **kwargs):
        pass
