"""
_helpers.py — Shared movement utilities used by BFS, A*, and Hamiltonian agents.
"""
from core.snake_env import Direction, Point, BLOCK_SIZE

_CW = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]


def _dir_to_action(current: Direction, desired: Direction) -> list:
    """Absolute desired direction → relative [straight, right, left]."""
    ci   = _CW.index(current)
    di   = _CW.index(desired)
    diff = (di - ci) % 4
    if diff == 0: return [1, 0, 0]
    if diff == 1: return [0, 1, 0]
    if diff == 3: return [0, 0, 1]
    return [1, 0, 0]   # 180° (illegal) → go straight


def _neighbours(pt: Point, w: int, h: int):
    return [
        (Direction.RIGHT, Point(pt.x + BLOCK_SIZE, pt.y)),
        (Direction.LEFT,  Point(pt.x - BLOCK_SIZE, pt.y)),
        (Direction.DOWN,  Point(pt.x, pt.y + BLOCK_SIZE)),
        (Direction.UP,    Point(pt.x, pt.y - BLOCK_SIZE)),
    ]


def _in_bounds(pt: Point, w: int, h: int) -> bool:
    return 0 <= pt.x < w and 0 <= pt.y < h


def _safe_fallback(game) -> list:
    """Return any action that avoids immediate collision."""
    ci = _CW.index(game.direction)
    for action, ni in [([1, 0, 0], ci), ([0, 1, 0], (ci + 1) % 4), ([0, 0, 1], (ci - 1) % 4)]:
        d = _CW[ni]
        x, y = game.head.x, game.head.y
        if d == Direction.RIGHT: x += BLOCK_SIZE
        elif d == Direction.LEFT:  x -= BLOCK_SIZE
        elif d == Direction.DOWN:  y += BLOCK_SIZE
        elif d == Direction.UP:    y -= BLOCK_SIZE
        pt = Point(x, y)
        if _in_bounds(pt, game.w, game.h) and pt not in game.snake[1:]:
            return action
    return [1, 0, 0]
