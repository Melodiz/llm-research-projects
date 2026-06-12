"""Utilities for deterministic 8x8 FrozenLake ground-truth state handling"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import gymnasium as gym
import numpy as np

Action = Literal[0, 1, 2, 3]
Outcome = Literal["safe", "hole", "goal", "wall"]
Position = tuple[int, int]
Grid = list[list[str]]

ACTION_DELTAS: dict[int, Position] = {
    0: (0, -1),  # left
    1: (1, 0),  # down
    2: (0, 1),  # right
    3: (-1, 0),  # up
}

ACTION_NAMES: dict[int, str] = {
    0: "left",
    1: "down",
    2: "right",
    3: "up",
}


@dataclass(frozen=True)
class FrozenLakeGTState:
    """Ground-truth state extracted from a FrozenLake map and observation."""

    player_position: Position
    goal_position: Position
    hole_positions: tuple[Position, ...]
    grid: tuple[tuple[str, ...], ...]


def generate_reachable_random_map(
    seed: int,
    size: int = 8,
    p_hole: float = 0.2,
    max_attempts: int = 10_000,
) -> list[str]:
    """Generate a random FrozenLake map with a BFS path from S to G.

    The start is fixed at `(0, 0)` and the goal at `(size - 1, size - 1)`.
    Each other cell independently becomes a hole with probability `p_hole`.
    """

    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        grid = [["F" for _ in range(size)] for _ in range(size)]
        grid[0][0] = "S"
        grid[size - 1][size - 1] = "G"

        holes = rng.random((size, size)) < p_hole
        holes[0, 0] = False
        holes[size - 1, size - 1] = False
        for row in range(size):
            for col in range(size):
                if holes[row, col]:
                    grid[row][col] = "H"

        desc = ["".join(row) for row in grid]
        if is_bfs_reachable(desc):
            return desc

    raise RuntimeError(f"failed to generate reachable map after {max_attempts} attempts")


def is_bfs_reachable(desc: Sequence[str]) -> bool:
    """Return True when S can reach G by moving through non-hole cells."""

    grid = desc_to_grid(desc)
    start = find_symbol(grid, "S")
    goal = find_symbol(grid, "G")
    queue: deque[Position] = deque([start])
    seen = {start}

    while queue:
        pos = queue.popleft()
        if pos == goal:
            return True
        for next_pos in neighbors(pos, len(grid), len(grid[0])):
            row, col = next_pos
            if next_pos not in seen and grid[row][col] != "H":
                seen.add(next_pos)
                queue.append(next_pos)
    return False


def make_frozenlake_env(desc: Sequence[str], seed: int | None = None) -> gym.Env:
    """Create the required deterministic Gymnasium FrozenLake environment."""

    env = gym.make(
        "FrozenLake-v1",
        desc=list(desc),
        map_name=None,
        is_slippery=False,
        render_mode="rgb_array",
    )
    env.reset(seed=seed)
    return env


def extract_gt_state(desc: Sequence[str], observation: int) -> FrozenLakeGTState:
    """Extract symbolic GT state from map description and flat observation."""

    grid = desc_to_grid(desc)
    width = len(grid[0])
    player_position = divmod(int(observation), width)
    goal_position = find_symbol(grid, "G")
    holes = tuple(
        (row, col)
        for row, line in enumerate(grid)
        for col, symbol in enumerate(line)
        if symbol == "H"
    )
    return FrozenLakeGTState(
        player_position=player_position,
        goal_position=goal_position,
        hole_positions=holes,
        grid=tuple(tuple(row) for row in grid),
    )


def render_rgb_array(env: gym.Env) -> np.ndarray:
    """Render the current environment frame as an RGB numpy array."""

    frame = env.render()
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected RGB array render, got {type(frame)} shape={getattr(frame, 'shape', None)}")
    return frame


def deterministic_transition(
    grid: Sequence[Sequence[str]] | Sequence[str],
    player_position: Position,
    action: int,
) -> tuple[Position, Outcome]:
    """Predict the deterministic next state for one FrozenLake action.

    Returns `(next_position, outcome)`. If the action would leave the map, the
    position stays unchanged and the outcome is `"wall"`.
    """

    if action not in ACTION_DELTAS:
        raise ValueError(f"unknown FrozenLake action {action}; expected one of {sorted(ACTION_DELTAS)}")

    normalized_grid = desc_to_grid(grid)
    row, col = player_position
    d_row, d_col = ACTION_DELTAS[action]
    next_pos = (row + d_row, col + d_col)

    if not in_bounds(next_pos, len(normalized_grid), len(normalized_grid[0])):
        return player_position, "wall"

    next_symbol = normalized_grid[next_pos[0]][next_pos[1]]
    if next_symbol == "H":
        return next_pos, "hole"
    if next_symbol == "G":
        return next_pos, "goal"
    return next_pos, "safe"


def desc_to_grid(desc: Sequence[Sequence[str]] | Sequence[str]) -> Grid:
    """Normalize Gymnasium desc/list rows into a mutable list-of-lists grid."""

    grid: Grid = []
    for row in desc:
        if isinstance(row, str):
            grid.append(list(row))
        else:
            grid.append([str(cell.decode() if isinstance(cell, bytes) else cell) for cell in row])
    return grid


def find_symbol(grid: Sequence[Sequence[str]], symbol: str) -> Position:
    """Find a single symbol in the map grid."""

    for row, line in enumerate(grid):
        for col, value in enumerate(line):
            if value == symbol:
                return (row, col)
    raise ValueError(f"symbol {symbol!r} not found")


def neighbors(position: Position, height: int, width: int) -> Iterable[Position]:
    """Yield in-bounds neighboring positions in action order."""

    for d_row, d_col in ACTION_DELTAS.values():
        next_pos = (position[0] + d_row, position[1] + d_col)
        if in_bounds(next_pos, height, width):
            yield next_pos


def in_bounds(position: Position, height: int, width: int) -> bool:
    """Return True when `(row, col)` lies inside the grid."""

    row, col = position
    return 0 <= row < height and 0 <= col < width
