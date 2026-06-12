"""Ground-truth greedy planner for deterministic FrozenLake."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from frozenlake_utils import ACTION_NAMES, Outcome, Position, deterministic_transition

# Deterministic tie-break order used whenever multiple safe actions have the
# same Manhattan distance to the goal.
GREEDY_TIE_BREAK_ORDER: tuple[int, ...] = (2, 1, 0, 3)  # right, down, left, up


@dataclass(frozen=True)
class GreedyDecision:
    action: int
    action_name: str
    next_position: Position
    outcome: Outcome
    manhattan_distance: int


def manhattan_distance(a: Position, b: Position) -> int:
    """Return Manhattan distance between two `(row, col)` positions."""

    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def choose_gt_greedy_action(
    grid: Sequence[Sequence[str]] | Sequence[str],
    player_position: Position,
    goal_position: Position,
) -> GreedyDecision | None:
    """Choose one greedy action using GT map knowledge.

    The planner evaluates actions in `GREEDY_TIE_BREAK_ORDER`, ignores actions
    predicted to hit holes or walls, and selects the remaining action whose
    predicted next position has the smallest Manhattan distance to the goal.
    Returns None if no safe/goal action exists.
    """

    best: GreedyDecision | None = None
    for action in GREEDY_TIE_BREAK_ORDER:
        next_position, outcome = deterministic_transition(grid, player_position, action)
        if outcome in {"hole", "wall"}:
            continue

        decision = GreedyDecision(
            action=action,
            action_name=ACTION_NAMES[action],
            next_position=next_position,
            outcome=outcome,
            manhattan_distance=manhattan_distance(next_position, goal_position),
        )
        if best is None or decision.manhattan_distance < best.manhattan_distance:
            best = decision

    return best
