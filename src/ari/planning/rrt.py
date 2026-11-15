"""M8: RRT-Connect motion planning in joint space, plus path shortcutting.

Why plan in joint space? Obstacles are easy to describe in the world (a box) but
impossible to describe in the 7-D space of joint angles. Collision CHECKING in joint
space is easy, though: set q, ask MuJoCo. Sampling-based planners only ever need that
yes/no question, which is why they dominate arm planning.

RRT-Connect (Kuffner & LaValle, 2000):
  - Grow two trees, one from the start and one from the goal.
  - Each iteration: sample a random q, EXTEND tree A one step toward it, then try to
    CONNECT tree B all the way to the new node (repeated steps until blocked or joined).
  - Swap the roles of A and B, repeat. When they join, read off the path.
Growing from both ends and greedily connecting makes it much faster than plain RRT.

The raw path zig-zags (random samples), so `shortcut` repeatedly picks two points on
the path and replaces the stretch between them with a straight segment if that segment
is collision-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ari.planning.collision import CollisionChecker
from ari.robot.ik import JOINT_LIMITS


@dataclass
class PlanResult:
    path: list | None      # list of (7,) joint configurations, start ... goal; None if failed
    iterations: int
    nodes: int
    checks: int            # collision checks used
    time: float            # s


class _Tree:
    def __init__(self, root: np.ndarray):
        self.nodes = [root]
        self.parent = [-1]

    def nearest(self, q: np.ndarray) -> int:
        d = np.linalg.norm(np.asarray(self.nodes) - q, axis=1)
        return int(np.argmin(d))

    def add(self, q: np.ndarray, parent: int) -> int:
        self.nodes.append(q)
        self.parent.append(parent)
        return len(self.nodes) - 1

    def path_to_root(self, i: int) -> list:
        out = []
        while i != -1:
            out.append(self.nodes[i])
            i = self.parent[i]
        return out


def rrt_connect(cc: CollisionChecker, q_start: np.ndarray, q_goal: np.ndarray, *,
                step: float = 0.15, max_iters: int = 3000, seed: int = 0,
                resolution: float = 0.03) -> PlanResult:
    import time

    t0, checks0 = time.time(), cc.check_count
    rng = np.random.default_rng(seed)
    lo, hi = JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]
    q_start, q_goal = np.asarray(q_start, float), np.asarray(q_goal, float)

    def result(path, it, nodes):
        return PlanResult(path, it, nodes, cc.check_count - checks0, time.time() - t0)

    if not cc.is_free(q_start) or not cc.is_free(q_goal):
        return result(None, 0, 0)
    if cc.segment_free(q_start, q_goal, resolution):
        return result([q_start, q_goal], 0, 2)  # straight line works: done

    def steer(q_from, q_to):
        d = q_to - q_from
        n = np.linalg.norm(d)
        return q_to.copy() if n <= step else q_from + d * (step / n)

    def extend(tree, q_target):
        """One step from the nearest node toward q_target. Returns (status, new_index)."""
        i = tree.nearest(q_target)
        q_new = steer(tree.nodes[i], q_target)
        if not cc.segment_free(tree.nodes[i], q_new, resolution):
            return "trapped", None
        j = tree.add(q_new, i)
        return ("reached" if np.allclose(q_new, q_target) else "advanced"), j

    def connect(tree, q_target):
        while True:
            status, j = extend(tree, q_target)
            if status != "advanced":
                return status, j

    a, b = _Tree(q_start), _Tree(q_goal)
    a_is_start = True
    for it in range(1, max_iters + 1):
        q_rand = rng.uniform(lo, hi)
        status, j = extend(a, q_rand)
        if status != "trapped":
            status_b, k = connect(b, a.nodes[j])
            if status_b == "reached":
                path_a, path_b = a.path_to_root(j)[::-1], b.path_to_root(k)
                path = path_a + path_b[1:]  # the joining node appears in both
                if not a_is_start:
                    path = path[::-1]
                return result(path, it, len(a.nodes) + len(b.nodes))
        a, b = b, a
        a_is_start = not a_is_start
    return result(None, max_iters, len(a.nodes) + len(b.nodes))


def shortcut(cc: CollisionChecker, path: list, iters: int = 150, seed: int = 0,
             resolution: float = 0.03) -> list:
    """Random shortcutting: try to replace path[i..j] with a straight segment."""
    rng = np.random.default_rng(seed)
    path = [np.asarray(q, float) for q in path]
    for _ in range(iters):
        if len(path) <= 2:
            break
        i, j = sorted(rng.choice(len(path), size=2, replace=False))
        if j - i < 2:
            continue
        if cc.segment_free(path[i], path[j], resolution):
            path = path[: i + 1] + path[j:]
    return path


def path_length(path: list) -> float:
    """Total joint-space distance (rad)."""
    return float(sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:])))
