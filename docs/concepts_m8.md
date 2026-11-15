# Concept notes: M8 collision-aware planning

## 1. Configuration space (C-space)
A robot's configuration is its 7 joint angles: one point in a 7-D space. Each obstacle
in the world becomes a complicated "forbidden region" in that space, which can't be written
down but can be *tested* point by point. Sampling-based planners only need that test.

## 2. Collision checking: `planning/collision.py`
- Set `qpos`, run `mj_kinematics` (poses only, no physics), query geometry.
- Exact clearance to obstacles with `mj_geomDistance`; overlap for self/floor via `mj_collision`.
- A private `MjData` keeps planning from disturbing the running simulation.
- The held cube is attached at its *measured* pose in the hand. If it slipped, the plan knows.

## 3. RRT-Connect: `planning/rrt.py`
Two trees (from start and from goal); extend one toward a random sample, then greedily
connect the other to the new node; swap roles. Probabilistically complete: if a path
exists, the chance of finding it goes to 1 with enough samples. It is not optimal, which is why
the path is then **shortcut** (replace any stretch with a straight segment if it's free).

## 4. Redundancy resolution: `planning/redundancy.py`
7 joints, 6 pose constraints, so a 1-D family of postures (the elbow swivel). Plain IK picks
one arbitrarily (via its null-space preference). Near obstacles, search the family:
IK from many start postures, keep a clear one. Anchor the choice where clearance is tightest.

## 5. Cartesian paths vs joint paths
A straight line for the *gripper* is a curve in joint space; a straight line in *joint space*
is a curve for the gripper. Use Cartesian lines where the gripper's path matters
(descending onto the cube) and joint-space plans in open space (carrying over the wall).
Both are converted to dense joint paths before execution, so what was checked is what runs.

## 6. Evaluate the failure mode, not just the success rate
"Clean success" separates "placed it" from "placed it without hitting anything". A planner
that declines (no_plan) is far safer than one that tries and collides, even at equal success.

## Check your understanding
1. Why can't we just describe the wall in joint space and plan around it directly?
2. RRT-Connect found paths in 30-115 iterations, yet 0 iterations in many episodes. What
   changed between those episodes? (Hint: look at which posture the planner chose.)
3. Why does the lift-high heuristic fail 24% of grasps when the planner fails none?
4. Proving "no plan exists" takes up to 22 s. Propose two ways to cut that.
