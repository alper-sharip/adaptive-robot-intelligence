# M8 results: collision-aware pick-and-place

Scene: a wall (3 cm thick, 50 cm long, random height 18-30 cm per episode) between the
cube region and the target region. 100 seeded episodes (seeds 0-99), identical for every row.
Cube pose from simulator ground truth.

**Clean success** = cube placed in the target AND nothing touched the wall at any point.

Reproduce: `python scripts/evaluate_wall.py --policy <naive|lift-high|planned> [--margin 0.01]`

| Policy | Clean success (95% CI) | Episodes touching wall | Failures | Mean episode time | Planning time |
|---|---|---|---|---|---|
| Naive (Stage 2 scripted, carries at 15 cm) | **0%** (0-3.7) | 100 | placement_miss 51, grasp_failed 24, dropped 17, not_on_table 8 | - | - |
| Lift-high heuristic (knows wall height, carries 12 cm above it) | **60%** (50.2-69.1) | 40 | grasp_failed 24, placement_miss 2 (+14 successes that scraped the wall) | 17.0 s | - |
| **Planned**, 2 cm margin | **96%** (90.2-98.4) | **0** | no_place_plan 3, no_grasp_plan 1 | 12.4 s | median 0.69 s, max 22 s |
| Planned, 1 cm margin | 97% (91.5-99.0) | 0 | no_place_plan 3 | 12.2 s | median 0.72 s, max 22 s |

## What the numbers say

1. **The obvious fix is not enough.** Lifting higher solves transport, but 24% of cubes sit
   close enough to the wall that the *elbow* hits it during the grasp. A hand-written policy
   can't see this; collision checking can.
2. **Planning is faster as well as safer** (12.4 s vs 17.0 s): it doesn't climb over every wall at
   maximum height, and in many episodes a well-chosen posture makes the transport a single
   straight joint-space move (0 RRT iterations).
3. **Failing safe:** every planned failure is a refusal ("no plan found"), never a collision.
   That is the property a real robot needs.
4. **Margin trade-off:** 1 cm vs 2 cm changes success by one episode, inside the noise. The
   remaining refusals are near-infeasible placements right against the wall.
5. **Planning time has a long tail.** Successful plans take well under a second; *proving*
   there is no plan takes up to 22 s (4 yaws × 41 postures × 25 IK solves). Worth fixing
   before anything runs in a loop (e.g. RL).

## Known limitations
- Collision checking samples segments at 0.03 rad. That is not a proof; the 2 cm margin
  absorbs it in practice (0 contacts in 100 episodes).
- Planned joint paths stop at every waypoint (simple, guaranteed to stay on checked
  segments). Spline smoothing would be faster but must be re-checked.
- The planner uses the true wall geometry. In Stage 3 it should come from perception.
