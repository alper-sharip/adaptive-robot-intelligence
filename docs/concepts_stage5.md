# Concept notes: Stage 5 reinforcement learning

## 1. What changes from imitation
Imitation needs an expert and copies it. Reinforcement learning needs only a **reward** and
improves by trial and error. That removes the ceiling ("never better than the teacher") and the
need for a teacher to label recoveries, at the price of far more interaction: PPO here uses
hundreds of thousands of simulated steps where cloning used a few hundred demonstrations.

## 2. The environment is four design decisions: `envs/gym_envs.py`
Observation, action, reward, termination. Observation and action are inherited from Stage 4 so
that policies are interchangeable. The two new ones:

**Reward.** A sparse "+5 when the cube is placed" is the true objective but gives no feedback
until the task is solved by accident. Shaping adds hints. *Potential-based* shaping pays
`γ·Φ(s') − Φ(s)`, the change in a hand-written progress estimate Φ. Its sum over any path
depends only on the endpoints (it telescopes; there is a test for this), so it cannot be farmed
and provably does not change which policy is best.

**Termination vs truncation.** Success *terminates* the episode: nothing follows. A time limit
*truncates* it: the world would have continued. For a truncated step the learner adds the
estimated value of the next state; otherwise the agent learns that the end of the clock is a cliff.

## 3. PPO in four ideas: `rl/ppo.py`
- **Stochastic policy.** The network outputs the mean of a Gaussian; sampling from it is the
  exploration. The standard deviation is learned too.
- **Value function.** A second network predicts the return from each state. It is the baseline
  against which an action is judged: *better than usual* or *worse than usual*.
- **Advantage (GAE).** `A_t = Σ (γλ)^k δ_{t+k}`, with `δ_t = r_t + γV(s_{t+1}) − V(s_t)`.
  λ trades variance (λ → 1, trust observed rewards) against bias (λ → 0, trust the critic).
- **Clipped update.** Increase the probability of positive-advantage actions, but clip the
  probability ratio to [0.8, 1.2] and stop early if the policy has moved too far (KL check).
  Data from the old policy only says something about nearby policies.

An implementation of this is easy to get subtly wrong in ways that still "run". It is therefore
checked on a standard benchmark (Pendulum-v1) and a toy task before being trusted on the robot.

## 4. Why not learn pick-and-place from scratch
Grasping requires a precise sequence (align, descend, close, lift) that random exploration almost
never produces, and nothing in the shaping can reward "closed the fingers at the right moment"
before it has happened. See the from-scratch result in `docs/results_stage5.md`.

## 5. Residual RL: `rl/tasks.py`
`action = clone(obs) + correction(obs)`, the clone frozen, the correction network initialized
to output exactly zero. Training starts at the clone's success rate rather than at zero;
exploration is millimetre-scale noise around a sensible action; RL only has to learn small fixes.
Two details that matter:
- **Critic warm-up.** At the start the policy is good and the value function knows nothing.
  Updating the policy from a random critic's advantages destroys it, so the critic trains alone first.
- **Small steps.** Lower learning rate and a tighter KL limit than from-scratch training.

## 6. Evaluate the mean, and control for the noise
Training statistics come from the noisy exploring policy; reported results use the mean action on
the fixed evaluation seeds. Here noise is not only a nuisance: jitter alone frees the clone from
millimetre-scale stalls. "Clone + noise, no learning" is therefore a required baseline; without
it, some of what RL appears to learn is just the noise it explores with.

## Check your understanding
1. Why does `Σ (γΦ(s') − Φ(s))` not depend on the path when γ = 1? What changes for γ < 1?
2. In `_worker`, what would the agent learn about the last 10 steps before the time limit if the
   truncation bootstrap were removed?
3. The residual starts at zero. Why is it also important that the clone's weights are frozen?
4. The noise-only baseline beats the deterministic clone. Does that mean the deployed robot should
   act with noise? What would you check before deciding?
