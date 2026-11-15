# Concept notes: Stage 7 benchmarking and reporting

## 1. In-distribution success says little: `benchmark.py`
A policy that scores 100% in the conditions it was built for has shown that those conditions
are easy for it. What separates policies is what happens when something changes. A
generalization benchmark is a list of such changes, applied to every policy on the same seeds.

## 2. One factor at a time
Each condition changes one thing (cube size, latency, friction...). If two things changed
together and success dropped, you could not say which one caused it. The price: real
deployments change several things at once, and effects can combine. One deliberate double
change is included (low friction + stiff gripper) because it answers a specific question: is
the friction failure a policy problem or a hardware limit?

## 3. Perturb where the policy reads, judge on the truth
Sensing errors are added in the one function every policy reads the cube from; success is
judged on the simulator's true state. Mixing the two up (judging success on the noisy reading)
would reward a policy for agreeing with its own sensor.

## 4. A mean over conditions is not a ranking
Three policies average 84.5% here and fail on completely different columns. The average
depends on which conditions were put in the table: add two more "moved cube" columns and the
scripted policy drops, add two more "odd cube size" columns and the reactive one does. Report
the profile; use a mean only as a summary of that specific table.

## 5. Cliffs and slopes: severity sweeps
A single strength per perturbation hides the shape of the failure. Sweeping the strength shows
whether a policy degrades gradually (a slope: there is margin, and you know how much) or
all at once (a cliff: fine until it suddenly is not). Hand-written rules tend to produce cliffs
at their assumptions; learned policies tend to produce slopes.

## 6. Hardware limits look like policy failures
Friction 0.5 gives 0% for every policy. Without the grip-force measurement from Stage 2 that
column reads as "nothing generalizes to slippery objects". With it, it reads as "this gripper
cannot hold this object", and the last column confirms it. Always ask whether *any* policy could
have succeeded.

## 7. The metric's tolerance hides differences
"Within 3 cm" made a policy with 12 mm median placement error look as good as one with 1.5 mm.
Re-scoring the same episodes at 1 cm and 5 mm costs nothing and showed what the RL reward had
traded away.

## 8. Reporting
A result is a number, the script that produced it, the seeds, the sample size and an interval.
A report also says what was tried and failed, what remains unexplained, and which of the
author's own conclusions turned out wrong. A reader can then decide how far to trust the rest.

## Check your understanding
1. The reactive policy scores 0% on the 5 cm cube and 100% once told the cube size. Is that a
   failure of generalization, of specification, or of testing? What would you change?
2. The 500-demo clone beats its own teacher on the 5 cm cube. How can a copy outperform the
   original, and why does DAgger undo it?
3. Latency costs the learned policies 26-44 points but not their teacher. Propose one experiment
   that would tell whether the cause is the delay itself or the errors the delay makes visible.
4. Why does 2 mm of sensing jitter raise the 100-demo clone's success?
5. You may add one column to the benchmark before a real-robot trial. Which, and why?
