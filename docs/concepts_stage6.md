# Concept notes: Stage 6 language and task-level autonomy

## 1. An instruction needs a choice: `envs/multi_object.py`
With one cube and one target, "put the cube on the target" carries no information. Three cubes
and three targets make language matter. Cube and target colours overlap (red, blue) so that the
same words in a different order name a different task.

**Focus.** Every earlier skill asks the environment "where is the cube / the target?".
`env.focus(cube, target)` decides which objects answer. Language chooses the focus; the skills
run unchanged. Reuse comes from a stable interface, not from rewriting skills.

## 2. Grounding: connecting words to things
Two different problems hide in "understanding an instruction":
- **Parsing:** which cube colour and which target colour does the sentence name? (text → symbols)
- **Grounding:** which object in the scene is that? (symbols → things)
`language/parser.py` does the first with labelled text. `language/policy.py` does both with no
labels at all: an attention layer matches a query computed from the sentence against each
object's colour tag, and is trained only through the action error.

## 3. Word order, and how a test can fail to test it: `language/grammar.py`
A bag-of-words model knows which words occurred; a recurrent network (GRU) reads them in order.
Whether a model *uses* order depends on the data: if only one of "red cube → blue target" and
"blue cube → red target" is in training, memorizing the word set is enough, and both models did
exactly that. A held-out split tests what the training set forces the model to learn. Design the
splits around the capability you want to measure.

## 4. Attention commits early: `language/policy.py`
Attention weights come from a softmax. Once it sits near 0/1 its gradient is nearly zero, so
whatever it settles on early is kept. Here every run's grounding after the first of 60 epochs
was within a few points of its final value: mostly right, exactly right, or (once) a consistent
wrong assignment. Two consequences. Start soft: normalize the sentence vector and keep the
initial query weights small, so the first commitment is made with some evidence. And expect
**seed variance**: which commitment a run makes depends on its random start, so the same code
gave 108, 144 and 146 successes out of 150. One training run is one sample; a conclusion drawn
from it is a conclusion about that sample.
At run time the attention is made hard (argmax): 1% of weight on a cube 15 cm away is a 1.5 mm error.

## 5. A plan is data; validate it before acting: `planning/task_planner.py`
The planner outputs a list of steps, not motion. Before anything moves, each step is checked
against the scene and the rest of the plan: understood? confident? target free? This gate does
not care whether the plan came from a 45-word network or a large language model. What it cannot
catch: a confidently wrong parse that is still a legal plan.

## 6. Post-conditions, monitoring, bounded recovery
"Put X on Y" is finished when X is *measured* on Y, not when the skill returns. After each step,
and again for all steps at the end (a later action can undo an earlier one), the executor checks
and retries from a known state, up to a limit, then stops and reports. Faults are injected on
purpose (a dropped cube, a knocked-off cube) because recovery that is never exercised is not tested.

## 7. Operating envelopes
A learned skill works where its training data was. After a mishap the scene is often somewhere
else (a cube inside the target area, 8 cm from its target), and retrying the same skill there
fails again. The classical skills have no such envelope. Hence: learned skill first, classical
fallback on failure. Knowing where a learned component should NOT be trusted is part of the design.

## Check your understanding
1. The first held-out pairing test gave both parsers exactly 48%. Without rerunning anything,
   explain where 48% comes from.
2. The language policy is never told which cube "the blue cube" is. Through which path does the
   action error reach the attention weights?
3. Validation refused 5 of 100 tasks; 2 refusals were justified and 3 were not. What would change
   if the confidence threshold were lowered from 0.8 to 0.5? What is the cost of each kind of error?
4. Monitoring recovered 100% of faults for the scripted skill but 43% for the learned one. Why does
   retrying help one and not the other?
5. Seed 0 of the language policy scored 108/150 and seeds 1 and 2 scored 144 and 146. Which
   statements about "end-to-end vs modular" can be made from seed 0 alone, and which need all three?
