"""M19: instructions. A small generated language with held-out test splits.

An instruction names one cube and one target zone: "put the blue cube on the red target".
There is no pretrained language model here; the vocabulary is whatever appears in the training
templates below (about 60 words). That makes the limits measurable:

    seen          a training template with a training colour pairing
    new phrasing  a template never used in training (some start with the target, one contains
                  a word that was never seen at all)
    new pairing   a (cube colour, target colour) combination never used in training

Cube and target colours overlap (red, blue), so the SAME words in a different order mean a
different task. A model that only counts words cannot solve this; see language/parser.py.
"""

from __future__ import annotations

import re

import numpy as np

from ari.envs.multi_object import CUBE_COLORS, TARGET_COLORS

CUBE_PHRASES = ["the {} cube", "the {} block", "the {} box", "{} cube", "the {} brick"]
TARGET_PHRASES = ["the {} target", "the {} zone", "the {} circle", "the {} spot", "{} target", "the {} area"]

TRAIN_TEMPLATES = [
    "put {c} on {t}",
    "place {c} on {t}",
    "move {c} to {t}",
    "set {c} down on {t}",
    "{c} goes on {t}",
    "pick up {c} and put it on {t}",
    "take {c} and place it on {t}",
    "i want {c} on {t}",
    "can you put {c} on {t}",
    "please move {c} onto {t}",
    "on {t} put {c}",
    "{t} is where {c} should go",
    "grab {c} then drop it on {t}",
    "bring {c} over to {t}",
    "{c} should be on {t}",
    "to {t} move {c}",
]
HELD_OUT_TEMPLATES = [
    "could you place {c} onto {t} please",
    "drop {c} on {t}",
    "lift {c} and set it on {t}",
    "{c} belongs on {t}",
    "onto {t} goes {c}",
    "make sure {c} ends up on {t}",
    "{t} should have {c} on it",
    "relocate {c} to {t}",                 # "relocate" never appears in training
]
ALL_PAIRS = [(c, t) for c in CUBE_COLORS for t in TARGET_COLORS]
# Both ("red", "blue") and ("blue", "red") must stay in training: they contain the same words,
# so they are the only examples that force a model to use word order. The first version of
# this file held out ("blue", "red"); both parsers then scored 100% on seen instructions and
# 48% on new pairings, having memorized "red and blue together => red cube, blue target"
# (devlog.md). FIRST_DESIGN_HELD_OUT is kept so that result can be reproduced.
HELD_OUT_PAIRS = [("yellow", "green"), ("blue", "blue")]
FIRST_DESIGN_HELD_OUT = [("blue", "red"), ("yellow", "green")]
TRAIN_PAIRS = [p for p in ALL_PAIRS if p not in HELD_OUT_PAIRS]

SPLITS = {
    "seen": (TRAIN_TEMPLATES, TRAIN_PAIRS),
    "new phrasing": (HELD_OUT_TEMPLATES, TRAIN_PAIRS),
    "new pairing": (TRAIN_TEMPLATES, HELD_OUT_PAIRS),
    # diagnostic: only the two instructions that differ in word order alone
    "order only": (TRAIN_TEMPLATES, [("red", "blue"), ("blue", "red")]),
}


def render(template: str, cube: str, target: str, rng: np.random.Generator) -> str:
    c = CUBE_PHRASES[rng.integers(len(CUBE_PHRASES))].format(cube)
    t = TARGET_PHRASES[rng.integers(len(TARGET_PHRASES))].format(target)
    return template.format(c=c, t=t)


def sample(split: str, rng: np.random.Generator, pair: tuple | None = None):
    """Returns (text, cube colour, target colour). `pair` fixes the colours (must belong to the split)."""
    templates, pairs = SPLITS[split]
    cube, target = pair if pair is not None else pairs[rng.integers(len(pairs))]
    return render(templates[rng.integers(len(templates))], cube, target, rng), cube, target


# ---------------------------------------------------------------- tokenizer
PAD, UNK = 0, 1


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def build_vocab() -> dict[str, int]:
    """Vocabulary = every word that can occur in TRAINING instructions. Anything else is <unk>."""
    words = set()
    for tpl in TRAIN_TEMPLATES:
        words.update(tokenize(tpl.replace("{c}", " ").replace("{t}", " ")))
    for ph in CUBE_PHRASES + TARGET_PHRASES:
        words.update(tokenize(ph.replace("{}", " ")))
    words.update(CUBE_COLORS + TARGET_COLORS)
    return {"<pad>": PAD, "<unk>": UNK, **{w: i + 2 for i, w in enumerate(sorted(words))}}


VOCAB = build_vocab()
MAX_LEN = 16


def encode(text: str) -> np.ndarray:
    ids = [VOCAB.get(w, UNK) for w in tokenize(text)][:MAX_LEN]
    return np.array(ids + [PAD] * (MAX_LEN - len(ids)), dtype=np.int64)
