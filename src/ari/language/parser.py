"""M19/M20: text -> (which cube, which target). Two models, to show what word order is for.

    BagOfWords  averages the word vectors: it knows WHICH words occurred, not their order.
    GRUParser   reads the words in sequence (a recurrent network): it can tell that in
                "the blue cube on the red target" blue belongs to the cube.

Both are trained from scratch on generated instructions and tested on the held-out splits of
language/grammar.py. The GRU parser is what the task planner (M20) uses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ari.envs.multi_object import CUBE_COLORS, TARGET_COLORS
from ari.language import grammar


class BagOfWords(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.emb = nn.Embedding(len(grammar.VOCAB), dim, padding_idx=grammar.PAD)
        self.body = nn.Sequential(nn.Linear(dim, 64), nn.ReLU())
        self.cube, self.target = nn.Linear(64, len(CUBE_COLORS)), nn.Linear(64, len(TARGET_COLORS))

    def forward(self, tokens):
        mask = (tokens != grammar.PAD).float().unsqueeze(-1)
        h = self.body((self.emb(tokens) * mask).sum(1) / mask.sum(1).clamp_min(1))
        return self.cube(h), self.target(h)


class GRUParser(nn.Module):
    def __init__(self, dim: int = 32, hidden: int = 64):
        super().__init__()
        self.emb = nn.Embedding(len(grammar.VOCAB), dim, padding_idx=grammar.PAD)
        self.gru = nn.GRU(dim, hidden, batch_first=True, bidirectional=True)
        self.cube, self.target = nn.Linear(2 * hidden, len(CUBE_COLORS)), nn.Linear(2 * hidden, len(TARGET_COLORS))

    def encode(self, tokens):
        """Sentence vector: the forward GRU's state at the last real word + the backward GRU's
        state at the first word (so padding never enters)."""
        out, _ = self.gru(self.emb(tokens))
        hid = out.shape[-1] // 2
        last = (tokens != grammar.PAD).sum(1).clamp_min(1) - 1
        fwd = out[torch.arange(len(tokens)), last, :hid]
        return torch.cat([fwd, out[:, 0, hid:]], dim=1)

    def forward(self, tokens):
        h = self.encode(tokens)
        return self.cube(h), self.target(h)


def make_batch(split: str, n: int, rng: np.random.Generator):
    rows = [grammar.sample(split, rng) for _ in range(n)]
    tokens = torch.as_tensor(np.stack([grammar.encode(r[0]) for r in rows]))
    cube = torch.as_tensor([CUBE_COLORS.index(r[1]) for r in rows])
    target = torch.as_tensor([TARGET_COLORS.index(r[2]) for r in rows])
    return tokens, cube, target


def train(model: nn.Module, steps: int = 1500, batch: int = 128, lr: float = 3e-3, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    model.train()
    for _ in range(steps):
        tokens, cube, target = make_batch("seen", batch, rng)        # fresh sentences every step
        lc, lt = model(tokens)
        loss = ce(lc, cube) + ce(lt, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


@torch.no_grad()
def accuracy(model: nn.Module, split: str, n: int = 2000, seed: int = 123) -> float:
    """Fraction of instructions where BOTH the cube and the target are right."""
    tokens, cube, target = make_batch(split, n, np.random.default_rng(seed))
    lc, lt = model(tokens)
    return float(((lc.argmax(1) == cube) & (lt.argmax(1) == target)).float().mean())


@torch.no_grad()
def parse(model: nn.Module, text: str):
    """Returns (cube colour, target colour, confidence in [0, 1])."""
    lc, lt = model(torch.as_tensor(grammar.encode(text)).unsqueeze(0))
    pc, pt = torch.softmax(lc, 1)[0], torch.softmax(lt, 1)[0]
    return CUBE_COLORS[int(pc.argmax())], TARGET_COLORS[int(pt.argmax())], float(pc.max() * pt.max())


def save(model: GRUParser, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "gru_parser", "state_dict": model.state_dict(), "vocab": grammar.VOCAB}, path)


def load(path: Path) -> GRUParser:
    ck = torch.load(path, weights_only=True)
    assert ck["vocab"] == grammar.VOCAB, "parser was trained with a different vocabulary"
    model = GRUParser()
    model.load_state_dict(ck["state_dict"])
    return model.eval()
