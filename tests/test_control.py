"""Joint control behaves as the theory predicts, and simulation is deterministic."""

import numpy as np
import pytest

from ari.control.joint import get_pd_gains, run_targets, set_pd_gains
from ari.robot.model import load_panda, reset_to_home


@pytest.fixture
def panda():
    model, data = load_panda()
    reset_to_home(model, data)
    return model, data


def hold_home(model, data, gravity_comp):
    q_home = data.ctrl[:7].copy()
    tr = run_targets(model, data, lambda t: q_home, 1.0, gravity_comp=gravity_comp)
    return np.abs(tr.q[-1] - q_home)


def test_gravity_sag_without_compensation(panda):
    """A pure PD servo must sag under gravity: error ~ gravity torque / kp."""
    err = hold_home(*panda, gravity_comp=False)
    assert err.max() > 1e-3


def test_gravity_compensation_removes_sag(panda):
    err = hold_home(*panda, gravity_comp=True)
    assert err.max() < 1e-5


def test_softer_gains_sag_more(panda):
    model, data = panda
    stiff = hold_home(model, data, gravity_comp=False).max()
    kp, kd = get_pd_gains(model)
    set_pd_gains(model, kp * 0.2, kd * 0.2)
    reset_to_home(model, data)
    soft = hold_home(model, data, gravity_comp=False).max()
    assert soft > 3 * stiff


def test_simulation_is_deterministic():
    """Same inputs, same outputs. Reproducible experiments depend on this."""
    finals = []
    for _ in range(2):
        model, data = load_panda()
        reset_to_home(model, data)
        q0 = data.ctrl[:7].copy()
        tr = run_targets(model, data, lambda t: q0 + 0.2 * np.sin(3 * t), 1.0)
        finals.append(tr.q[-1])
    assert np.array_equal(finals[0], finals[1])
