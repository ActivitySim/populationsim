# PopulationSim
# See full license in LICENSE.txt.

import copy

import numpy as np
import pytest

from populationsim.integerizing import lp_ortools
from tests.integerizer_lp_reference import (
    capture_solves,
    legacy_integerizer_ortools,
    legacy_simul_integerizer_ortools,
    simul_problem,
    single_problem,
)


def _run(func, args, kwargs):
    """Call func on a copy of the inputs; return its result, exported model and inputs."""
    args, kwargs = copy.deepcopy((args, kwargs))
    with capture_solves() as records:
        result = func(*args, **kwargs)
    assert len(records) == 1
    return result, records[0]["model"], (args, kwargs)


def _assert_same(legacy, new):
    (old_weights, old_status), old_model, old_inputs = legacy
    (new_weights, new_status), new_model, new_inputs = new
    assert new_model == old_model
    assert new_status == old_status
    assert new_weights.dtype == old_weights.dtype
    assert new_weights.shape == old_weights.shape
    assert np.array_equal(new_weights, old_weights)
    # inputs the integerizer mutates (simul importance weights) change identically
    for old_arg, new_arg in zip(old_inputs[0], new_inputs[0], strict=True):
        assert np.array_equal(old_arg, new_arg)


SINGLE_CASES = {
    "backstopped": dict(sample_count=300),
    "unbackstopped": dict(sample_count=300, backstopped=False),
    "total_hh_last": dict(sample_count=200, control_count=9, total_hh_control_index=8),
    "total_hh_middle": dict(sample_count=200, total_hh_control_index=4, seed=3),
    "many_zero_resid": dict(sample_count=150, zero_resid_share=0.6, seed=1),
    "tiny": dict(sample_count=3, control_count=4, seed=2),
    "float32": dict(sample_count=300, objective_dtype=np.float32, seed=4),
}


@pytest.mark.parametrize("case", SINGLE_CASES.values(), ids=SINGLE_CASES.keys())
def test_single_model_matches_legacy(case):
    kwargs = single_problem(**case)
    legacy = _run(legacy_integerizer_ortools, (), kwargs)
    new = _run(lp_ortools.np_integerizer_ortools, (), kwargs)
    _assert_same(legacy, new)
    assert new[0][1] == "OPTIMAL"


SIMUL_CASES = {
    "default": dict(sample_count=60, sub_zone_count=4),
    "parent_total_hh": dict(
        sample_count=60, sub_zone_count=3, total_hh_parent_control_index=2, seed=1
    ),
    "one_zone": dict(sample_count=80, sub_zone_count=1, seed=2),
    "many_zero_resid": dict(
        sample_count=50, sub_zone_count=5, zero_resid_share=0.8, seed=3
    ),
    "float32": dict(
        sample_count=60, sub_zone_count=4, objective_dtype=np.float32, seed=4
    ),
}


@pytest.mark.parametrize("case", SIMUL_CASES.values(), ids=SIMUL_CASES.keys())
def test_simul_model_matches_legacy(case):
    args = simul_problem(**case)
    legacy = _run(legacy_simul_integerizer_ortools, args, {})
    new = _run(lp_ortools.np_simul_integerizer_ortools, args, {})
    _assert_same(legacy, new)
    assert new[0][1] == "OPTIMAL"


def test_failed_solve_returns_inputs_unchanged():
    """An infeasible total households constraint falls back to the input weights."""
    kwargs = single_problem(sample_count=50, seed=5)
    kwargs["lp_right_hand_side"][kwargs["total_hh_control_index"]] = 10**6
    legacy = _run(legacy_integerizer_ortools, (), kwargs)
    new = _run(lp_ortools.np_integerizer_ortools, (), kwargs)
    _assert_same(legacy, new)
    assert new[0][1] == "INFEASIBLE"
    assert np.array_equal(new[0][0], kwargs["resid_weights"])
