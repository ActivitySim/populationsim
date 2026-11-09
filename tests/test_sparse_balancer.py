# PopulationSim
# See full license in LICENSE.txt.
"""
The numba balancers iterate over nonzero incidence entries only. These tests
check them against dense reference loops on sparse, categorical incidence
tables, including the cases where the bounds clamp touches zero-incidence
rows: initial weights outside the bounds, a first control that makes no
update, and (simultaneous balancer) weights pushed outside the bounds by the
rescale to parent weights.

The references below restate the dense numba loops in numpy. They are used
instead of ``np_balancer_py`` / ``np_simul_balancer_py`` because those clamp
weights at every control, including controls that make no update, and use a
tighter convergence threshold, so they differ from the numba balancers
whenever a weight starts outside its bounds.
"""

import numpy as np
import numpy.testing as npt
import pytest

from populationsim.balancing.balancers_numba import (
    incidence_to_sparse_rows,
    np_balancer_numba,
    np_simul_balancer_numba,
)
from populationsim.balancing.constants import (
    ALT_MAX_DELTA,
    IMPORTANCE_ADJUST,
    IMPORTANCE_ADJUST_COUNT,
    MAX_DELTA32,
    MAX_GAMMA,
    MAX_RELAXATION_FACTOR,
    MIN_CONTROL_VALUE,
    MIN_GAMMA,
    MIN_IMPORTANCE,
)


def control_order(control_count, master_control_index):
    order = [c for c in range(control_count) if c != master_control_index]
    if master_control_index >= 0:
        order.append(master_control_index)
    return order


def control_importance(controls_importance, c, master_control_index, adjustment):
    if c == master_control_index:
        return controls_importance[c]
    return max(controls_importance[c] * adjustment, MIN_IMPORTANCE)


def gamma_update(xx, yy, control, relaxation_factor, importance):
    relaxed = max(control * relaxation_factor, MIN_CONTROL_VALUE)
    return max(1.0 - (xx - relaxed) / (yy + relaxed / importance), MIN_GAMMA)


def dense_reference_balancer(
    sample_count,
    control_count,
    master_control_index,
    incidence,
    weights_initial,
    weights_lower_bound,
    weights_upper_bound,
    controls_constraint,
    controls_importance,
    max_iterations,
):
    weights = weights_initial.astype(np.float64)
    relaxation_factors = np.ones(control_count)
    adjustment = 1.0
    for iter in range(max_iterations):
        delta = 0.0
        gamma = np.ones(control_count)
        if iter > 0 and iter % IMPORTANCE_ADJUST_COUNT == 0:
            adjustment /= IMPORTANCE_ADJUST
        for c in control_order(control_count, master_control_index):
            xx = (weights * incidence[c]).sum()
            yy = (weights * incidence[c] * incidence[c]).sum()
            importance = control_importance(
                controls_importance, c, master_control_index, adjustment
            )
            if xx > 0.0:
                gamma[c] = gamma_update(
                    xx, yy, controls_constraint[c], relaxation_factors[c], importance
                )
                updated = np.clip(
                    weights * np.exp(np.log(gamma[c]) * incidence[c]),
                    weights_lower_bound,
                    weights_upper_bound,
                )
                delta += np.abs(updated - weights).sum()
                weights = updated
                relaxation_factors[c] = min(
                    relaxation_factors[c] * (1.0 / gamma[c]) ** (1.0 / importance),
                    MAX_RELAXATION_FACTOR,
                )
        delta /= sample_count
        max_gamma_dif = np.abs(gamma - 1.0).max()
        if (delta < MAX_DELTA32 and max_gamma_dif < MAX_GAMMA) or delta < ALT_MAX_DELTA:
            return weights, relaxation_factors, (True, iter, delta, max_gamma_dif)
    return weights, relaxation_factors, (False, max_iterations, delta, max_gamma_dif)


def dense_reference_simul_balancer(
    sample_count,
    control_count,
    zone_count,
    master_control_index,
    incidence,
    parent_weights,
    weights_lower_bound,
    weights_upper_bound,
    sub_weights,
    parent_controls,
    controls_importance,
    sub_controls,
    max_iterations,
):
    totals = sub_controls.sum(axis=0)
    scale = np.where(totals > 0, parent_controls / np.where(totals > 0, totals, 1), 1)
    sub_controls = sub_controls * scale
    weights = sub_weights.astype(np.float64)
    relaxation_factors = np.ones((zone_count, control_count))
    adjustment = 1.0
    for iter in range(max_iterations):
        weights_previous = weights.copy()
        gamma = np.ones((zone_count, control_count))
        if iter > 0 and iter % IMPORTANCE_ADJUST_COUNT == 0:
            adjustment /= IMPORTANCE_ADJUST
        for c in control_order(control_count, master_control_index):
            importance = control_importance(
                controls_importance, c, master_control_index, adjustment
            )
            for z in range(zone_count):
                xx = (weights[z] * incidence[c]).sum()
                yy = (weights[z] * incidence[c] * incidence[c]).sum()
                if xx > 0.0:
                    gamma[z, c] = gamma_update(
                        xx,
                        yy,
                        sub_controls[z, c],
                        relaxation_factors[z, c],
                        importance,
                    )
                    weights[z] = np.clip(
                        weights[z] * np.exp(np.log(gamma[z, c]) * incidence[c]),
                        weights_lower_bound,
                        weights_upper_bound,
                    )
                    relaxation_factors[z, c] = min(
                        relaxation_factors[z, c]
                        * (1.0 / gamma[z, c]) ** (1.0 / importance),
                        MAX_RELAXATION_FACTOR,
                    )
        zone_sums = weights.sum(axis=0)
        weights *= np.where(
            zone_sums > 0, parent_weights / np.where(zone_sums > 0, zone_sums, 1), 1.0
        )
        max_gamma_dif = np.abs(gamma - 1).max()
        delta = np.abs(weights - weights_previous).sum() / sample_count
        converged = delta < MAX_DELTA32 and max_gamma_dif < MAX_GAMMA
        if converged or delta < ALT_MAX_DELTA:
            return weights, relaxation_factors, (converged, iter, delta, max_gamma_dif)
    return weights, relaxation_factors, (False, max_iterations, delta, max_gamma_dif)


def categorical_incidence(sample_count, rng):
    """
    Incidence table shaped like a household seed: one-hot household
    categories plus person counts by category, with an all-zero control
    first so the first control in the iteration order makes no update.
    """
    blocks = [np.zeros((1, sample_count))]
    blocks.append(np.ones((1, sample_count)))
    for category_count in (8, 5, 4):
        category = rng.integers(0, category_count, sample_count)
        blocks.append(np.eye(category_count)[category].T)
    persons = rng.integers(1, 6, sample_count)
    for category_count in (6, 4):
        counts = np.zeros((category_count, sample_count))
        for j in range(sample_count):
            np.add.at(counts[:, j], rng.integers(0, category_count, persons[j]), 1.0)
        blocks.append(counts)
    return np.vstack(blocks)


def single_problem(seed, initial_out_of_bounds):
    rng = np.random.default_rng(seed)
    sample_count = 400
    incidence = categorical_incidence(sample_count, rng)
    control_count = incidence.shape[0]
    weights_initial = rng.uniform(5.0, 50.0, sample_count)
    weights_lower_bound = weights_initial * 0.5
    weights_upper_bound = weights_initial * 3.0
    if initial_out_of_bounds:
        weights_lower_bound[::7] = weights_initial[::7] * 1.5
        weights_upper_bound[1::7] = weights_initial[1::7] * 0.75
    controls_constraint = (incidence * weights_initial).sum(axis=1) * rng.uniform(
        0.8, 1.2, control_count
    )
    controls_importance = rng.uniform(10.0, 1000.0, control_count)
    return (
        sample_count,
        control_count,
        1,
        incidence,
        weights_initial,
        weights_lower_bound,
        weights_upper_bound,
        controls_constraint,
        controls_importance,
    )


def simul_problem(seed, sample_count=300, zone_count=4):
    rng = np.random.default_rng(seed)
    incidence = categorical_incidence(sample_count, rng)
    control_count = incidence.shape[0]
    parent_weights = rng.uniform(5.0, 50.0, sample_count)
    zone_shares = rng.dirichlet(np.ones(zone_count), sample_count).T
    sub_weights = parent_weights * zone_shares
    weights_lower_bound = parent_weights * 0.05
    weights_upper_bound = parent_weights * 0.9
    sub_controls = (sub_weights[:, None, :] * incidence[None, :, :]).sum(
        axis=2
    ) * rng.uniform(0.7, 1.3, (zone_count, control_count))
    parent_controls = sub_controls.sum(axis=0)
    controls_importance = rng.uniform(10.0, 1000.0, control_count)
    return (
        sample_count,
        control_count,
        zone_count,
        1,
        incidence,
        parent_weights,
        weights_lower_bound,
        weights_upper_bound,
        sub_weights,
        parent_controls,
        controls_importance,
        sub_controls,
    )


def copy_args(args):
    return tuple(a.copy() if isinstance(a, np.ndarray) else a for a in args)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_incidence_to_sparse_rows_round_trip(dtype):
    rng = np.random.default_rng(0)
    incidence = categorical_incidence(200, rng).astype(dtype)
    incidence[3, 5] = 0.3

    row_start, sample_index, value = incidence_to_sparse_rows(incidence)

    rebuilt = np.zeros(incidence.shape)
    for c in range(incidence.shape[0]):
        entries = slice(row_start[c], row_start[c + 1])
        assert np.all(np.diff(sample_index[entries]) > 0)
        rebuilt[c, sample_index[entries]] = value[entries]
    npt.assert_array_equal(rebuilt, incidence.astype(np.float64))
    assert value.dtype == incidence.dtype
    assert row_start[0] == 0 and row_start[1] == 0
    assert row_start[-1] == np.count_nonzero(incidence)


@pytest.mark.parametrize("initial_out_of_bounds", [False, True])
@pytest.mark.parametrize("iterations", [1, 3, 150, 10000])
def test_single_balancer_matches_dense_reference(initial_out_of_bounds, iterations):
    args = single_problem(1, initial_out_of_bounds)

    w_numba, r_numba, s_numba = np_balancer_numba(*copy_args(args), iterations)
    w_ref, r_ref, s_ref = dense_reference_balancer(*copy_args(args), iterations)

    npt.assert_allclose(w_numba, w_ref, rtol=1e-9)
    npt.assert_allclose(r_numba, r_ref, rtol=1e-9)
    assert s_numba[:2] == s_ref[:2]
    lower, upper = args[5], args[6]
    assert np.all(w_numba >= lower) and np.all(w_numba <= upper)


@pytest.mark.parametrize("iterations", [1, 3, 150])
def test_simul_balancer_matches_dense_reference(iterations):
    args = simul_problem(2)
    parent_weights, lower, sub_weights = args[5], args[6], args[8]
    assert np.any(sub_weights < lower), "test needs out-of-bounds sub weights"

    w_numba, r_numba, s_numba = np_simul_balancer_numba(*copy_args(args), iterations)
    w_ref, r_ref, s_ref = dense_reference_simul_balancer(*copy_args(args), iterations)

    npt.assert_allclose(w_numba, w_ref, rtol=1e-9)
    npt.assert_allclose(r_numba, r_ref, rtol=1e-9)
    npt.assert_allclose(w_numba.sum(axis=0), parent_weights, rtol=1e-12)


def test_results_do_not_depend_on_thread_count():
    import numba

    from populationsim.balancing.balancers_numba import PARALLEL_MIN_ENTRIES

    rng = np.random.default_rng(3)
    sample_count = PARALLEL_MIN_ENTRIES + 1234
    incidence = categorical_incidence(sample_count, rng)
    assert np.count_nonzero(incidence[1]) >= PARALLEL_MIN_ENTRIES
    weights = rng.uniform(5.0, 50.0, sample_count)
    controls = (incidence * weights).sum(axis=1) * rng.uniform(
        0.8, 1.2, incidence.shape[0]
    )
    single_args = (
        sample_count,
        incidence.shape[0],
        1,
        incidence,
        weights,
        weights * 0.5,
        weights * 3.0,
        controls,
        rng.uniform(10.0, 1000.0, incidence.shape[0]),
    )
    simul_args = simul_problem(4, sample_count=PARALLEL_MIN_ENTRIES // 4, zone_count=6)
    assert np.count_nonzero(simul_args[4][1]) * 6 >= PARALLEL_MIN_ENTRIES

    threads = numba.get_num_threads()
    try:
        results = []
        for count in (1, threads):
            numba.set_num_threads(count)
            results.append(
                (
                    np_balancer_numba(*copy_args(single_args), 20),
                    np_simul_balancer_numba(*copy_args(simul_args), 20),
                )
            )
    finally:
        numba.set_num_threads(threads)

    (single_1, simul_1), (single_n, simul_n) = results
    for one, many in ((single_1, single_n), (simul_1, simul_n)):
        npt.assert_array_equal(one[0], many[0])
        npt.assert_array_equal(one[1], many[1])
        assert one[2] == many[2]
