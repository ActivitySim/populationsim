"""Numba list balancers that iterate over the nonzero incidence entries only.

Most incidence tables are sparse: a household falls in one income band, one
household-size band, one age band per person, and so on. The balancing
update for a control only changes the weights of rows with a nonzero
incidence for that control (``exp(log_gamma * 0) == 1``), and those rows are
the only ones that contribute to the control's weighted sums. Each control's
nonzero entries are therefore gathered once into a compressed-row layout and
the iterations visit only those entries. For entries equal to 1, the update
factor ``exp(log_gamma * 1)`` is ``gamma`` itself, so ``exp`` is skipped.

The one place where a zero-incidence row is changed is the bounds clamp: if a
weight is outside ``[weights_lower_bound, weights_upper_bound]``, the dense
loop clamps it at the first control that updates weights. That happens to the
initial weights and, in the simultaneous balancer, after the per-iteration
rescale to the parent weights. Weights are checked against the bounds at
those points, and if any is out of bounds the first updating control runs
over every row, exactly as before. The algorithm and its results are
unchanged; only floating-point rounding can differ.

Large controls are split across Numba threads. In ``np_balancer_numba`` each
control's entries are cut into fixed chunks of ``CHUNK_SIZE`` whose partial
sums are added in chunk order; in ``np_simul_balancer_numba`` the sub-zones,
which share no weights, are updated in parallel. Either way the arithmetic
does not depend on the thread count, so one thread and many threads give
identical results.
"""

import logging
import numpy as np
import numba
from numba import njit, prange
from populationsim.balancing.constants import (
    DEFAULT_MAX_ITERATIONS,
    MAX_DELTA32,
    MAX_GAMMA,
    MIN_GAMMA,
    IMPORTANCE_ADJUST,
    IMPORTANCE_ADJUST_COUNT,
    MIN_IMPORTANCE,
    MAX_RELAXATION_FACTOR,
    MIN_CONTROL_VALUE,
    ALT_MAX_DELTA,
)

logger = logging.getLogger(__name__)

# Entries per chunk when a control's sums are split across threads. The
# chunks are fixed by position and their partial sums are added in chunk
# order, so results do not depend on the number of threads.
CHUNK_SIZE = 16384

# Loops with fewer entries than this run on the calling thread; starting a
# parallel loop costs more than it saves on small controls. With one Numba
# thread every loop runs on the calling thread. The serial and parallel paths
# do the same arithmetic in the same order.
PARALLEL_MIN_ENTRIES = 4 * CHUNK_SIZE
SERIAL_ONLY = np.iinfo(np.int64).max


def _parallel_min_entries():
    return PARALLEL_MIN_ENTRIES if numba.get_num_threads() > 1 else SERIAL_ONLY


@njit(cache=True)
def incidence_to_sparse_rows(incidence: np.ndarray):
    """
    Gather the nonzero entries of each control row of a dense incidence matrix.

    Parameters
    ----------
    incidence : numpy.ndarray
        2-D array with one row per control and one column per sample.

    Returns
    -------
    row_start : numpy.ndarray of int64, shape (control_count + 1,)
        Entries of control ``c`` are at positions ``row_start[c]`` to
        ``row_start[c + 1] - 1`` of the other arrays.
    sample_index : numpy.ndarray of int32
        Sample (column) index of each nonzero entry, ascending within a control.
    value : numpy.ndarray
        Incidence value of each nonzero entry, in the dtype of ``incidence``.
    """
    control_count, sample_count = incidence.shape
    row_start = np.zeros(control_count + 1, dtype=np.int64)
    for c in range(control_count):
        count = 0
        for j in range(sample_count):
            if incidence[c, j] != 0:
                count += 1
        row_start[c + 1] = row_start[c] + count

    nonzero_count = row_start[control_count]
    sample_index = np.empty(nonzero_count, dtype=np.int32)
    value = np.empty(nonzero_count, dtype=incidence.dtype)
    k = 0
    for c in range(control_count):
        for j in range(sample_count):
            if incidence[c, j] != 0:
                sample_index[k] = j
                value[k] = incidence[c, j]
                k += 1

    return row_start, sample_index, value


@njit(cache=True)
def _weights_within_bounds(weights, weights_lower_bound, weights_upper_bound):
    for j in range(weights.shape[0]):
        w = float(weights[j])
        if w < float(weights_lower_bound[j]) or w > float(weights_upper_bound[j]):
            return False
    return True


@njit(cache=True)
def _control_order(control_count, master_control_index):
    control_indexes = np.empty(control_count, dtype=np.int32)
    k = 0
    for i in range(control_count):
        if i != master_control_index:
            control_indexes[k] = i
            k += 1
    if master_control_index >= 0:
        control_indexes[k] = master_control_index
    return control_indexes


@njit(fastmath=True, cache=True)
def _chunk_sums(weights, sample_index, value, lo, hi):
    xx = 0.0
    yy = 0.0
    for t in range(lo, hi):
        w = float(weights[sample_index[t]])
        inc = value[t]
        xx += w * float(inc)
        yy += w * float(inc * inc)
    return xx, yy


@njit(fastmath=True, cache=True)
def _chunk_update(
    weights,
    sample_index,
    value,
    weights_lower_bound,
    weights_upper_bound,
    lo,
    hi,
    gamma_val,
    log_gamma,
):
    delta = 0.0
    for t in range(lo, hi):
        j = sample_index[t]
        w_old = float(weights[j])
        inc = float(value[t])
        factor = gamma_val if inc == 1.0 else np.exp(log_gamma * inc)
        new_w = w_old * factor

        lb = float(weights_lower_bound[j])
        ub = float(weights_upper_bound[j])
        new_w = min(max(new_w, lb), ub)

        delta += abs(new_w - w_old)
        weights[j] = new_w
    return delta


@njit(fastmath=True, cache=True, parallel=True)
def _sparse_sums(
    weights, sample_index, value, start, end, partial_xx, partial_yy, parallel_min
):
    chunk_count = (end - start + CHUNK_SIZE - 1) // CHUNK_SIZE
    if end - start >= parallel_min:
        for k in prange(chunk_count):
            lo = start + k * CHUNK_SIZE
            partial_xx[k], partial_yy[k] = _chunk_sums(
                weights, sample_index, value, lo, min(lo + CHUNK_SIZE, end)
            )
    else:
        for k in range(chunk_count):
            lo = start + k * CHUNK_SIZE
            partial_xx[k], partial_yy[k] = _chunk_sums(
                weights, sample_index, value, lo, min(lo + CHUNK_SIZE, end)
            )
    xx = 0.0
    yy = 0.0
    for k in range(chunk_count):
        xx += partial_xx[k]
        yy += partial_yy[k]
    return xx, yy


@njit(fastmath=True, cache=True, parallel=True)
def _sparse_update(
    weights,
    sample_index,
    value,
    weights_lower_bound,
    weights_upper_bound,
    start,
    end,
    gamma_val,
    log_gamma,
    partial_delta,
    parallel_min,
):
    chunk_count = (end - start + CHUNK_SIZE - 1) // CHUNK_SIZE
    if end - start >= parallel_min:
        for k in prange(chunk_count):
            lo = start + k * CHUNK_SIZE
            partial_delta[k] = _chunk_update(
                weights,
                sample_index,
                value,
                weights_lower_bound,
                weights_upper_bound,
                lo,
                min(lo + CHUNK_SIZE, end),
                gamma_val,
                log_gamma,
            )
    else:
        for k in range(chunk_count):
            lo = start + k * CHUNK_SIZE
            partial_delta[k] = _chunk_update(
                weights,
                sample_index,
                value,
                weights_lower_bound,
                weights_upper_bound,
                lo,
                min(lo + CHUNK_SIZE, end),
                gamma_val,
                log_gamma,
            )
    delta = 0.0
    for k in range(chunk_count):
        delta += partial_delta[k]
    return delta


def np_balancer_numba(
    sample_count: int,
    control_count: int,
    master_control_index: int,
    incidence: np.ndarray,
    weights_initial: np.ndarray,
    weights_lower_bound: np.ndarray,
    weights_upper_bound: np.ndarray,
    controls_constraint: np.ndarray,
    controls_importance: np.ndarray,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> tuple[np.ndarray, np.ndarray, tuple[bool, int, float, float]]:
    return _balance(
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
        _parallel_min_entries(),
    )


@njit(fastmath=True, cache=True)
def _balance(
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
    parallel_min,
):
    weights_final = weights_initial.copy()
    relaxation_factors = np.ones(control_count, dtype=np.float64)
    importance_adjustment = 1.0

    row_start, sample_index, value = incidence_to_sparse_rows(incidence)
    control_indexes = _control_order(control_count, master_control_index)

    max_chunks = 1
    for c in range(control_count):
        chunks = (row_start[c + 1] - row_start[c] + CHUNK_SIZE - 1) // CHUNK_SIZE
        max_chunks = max(max_chunks, chunks)
    partial_xx = np.zeros(max_chunks)
    partial_yy = np.zeros(max_chunks)
    partial_delta = np.zeros(max_chunks)

    # Weights only change through clamped updates, so once every weight is
    # within bounds the clamp is a no-op for zero-incidence rows.
    within_bounds = _weights_within_bounds(
        weights_final, weights_lower_bound, weights_upper_bound
    )

    for iter in range(max_iterations):
        delta = 0.0  # float64
        gamma = np.ones(control_count, dtype=np.float64)

        if iter > 0 and iter % IMPORTANCE_ADJUST_COUNT == 0:
            importance_adjustment /= IMPORTANCE_ADJUST

        for i in range(control_count):
            c = control_indexes[i]
            start = row_start[c]
            end = row_start[c + 1]

            xx, yy = _sparse_sums(
                weights_final,
                sample_index,
                value,
                start,
                end,
                partial_xx,
                partial_yy,
                parallel_min,
            )

            imp = (
                float(controls_importance[c])
                if c == master_control_index
                else max(
                    float(controls_importance[c]) * importance_adjustment,
                    MIN_IMPORTANCE,
                )
            )

            if xx > 0.0:
                relaxed = float(controls_constraint[c]) * relaxation_factors[c]
                if relaxed < MIN_CONTROL_VALUE:
                    relaxed = MIN_CONTROL_VALUE

                gamma_val = 1.0 - (xx - relaxed) / (yy + relaxed / imp)
                gamma_val = max(gamma_val, MIN_GAMMA)
                gamma[c] = gamma_val
                log_gamma = np.log(gamma_val)

                if within_bounds:
                    delta += _sparse_update(
                        weights_final,
                        sample_index,
                        value,
                        weights_lower_bound,
                        weights_upper_bound,
                        start,
                        end,
                        gamma_val,
                        log_gamma,
                        partial_delta,
                        parallel_min,
                    )
                else:
                    for j in range(sample_count):
                        w_old = float(weights_final[j])
                        inc = float(incidence[c, j])
                        new_w = w_old * np.exp(log_gamma * inc)

                        lb = float(weights_lower_bound[j])
                        ub = float(weights_upper_bound[j])
                        new_w = min(max(new_w, lb), ub)

                        delta += abs(new_w - w_old)
                        weights_final[j] = new_w
                    within_bounds = True

                relax_factor = relaxation_factors[c] * (1.0 / gamma_val) ** (1.0 / imp)
                if relax_factor > MAX_RELAXATION_FACTOR:
                    relax_factor = MAX_RELAXATION_FACTOR
                relaxation_factors[c] = relax_factor

        delta /= sample_count
        max_gamma_dif = 0.0
        for i in range(control_count):
            g_dif = abs(gamma[i] - 1.0)
            if g_dif > max_gamma_dif:
                max_gamma_dif = g_dif

        converged = delta < MAX_DELTA32 and max_gamma_dif < MAX_GAMMA
        no_progress = delta < ALT_MAX_DELTA

        if converged or no_progress:
            return weights_final, relaxation_factors, (True, iter, delta, max_gamma_dif)

    return (
        weights_final,
        relaxation_factors,
        (False, max_iterations, delta, max_gamma_dif),
    )


@njit(fastmath=True, cache=True)
def _simul_zone_update(
    z,
    c,
    importance,
    sample_count,
    start,
    end,
    sample_index,
    value,
    incidence,
    sub_weights,
    sub_controls,
    relaxation_factors,
    gamma,
    within_bounds,
    weights_lower_bound,
    weights_upper_bound,
):
    xx = 0.0
    yy = 0.0
    for t in range(start, end):
        w = float(sub_weights[z, sample_index[t]])
        inc = value[t]
        xx += w * float(inc)
        yy += w * float(inc * inc)

    if xx > 0.0:
        relaxed = float(sub_controls[z, c]) * relaxation_factors[z, c]
        if relaxed < MIN_CONTROL_VALUE:
            relaxed = MIN_CONTROL_VALUE

        gamma_val = 1.0 - (xx - relaxed) / (yy + relaxed / importance)
        gamma_val = max(gamma_val, MIN_GAMMA)
        gamma[z, c] = gamma_val
        log_gamma = np.log(gamma_val)

        if within_bounds[z]:
            for t in range(start, end):
                j = sample_index[t]
                w_old = float(sub_weights[z, j])
                inc = float(value[t])
                factor = gamma_val if inc == 1.0 else np.exp(log_gamma * inc)
                new_w = w_old * factor

                lb = float(weights_lower_bound[j])
                ub = float(weights_upper_bound[j])
                new_w = min(max(new_w, lb), ub)
                sub_weights[z, j] = new_w
        else:
            for j in range(sample_count):
                w_old = float(sub_weights[z, j])
                inc = float(incidence[c, j])
                new_w = w_old * np.exp(log_gamma * inc)

                lb = float(weights_lower_bound[j])
                ub = float(weights_upper_bound[j])
                new_w = min(max(new_w, lb), ub)
                sub_weights[z, j] = new_w
            within_bounds[z] = True

        relax_factor = relaxation_factors[z, c] * (1.0 / gamma_val) ** (
            1.0 / importance
        )
        relaxation_factors[z, c] = min(relax_factor, MAX_RELAXATION_FACTOR)


@njit(fastmath=True, cache=True, parallel=True)
def _simul_control_update(
    c,
    importance,
    zone_count,
    sample_count,
    start,
    end,
    sample_index,
    value,
    incidence,
    sub_weights,
    sub_controls,
    relaxation_factors,
    gamma,
    within_bounds,
    weights_lower_bound,
    weights_upper_bound,
):
    # Zones share no weights, so they update independently and the result
    # does not depend on the number of threads.
    for z in prange(zone_count):
        _simul_zone_update(
            z,
            c,
            importance,
            sample_count,
            start,
            end,
            sample_index,
            value,
            incidence,
            sub_weights,
            sub_controls,
            relaxation_factors,
            gamma,
            within_bounds,
            weights_lower_bound,
            weights_upper_bound,
        )


def np_simul_balancer_numba(
    sample_count: int,
    control_count: int,
    zone_count: int,
    master_control_index: int,
    incidence: np.ndarray,
    parent_weights: np.ndarray,
    weights_lower_bound: np.ndarray,
    weights_upper_bound: np.ndarray,
    sub_weights: np.ndarray,
    parent_controls: np.ndarray,
    controls_importance: np.ndarray,
    sub_controls: np.ndarray,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> tuple[np.ndarray, np.ndarray, tuple[bool, int, float, float]]:
    return _simul_balance(
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
        _parallel_min_entries(),
    )


@njit(fastmath=True, cache=True)
def _simul_balance(
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
    parallel_min,
):
    if parent_controls is not None:
        # Normalize sub_controls to match parent_controls if provided
        totals = sub_controls.sum(axis=0)
        scale_factors = np.ones_like(totals, dtype=np.float64)
        valid = totals > 0
        scale_factors[valid] = parent_controls[valid] / totals[valid]
        sub_controls *= scale_factors

    relaxation_factors = np.ones((zone_count, control_count), dtype=np.float64)
    importance_adjustment = 1.0

    row_start, sample_index, value = incidence_to_sparse_rows(incidence)
    control_indexes = _control_order(control_count, master_control_index)

    # The rescale to parent weights at the end of each iteration can move
    # weights outside their bounds, so each zone is checked before every
    # iteration rather than once.
    within_bounds = np.empty(zone_count, dtype=np.bool_)
    for z in range(zone_count):
        within_bounds[z] = _weights_within_bounds(
            sub_weights[z], weights_lower_bound, weights_upper_bound
        )

    for iter in range(max_iterations):
        weights_previous = sub_weights.copy()
        gamma = np.ones((zone_count, control_count), dtype=np.float64)

        if iter > 0 and iter % IMPORTANCE_ADJUST_COUNT == 0:
            importance_adjustment /= IMPORTANCE_ADJUST

        for i in range(control_count):
            c = control_indexes[i]

            if c == master_control_index:
                importance = float(controls_importance[c])
            else:
                importance = max(
                    float(controls_importance[c]) * importance_adjustment,
                    MIN_IMPORTANCE,
                )

            start = row_start[c]
            end = row_start[c + 1]
            if zone_count > 1 and (end - start) * zone_count >= parallel_min:
                _simul_control_update(
                    c,
                    importance,
                    zone_count,
                    sample_count,
                    start,
                    end,
                    sample_index,
                    value,
                    incidence,
                    sub_weights,
                    sub_controls,
                    relaxation_factors,
                    gamma,
                    within_bounds,
                    weights_lower_bound,
                    weights_upper_bound,
                )
            else:
                for z in range(zone_count):
                    _simul_zone_update(
                        z,
                        c,
                        importance,
                        sample_count,
                        start,
                        end,
                        sample_index,
                        value,
                        incidence,
                        sub_weights,
                        sub_controls,
                        relaxation_factors,
                        gamma,
                        within_bounds,
                        weights_lower_bound,
                        weights_upper_bound,
                    )

        # Rescale
        zone_sums = np.sum(sub_weights, axis=0)
        for j in range(sample_count):
            scale = parent_weights[j] / zone_sums[j] if zone_sums[j] > 0 else 1.0
            for z in range(zone_count):
                sub_weights[z, j] *= scale

        for z in range(zone_count):
            within_bounds[z] = _weights_within_bounds(
                sub_weights[z], weights_lower_bound, weights_upper_bound
            )

        max_gamma_dif = np.abs(gamma - 1).max()
        delta = np.abs(sub_weights - weights_previous).sum() / float(sample_count)

        converged = delta < MAX_DELTA32 and max_gamma_dif < MAX_GAMMA
        no_progress = delta < ALT_MAX_DELTA

        if converged or no_progress:
            return (
                sub_weights,
                relaxation_factors,
                (converged, iter, delta, max_gamma_dif),
            )

    return (
        sub_weights,
        relaxation_factors,
        (False, max_iterations, delta, max_gamma_dif),
    )
