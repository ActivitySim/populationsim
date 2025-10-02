import logging
import numpy as np
from numba import njit
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


@njit(fastmath=True, cache=True)
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
    # Upcast key scalars to float64 for stability
    weights_final = weights_initial.copy()
    relaxation_factors = np.empty(control_count, dtype=np.float64)

    for i in range(control_count):
        relaxation_factors[i] = 1.0

    # Precompute incidence squared
    incidence2 = incidence * incidence
    importance_adjustment = 1.0

    # Manual control reordering
    control_indexes = np.empty(control_count, dtype=np.int32)
    k = 0
    for i in range(control_count):
        if i != master_control_index:
            control_indexes[k] = i
            k += 1
    if master_control_index >= 0:
        control_indexes[k] = master_control_index

    for iter in range(max_iterations):
        delta = 0.0  # float64
        gamma = np.ones(control_count, dtype=np.float64)

        if iter > 0 and iter % IMPORTANCE_ADJUST_COUNT == 0:
            importance_adjustment /= IMPORTANCE_ADJUST

        for i in range(control_count):
            c = control_indexes[i]
            xx = 0.0
            yy = 0.0
            for j in range(sample_count):
                w = float(weights_final[j])
                inc = float(incidence[c, j])
                xx += w * inc
                yy += w * float(incidence2[c, j])

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

                for j in range(sample_count):
                    w_old = float(weights_final[j])
                    inc = float(incidence[c, j])
                    new_w = w_old * np.exp(log_gamma * inc)

                    lb = float(weights_lower_bound[j])
                    ub = float(weights_upper_bound[j])
                    new_w = min(max(new_w, lb), ub)

                    delta += abs(new_w - w_old)
                    weights_final[j] = new_w

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
    if parent_controls is not None:
        # Normalize sub_controls to match parent_controls if provided
        totals = sub_controls.sum(axis=0)
        scale_factors = np.ones_like(totals, dtype=np.float64)
        valid = totals > 0
        scale_factors[valid] = parent_controls[valid] / totals[valid]
        sub_controls *= scale_factors

    relaxation_factors = np.ones((zone_count, control_count), dtype=np.float64)
    incidence2 = incidence * incidence
    importance_adjustment = 1.0

    control_indexes = np.empty(control_count, dtype=np.int32)
    k = 0
    for i in range(control_count):
        if i != master_control_index:
            control_indexes[k] = i
            k += 1
    if master_control_index >= 0:
        control_indexes[k] = master_control_index

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

            for z in range(zone_count):
                xx = 0.0
                yy = 0.0
                for j in range(sample_count):
                    w = float(sub_weights[z, j])
                    inc = float(incidence[c, j])
                    xx += w * inc
                    yy += w * float(incidence2[c, j])

                if xx > 0.0:
                    relaxed = float(sub_controls[z, c]) * relaxation_factors[z, c]
                    if relaxed < MIN_CONTROL_VALUE:
                        relaxed = MIN_CONTROL_VALUE

                    gamma_val = 1.0 - (xx - relaxed) / (yy + relaxed / importance)
                    gamma_val = max(gamma_val, MIN_GAMMA)
                    gamma[z, c] = gamma_val
                    log_gamma = np.log(gamma_val)

                    for j in range(sample_count):
                        w_old = float(sub_weights[z, j])
                        inc = float(incidence[c, j])
                        new_w = w_old * np.exp(log_gamma * inc)

                        lb = float(weights_lower_bound[j])
                        ub = float(weights_upper_bound[j])
                        new_w = min(max(new_w, lb), ub)
                        sub_weights[z, j] = new_w

                    relax_factor = relaxation_factors[z, c] * (1.0 / gamma_val) ** (
                        1.0 / importance
                    )
                    relaxation_factors[z, c] = min(relax_factor, MAX_RELAXATION_FACTOR)

        # Rescale
        zone_sums = np.sum(sub_weights, axis=0)
        for j in range(sample_count):
            scale = parent_weights[j] / zone_sums[j] if zone_sums[j] > 0 else 1.0
            for z in range(zone_count):
                sub_weights[z, j] *= scale

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
