import logging
import numpy as np
from numba import njit
from populationsim.balancer.constants import (
    MAX_DELTA,
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


# Original unoptimized Python code for single list balancing weights
def np_balancer_py(
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
    max_delta,
):

    # initial relaxation factors
    relaxation_factors = np.repeat(1.0, control_count)

    # Note: importance_adjustment must always be a float to ensure
    # correct "true division" in both Python 2 and 3
    importance_adjustment = 1.0

    # make a copy as we change this
    weights_final = weights_initial.copy()

    # array of control indexes for iterating over controls
    control_indexes = list(range(control_count))
    if master_control_index is not None:
        # reorder indexes so we handle master_control_index last
        control_indexes.append(control_indexes.pop(master_control_index))

    # precompute incidence squared
    incidence2 = incidence * incidence

    for iter in range(max_iterations):

        delta = 0.0  # float64
        weights_previous = weights_final.copy()

        # reset gamma every iteration
        gamma = np.repeat(1.0, control_count)

        # importance adjustment as number of iterations progress
        if iter > 0 and iter % IMPORTANCE_ADJUST_COUNT == 0:
            # always a float
            importance_adjustment = importance_adjustment / IMPORTANCE_ADJUST

        # for each control
        for c in control_indexes:

            xx = (weights_final * incidence[c]).sum()
            yy = (weights_final * incidence2[c]).sum()

            # adjust importance (unless this is master_control)
            if c == master_control_index:
                importance = controls_importance[c]
            else:
                importance = max(
                    controls_importance[c] * importance_adjustment, MIN_IMPORTANCE
                )

            # calculate constraint balancing factors, gamma
            if xx > 0:
                relaxed_constraint = controls_constraint[c] * relaxation_factors[c]
                relaxed_constraint = max(relaxed_constraint, MIN_CONTROL_VALUE)
                # ensure float division
                gamma[c] = 1.0 - (xx - relaxed_constraint) / (
                    yy + relaxed_constraint / float(importance)
                )

            # update HH weights
            weights_final *= pow(gamma[c], incidence[c])

            # clip weights to upper and lower bounds
            weights_final = np.clip(
                weights_final, weights_lower_bound, weights_upper_bound
            )

            relaxation_factors[c] *= pow(1.0 / gamma[c], 1.0 / importance)

            # clip relaxation_factors
            relaxation_factors = np.minimum(relaxation_factors, MAX_RELAXATION_FACTOR)

        max_gamma_dif = np.absolute(gamma - 1).max()

        # ensure float64 division
        delta = np.absolute(weights_final - weights_previous).sum() / float(
            sample_count
        )

        converged = delta < max_delta and max_gamma_dif < MAX_GAMMA
        no_progress = delta < ALT_MAX_DELTA

        if converged or no_progress:
            return weights_final, relaxation_factors, (True, iter, delta, max_gamma_dif)

    return (
        weights_final,
        relaxation_factors,
        (False, max_iterations, delta, max_gamma_dif),
    )


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
    max_iterations: int,
    max_delta: float,
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

        converged = delta < max_delta and max_gamma_dif < MAX_GAMMA
        no_progress = delta < ALT_MAX_DELTA

        if converged or no_progress:
            return weights_final, relaxation_factors, (True, iter, delta, max_gamma_dif)

    return (
        weights_final,
        relaxation_factors,
        (False, max_iterations, delta, max_gamma_dif),
    )


# single list balancing wrapper function
def np_balancer(
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
    use_numba,
    numba_precision="float32",
):
    """
    Wrapper function for balancing weights using either Numba or pure Python.

    Intent is to validate inputs and then call the appropriate implementation, which
    balances weights using the Newton-Raphson method with control relaxation.

    Parameters
    ----------
    sample_count : int
        Number of samples (households).
    control_count : int
        Number of controls (variables to balance).
    master_control_index : int
        Index of the master control variable, or -1 if there is no master control.
    incidence : np.ndarray
        Incidence matrix of shape (control_count, sample_count) where each row corresponds to a control variable and each column corresponds to a sample.
    weights_initial : np.ndarray
        Initial weights for each sample, shape (sample_count,).
    weights_lower_bound : np.ndarray
        Lower bounds for the weights, shape (sample_count,).
    weights_upper_bound : np.ndarray
        Upper bounds for the weights, shape (sample_count,).
    controls_constraint : np.ndarray
        Constraints for each control variable, shape (control_count,).
    controls_importance : np.ndarray
        Importance weights for each control variable, shape (control_count,).
    max_iterations : int
        Maximum number of iterations for the balancing algorithm.
    use_numba : bool
        Whether to use Numba for performance optimization.
        Numba is about ~4x faster than pure Python when using float64, and about ~11x faster when using float32.
    numba_precision : str
        Precision of the Numba calculations, either 'float64' or 'float32'. Default is 'float64'.

    Returns
    -------
    tuple
        A tuple containing:
        - weights: Final balanced weights for each sample.
        - relax: Relaxation factors for each control variable.
        - result: A dictionary with convergence status, iteration count, delta, and max gamma difference.
    """
    # Argument Validation
    if master_control_index is None:
        master_control_index = -1
    elif not (
        (master_control_index == -1) or (0 <= master_control_index < control_count)
    ):
        raise ValueError(
            f"master_control_index={master_control_index} is out of bounds"
        )

    if incidence.shape != (control_count, sample_count):
        raise ValueError(
            f"Expected incidence shape {(control_count, sample_count)}, got {incidence.shape}"
        )

    if weights_initial.shape[0] != sample_count:
        raise ValueError(f"weights_initial must have length {sample_count}")

    if controls_constraint.shape[0] != control_count:
        raise ValueError(f"controls_constraint must have length {control_count}")
    if controls_importance.shape[0] != control_count:
        raise ValueError(f"controls_importance must have length {control_count}")

    # Broadcast scalar bounds if needed
    if np.isscalar(weights_lower_bound) or weights_lower_bound.size == 1:
        weights_lower_bound = np.full(
            sample_count, weights_lower_bound, dtype=np.float64
        )
    if np.isscalar(weights_upper_bound) or weights_upper_bound.size == 1:
        weights_upper_bound = np.full(
            sample_count, weights_upper_bound, dtype=np.float64
        )

    # Decide whether to use Numba or not
    max_delta = MAX_DELTA

    if use_numba:
        _balancer = np_balancer_numba
        # Cast to float32 for performance and memory efficiency
        if numba_precision == "float32":
            incidence = incidence.astype(np.float32)
            weights_initial = weights_initial.astype(np.float32)
            weights_lower_bound = weights_lower_bound.astype(np.float32)
            weights_upper_bound = weights_upper_bound.astype(np.float32)
            controls_constraint = controls_constraint.astype(np.float32)
            controls_importance = controls_importance.astype(np.float32)
            max_delta = 1e-5
    else:
        _balancer = np_balancer_py

    logger.info("Balancing with Numba=%s, precision=%s", use_numba, numba_precision)

    # Send to Numba for processing
    weights, relax, result = _balancer(
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
        max_delta,
    )

    converged, iter_, delta, max_gamma_dif = result
    return (
        weights,
        relax,
        {
            "converged": converged,
            "iter": iter_,
            "delta": delta,
            "max_gamma_dif": max_gamma_dif,
        },
    )


# Original unoptimized Python code for simultaneous list balancing
def np_simul_balancer_py(
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
) -> tuple[np.ndarray, np.ndarray, tuple[bool, int, float, float]]:
    """
    Simultaneous balancer using only numpy (no pandas) data types.
    Separate function to ensure that no pandas data types leak in from object instance variables
    since they are often silently accepted as numpy arguments but slow things down
    """

    logger.debug(
        "np_simul_balancer sample_count %s control_count %s zone_count %s"
        % (sample_count, control_count, zone_count)
    )

    # Normalize sub_controls to match parent_controls if provided
    if parent_controls is not None:
        totals = sub_controls.sum(axis=0)
        scale_factors = np.ones_like(totals)
        valid = totals > 0
        scale_factors[valid] = parent_controls[valid] / totals[valid]
        sub_controls = sub_controls * scale_factors  # Safe broadcasting

    # initial relaxation factors
    relaxation_factors = np.ones((zone_count, control_count))

    # Note: importance_adjustment must always be a float to ensure
    # correct "true division" in both Python 2 and 3
    importance_adjustment = 1.0

    # FIXME - make a copy as we change this (not really necessary as caller doesn't use it...)
    sub_weights = sub_weights.copy()

    # array of control indexes for iterating over controls
    control_indexes = list(range(control_count))
    if master_control_index is not None:
        # reorder indexes so we handle master_control_index last
        control_indexes.append(control_indexes.pop(master_control_index))

    # precompute incidence squared
    incidence2 = incidence * incidence

    for iter in range(max_iterations):

        weights_previous = sub_weights.copy()

        # reset gamma every iteration
        gamma = np.ones((zone_count, control_count))

        # importance adjustment as number of iterations progress
        if iter > 0 and iter % IMPORTANCE_ADJUST_COUNT == 0:
            importance_adjustment = importance_adjustment / IMPORTANCE_ADJUST

        # for each control
        for c in control_indexes:

            # adjust importance (unless this is master_control)
            if c == master_control_index:
                importance = controls_importance[c]
            else:
                importance = max(
                    controls_importance[c] * importance_adjustment, MIN_IMPORTANCE
                )

            for z in range(zone_count):

                xx = (sub_weights[z] * incidence[c]).sum()

                # calculate constraint balancing factors, gamma
                if xx > 0:
                    yy = (sub_weights[z] * incidence2[c]).sum()

                    # calculate relaxed constraint, within bounds to avoid NaN
                    relaxed = sub_controls[z, c] * relaxation_factors[z, c]
                    relaxed = max(relaxed, MIN_CONTROL_VALUE)

                    # calculate gamma value, within bounds to avoid NaN
                    gamma[z, c] = 1.0 - (xx - relaxed) / (yy + (relaxed / importance))
                    gamma[z, c] = max(gamma[z, c], MIN_GAMMA)

                # update HH weights
                # sub_weights[z] *= pow(gamma[z, c], incidence[c])
                log_gamma = np.log(gamma[z, c])
                sub_weights[z] *= np.exp(log_gamma * incidence[c])

                # clip weights to upper and lower bounds
                sub_weights[z] = np.clip(
                    sub_weights[z], weights_lower_bound, weights_upper_bound
                )

                # relaxation_factors[z, c] *= pow(1.0 / gamma[z, c], 1.0 / importance)
                inv_log_gamma = np.log(1.0 / gamma[z, c])
                relaxation_factors[z, c] *= np.exp(inv_log_gamma / importance)

                # clip relaxation_factors
                relaxation_factors[z] = np.minimum(
                    relaxation_factors[z], MAX_RELAXATION_FACTOR
                )

        # FIXME - can't rescale weights and expect to converge

        # Rescale sub_weights so weight of each hh across sub zones sums to parent_weight
        # Create scaling vector with zero sums left as 1.0 to avoid division by zero
        zone_sums = np.sum(sub_weights, axis=0)
        scale = np.ones_like(zone_sums)
        valid = zone_sums > 0
        scale[valid] = parent_weights[valid] / zone_sums[valid]

        # Apply scaling to sub_weights
        sub_weights *= scale

        max_gamma_dif = np.absolute(gamma - 1).max()
        assert not np.isnan(max_gamma_dif)

        # ensure float division
        delta = np.absolute(sub_weights - weights_previous).sum() / float(sample_count)
        assert not np.isnan(delta)

        # standard convergence criteria
        converged = delta < MAX_DELTA and max_gamma_dif < MAX_GAMMA

        # even if not converged, no point in further iteration if weights aren't changing
        no_progress = delta < ALT_MAX_DELTA

        if (iter % 100) == 0:
            logger.debug(
                "np_simul_balancer iteration %s delta %s max_gamma_dif %s"
                % (iter, delta, max_gamma_dif)
            )

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
    max_iterations: int,
    max_delta: float,
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

        converged = delta < max_delta and max_gamma_dif < MAX_GAMMA
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
