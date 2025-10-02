import logging
import numpy as np
from populationsim.balancing.constants import (
    DEFAULT_MAX_ITERATIONS,
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


# Original unoptimized balancer code
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
    max_iterations=DEFAULT_MAX_ITERATIONS,
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

        converged = delta < MAX_DELTA and max_gamma_dif < MAX_GAMMA
        no_progress = delta < ALT_MAX_DELTA

        if converged or no_progress:
            return weights_final, relaxation_factors, (True, iter, delta, max_gamma_dif)

    return (
        weights_final,
        relaxation_factors,
        (False, max_iterations, delta, max_gamma_dif),
    )


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
    max_iterations=DEFAULT_MAX_ITERATIONS,
    max_delta=MAX_DELTA,
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
        converged = delta < max_delta and max_gamma_dif < MAX_GAMMA

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
