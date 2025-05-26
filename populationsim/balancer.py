# PopulationSim
# See full license in LICENSE.txt.

import logging
import numpy as np
from numba import njit
import pandas as pd

from populationsim.core.config import setting


logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 10000

MAX_GAP = 1.0e-9

IMPORTANCE_ADJUST = 2
IMPORTANCE_ADJUST_COUNT = 100
MIN_IMPORTANCE = 1.0
MAX_RELAXATION_FACTOR = 1000000
MIN_CONTROL_VALUE = 0.1
MAX_INT = 1 << 31


class ListBalancer:
    """
    Single-geography list balancer using Newton-Raphson method with control relaxation.

    Takes a list of households with initial weights assigned to each household, and updates those
    weights in such a way as to match the marginal distribution of control variables while
    minimizing the change to the initial weights. Uses Newton-Raphson method with control
    relaxation.

    The resulting weights are float weights, so need to be integerized to integer household weights
    """

    def __init__(
        self,
        incidence_table,
        initial_weights,
        control_totals,
        control_importance_weights,
        lb_weights,
        ub_weights,
        master_control_index,
        max_iterations,
        use_numba,
    ):
        """
        Parameters
        ----------
        incidence_table : pandas DataFrame
            incidence table with only columns for controls to balance
        initial_weights : pandas Series
            initial weights of households in incidence table (in same order)
        control_totals : pandas Series or numpy array
            control totals (in same order as incidence_table columns)
        control_importance_weights : pandas Series
            importance weights of controls (in same order as incidence_table columns)
        lb_weights : pandas Series, numpy array, or scalar
            upper bound on balanced weights for hhs in incidence_table (in same order)
        ub_weights : pandas Series, numpy array, or scalar
            lower bound on balanced weights for hhs in incidence_table (in same order)
        master_control_index
            index of the total_hh_controsl column in controls (and incidence_table columns)
        """

        assert isinstance(incidence_table, pd.DataFrame)

        self.incidence_table = incidence_table

        assert len(initial_weights) == len(self.incidence_table.index)

        self.control_totals = control_totals
        self.initial_weights = initial_weights
        self.control_importance_weights = control_importance_weights
        self.lb_weights = lb_weights
        self.ub_weights = ub_weights
        self.master_control_index = master_control_index
        self.use_numba = use_numba
        self.max_iterations = max_iterations

        assert len(self.incidence_table.columns) == len(self.control_totals)
        assert len(self.incidence_table.columns) == len(self.control_importance_weights)

    def balance(self):

        assert len(self.incidence_table.columns) == len(self.control_totals)
        assert np.isscalar(self.control_importance_weights) or len(
            self.incidence_table.columns
        ) == len(self.control_importance_weights)

        # default values
        if self.control_importance_weights is None:
            self.control_importance_weights = min(1, MIN_IMPORTANCE)
        if self.lb_weights is None:
            self.lb_weights = 0.0
        if self.ub_weights is None:
            self.ub_weights = MAX_INT

        # prepare inputs as numpy (no pandas)
        sample_count = len(self.incidence_table.index)
        control_count = len(self.incidence_table.columns)
        master_control_index = self.master_control_index
        incidence = self.incidence_table.values.transpose()
        weights_initial = np.asanyarray(self.initial_weights).astype(np.float64)
        weights_lower_bound = np.asanyarray(self.lb_weights).astype(np.float64)
        weights_upper_bound = np.asanyarray(self.ub_weights).astype(np.float64)
        controls_constraint = np.maximum(
            np.asanyarray(self.control_totals), MIN_CONTROL_VALUE
        )
        controls_importance = np.maximum(
            np.asanyarray(self.control_importance_weights), MIN_IMPORTANCE
        )

        # balance | Default to numba
        weights_final, relaxation_factors, status = np_balancer(
            sample_count,
            control_count,
            master_control_index,
            incidence,
            weights_initial,
            weights_lower_bound,
            weights_upper_bound,
            controls_constraint,
            controls_importance,
            self.max_iterations,
            use_numba=self.use_numba,
        )

        # weights dataframe
        weights = pd.DataFrame(index=self.incidence_table.index)
        weights["initial"] = self.initial_weights
        weights["final"] = weights_final

        # controls dataframe
        controls = pd.DataFrame(index=self.incidence_table.columns.tolist())
        controls["control"] = np.maximum(self.control_totals, MIN_CONTROL_VALUE)
        controls["relaxation_factor"] = relaxation_factors
        controls["relaxed_control"] = controls.control * relaxation_factors
        controls["weight_totals"] = [
            round((self.incidence_table.loc[:, c] * weights["final"]).sum(), 2)
            for c in controls.index
        ]

        return status, weights, controls


def do_balancing(
    control_spec,
    total_hh_control_col,
    max_expansion_factor,
    min_expansion_factor,
    absolute_upper_bound,
    absolute_lower_bound,
    incidence_df,
    control_totals,
    initial_weights,
    use_hard_constraints,
    use_numba,
):

    # incidence table should only have control columns
    incidence_df = incidence_df[control_spec.target]

    # master_control_index is total_hh_control_col
    if total_hh_control_col not in incidence_df.columns:
        raise RuntimeError(
            "total_hh_control column '%s' not found in incidence table"
            % total_hh_control_col
        )
    total_hh_control_index = incidence_df.columns.get_loc(total_hh_control_col)

    # control_totals series rows and incidence_df columns should be aligned
    assert total_hh_control_index == control_totals.index.get_loc(total_hh_control_col)

    control_totals = control_totals.values

    control_importance_weights = control_spec.importance

    if min_expansion_factor:

        # number_of_households in this seed geograpy as specified in seed_controls
        number_of_households = control_totals[total_hh_control_index]

        total_weights = initial_weights.sum()
        lb_ratio = (
            min_expansion_factor * float(number_of_households) / float(total_weights)
        )

        # Added hard limit of min_expansion_factor value that would otherwise drift
        # due to the float(number_of_households) / float(total_weights) calculation
        if use_hard_constraints:
            lb_ratio = np.clip(lb_ratio, a_min=min_expansion_factor, a_max=None)

        lb_weights = initial_weights * lb_ratio

        if absolute_lower_bound:
            lb_weights = lb_weights.clip(lower=absolute_lower_bound)
        else:
            lb_weights = lb_weights.clip(lower=0)

    elif absolute_lower_bound:
        lb_weights = initial_weights.clip(lower=absolute_lower_bound)

    else:
        lb_weights = None

    if max_expansion_factor:

        # number_of_households in this seed geograpy as specified in seed_controlss
        number_of_households = control_totals[total_hh_control_index]

        total_weights = initial_weights.sum()
        ub_ratio = (
            max_expansion_factor * float(number_of_households) / float(total_weights)
        )

        # Added hard limit of max_expansion_factor value that would otherwise drift
        # due to the float(number_of_households) / float(total_weights) calculation
        if use_hard_constraints:
            ub_ratio = np.clip(ub_ratio, a_max=max_expansion_factor, a_min=None)

        ub_weights = initial_weights * ub_ratio

        if absolute_upper_bound:
            ub_weights = (
                ub_weights.round().clip(upper=absolute_upper_bound, lower=1).astype(int)
            )
        else:
            ub_weights = ub_weights.round().clip(lower=1).astype(int)

    elif absolute_upper_bound:
        ub_weights = (
            ub_weights.round().clip(upper=absolute_upper_bound, lower=1).astype(int)
        )

    else:
        ub_weights = None

    max_iterations = setting(
        "MAX_BALANCE_ITERATIONS_SEQUENTIAL", DEFAULT_MAX_ITERATIONS
    )

    balancer = ListBalancer(
        incidence_table=incidence_df,
        initial_weights=initial_weights,
        control_totals=control_totals,
        control_importance_weights=control_importance_weights,
        lb_weights=lb_weights,
        ub_weights=ub_weights,
        master_control_index=total_hh_control_index,
        max_iterations=max_iterations,
        use_numba=use_numba,
    )

    status, weights, controls = balancer.balance()

    return status, weights, controls


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

        weights_previous = weights_final.copy()

        # reset gamma every iteration
        gamma = np.array([1.0] * control_count)

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

        # ensure float division
        delta = np.absolute(weights_final - weights_previous).sum() / float(
            sample_count
        )

        converged = delta < MAX_GAP and max_gamma_dif < MAX_GAP

        # logger.debug("iter %s delta %s max_gamma_dif %s" % (iter, delta, max_gamma_dif))

        if converged:
            return weights_final, relaxation_factors, (True, iter, delta, max_gamma_dif)

    return (
        weights_final,
        relaxation_factors,
        (False, max_iterations, delta, max_gamma_dif),
    )


@njit(parallel=True, fastmath=True, cache=True)
def np_balancer_numba(
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
    """
    Numba-optimized version of the balancing algorithm.
    This function performs the balancing using the Newton-Raphson method with control relaxation.
    It iteratively adjusts the weights of samples to match the control totals while minimizing
    the change from the initial weights.
    Parameters
    ----------
    sample_count : int
        Number of samples (households).
    control_count : int
        Number of controls (variables to balance).
    master_control_index : int
        Index of the master control variable, or -1 if there is no master control.
        This is the index of the primary control variable (e.g., total_hh) and is placed last
        in the control indexes for processing to ensure it is handled last and given best chance of fitting.
    incidence : np.ndarray
        2D array of shape (control_count, sample_count) representing the incidence of controls in samples.
    weights_initial : np.ndarray
        1D array of shape (sample_count,) representing the initial weights of samples.
    weights_lower_bound : np.ndarray
        1D array of shape (sample_count,) representing the lower bounds for weights.
    weights_upper_bound : np.ndarray
        1D array of shape (sample_count,) representing the upper bounds for weights.
    controls_constraint : np.ndarray
        1D array of shape (control_count,) representing the constraints for each control.
    controls_importance : np.ndarray
        1D array of shape (control_count,) representing the importance weights for each control.
    max_iterations : int
        Maximum number of iterations to perform for balancing.
    Returns
    -------
    weights_final : np.ndarray
        1D array of shape (sample_count,) representing the final balanced weights.
    relaxation_factors : np.ndarray
        1D array of shape (control_count,) representing the relaxation factors for each control.
    status : tuple
        A tuple containing:
        - converged (bool): Whether the balancing converged.
        - iter (int): Number of iterations performed.
        - delta (float): Average change in weights across samples.
        - max_gamma_dif (float): Maximum difference in gamma values from 1.0.
    """
    # Initialization
    weights_final = weights_initial.copy()
    relaxation_factors = np.ones(control_count)
    incidence2 = incidence * incidence
    importance_adjustment = 1.0

    control_indexes = np.arange(control_count)
    if master_control_index >= 0:
        tmp = control_indexes[master_control_index]
        control_indexes = np.delete(control_indexes, master_control_index)
        control_indexes = np.append(control_indexes, tmp)

    for iter in range(max_iterations):
        delta = 0.0
        gamma = np.ones(control_count)

        if iter > 0 and iter % IMPORTANCE_ADJUST_COUNT == 0:
            importance_adjustment /= IMPORTANCE_ADJUST

        for i in range(control_count):
            c = control_indexes[i]

            xx = 0.0
            yy = 0.0
            for j in range(sample_count):
                w = weights_final[j]
                xx += w * incidence[c, j]
                yy += w * incidence2[c, j]

            importance = (
                controls_importance[c]
                if c == master_control_index
                else max(controls_importance[c] * importance_adjustment, MIN_IMPORTANCE)
            )

            if xx > 0.0:
                relaxed = controls_constraint[c] * relaxation_factors[c]
                if relaxed < MIN_CONTROL_VALUE:
                    relaxed = MIN_CONTROL_VALUE
                gamma_val = 1.0 - (xx - relaxed) / (yy + relaxed / importance)
                gamma[c] = gamma_val
                log_gamma = np.log(gamma_val)

                for j in range(sample_count):
                    new_w = weights_final[j] * np.exp(log_gamma * incidence[c, j])
                    # Fuse clip in-place
                    if new_w < weights_lower_bound[j]:
                        new_w = weights_lower_bound[j]
                    elif new_w > weights_upper_bound[j]:
                        new_w = weights_upper_bound[j]

                    # Track convergence delta (sum of abs diffs)
                    delta += abs(new_w - weights_final[j])
                    weights_final[j] = new_w

                # Update relaxation factor
                update = 1.0 / gamma_val
                relax_factor = relaxation_factors[c] * np.exp(
                    np.log(update) / importance
                )
                if relax_factor > MAX_RELAXATION_FACTOR:
                    relax_factor = MAX_RELAXATION_FACTOR
                relaxation_factors[c] = relax_factor

        delta /= sample_count
        max_gamma_dif = np.max(np.abs(gamma - 1.0))

        if delta < MAX_GAP and max_gamma_dif < MAX_GAP:
            return weights_final, relaxation_factors, (True, iter, delta, max_gamma_dif)

    return (
        weights_final,
        relaxation_factors,
        (False, max_iterations, delta, max_gamma_dif),
    )


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
):
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
    _balancer = np_balancer_numba if use_numba else np_balancer_py

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
