
# PopulationSim
# See full license in LICENSE.txt.

from builtins import range
from builtins import object
import logging
import numpy as np

import pandas as pd

from activitysim.core.config import setting


logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 10000

MAX_GAP = 1.0e-9

IMPORTANCE_ADJUST = 2
IMPORTANCE_ADJUST_COUNT = 100
MIN_IMPORTANCE = 1.0
MAX_RELAXATION_FACTOR = 1000000
MIN_CONTROL_VALUE = 0.1
MAX_INT = (1 << 31)


class ListBalancer(object):
    """
    Single-geography list balancer using Newton-Raphson method with control relaxation.

    Takes a list of households with initial weights assigned to each household, and updates those
    weights in such a way as to match the marginal distribution of control variables while
    minimizing the change to the initial weights. Uses Newton-Raphson method with control
    relaxation.

    The resulting weights are float weights, so need to be integerized to integer household weights
    """

    def __init__(self,
                 incidence_table,
                 initial_weights,
                 control_totals,
                 control_importance_weights,
                 lb_weights,
                 ub_weights,
                 master_control_index,
                 max_iterations):
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

        self.max_iterations = max_iterations

        assert len(self.incidence_table.columns) == len(self.control_totals)
        assert len(self.incidence_table.columns) == len(self.control_importance_weights)

    def balance(self):

        assert len(self.incidence_table.columns) == len(self.control_totals)
        assert \
            np.isscalar(self.control_importance_weights) \
            or len(self.incidence_table.columns) == len(self.control_importance_weights)

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
        controls_constraint = \
            np.maximum(np.asanyarray(self.control_totals), MIN_CONTROL_VALUE)
        controls_importance = \
            np.maximum(np.asanyarray(self.control_importance_weights), MIN_IMPORTANCE)

        # balance
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
            self.max_iterations)

        # weights dataframe
        weights = pd.DataFrame(index=self.incidence_table.index)
        weights['initial'] = self.initial_weights
        weights['final'] = weights_final

        # controls dataframe
        controls = pd.DataFrame(index=self.incidence_table.columns.tolist())
        controls['control'] = np.maximum(self.control_totals, MIN_CONTROL_VALUE)
        controls['relaxation_factor'] = relaxation_factors
        controls['relaxed_control'] = controls.control * relaxation_factors
        controls['weight_totals'] = \
            [round((self.incidence_table.loc[:, c] * weights['final']).sum(), 2)
             for c in controls.index]

        return status, weights, controls


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
        max_iterations):

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
                importance = max(controls_importance[c] * importance_adjustment,
                                 MIN_IMPORTANCE)

            # calculate constraint balancing factors, gamma
            if xx > 0:
                relaxed_constraint = controls_constraint[c] * relaxation_factors[c]
                relaxed_constraint = max(relaxed_constraint, MIN_CONTROL_VALUE)
                # ensure float division
                gamma[c] = 1.0 - (xx - relaxed_constraint) / (
                    yy + relaxed_constraint / float(importance))

            # update HH weights
            weights_final *= pow(gamma[c], incidence[c])

            # clip weights to upper and lower bounds
            weights_final = np.clip(weights_final, weights_lower_bound, weights_upper_bound)

            relaxation_factors[c] *= pow(1.0 / gamma[c], 1.0 / importance)

            # clip relaxation_factors
            relaxation_factors = np.minimum(relaxation_factors, MAX_RELAXATION_FACTOR)

        max_gamma_dif = np.absolute(gamma - 1).max()

        # ensure float division
        delta = np.absolute(weights_final - weights_previous).sum() / float(sample_count)

        converged = delta < MAX_GAP and max_gamma_dif < MAX_GAP

        # logger.debug("iter %s delta %s max_gamma_dif %s" % (iter, delta, max_gamma_dif))

        if converged:
            break

    status = {
        'converged': converged,
        'iter': iter,
        'delta': delta,
        'max_gamma_dif': max_gamma_dif,
    }

    return weights_final, relaxation_factors, status


def do_balancing(control_spec,
                 total_hh_control_col,
                 max_expansion_factor, min_expansion_factor,
                 absolute_upper_bound, absolute_lower_bound,
                 incidence_df, control_totals, initial_weights):

    # incidence table should only have control columns
    incidence_df = incidence_df[control_spec.target]

    # master_control_index is total_hh_control_col
    if total_hh_control_col not in incidence_df.columns:
        raise RuntimeError("total_hh_control column '%s' not found in incidence table"
                           % total_hh_control_col)
    total_hh_control_index = incidence_df.columns.get_loc(total_hh_control_col)

    # control_totals series rows and incidence_df columns should be aligned
    assert total_hh_control_index == control_totals.index.get_loc(total_hh_control_col)

    control_totals = control_totals.values

    control_importance_weights = control_spec.importance

    if min_expansion_factor:

        # number_of_households in this seed geograpy as specified in seed_controls
        number_of_households = control_totals[total_hh_control_index]

        total_weights = initial_weights.sum()
        lb_ratio = min_expansion_factor * float(number_of_households) / float(total_weights)

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
        ub_ratio = max_expansion_factor * float(number_of_households) / float(total_weights)

        ub_weights = initial_weights * ub_ratio

        if absolute_upper_bound:
            ub_weights = ub_weights.round().clip(upper=absolute_upper_bound, lower=1).astype(int)
        else:
            ub_weights = ub_weights.round().clip(lower=1).astype(int)

    elif absolute_upper_bound:
        ub_weights = ub_weights.round().clip(upper=absolute_upper_bound, lower=1).astype(int)

    else:
        ub_weights = None

    max_iterations = setting('MAX_BALANCE_ITERATIONS_SEQUENTIAL', DEFAULT_MAX_ITERATIONS)

    balancer = ListBalancer(
        incidence_table=incidence_df,
        initial_weights=initial_weights,
        control_totals=control_totals,
        control_importance_weights=control_importance_weights,
        lb_weights=lb_weights,
        ub_weights=ub_weights,
        master_control_index=total_hh_control_index,
        max_iterations=max_iterations
    )

    status, weights, controls = balancer.balance()

    return status, weights, controls
