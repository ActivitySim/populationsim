
# PopulationSim
# See full license in LICENSE.txt.

from builtins import object
import logging

import os

import numpy as np
import pandas as pd
from activitysim.core.config import setting

from .lp import get_single_integerizer
from .lp import STATUS_SUCCESS
from .lp import STATUS_OPTIMAL


logger = logging.getLogger(__name__)


def smart_round(int_weights, resid_weights, target_sum):
    """
    Round weights while ensuring (as far as possible that result sums to target_sum)

    Parameters
    ----------
    int_weights : numpy.ndarray(int)
    resid_weights : numpy.ndarray(float)
    target_sum : int

    Returns
    -------
    rounded_weights : numpy.ndarray array of ints
    """
    assert len(int_weights) == len(resid_weights)
    assert (int_weights == int_weights.astype(int)).all()
    assert target_sum == int(target_sum)

    target_sum = int(target_sum)

    # integer part of numbers to round (astype both copies and coerces)
    rounded_weights = int_weights.astype(int)

    # find number of residuals that we need to round up
    int_shortfall = target_sum - rounded_weights.sum()

    # clip to feasible, in case target was not achievable by rounding
    int_shortfall = np.clip(int_shortfall, 0, len(resid_weights))

    # Order the residual weights and round at the tipping point where target_sum is achieved
    if int_shortfall > 0:
        # indices of the int_shortfall highest resid_weights
        i = np.argsort(resid_weights)[-int_shortfall:]

        # add 1 to the integer weights that we want to round upwards
        rounded_weights[i] += 1

    return rounded_weights


class Integerizer(object):
    def __init__(self,
                 incidence_table,
                 control_importance_weights,
                 float_weights,
                 relaxed_control_totals,
                 total_hh_control_value,
                 total_hh_control_index,
                 control_is_hh_based,
                 trace_label=''):
        """

        Parameters
        ----------
        control_totals : pandas.Series
            targeted control totals (either explict or backstopped) we are trying to hit
        incidence_table : pandas.Dataframe
            incidence table with columns only for targeted controls
        control_importance_weights : pandas.Series
            importance weights (from control_spec) of targeted controls
        float_weights
            blanaced float weights to integerize
        relaxed_control_totals
        total_hh_control_index : int
        control_is_hh_based : bool
        """

        self.incidence_table = incidence_table
        self.control_importance_weights = control_importance_weights
        self.float_weights = float_weights
        self.relaxed_control_totals = relaxed_control_totals

        self.total_hh_control_value = total_hh_control_value
        self.total_hh_control_index = total_hh_control_index
        self.control_is_hh_based = control_is_hh_based

        self.trace_label = trace_label

    def integerize(self):

        sample_count = len(self.incidence_table.index)
        control_count = len(self.incidence_table.columns)

        incidence = self.incidence_table.values.transpose().astype(np.float64)
        float_weights = np.asanyarray(self.float_weights).astype(np.float64)
        relaxed_control_totals = np.asanyarray(self.relaxed_control_totals).astype(np.float64)
        control_is_hh_based = np.asanyarray(self.control_is_hh_based).astype(bool)
        control_importance_weights = \
            np.asanyarray(self.control_importance_weights).astype(np.float64)

        assert len(float_weights) == sample_count
        assert len(relaxed_control_totals) == control_count
        assert len(control_is_hh_based) == control_count
        assert len(self.incidence_table.columns) == control_count
        assert (relaxed_control_totals == np.round(relaxed_control_totals)).all()
        assert not np.isnan(incidence).any()
        assert not np.isnan(float_weights).any()
        assert (incidence[self.total_hh_control_index] == 1).all()

        int_weights = float_weights.astype(int)
        resid_weights = float_weights % 1.0

        if (resid_weights == 0.0).all():
            # not sure this matters...
            logger.info("Integerizer: all %s resid_weights zero. Returning success." %
                        ((resid_weights == 0).sum(), ))

            integerized_weights = int_weights
            status = STATUS_OPTIMAL

        else:

            # - lp_right_hand_side - relaxed_control_shortfall
            lp_right_hand_side = relaxed_control_totals - np.dot(int_weights, incidence.T)
            lp_right_hand_side = np.maximum(lp_right_hand_side, 0.0)

            # - max_incidence_value of each control
            max_incidence_value = np.amax(incidence, axis=1)
            assert (max_incidence_value[control_is_hh_based] <= 1).all()

            # - create the inequality constraint upper bounds
            num_households = relaxed_control_totals[self.total_hh_control_index]
            relax_ge_upper_bound = \
                np.maximum(max_incidence_value * num_households - lp_right_hand_side, 0)
            hh_constraint_ge_bound = \
                np.maximum(self.total_hh_control_value * max_incidence_value, lp_right_hand_side)

            # popsim3 does does something rather peculiar, which I am not sure is right
            # it applies a huge penalty to rounding a near-zero residual upwards
            # the documentation justifying this is sparse and possibly confused:
            # // Set objective: min sum{c(n)*x(n)} + 999*y(i) - 999*z(i)}
            # objective_function_coefficients = -1.0 * np.log(resid_weights)
            # objective_function_coefficients[(resid_weights <= np.exp(-999))] = 999
            # We opt for an alternate interpretation of what they meant to do: avoid log overflow
            # There is not much difference in effect...
            LOG_OVERFLOW = -725
            log_resid_weights = np.log(np.maximum(resid_weights, np.exp(LOG_OVERFLOW)))
            assert not np.isnan(log_resid_weights).any()

            if (float_weights == 0).any():
                # not sure this matters...
                logger.warning("Integerizer: %s zero weights out of %s" %
                               ((float_weights == 0).sum(), sample_count))
                assert False

            if (resid_weights == 0.0).any():
                # not sure this matters...
                logger.info("Integerizer: %s zero resid_weights out of %s" %
                            ((resid_weights == 0).sum(), sample_count))
                # assert False

            integerizer_func = get_single_integerizer()

            resid_weights, status = integerizer_func(
                incidence=incidence,
                resid_weights=resid_weights,
                log_resid_weights=log_resid_weights,
                control_importance_weights=control_importance_weights,
                total_hh_control_index=self.total_hh_control_index,
                lp_right_hand_side=lp_right_hand_side,
                relax_ge_upper_bound=relax_ge_upper_bound,
                hh_constraint_ge_bound=hh_constraint_ge_bound
            )

            integerized_weights = \
                smart_round(int_weights, resid_weights, self.total_hh_control_value)

        self.weights = pd.DataFrame(index=self.incidence_table.index)
        self.weights['integerized_weight'] = integerized_weights

        delta = (integerized_weights != np.round(float_weights)).sum()
        logger.debug("Integerizer: %s out of %s different from round" % (delta, len(float_weights)))

        return status


def do_integerizing(
        trace_label,
        control_spec,
        control_totals,
        incidence_table,
        float_weights,
        total_hh_control_col):
    """

    Parameters
    ----------
    trace_label : str
        trace label indicating geography zone being integerized (e.g. PUMA_600)
    control_spec : pandas.Dataframe
        full control spec with columns 'target', 'seed_table', 'importance', ...
    control_totals : pandas.Series
        control totals explicitly specified for this zone
    incidence_table : pandas.Dataframe
    float_weights : pandas.Series
        balanced float weights to integerize
    total_hh_control_col : str
        name of total_hh column (preferentially constrain to match this control)

    Returns
    -------
    integerized_weights : pandas.Series
    status : str
        as defined in integerizer.STATUS_TEXT and STATUS_SUCCESS
    """

    # incidence table should only have control columns
    incidence_table = incidence_table[control_spec.target]

    if total_hh_control_col not in incidence_table.columns:
        raise RuntimeError("total_hh_control column '%s' not found in incidence table"
                           % total_hh_control_col)

    zero_weight_rows = (float_weights == 0)
    if zero_weight_rows.any():
        logger.debug("omitting %s zero weight rows out of %s"
                     % (zero_weight_rows.sum(), len(incidence_table.index)))
        incidence_table = incidence_table[~zero_weight_rows]
        float_weights = float_weights[~zero_weight_rows]

    total_hh_control_value = control_totals[total_hh_control_col]

    status = None
    if setting('INTEGERIZE_WITH_BACKSTOPPED_CONTROLS') \
            and len(control_totals) < len(incidence_table.columns):

        ##########################################
        # - backstopped control_totals
        # Use balanced float weights to establish target values for all control values
        # note: this more frequently results in infeasible solver results
        ##########################################

        relaxed_control_totals = \
            np.round(np.dot(np.asanyarray(float_weights), incidence_table.values))
        relaxed_control_totals = \
            pd.Series(relaxed_control_totals, index=incidence_table.columns.values)

        # if the incidence table has only one record, then the final integer weights
        # should be just an array with 1 element equal to the total number of households;
        assert len(incidence_table.index) > 1

        integerizer = Integerizer(
            incidence_table=incidence_table,
            control_importance_weights=control_spec.importance,
            float_weights=float_weights,
            relaxed_control_totals=relaxed_control_totals,
            total_hh_control_value=total_hh_control_value,
            total_hh_control_index=incidence_table.columns.get_loc(total_hh_control_col),
            control_is_hh_based=control_spec['seed_table'] == 'households',
            trace_label='backstopped_%s' % trace_label
        )

        # otherwise, solve for the integer weights using the Mixed Integer Programming solver.
        status = integerizer.integerize()

        logger.debug("Integerizer status for backstopped %s: %s" % (trace_label, status))

    # if we either tried backstopped controls or failed, or never tried at all
    if status not in STATUS_SUCCESS:

        ##########################################
        # - unbackstopped partial control_totals
        # Use balanced weights to establish control totals only for explicitly specified controls
        # note: this usually results in feasible solver results, except for some single hh zones
        ##########################################

        balanced_control_cols = control_totals.index
        incidence_table = incidence_table[balanced_control_cols]
        control_spec = control_spec[control_spec.target.isin(balanced_control_cols)]

        relaxed_control_totals = \
            np.round(np.dot(np.asanyarray(float_weights), incidence_table.values))
        relaxed_control_totals = \
            pd.Series(relaxed_control_totals, index=incidence_table.columns.values)

        integerizer = Integerizer(
            incidence_table=incidence_table,
            control_importance_weights=control_spec.importance,
            float_weights=float_weights,
            relaxed_control_totals=relaxed_control_totals,
            total_hh_control_value=total_hh_control_value,
            total_hh_control_index=incidence_table.columns.get_loc(total_hh_control_col),
            control_is_hh_based=control_spec['seed_table'] == 'households',
            trace_label=trace_label
        )

        status = integerizer.integerize()

        logger.debug("Integerizer status for unbackstopped %s: %s" % (trace_label, status))

    if status not in STATUS_SUCCESS:
        logger.error("Integerizer failed for %s status %s. "
                     "Returning smart-rounded original weights" % (trace_label, status))
    elif status != 'OPTIMAL':
        logger.warning("Integerizer status non-optimal for %s status %s." % (trace_label, status))

    integerized_weights = pd.Series(0, index=zero_weight_rows.index)
    integerized_weights.update(integerizer.weights['integerized_weight'])
    return integerized_weights, status
