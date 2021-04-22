
# PopulationSim
# See full license in LICENSE.txt.

from builtins import range
from builtins import object
import logging
import os

import numpy as np
import pandas as pd


from activitysim.core.config import setting

from .lp import get_simul_integerizer
from .lp import STATUS_SUCCESS

from .integerizer import smart_round
from .integerizer import do_integerizing

logger = logging.getLogger(__name__)


def use_simul_integerizer():

    # use_simul_integerizer it if we can it unless told not to
    return setting('USE_SIMUL_INTEGERIZER', True)


class SimulIntegerizer(object):

    def __init__(self,
                 incidence_df,
                 sub_weights, sub_controls_df,
                 control_spec, total_hh_control_col,
                 trace_label=''
                 ):

        sample_count = len(sub_weights.index)
        sub_zone_count = len(sub_weights.columns)

        assert len(sub_weights.index) == sample_count
        assert len(incidence_df.index) == sample_count
        assert len(sub_controls_df.index) == sub_zone_count
        assert len(sub_weights.columns) == sub_zone_count
        assert total_hh_control_col in sub_controls_df.columns
        assert total_hh_control_col in incidence_df.columns
        assert (incidence_df.columns == control_spec.target).all()

        self.incidence_df = incidence_df
        self.sub_weights = sub_weights
        self.total_hh_control_col = total_hh_control_col

        # control spec rows and control_df columns should be same and in same order
        sub_countrol_cols = list(sub_controls_df.columns)
        sub_control_spec = control_spec[control_spec.target.isin(sub_countrol_cols)]
        self.sub_controls_df = sub_controls_df[sub_control_spec.target.values]
        self.sub_countrol_importance = sub_control_spec.importance

        # only care about parent control columns NOT in sub_controls
        # control spec rows and control_df columns should be same and in same order
        parent_control_spec = control_spec[~control_spec.target.isin(self.sub_controls_df.columns)]
        self.parent_countrol_cols = parent_control_spec.target.values
        self.parent_countrol_importance = parent_control_spec.importance

        assert total_hh_control_col not in self.parent_countrol_cols

        self.trace_label = trace_label

    def integerize(self):

        # - subzone

        total_hh_sub_control_index = \
            self.sub_controls_df.columns.get_loc(self.total_hh_control_col)

        # FIXME - shouldn't need this?
        total_hh_parent_control_index = -1

        sub_incidence = self.incidence_df[self.sub_controls_df.columns]
        sub_incidence = sub_incidence.values.astype(np.float64)

        sub_float_weights = self.sub_weights.values.transpose().astype(np.float64)
        sub_int_weights = sub_float_weights.astype(int)
        sub_resid_weights = sub_float_weights % 1.0

        # print "sub_float_weights\n", sub_float_weights
        # print "sub_int_weights\n", sub_int_weights
        # print "sub_resid_weights\n", sub_resid_weights

        sub_control_totals = np.asanyarray(self.sub_controls_df).astype(np.int)
        sub_countrol_importance = np.asanyarray(self.sub_countrol_importance).astype(np.float64)

        relaxed_sub_control_totals = np.dot(sub_float_weights, sub_incidence)

        # lp_right_hand_side
        lp_right_hand_side = \
            np.round(relaxed_sub_control_totals) - np.dot(sub_int_weights, sub_incidence)
        lp_right_hand_side = np.maximum(lp_right_hand_side, 0.0)

        # inequality constraint upper bounds
        sub_num_households = relaxed_sub_control_totals[:, (total_hh_sub_control_index,)]
        sub_max_control_values = np.amax(sub_incidence, axis=0) * sub_num_households
        relax_ge_upper_bound = np.maximum(sub_max_control_values - lp_right_hand_side, 0)
        hh_constraint_ge_bound = np.maximum(sub_max_control_values, lp_right_hand_side)

        # equality constraint for the total households control
        total_hh_right_hand_side = lp_right_hand_side[:, total_hh_sub_control_index]

        # - parent
        parent_incidence = self.incidence_df[self.parent_countrol_cols]
        parent_incidence = parent_incidence.values.astype(np.float64)

        # note:
        # sum(sub_int_weights) might be different from parent_float_weights.astype(int)
        # parent_resid_weights might be > 1.0, OK as we are using in objective, not for rounding
        parent_float_weights = np.sum(sub_float_weights, axis=0)
        parent_int_weights = np.sum(sub_int_weights, axis=0)
        parent_resid_weights = np.sum(sub_resid_weights, axis=0)

        # print "parent_float_weights\n", parent_float_weights
        # print "parent_int_weights\n", parent_int_weights
        # print "parent_resid_weights\n", parent_resid_weights

        # - parent control totals based on sub_zone balanced weights
        relaxed_parent_control_totals = np.dot(parent_float_weights, parent_incidence)

        parent_countrol_importance = \
            np.asanyarray(self.parent_countrol_importance).astype(np.float64)

        parent_lp_right_hand_side = \
            np.round(relaxed_parent_control_totals) - np.dot(parent_int_weights, parent_incidence)
        parent_lp_right_hand_side = np.maximum(parent_lp_right_hand_side, 0.0)

        # - create the inequality constraint upper bounds
        parent_num_households = np.sum(sub_num_households)

        parent_max_possible_control_values = \
            np.amax(parent_incidence, axis=0) * parent_num_households
        parent_relax_ge_upper_bound = \
            np.maximum(parent_max_possible_control_values - parent_lp_right_hand_side, 0)
        parent_hh_constraint_ge_bound = \
            np.maximum(parent_max_possible_control_values, parent_lp_right_hand_side)

        # how could this not be the case?
        if not (parent_hh_constraint_ge_bound == parent_max_possible_control_values).all():
            print("\nSimulIntegerizer integerizing", self.trace_label)
            logger.warning("parent_hh_constraint_ge_bound != parent_max_possible_control_values")
            logger.warning("parent_hh_constraint_ge_bound:      %s" %
                           parent_hh_constraint_ge_bound)
            logger.warning("parent_max_possible_control_values: %s" %
                           parent_max_possible_control_values)
            print("\n")
            # assert (parent_hh_constraint_ge_bound == parent_max_possible_control_values).all()

        integerizer_func = get_simul_integerizer()

        resid_weights_out, status_text = \
            integerizer_func(sub_int_weights,
                             parent_countrol_importance,
                             parent_relax_ge_upper_bound,
                             sub_countrol_importance,
                             sub_float_weights,
                             sub_resid_weights,
                             lp_right_hand_side,
                             parent_hh_constraint_ge_bound,
                             sub_incidence,
                             parent_incidence,
                             total_hh_right_hand_side,
                             relax_ge_upper_bound,
                             parent_lp_right_hand_side,
                             hh_constraint_ge_bound,
                             parent_resid_weights,
                             total_hh_sub_control_index,
                             total_hh_parent_control_index)

        # smart round resid_weights_out for each sub_zone
        total_household_controls = sub_control_totals[:, total_hh_sub_control_index].flatten()
        integerized_weights = np.empty_like(sub_int_weights)

        sub_zone_count = len(self.sub_weights.columns)
        for i in range(sub_zone_count):
            integerized_weights[i] = \
                smart_round(sub_int_weights[i], resid_weights_out[i], total_household_controls[i])

        # integerized_weights df: one column of integerized weights per sub_zone
        self.integerized_weights = pd.DataFrame(data=integerized_weights.T,
                                                columns=self.sub_weights.columns,
                                                index=self.incidence_df.index)

        return status_text


def try_simul_integerizing(
        trace_label,
        incidence_df,
        sub_weights, sub_controls_df,
        sub_geography,
        control_spec, total_hh_control_col,
        sub_control_zones):
    """
    Attempt simultaneous integerization and return integerized weights if successful

    Parameters
    ----------
    incidence_df
    sub_weights
    sub_controls_df
    sub_geography
    control_spec
    total_hh_control_col
    sub_control_zones

    Returns
    -------
    status : str
        str value of integerizer status from STATUS_TEXT dict
        integerization was successful if status in STATUS_SUCCESS list

    integerized_weights_df : pandas.DataFrame or None
        canonical form weight table, with columns for 'balanced_weight', 'integer_weight'
        or None if integerization failed
    """

    zero_weight_rows = sub_weights.sum(axis=1) == 0

    if zero_weight_rows.any():
        logger.info("omitting %s zero weight rows out of %s"
                    % (zero_weight_rows.sum(), len(incidence_df.index)))

    integerizer = SimulIntegerizer(
        incidence_df[~zero_weight_rows],
        sub_weights[~zero_weight_rows],
        sub_controls_df,
        control_spec,
        total_hh_control_col,
        trace_label
    )

    status = integerizer.integerize()

    if status not in STATUS_SUCCESS:
        return status, None

    if zero_weight_rows.any():
        # restore zero_weight_rows to integerized_weights
        logger.info("restoring %s zero weight rows" % zero_weight_rows.sum())
        integerized_weights = \
            pd.DataFrame(data=np.zeros(
                sub_weights.shape, dtype=np.int),
                columns=sub_weights.columns,
                index=sub_weights.index)
        integerized_weights.update(integerizer.integerized_weights)
    else:
        integerized_weights = integerizer.integerized_weights

    integerized_weights_df = \
        reshape_result(sub_weights, integerized_weights, sub_geography, sub_control_zones)

    logger.debug("SimulIntegerizer status %s" % (status,))

    return status, integerized_weights_df


def reshape_result(float_weights, integerized_weights, sub_geography, sub_control_zones):
    """
    Reshape results into unstacked form - (same as that returned by sequential integerizer)
    with columns for 'balanced_weight', 'integer_weight'
    plus columns for household id, and sub_geography zone ids

    Parameters
    ----------
    float_weights : pandas.DataFrame
        dataframe with one row per sample hh and one column per sub_zone
    integerized_weights : pandas.DataFrame
        dataframe with one row per sample hh and one column per sub_zone
    sub_geography : str
        name of sub_geography for result column name
    sub_control_zones : pandas.Series
        series mapping zone_id (index) to zone label (value)

    Returns
    -------
    integer_weights_df : pandas.DataFrame
        canonical form weight table, with columns for 'balanced_weight', 'integer_weight'
        plus columns for household id, and sub_geography zone ids
    """

    # integerize the sub_zone weights
    integer_weights_list = []
    for zone_id, zone_name in list(sub_control_zones.items()):

        weights = float_weights[zone_name]

        zone_weights_df = pd.DataFrame(index=list(range(0, len(integerized_weights.index))))
        zone_weights_df[weights.index.name] = float_weights.index
        zone_weights_df[sub_geography] = zone_id
        zone_weights_df['balanced_weight'] = float_weights[zone_name].values
        zone_weights_df['integer_weight'] = integerized_weights[zone_name].astype(int).values

        integer_weights_list.append(zone_weights_df)

    integer_weights_df = pd.concat(integer_weights_list)

    return integer_weights_df


def do_simul_integerizing(
        trace_label,
        incidence_df,
        sub_weights,
        sub_controls_df,
        control_spec,
        total_hh_control_col,
        sub_geography,
        sub_control_zones):
    """

    Wrapper around simultaneous integerizer to handle solver failure for infeasible subzones.

    Simultaneous integerize balanced float sub_weights,
    If simultaneous integerization fails, integerize serially to identify infeasible subzones,
    remove and smart_round infeasible subzones, and try simultaneous integerization again.
    (That ought to succeed, but if not, then fall back to all sequential integerization)
    Finally combine all results into a single result dataframe.

    Parameters
    ----------
    incidence_df : pandas.Dataframe
        full incidence_df for all hh samples in seed zone
    sub_zone_weights : pandas.DataFame
        balanced subzone household sample weights to integerize
    sub_controls_df : pandas.Dataframe
        sub_geography controls (one row per zone indexed by sub_zone id)
    control_spec : pandas.Dataframe
        full control spec with columns 'target', 'seed_table', 'importance', ...
    total_hh_control_col : str
        name of total_hh column (so we can preferentially match this control)
    sub_geography : str
        subzone geography name (e.g. 'TAZ')
    sub_control_zones : pandas.Series
        index is zone id and value is zone label (e.g. TAZ_101)
        for use in sub_controls_df column names

    Returns
    -------
    integer_weights_df : pandas.DataFrame
        canonical form weight table, with columns for 'balanced_weight', 'integer_weight'
        plus columns for household id, and sub_geography zone ids
    """

    # try simultaneous integerization of all subzones
    status,  integerized_weights_df = try_simul_integerizing(
        trace_label,
        incidence_df,
        sub_weights, sub_controls_df,
        sub_geography,
        control_spec, total_hh_control_col,
        sub_control_zones)

    if status in STATUS_SUCCESS:
        logger.info("do_simul_integerizing succeeded for %s status %s. " % (trace_label, status))
        return integerized_weights_df

    logger.warning("do_simul_integerizing failed for %s status %s. " % (trace_label, status))

    # if simultaneous integerization failed, sequentially integerize to detect infeasible subzones
    # infeasible zones will be smart rounded and returned in rounded_weights_df
    feasible_zone_ids, rounded_zone_ids, sequentially_integerized_weights_df, rounded_weights_df = \
        do_sequential_integerizing(
            trace_label,
            incidence_df,
            sub_weights, sub_controls_df,
            control_spec, total_hh_control_col,
            sub_control_zones,
            sub_geography,
            combine_results=False)

    if len(feasible_zone_ids) == 0:
        # if all subzones are infeasible, then we don't have any feasible zones to try
        # so the best we can do is return rounded_weights_df
        logger.warning("do_sequential_integerizing failed for all subzones %s. " % trace_label)
        logger.info("do_simul_integerizing returning smart rounded weights for %s."
                    % trace_label)
        return rounded_weights_df

    if len(rounded_zone_ids) == 0:
        # if all subzones are feasible, then there are no zones to remove in order to retry
        # so the best we can do is return sequentially_integerized_weights_df
        logger.warning("do_simul_integerizing failed but found no infeasible sub zones %s. "
                       % trace_label)
        logger.info("do_simul_integerizing falling back to sequential integerizing for %s."
                    % trace_label)
        return sequentially_integerized_weights_df

    if len(feasible_zone_ids) == 1:
        # if only one zone is feasible, not much point in simul_integerizing it
        # so the best we can do is return do_sequential_integerizing combined results
        logger.warning("do_simul_integerizing failed but found no infeasible sub zones %s. "
                       % trace_label)
        return pd.concat([sequentially_integerized_weights_df, rounded_weights_df])

    # - remove the infeasible subzones and retry simul_integerizing

    sub_controls_df = sub_controls_df.loc[feasible_zone_ids]
    sub_control_zones = sub_control_zones.loc[sub_control_zones.index.isin(feasible_zone_ids)]
    sub_weights = sub_weights[sub_control_zones]

    logger.info("do_simul_integerizing %s infeasable subzones for %s. "
                % (len(rounded_zone_ids), trace_label))

    status, integerized_weights_df = try_simul_integerizing(
        "retry_%s" % trace_label,
        incidence_df,
        sub_weights, sub_controls_df,
        sub_geography,
        control_spec, total_hh_control_col,
        sub_control_zones)

    if status in STATUS_SUCCESS:
        # we successfully simul_integerized the sequentially feasible sub zones, so we can
        # return the simul_integerized results along with the rounded_weights for the infeasibles
        logger.info("do_simul_integerizing retry succeeded for %s status %s. "
                    % (trace_label, status))
        return pd.concat([integerized_weights_df, rounded_weights_df])

    # haven't seen this happen, but I suppose it could...
    logger.error("do_simul_integerizing retry failed for %s status %s." % (trace_label, status))
    logger.info("do_simul_integerizing falling back to sequential integerizing for %s."
                % trace_label)

    # nothing to do but return do_sequential_integerizing combined results
    return pd.concat([sequentially_integerized_weights_df, rounded_weights_df])


def do_sequential_integerizing(
        trace_label,
        incidence_df,
        sub_weights, sub_controls_df,
        control_spec, total_hh_control_col,
        sub_control_zones,
        sub_geography,
        combine_results=True):
    """

    note: this method returns different results depending on the value of combine_results

    Parameters
    ----------
    incidence_df : pandas.Dataframe
        full incidence_df for all hh samples in seed zone
    sub_zone_weights : pandas.DataFame
        balanced subzone household sample weights to integerize
    sub_controls_df : pandas.Dataframe
        sub_geography controls (one row per zone indexed by sub_zone id)
    control_spec : pandas.Dataframe
        full control spec with columns 'target', 'seed_table', 'importance', ...
    total_hh_control_col : str
        name of total_hh column (so we can preferentially match this control)
    sub_geography : str
        subzone geography name (e.g. 'TAZ')
    sub_control_zones : pandas.Series
        series mapping zone_id (index) to zone label (value)
        for use in sub_controls_df column names
    combine_results : bool
        return all results in a single frame or return infeasible rounded results separately?
    Returns
    -------

    For combined results:

        integerized_weights_df : pandas.DataFrame
            canonical form weight table, with columns for 'balanced_weight', 'integer_weight'
            plus columns for household id, and sub_geography zone ids

    for segregated results:

        integerized_zone_ids : array(int)
            zone_ids of feasible (integerized) zones
        rounded_zone_ids : array(int)
            zone_ids of infeasible (rounded) zones
        integerized_weights_df : pandas.DataFrame or None if all zones infeasible
            integerized weights for feasible zones
        rounded_weights_df : pandas.DataFrame or None if all zones feasible
            rounded weights for infeasible aones

    Results dataframes are canonical form weight table,
    with columns for 'balanced_weight', 'integer_weight'
    plus columns for household id, and sub_geography zone ids

    """
    integerized_weights_list = []
    rounded_weights_list = []
    integerized_zone_ids = []
    rounded_zone_ids = []
    for zone_id, zone_name in list(sub_control_zones.items()):

        logger.info("sequential_integerizing zone_id %s zone_name %s" % (zone_id, zone_name))

        weights = sub_weights[zone_name]

        sub_trace_label = "%s_%s_%s" % (trace_label, sub_geography, zone_id)

        integer_weights, status = do_integerizing(
            trace_label=sub_trace_label,
            control_spec=control_spec,
            control_totals=sub_controls_df.loc[zone_id],
            incidence_table=incidence_df[control_spec.target],
            float_weights=weights,
            total_hh_control_col=total_hh_control_col
        )

        zone_weights_df = pd.DataFrame(index=list(range(0, len(integer_weights.index))))
        zone_weights_df[weights.index.name] = weights.index
        zone_weights_df[sub_geography] = zone_id
        zone_weights_df['balanced_weight'] = weights.values
        zone_weights_df['integer_weight'] = integer_weights.astype(int).values

        if status in STATUS_SUCCESS:
            integerized_weights_list.append(zone_weights_df)
            integerized_zone_ids.append(zone_id)
        else:
            rounded_weights_list.append(zone_weights_df)
            rounded_zone_ids.append(zone_id)

    if combine_results:
        integerized_weights_df = pd.concat(integerized_weights_list + rounded_weights_list)
        return integerized_weights_df

    integerized_weights_df = pd.concat(integerized_weights_list) if integerized_zone_ids else None
    rounded_weights_df = pd.concat(rounded_weights_list) if rounded_zone_ids else None

    return integerized_zone_ids, rounded_zone_ids, integerized_weights_df, rounded_weights_df


def do_no_integerizing(
        trace_label,
        incidence_df,
        sub_weights, sub_controls_df,
        control_spec, total_hh_control_col,
        sub_control_zones,
        sub_geography):
    """

    """
    integerized_weights_list = []
    rounded_weights_list = []
    integerized_zone_ids = []
    for zone_id, zone_name in sub_control_zones.iteritems():

        logger.info("sequential_integerizing zone_id %s zone_name %s" % (zone_id, zone_name))

        weights = sub_weights[zone_name]

        zone_weights_df = pd.DataFrame(index=range(0, len(weights.index)))
        zone_weights_df[weights.index.name] = weights.index
        zone_weights_df[sub_geography] = zone_id
        zone_weights_df['balanced_weight'] = weights.values

        integerized_weights_list.append(zone_weights_df)
        integerized_zone_ids.append(zone_id)

    integerized_weights_df = pd.concat(integerized_weights_list + rounded_weights_list)
    return integerized_weights_df


def multi_integerize(
        incidence_df,
        sub_zone_weights,
        sub_controls_df,
        control_spec,
        total_hh_control_col,
        parent_geography,
        parent_id,
        sub_geography,
        sub_control_zones):
    """

    Parameters
    ----------
    incidence_df : pandas.Dataframe
        full incidence_df for all hh samples in seed zone
    sub_zone_weights : pandas.DataFame
        balanced subzone household sample weights to integerize
    sub_controls_df : pandas.Dataframe
        sub_geography controls (one row per zone indexed by sub_zone id)
    control_spec : pandas.Dataframe
        full control spec with columns 'target', 'seed_table', 'importance', ...
    total_hh_control_col : str
        name of total_hh column (so we can preferentially match this control)
    parent_geography : str
        parent geography zone name
    parent_id : int
        parent geography zone id
    sub_geography : str
        subzone geography name (e.g. 'TAZ')
    sub_control_zones : pandas.Series
        index is zone id and value is zone label (e.g. TAZ_101)
        for use in sub_controls_df column names

    Returns
    -------
    integer_weights_df : pandas.DataFrame
        canonical form weight table, with columns for 'balanced_weight', 'integer_weight'
        plus columns for household id, parent and sub_geography zone ids
    """

    trace_label = "%s_%s" % (parent_geography, parent_id)

    if setting('NO_INTEGERIZATION_EVER', False):
        integerizer = do_no_integerizing
    elif use_simul_integerizer():
        integerizer = do_simul_integerizing
    else:
        integerizer = do_sequential_integerizing

    integer_weights_df = integerizer(
        trace_label=trace_label,
        incidence_df=incidence_df,
        sub_weights=sub_zone_weights,
        sub_controls_df=sub_controls_df,
        control_spec=control_spec,
        total_hh_control_col=total_hh_control_col,
        sub_geography=sub_geography,
        sub_control_zones=sub_control_zones,
    )

    return integer_weights_df
